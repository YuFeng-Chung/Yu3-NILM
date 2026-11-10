"""Evaluate fixed E5 checkpoints on validated waveform corruptions; no retraining."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_e5_ablation import Ablation, metrics
from prepare_e5_corruptions import CONDITIONS, CORRUPTION_SEEDS


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--models', type=Path, required=True)
    p.add_argument('--tests', type=Path, required=True); p.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    p.add_argument('--wait-for-models', action='store_true')
    args = p.parse_args(); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = json.loads((args.models / 'config.json').read_text())
    test_config = json.loads((args.tests / 'corruption_config.json').read_text())
    assert config['seeds'] == list(range(42, 52))
    assert test_config['clean_images_exactly_match_supplied_test']
    cache = np.load(args.tests / 'corrupted_images.npz'); y = cache['labels']
    data = {name: torch.from_numpy(cache[name].astype(np.float32) / 255.)
            for name in cache.files if name == 'clean_images' or '_seed' in name}
    destination = args.models / 'corrupted_evaluation'; destination.mkdir(exist_ok=True)
    eval_config = {'conditions': [c for c, _ in CONDITIONS], 'corruption_seeds': CORRUPTION_SEEDS,
        'model_config_sha256': hashlib.sha256((args.models / 'config.json').read_bytes()).hexdigest(),
        'test_config_sha256': hashlib.sha256((args.tests / 'corruption_config.json').read_bytes()).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'device': args.device,
        'n_training_seeds': 10, 'normalization': 'E5 per-waveform; not Exp4 normalization',
        'checkpoint_policy': 'Evaluate fixed19-epoch checkpoints; no tuning or selection.'}
    path = destination / 'config.json'
    if path.exists(): assert json.loads(path.read_text()) == eval_config
    else: path.write_text(json.dumps(eval_config, indent=2))
    all_rows, averaged = [], []
    for seed in config['seeds']:
        for variant in config['variants']:
            out = destination / f'{variant}_seed{seed}'; out.mkdir(exist_ok=True)
            finished = out / 'averaged_metrics.json'
            if finished.exists():
                averaged.extend(json.loads(finished.read_text()))
                all_rows.extend(json.loads((out / 'per_corruption_seed_metrics.json').read_text())); continue
            source = args.models / f'{variant}_seed{seed}'
            waited_from = time.monotonic()
            while not (source / 'metrics.json').exists():
                if not args.wait_for_models: raise FileNotFoundError(f'Model training not complete: {source}')
                if time.monotonic() - waited_from > 7200: raise TimeoutError(f'Model wait exceeded2h: {source}')
                time.sleep(5)
            checkpoint = torch.load(source / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert checkpoint['variant'] == variant and checkpoint['seed'] == seed and checkpoint['epoch'] == 19
            model = Ablation(variant).to(args.device); model.load_state_dict(checkpoint['state_dict']); model.eval()

            def predict(images):
                first, last = [], []
                with torch.no_grad():
                    for x in images.split(32):
                        logits, _ = model(x.to(args.device))
                        first.extend(logits[0].argmax(1).cpu().tolist()); last.extend(logits[-1].argmax(1).cpu().tolist())
                return np.asarray(first), np.asarray(last)

            clean_first, clean_pred = predict(data['clean_images'])
            with (source / 'pred_clean.csv').open() as f:
                old = {r['sample_id']: int(r['y_pred']) for r in csv.DictReader(f)}
            assert all(old[str(path)] == int(pred) for path, pred in zip(cache['original_png_paths'], clean_pred)), 'Clean prediction mismatch'
            clean = metrics(y, clean_pred)
            assert abs(clean['macro_f1'] - json.loads((source / 'metrics.json').read_text())['macro_f1']) < 1e-12
            run_rows, run_average = [], []
            for condition, _ in CONDITIONS:
                predictions = []; condition_rows = []
                for corruption_seed in CORRUPTION_SEEDS:
                    first, pred = predict(data[f'{condition}_seed{corruption_seed}'])
                    m = metrics(y, pred)
                    row = {'variant': variant, 'training_seed': seed, 'condition': condition,
                           'corruption_seed': corruption_seed,
                           **{k: v for k, v in m.items() if k != 'class_f1'}}
                    condition_rows.append(row)
                    predictions.extend({'sample_id': str(sid), 'original_png_path': str(path),
                        'training_seed': seed, 'corruption_seed': corruption_seed, 'y_true': int(label),
                        'first_pass_pred': int(p1), 'y_pred': int(p2)} for sid, path, label, p1, p2 in
                        zip(cache['sample_ids'], cache['original_png_paths'], y, first, pred))
                write_csv(out / f'pred_{condition}.csv', predictions)
                run_rows.extend(condition_rows)
                mean_f1 = float(np.mean([r['macro_f1'] for r in condition_rows]))
                drop = clean['macro_f1'] - mean_f1
                run_average.append({'variant': variant, 'training_seed': seed, 'condition': condition,
                    'clean_macro_f1': clean['macro_f1'], 'corrupted_macro_f1': mean_f1,
                    'absolute_drop_pp': drop * 100,
                    'relative_drop_fraction': drop / clean['macro_f1'] if clean['macro_f1'] > 1e-12 else None,
                    'n_corruption_seeds': len(condition_rows)})
            (out / 'per_corruption_seed_metrics.json').write_text(json.dumps(run_rows, indent=2))
            (out / 'clean_prediction_check.json').write_text(json.dumps({'all219match': True,
                'checkpoint_sha256': hashlib.sha256((source / 'checkpoint.pt').read_bytes()).hexdigest()}))
            finished.write_text(json.dumps(run_average, indent=2))
            all_rows.extend(run_rows); averaged.extend(run_average)
            write_csv(destination / 'results_per_corruption_seed.csv', all_rows)
            write_csv(destination / 'results_averaged.csv', averaged)
            print(f'PASS {variant} seed{seed}: clean predictions identical; two conditions x10 seeds evaluated.', flush=True)
            del model
            if args.device == 'mps': torch.mps.empty_cache()
    assert len(all_rows) == 2200 and len(averaged) == 220
    print('COMPLETE 110 models, 2200 corruption evaluations, n=10 training seeds per comparison.', flush=True)


if __name__ == '__main__': main()
