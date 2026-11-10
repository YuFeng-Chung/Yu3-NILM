"""Recompute all per-corruption metrics from predictions and audit seed averaging."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_e5_corruptions import CONDITIONS, CORRUPTION_SEEDS
from run_e5_ablation import metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); folder = args.output / 'corrupted_evaluation'
    config = json.loads((args.output / 'config.json').read_text())
    cache = np.load(args.output / 'waveform_tests/corrupted_images.npz')
    expected = {str(sid): (str(path), int(label)) for sid, path, label in
                zip(cache['sample_ids'], cache['original_png_paths'], cache['labels'])}
    assert len(expected) == 219
    total_rows, total_conditions, models = 0, 0, 0
    for seed in config['seeds']:
        for variant in config['variants']:
            run = folder / f'{variant}_seed{seed}'
            assert json.loads((run / 'clean_prediction_check.json').read_text())['all219match']
            saved = json.loads((run / 'per_corruption_seed_metrics.json').read_text())
            averaged = json.loads((run / 'averaged_metrics.json').read_text())
            for condition, _ in CONDITIONS:
                with (run / f'pred_{condition}.csv').open() as f: rows = list(csv.DictReader(f))
                assert len(rows) == 2190
                per_seed = []
                for corruption_seed in CORRUPTION_SEEDS:
                    selected = [r for r in rows if int(r['corruption_seed']) == corruption_seed]
                    assert len(selected) == 219 and len({r['sample_id'] for r in selected}) == 219
                    assert {r['sample_id'] for r in selected} == set(expected)
                    for r in selected:
                        assert int(r['training_seed']) == seed
                        assert expected[r['sample_id']] == (r['original_png_path'], int(r['y_true']))
                    y = np.array([int(r['y_true']) for r in selected]); pred = [int(r['y_pred']) for r in selected]
                    actual = metrics(y, pred)
                    target = next(r for r in saved if r['condition'] == condition and r['corruption_seed'] == corruption_seed)
                    for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
                        assert abs(actual[key] - target[key]) < 1e-12
                    per_seed.append(actual['macro_f1']); total_conditions += 1
                aggregate = next(r for r in averaged if r['condition'] == condition)
                assert aggregate['n_corruption_seeds'] == 10
                assert abs(aggregate['corrupted_macro_f1'] - np.mean(per_seed)) < 1e-12
                assert abs(aggregate['absolute_drop_pp'] - (aggregate['clean_macro_f1'] - np.mean(per_seed)) * 100) < 1e-12
                total_rows += len(rows)
            models += 1
    assert models == 110 and total_conditions == 2200 and total_rows == 481800
    audit = {'all_checks_passed': True, 'models': models, 'per_corruption_evaluations': total_conditions,
        'prediction_rows': total_rows, 'replicate_unit': '10 paired training seeds; corruption metrics averaged first',
        'checks': ['all219clean predictions matched for every checkpoint', 'allraw sample IDs and labels match cache',
                   'all2200 metrics recomputed from predictions', 'all220 model-condition averages and drops verified']}
    (folder / 'completed_audit.json').write_text(json.dumps(audit, indent=2))
    print('PASS 110 models, 2200 metric rows, 481800 predictions, all seed averages and degradation values.')


if __name__ == '__main__': main()
