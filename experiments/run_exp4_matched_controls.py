"""Supplementary Exp4 matched-budget controls; never select models on held-out data.

Uses audited raw windows and supplied Exp4 functions via AST (no Windows paths).
RecAux and no-feedback retain Yu3's two-channel encoder with a zero second input.
They are E5-matched controls, not the proposed one-channel M1/RecAux experiment.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from run_e5_ablation import Ablation, loss_function, metrics, seed_all

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '實驗四'
DATA = ROOT / 'data/plaid_2014_published'
VARIANTS = ['err_detach', 'rec_aux', 'no_feedback']
STRATEGIES = {'no_defense': (23, 0.), 'S3': (81, 0.), 'combined': (51, 1e-4)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def load_functions():
    ns = {'np': np}
    names = {'_zscore_fixed', '_scatter_vi_image', 'generate_vi_image_fixed_norm',
             'add_awgn_waveform', 'add_salt_pepper_waveform', '_interpolate_missing',
             'add_missing_random_waveform', 'add_missing_consecutive_waveform',
             'apply_waveform_corruption', 'CORRUPTION_FUNCS', 'SEVERITY_GRIDS'}
    for filename, selected in [('exp4_raw_data_V2.py', names),
            ('exp4_defense_strategies_V2.py', {'NOISE_AUGMENT_CONFIG',
             'MISSING_AUGMENT_CONFIG', 'dynamic_noise_augment', 'dynamic_missing_augment'})]:
        tree = ast.parse((SOURCE / filename).read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if
                 isinstance(n, ast.FunctionDef) and n.name in selected or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in selected for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / filename), 'exec'), ns)
    return ns


FNS = load_functions()


def rng_for(*keys):
    value = int.from_bytes(hashlib.sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'little')
    return np.random.default_rng(value)


def scatter_fast(c, v, stats):
    """Exact float32 repeated accumulation, checked against supplied Python loop."""
    c = FNS['_zscore_fixed'](c, stats['i_mean'], stats['i_std'])
    v = FNS['_zscore_fixed'](v, stats['v_mean'], stats['v_std'])
    iy = np.clip(((c + 3) / 6 * 63).astype(int), 0, 63)
    ix = np.clip(((v + 3) / 6 * 63).astype(int), 0, 63)
    out = np.zeros((64, 64), dtype=np.float32)
    np.add.at(out, (iy, ix), np.float32(.1))
    return np.clip(out, 0, 1)[None]


def images(raw, indices, stats, strategy='no_defense', seed=0, epoch=0, condition=None):
    result = []
    for idx in indices:
        c, v = raw['current'][idx].astype(np.float32), raw['voltage'][idx].astype(np.float32)
        sid = str(raw['sample_ids'][idx])
        if condition:
            typ, severity, pollution_seed = condition
            c, v = FNS['apply_waveform_corruption'](c, v, typ, severity, stats,
                                                   rng_for('test', sid, typ, severity, pollution_seed))
        elif strategy != 'no_defense':
            fn = 'dynamic_noise_augment' if strategy == 'S3' else 'dynamic_missing_augment'
            c, v = FNS[fn](c, v, stats, rng_for('train', sid, strategy, seed, epoch))
        result.append(scatter_fast(c, v, stats))
    return np.stack(result)


def predict(model, array, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x.to(device))[0][-1].argmax(1).cpu().numpy()
                               for x in torch.from_numpy(array).split(32)])


def smoke(raw, train, stats, device):
    for idx in train[:30]:
        c, v = raw['current'][idx].astype(np.float32), raw['voltage'][idx].astype(np.float32)
        for typ, severity in [('clean', None), ('awgn', 10), ('salt_pepper', .2), ('missing_consecutive', .2)]:
            a, b = FNS['apply_waveform_corruption'](c, v, typ, severity, stats, rng_for(idx, typ))
            assert np.array_equal(scatter_fast(a, b, stats), FNS['generate_vi_image_fixed_norm'](a, b, stats))
    x = torch.from_numpy(images(raw, train[:4], stats)).to(device)
    y = torch.from_numpy(raw['labels'][train[:4]]).long().to(device)
    from run_e5_ablation import original
    seed_all(42)
    original_model = original.M5_Model().to(device).eval()
    matched = Ablation('err_detach').to(device).eval()
    matched.load_state_dict(original_model.state_dict())
    with torch.no_grad():
        a, b = original_model(x), matched(x)
        for left, right in zip(a, [b[0][0], b[0][1], b[1][0]]):
            assert torch.equal(left, right)
    for variant in VARIANTS:
        model = Ablation(variant).to(device).train()
        loss = loss_function(model(x), x, y, variant)
        loss.backward()
        assert torch.isfinite(loss)
        assert int(model.encoder.bn1.num_batches_tracked) == (2 if variant == 'err_detach' else 1)
        assert (model.dec_conv1.weight.grad is None) == (variant == 'no_feedback')
    print('PASS smoke: 120 exact image comparisons, original forward equivalence, gradients and BN counts.', flush=True)


def protocol(args, raw, stats):
    return {'study': 'Exp4 supplementary matched-budget mechanism controls',
        'variants': VARIANTS, 'strategies': STRATEGIES, 'seeds': args.seeds,
        'pollution_seeds': list(range(1000, 1010)), 'severity_grids': FNS['SEVERITY_GRIDS'],
        'stats': stats, 'optimizer': 'Adam', 'lr': .0005, 'batch_size': 32,
        'dropout': .5, 'device': args.device, 'torch': torch.__version__,
        'train_n': int((raw['partitions'] == 'train').sum()), 'test_n': int((raw['partitions'] == 'val').sum()),
        'epoch_source': 'Supplied Yu3 fixed epochs: clean23, S3 81, combined51. Same budget for all three controls within strategy; no new pilot or held-out selection.',
        'stats_source': 'Supplied exp4_fixed_norm_stats.json, cross-checked against audited training-only float32 windows; tiny reduction-order differences retained.',
        'augmentation': 'Supplied default p=.5; equiprobable corruption types; uniform supplied severity grids; combined wd=1e-4. Historical fidelity of these defaults remains unconfirmed.',
        'pairing': 'SHA256 sample-ID keyed train augmentation shared across models, epoch permutation shared across models; test corruption shared across all runs.',
        'controls': 'Both single-pass controls use two-channel encoder [image,zero]. RecAux CE+.5MSE; no_feedback CE. Yu3 .5CE1+CE2+.5MSE, detached residual. No-feedback decoder allocated but not executed.',
        'limits': 'Not one-channel M1/RecAux, no per-model HPO, no strategy reselection, no original numeric reproduction claim. Yu3 vs RecAux changes pass count and classification loss composition.',
        'endpoints': 'Absolute all-11-class Macro-F1 and clean minus corrupted F1 (percentage points); average ten pollution realizations within each training seed.',
        'primary_tests': 'Yu3 vs each of two controls: clean F1 for each of 3 strategies (6), S3 AWGN10 absolute F1/drop (4), combined consecutive20 absolute F1/drop (4). Two-sided exhaustive Wilcoxon; Holm across14. All severity curves descriptive; no architecture selection on test.',
        'disclosure': 'Follow-up planned after seeing E5 results. Plan frozen before this follow-up training/evaluation.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            Path(__file__).with_name('run_e5_ablation.py'), ROOT/'實驗一/models_m1_to_m5.py',
            Path(__file__).with_name('analyze_exp4_matched_controls.py'),
            SOURCE/'exp4_raw_data_V2.py', SOURCE/'exp4_defense_strategies_V2.py',
            SOURCE/'best_epochs_exp4.json', SOURCE/'best_epochs_defense.json', SOURCE/'exp4_fixed_norm_stats.json',
            DATA/'waveform_image_audit.json', DATA/'raw_windows.npz']}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    raw = dict(np.load(DATA/'raw_windows.npz'))
    train = np.flatnonzero(raw['partitions'] == 'train')
    test = np.flatnonzero(raw['partitions'] == 'val')
    assert len(train) == 855 and len(test) == 219
    assert not set(raw['groups'][train]) & set(raw['groups'][test])
    stats = json.loads((SOURCE/'exp4_fixed_norm_stats.json').read_text())
    for prefix, column in [('i', 'current'), ('v', 'voltage')]:
        values = raw[column][train].astype(np.float32)
        for suffix, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
            assert np.isclose(stats[f'{prefix}_{suffix}'], float(fn(values)), rtol=2e-6, atol=1e-7)
    assert json.loads((SOURCE/'best_epochs_exp4.json').read_text())['M5']['fixed_epoch'] == 23
    supplied = json.loads((SOURCE/'best_epochs_defense.json').read_text())
    assert supplied['Yu3_S3']['fixed_epoch'] == 81 and supplied['Yu3_combined']['fixed_epoch'] == 51
    smoke(raw, train, stats, device)
    if args.smoke_only: return
    args.output.mkdir(parents=True, exist_ok=True)
    plan = protocol(args, raw, stats)
    planfile = args.output/'protocol.json'
    if planfile.exists(): assert json.loads(planfile.read_text()) == json.loads(json.dumps(plan))
    else: save_json(planfile, plan)
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for relative in plan['source_sha256']:
        if relative.endswith('.npz'): continue
        dest = snapshot/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): shutil.copy2(ROOT/relative, dest)
    clean_test = images(raw, test, stats)
    labels = raw['labels'][test]
    cache = args.output/'test_cache'
    cache.mkdir(exist_ok=True)
    conditions = [(typ, sev, ps) for typ, grid in FNS['SEVERITY_GRIDS'].items()
                  for sev in grid for ps in range(1000, 1010)]
    for i, condition in enumerate(conditions):
        dest = cache/f'{i:03}.npy'
        if not dest.exists():
            with dest.with_suffix('.tmp').open('wb') as stream:
                np.save(stream, images(raw, test, stats, condition=condition))
            os.replace(dest.with_suffix('.tmp'), dest)
    save_json(cache/'conditions.json', conditions)
    all_start = time.monotonic()
    for seed in args.seeds:
        for strategy, (epochs, wd) in STRATEGIES.items():
            for variant in VARIANTS:
                folder = args.output/f'{strategy}_{variant}_seed{seed}'
                folder.mkdir(exist_ok=True)
                done = folder/'results.json'
                if done.exists(): continue
                seed_all(seed)
                model = Ablation(variant).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=.0005, weight_decay=wd)
                start, logs = 1, []
                resume = folder/'resume.pt'
                if resume.exists():
                    state = torch.load(resume, map_location=device, weights_only=False)
                    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
                    start = state['epoch'] + 1; logs = state['logs']
                run_start = time.monotonic()
                y = torch.from_numpy(raw['labels'][train]).long().to(device)
                for epoch in range(start, epochs + 1):
                    seed_all(seed * 10000 + epoch)
                    x = torch.from_numpy(images(raw, train, stats, strategy, seed, epoch)).to(device)
                    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed * 10000 + epoch))
                    model.train(); total = 0.
                    for indices in order.split(32):
                        indices = indices.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(model(x[indices]), x[indices], y[indices], variant)
                        assert torch.isfinite(loss)
                        loss.backward(); optimizer.step()
                        total += float(loss.detach()) * len(indices)
                    logs.append({'epoch': epoch, 'train_loss': total / len(train)})
                    if epoch % 5 == 0 or epoch == epochs:
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                    'epoch': epoch, 'logs': logs}, folder/'resume.tmp')
                        os.replace(folder/'resume.tmp', resume)
                    if epoch % 10 == 0 or epoch == epochs:
                        print(f'{strategy} {variant} seed{seed}: epoch {epoch}/{epochs}, loss {total / len(train):.5f}', flush=True)
                expected = 27 * epochs * (2 if variant == 'err_detach' else 1)
                assert int(model.encoder.bn1.num_batches_tracked) == expected
                torch.save({'state_dict': model.state_dict(), 'variant': variant, 'seed': seed,
                            'strategy': strategy, 'epoch': epochs}, folder/'checkpoint.pt')
                with (folder/'train_log.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss'])
                    writer.writeheader(); writer.writerows(logs)
                clean_pred = predict(model, clean_test, device)
                result = {'variant': variant, 'seed': seed, 'strategy': strategy, 'epochs': epochs,
                          'clean': metrics(labels, clean_pred), 'conditions': [],
                          'seconds_this_invocation': time.monotonic() - run_start}
                preds = {'clean': clean_pred, 'labels': labels, 'sample_ids': raw['sample_ids'][test]}
                for i, (typ, sev, ps) in enumerate(conditions):
                    pred = predict(model, np.load(cache/f'{i:03}.npy'), device)
                    score = metrics(labels, pred)
                    preds[f'condition_{i:03}'] = pred
                    result['conditions'].append({'type': typ, 'severity': sev, 'pollution_seed': ps,
                        **score, 'drop_pp': 100 * (result['clean']['macro_f1'] - score['macro_f1'])})
                np.savez_compressed(folder/'predictions.npz', **preds)
                save_json(done, result)
                resume.unlink(missing_ok=True)
                count = len(list(args.output.glob('*/results.json')))
                save_json(args.output/'progress.json', {'completed': count, 'planned': len(args.seeds)*9,
                    'last_run': folder.name, 'elapsed_this_invocation_seconds': time.monotonic()-all_start})
                print(f'COMPLETE {count}/{len(args.seeds)*9}: {folder.name}; clean F1={result["clean"]["macro_f1"]:.4f}', flush=True)
    print('ALL TRAINING AND EVALUATION COMPLETE', flush=True)
    from analyze_exp4_matched_controls import analyze
    analyze(args.output)


if __name__ == '__main__':
    main()
