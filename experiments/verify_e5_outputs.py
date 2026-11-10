"""Audit completed E5 outputs against config, cached labels and model BN counters."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_e5_ablation import metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.output / 'config.json').read_text())
    cache = np.load(args.output / 'e5_images.npz')
    batches = (config['train_size'] + config['batch_size'] - 1) // config['batch_size']
    steps = batches * config['epochs']
    checked = []
    for seed in config['seeds']:
        for variant in config['variants']:
            run = args.output / f'{variant}_seed{seed}'
            result = json.loads((run / 'metrics.json').read_text())
            with (run / 'train_log.csv').open() as f: logs = list(csv.DictReader(f))
            assert [int(r['epoch']) for r in logs] == list(range(1, config['epochs'] + 1))
            assert all(np.isfinite(float(r['train_loss'])) for r in logs)
            with (run / 'pred_clean.csv').open() as f: pred = list(csv.DictReader(f))
            assert [r['sample_id'] for r in pred] == list(cache['val_ids'])
            y = np.asarray([int(r['y_true']) for r in pred])
            assert np.array_equal(y, cache['val_labels'])
            recomputed = metrics(y, [int(r['y_pred']) for r in pred])
            for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1'):
                assert abs(result[key] - recomputed[key]) < 1e-12, (run, key)
            ck = torch.load(run / 'checkpoint.pt', map_location='cpu', weights_only=True)
            assert (ck['variant'], ck['seed'], ck['epoch']) == (variant, seed, config['epochs'])
            state = ck['state_dict']
            expected = steps * (3 if variant == 'three_pass' else 1 if variant in ('rec_aux', 'no_feedback', 'pass_specific_bn') else 2)
            for layer in (1, 2, 3):
                assert int(state[f'encoder.bn{layer}.num_batches_tracked']) == expected
            if variant == 'pass_specific_bn':
                for layer in range(3): assert int(state[f'pass2_bn.{layer}.num_batches_tracked']) == steps
            assert int(state['dec_bn1.num_batches_tracked']) == (0 if variant == 'no_feedback' else steps * (2 if variant == 'three_pass' else 1))
            checked.append({'variant': variant, 'seed': seed, 'epochs': len(logs),
                            'prediction_rows': len(pred), 'encoder_bn_updates': expected})
    result = {'all_checks_passed': True, 'completed_runs': len(checked), 'checks': checked,
              'scope': 'Logs, labels, recomputed metrics, checkpoint metadata, BN update counts. Not a claim of scientific validity or cross-platform numerical reproduction.'}
    (args.output / 'completed_audit.json').write_text(json.dumps(result, indent=2))
    print(f'PASS: {len(checked)} runs; every epoch/prediction/metric/checkpoint BN count verified.')


if __name__ == '__main__': main()
