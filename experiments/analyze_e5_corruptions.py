"""Secondary paired degradation analysis; never treat corruption seeds as replicates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_e5_full import COMPARISONS, holm, paired_test


def plan():
    return {'conditions': ['awgn_10db', 'consecutive_missing_20pct'],
        'endpoint': 'D_abs = clean Macro-F1 - mean corrupted Macro-F1; smaller degradation is better.',
        'replicate_unit': '10 paired training seeds42-51; average each metric over contamination seeds1000-1009 first.',
        'multiplicity': 'Holm across24 secondary D_abs contrasts (12 per condition); additionally show36-test Holm combining12 clean primary contrasts.',
        'selection': 'No architecture, hyperparameter or epoch selection from these held-out results.',
        'scope': 'Supplementary E5 fixed19-epoch clean-trained models; does not replace Exp4 defense experiments.',
        'frozen_before': 'Corrupted predictions were generated.'}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--write-plan', action='store_true'); args = p.parse_args()
    plan_path = args.output / 'secondary_analysis_plan.json'
    if plan_path.exists(): assert json.loads(plan_path.read_text()) == plan()
    else: plan_path.write_text(json.dumps(plan(), indent=2))
    if args.write_plan: print('Saved secondary degradation plan before corrupted predictions.'); return
    folder = args.output / 'corrupted_evaluation'
    with (folder / 'results_averaged.csv').open() as f: rows = list(csv.DictReader(f))
    assert len(rows) == 220
    config = json.loads((args.output / 'config.json').read_text()); seeds = config['seeds']
    assert seeds == list(range(42, 52))
    summaries, tests, differences = [], [], []
    for condition in plan()['conditions']:
        chosen = [r for r in rows if r['condition'] == condition]
        by_variant = {}
        for variant in config['variants']:
            records = sorted([r for r in chosen if r['variant'] == variant], key=lambda r: int(r['training_seed']))
            assert [int(r['training_seed']) for r in records] == seeds
            f1 = np.array([float(r['corrupted_macro_f1']) for r in records])
            drop = np.array([float(r['absolute_drop_pp']) for r in records])
            by_variant[variant] = drop / 100.
            summaries.append({'condition': condition, 'variant': variant, 'n_training_seeds': 10,
                'corrupted_macro_f1_mean_pct': float(f1.mean() * 100), 'corrupted_macro_f1_sd_pct': float(f1.std(ddof=1) * 100),
                'absolute_drop_mean_pp': float(drop.mean()), 'absolute_drop_sd_pp': float(drop.std(ddof=1))})
        for family, name, coefficients in COMPARISONS:
            d = sum(coef * by_variant[v] for v, coef in coefficients.items())
            statistic, pv = paired_test(d)
            tests.append({'condition': condition, 'family': family, 'comparison': name,
                'n_training_seeds': 10, 'mean_difference_drop_pp': float(d.mean() * 100),
                'n_nonzero_differences': int((np.round(d, 12) != 0).sum()),
                'statistic': statistic, 'p_raw': pv,
                'direction': 'Positive means greater degradation on left; reverse of positive clean-F1 advantage.'})
            differences.extend({'condition': condition, 'comparison': name, 'training_seed': s,
                'difference_drop_pp': float(value * 100)} for s, value in zip(seeds, d))
    for row, pv in zip(tests, holm([r['p_raw'] for r in tests])): row['p_holm_secondary_24'] = float(pv)
    with (args.output / 'paired_tests_clean.csv').open() as f: clean = list(csv.DictReader(f))
    global36 = holm([float(r['p_raw']) for r in clean] + [r['p_raw'] for r in tests])
    for row, pv in zip(tests, global36[len(clean):]): row['p_holm_all36_sensitivity'] = float(pv)
    clean36 = [{'comparison': r['comparison'], 'p_holm_all36_sensitivity': float(pv)} for r, pv in zip(clean, global36[:len(clean)])]
    for name, data in [('summary_corrupted.csv', summaries), ('paired_tests_degradation.csv', tests),
                       ('paired_differences_degradation.csv', differences), ('clean_all36_sensitivity.csv', clean36)]:
        with (folder / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps({'summary': summaries, 'tests': tests}, indent=2))


if __name__ == '__main__': main()
