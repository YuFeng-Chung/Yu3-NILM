"""Paired ten-seed E5 analysis; global Holm is primary, family Holm secondary."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import PermutationMethod, wilcoxon

COMPARISONS = [
    ('A', 'err_detach_vs_err_no_detach', {'err_detach': 1, 'err_no_detach': -1}),
    ('A', 'err_detach_vs_img_no_detach', {'err_detach': 1, 'img_no_detach': -1}),
    ('A', 'err_detach_vs_img_detach', {'err_detach': 1, 'img_detach': -1}),
    ('A', 'signal_detach_interaction', {'err_detach': 1, 'err_no_detach': -1, 'img_detach': -1, 'img_no_detach': 1}),
    ('B', 'err_detach_vs_no_first_ce', {'err_detach': 1, 'no_first_ce': -1}),
    ('B', 'err_detach_vs_no_recon_loss', {'err_detach': 1, 'no_recon_loss': -1}),
    ('B', 'err_detach_vs_recon_weight_one', {'err_detach': 1, 'recon_weight_one': -1}),
    ('C', 'err_detach_vs_rec_aux', {'err_detach': 1, 'rec_aux': -1}),
    ('C', 'err_detach_vs_three_pass', {'err_detach': 1, 'three_pass': -1}),
    ('C', 'err_detach_vs_no_feedback', {'err_detach': 1, 'no_feedback': -1}),
    ('C', 'rec_aux_vs_no_feedback', {'rec_aux': 1, 'no_feedback': -1}),
    ('D', 'err_detach_vs_pass_specific_bn', {'err_detach': 1, 'pass_specific_bn': -1}),
]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float); order = np.argsort(p, kind='stable')
    out = np.empty_like(p)
    out[order] = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    return out


def paired_test(d):
    d = np.round(np.asarray(d, dtype=float), 12)
    if np.all(d == 0): return 0., 1.
    r = wilcoxon(d, zero_method='wilcox', alternative='two-sided',
                 method=PermutationMethod(n_resamples=np.inf))
    return float(r.statistic), float(r.pvalue)


def analysis_plan():
    return {'purpose': 'Supplementary evidence about original Yu3; no architecture selection on test.',
        'seeds': list(range(42, 52)), 'endpoint': 'Clean all-11-class Macro-F1',
        'families': {'A': 'feedback signal and detach, including interaction', 'B': 'loss weights',
                     'C': 'reconstruction supervision and pass count', 'D': 'shared vs pass-specific BN'},
        'comparisons': [{'family': f, 'name': n, 'coefficients': c} for f, n, c in COMPARISONS],
        'primary_correction': 'Holm across all 12 unique tests; alpha=0.05',
        'secondary_correction': 'Within-family Holm, shown in full without replacing primary adjustment.',
        'test': 'Two-sided Wilcoxon signed-rank; exhaustive sign permutations via SciPy PermutationMethod(n_resamples=inf).',
        'zeros': 'wilcox; all-zero differences assigned p=1', 'difference_rounding_decimals': 12,
        'disclosure': 'Initial seeds42-44 had been inspected. This plan was fixed before remaining seeds45-51 completed; it is not a pristine preregistration.',
        'limits': 'Single held-out split; randomness over training seeds only. Non-significance does not establish equivalence; clean results do not establish robustness.',
        'method_reference': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-plan', action='store_true'); args = parser.parse_args()
    plan = analysis_plan(); plan_path = args.output / 'analysis_plan.json'
    if plan_path.exists(): assert json.loads(plan_path.read_text()) == plan
    else: plan_path.write_text(json.dumps(plan, indent=2))
    if args.write_plan: print('Saved fixed analysis plan; no results inspected.'); return
    config = json.loads((args.output / 'config.json').read_text())
    assert config['seeds'] == plan['seeds']
    rows = {}
    for v in config['variants']:
        rows[v] = [json.loads((args.output / f'{v}_seed{s}' / 'metrics.json').read_text()) for s in plan['seeds']]
    f1 = {v: np.array([r['macro_f1'] for r in rs]) for v, rs in rows.items()}
    summary = []
    for v, rs in rows.items():
        delta = (f1[v] - f1['err_detach']) * 100
        summary.append({'variant': v, 'n': len(rs), 'macro_f1_mean_pct': float(f1[v].mean() * 100),
                        'macro_f1_sd_pct': float(f1[v].std(ddof=1) * 100),
                        'paired_delta_vs_yu3_pp': float(delta.mean()), 'paired_delta_sd_pp': float(delta.std(ddof=1)),
                        'wins_vs_yu3': int((delta > 1e-10).sum()), 'losses_vs_yu3': int((delta < -1e-10).sum()),
                        'accuracy_mean_pct': float(np.mean([r['accuracy'] for r in rs]) * 100)})
    contrasts, paired = [], []
    for family, name, coefficients in COMPARISONS:
        d = sum(coef * f1[v] for v, coef in coefficients.items())
        statistic, p = paired_test(d)
        contrasts.append({'family': family, 'comparison': name, 'n': len(d),
            'n_nonzero_differences': int((np.round(d, 12) != 0).sum()),
            'mean_difference_pp': float(d.mean() * 100), 'sd_difference_pp': float(d.std(ddof=1) * 100),
            'median_difference_pp': float(np.median(d) * 100), 'wilcoxon_statistic': statistic, 'p_raw': p})
        paired.extend({'comparison': name, 'seed': s, 'difference_pp': float(value * 100)} for s, value in zip(plan['seeds'], d))
    adjusted = holm([r['p_raw'] for r in contrasts])
    for r, adj in zip(contrasts, adjusted): r['p_holm_global_primary'] = float(adj)
    for family in ['A', 'B', 'C', 'D']:
        selected = [r for r in contrasts if r['family'] == family]
        for r, adj in zip(selected, holm([r['p_raw'] for r in selected])):
            r['p_holm_family_secondary'] = float(adj)
    bn_rows, bn_pass_differences = [], []
    for seed in plan['seeds']:
        diagnosis = json.loads((args.output / f'err_detach_seed{seed}/bn_diagnostic.json').read_text())
        for layer in (1, 2, 3):
            passes = [diagnosis[f'layer{layer}_pass{j}'] for j in (1, 2)]
            assert np.array_equal(passes[0]['running_mean'], passes[1]['running_mean'])
            assert np.array_equal(passes[0]['running_variance'], passes[1]['running_variance'])
            for j, record in enumerate(passes, 1):
                gap = np.abs(np.asarray(record['population_mean']) - np.asarray(record['running_mean']))
                scale = np.sqrt(np.asarray(record['running_variance']) + 1e-5)
                bn_rows.append({'seed': seed, 'layer': layer, 'pass': j,
                    'mean_absolute_mean_gap': record['mean_absolute_mean_gap'],
                    'mean_absolute_variance_gap': record['mean_absolute_variance_gap'],
                    'standardized_mean_gap': float(np.mean(gap / scale)), 'mode': 'eval'})
            mean1, mean2 = [np.asarray(r['population_mean']) for r in passes]
            var1, var2 = [np.asarray(r['population_variance']) for r in passes]
            scale = np.sqrt((var1 + var2) / 2 + 1e-5)
            bn_pass_differences.append({'seed': seed, 'layer': layer,
                'mean_abs_pass_mean_difference': float(np.abs(mean1 - mean2).mean()),
                'standardized_pass_mean_difference': float(np.mean(np.abs(mean1 - mean2) / scale)),
                'mean_abs_pass_variance_difference': float(np.abs(var1 - var2).mean()),
                'interpretation': 'Descriptive eval input distributions only; not a performance-effect test.'})
    for name, data in [('summary_full.csv', summary), ('paired_tests_clean.csv', contrasts), ('paired_differences_clean.csv', paired),
                       ('bn_input_summary.csv', bn_rows), ('bn_pass_distribution_difference.csv', bn_pass_differences)]:
        with (args.output / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(json.dumps({'summary': summary, 'paired_tests': contrasts}, indent=2))


if __name__ == '__main__': main()
