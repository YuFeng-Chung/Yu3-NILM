"""Audit every prediction, summarize paired Exp4 controls, export supplementary report."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from run_e5_ablation import metrics
from analyze_e5_full import paired_test, holm


def write_csv(path, rows):
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def analyze(output):
    plan = json.loads((output/'protocol.json').read_text())
    seeds, variants = plan['seeds'], plan['variants']
    assert seeds == list(range(42, 52)), 'Primary statistical report requires all ten specified seeds.'
    expected_conditions = json.loads((output/'test_cache/conditions.json').read_text())
    records, ids, labels, audit_n = {}, None, None, 0
    for strategy, (epochs, _) in plan['strategies'].items():
        for variant in variants:
            rows = []
            for seed in seeds:
                folder = output/f'{strategy}_{variant}_seed{seed}'
                r = json.loads((folder/'results.json').read_text())
                assert (r['seed'], r['variant'], r['strategy'], r['epochs']) == (seed, variant, strategy, epochs)
                preds = np.load(folder/'predictions.npz')
                if ids is None: ids, labels = preds['sample_ids'], preds['labels']
                assert np.array_equal(ids, preds['sample_ids']) and np.array_equal(labels, preds['labels'])
                assert len(ids) == 219 and len(set(ids)) == 219
                assert np.allclose(metrics(labels, preds['clean'])['macro_f1'], r['clean']['macro_f1'], atol=1e-12)
                assert len(r['conditions']) == len(expected_conditions)
                for i, c in enumerate(r['conditions']):
                    assert [c['type'], c['severity'], c['pollution_seed']] == expected_conditions[i]
                    actual = metrics(labels, preds[f'condition_{i:03}'])
                    for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'class_f1']:
                        assert np.allclose(actual[key], c[key], rtol=0, atol=1e-12)
                    assert np.isclose(c['drop_pp'], 100*(r['clean']['macro_f1']-actual['macro_f1']), atol=1e-10)
                    audit_n += 1
                checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
                for layer in (1, 2, 3):
                    assert int(checkpoint['state_dict'][f'encoder.bn{layer}.num_batches_tracked']) == 27*epochs*(2 if variant == 'err_detach' else 1)
                with (folder/'train_log.csv').open() as f: logs = list(csv.DictReader(f))
                assert [int(x['epoch']) for x in logs] == list(range(1, epochs+1))
                assert all(np.isfinite(float(x['train_loss'])) for x in logs)
                rows.append(r)
            records[strategy, variant] = rows

    def endpoint(strategy, variant, typ=None, severity=None, drop=False):
        rows = records[strategy, variant]
        if typ is None: return np.array([r['clean']['macro_f1']*100 for r in rows])
        key = 'drop_pp' if drop else 'macro_f1'
        values = []
        for r in rows:
            selected = [c for c in r['conditions'] if c['type'] == typ and c['severity'] == severity]
            assert len(selected) == 10
            values.append(np.mean([c[key] for c in selected]) * (1 if drop else 100))
        return np.array(values)

    clean, curves = [], []
    for strategy in plan['strategies']:
        for variant in variants:
            values = endpoint(strategy, variant)
            clean.append({'strategy': strategy, 'variant': variant, 'n': 10,
                          'f1_mean_pct': values.mean(), 'f1_sd_pct': values.std(ddof=1)})
            for typ, grid in plan['severity_grids'].items():
                for severity in grid:
                    f1 = endpoint(strategy, variant, typ, severity)
                    drop = endpoint(strategy, variant, typ, severity, True)
                    curves.append({'strategy': strategy, 'variant': variant, 'corruption': typ,
                        'severity': severity, 'n_training_seeds': 10, 'realizations_per_seed': 10,
                        'f1_mean_pct': f1.mean(), 'f1_sd_pct': f1.std(ddof=1),
                        'drop_mean_pp': drop.mean(), 'drop_sd_pp': drop.std(ddof=1)})
    contrasts = []
    for strategy in plan['strategies']:
        for control in variants[1:]:
            contrasts.append((strategy, control, None, None, False))
    for strategy, typ, severity in [('S3', 'awgn', 10), ('combined', 'missing_consecutive', .2)]:
        for control in variants[1:]:
            for drop in (False, True): contrasts.append((strategy, control, typ, severity, drop))
    assert len(contrasts) == 14
    tests, paired = [], []
    for strategy, control, typ, severity, drop in contrasts:
        a = endpoint(strategy, 'err_detach', typ, severity, drop)
        b = endpoint(strategy, control, typ, severity, drop)
        d = a-b
        _, p = paired_test(d)
        name = f'{strategy}:Yu3-{control}:{typ or "clean"}:{severity}:{"drop" if drop else "F1"}'
        tests.append({'comparison': name, 'mean_difference_pp': d.mean(),
                      'sd_difference_pp': d.std(ddof=1), 'p_raw': p,
                      'direction': 'negative favors Yu3' if drop else 'positive favors Yu3'})
        paired.extend({'comparison': name, 'seed': seed, 'difference_pp': value} for seed, value in zip(seeds, d))
    for row, p in zip(tests, holm([t['p_raw'] for t in tests])): row['p_holm_14'] = p
    write_csv(output/'clean_summary.csv', clean)
    write_csv(output/'severity_summary.csv', curves)
    write_csv(output/'primary_paired_tests.csv', tests)
    write_csv(output/'paired_differences.csv', paired)
    audit = {'models': 90, 'corruption_evaluations': audit_n,
             'corrupted_predictions_recomputed': audit_n*219,
             'clean_predictions_recomputed': 90*219, 'all_checks_passed': True,
             'checks': 'IDs, labels, all metrics, drops, epoch logs, checkpoint BN counts',
             'manuscript_modified': False}
    (output/'completed_audit.json').write_text(json.dumps(audit, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
    colors = {'err_detach': '#0072B2', 'rec_aux': '#D55E00', 'no_feedback': '#009E73'}
    names = {'err_detach': 'Yu3', 'rec_aux': 'RecAux (2ch)', 'no_feedback': 'No feedback (2ch)'}
    for i, strategy in enumerate(plan['strategies']):
        for j, (typ, grid) in enumerate(plan['severity_grids'].items()):
            ax = axes[i, j]
            for variant in variants:
                vals = np.stack([endpoint(strategy, variant, typ, severity) for severity in grid])
                mean, sd = vals.mean(1), vals.std(1, ddof=1)
                x = np.arange(len(grid))
                ax.plot(x, mean, '.-', color=colors[variant], label=names[variant])
                ax.fill_between(x, np.maximum(0, mean-sd), np.minimum(100, mean+sd), alpha=.10, color=colors[variant])
            ax.set_xticks(np.arange(len(grid)), [str(g) for g in grid])
            ax.set_title(f'{strategy} / {typ}')
            ax.set_ylim(0, 100); ax.grid(alpha=.2)
            if j == 0: ax.set_ylabel('Macro-F1 (%)')
            if i == 2: ax.set_xlabel('SNR (dB)' if typ == 'awgn' else 'Corrupted fraction')
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside upper center', ncol=3)
    for suffix in ['png', 'pdf', 'svg']: fig.savefig(output/f'exp4_matched_curves.{suffix}', dpi=300)
    plt.close(fig)
    lines = ['# Experiment 4 受控補充實驗', '',
        '已完成 90 個模型：10 個訓練 seed × 3 個策略 × 3 個模型。每個模型評估 18 個污染強度，每個強度使用 10 個污染 seed。', '',
        '本輪保留原 Yu3。RecAux 與無回饋模型均採兩通道編碼器，第二輸入固定為零，與 E5 控制一致；並非一通道 M1／RecAux 架構實驗。', '',
        '訓練預算沿用補交紀錄中 Yu3 的回合數，各策略內三模型相同：未防禦 23、S3 81、Combined 51。使用原 Experiment 4 的訓練集固定正規化參數、float32 影像與波形級擾動。', '',
        '**限制：動態增強的強度抽樣規則及 Combined 的 weight decay 沿用程式標示的合理預設值，歷史一致性尚未確認。因此這是新增受控實驗，不能聲稱重現原投稿數字。未重新調參或重新選策略，也未根據測試結果選模型。**', '',
        '## 乾淨資料 Macro-F1（平均 ± 訓練 seed 標準差，%）', '',
        '| 策略 | Yu3 | RecAux（兩通道） | 無回饋（兩通道） |', '|---|---:|---:|---:|']
    for strategy in plan['strategies']:
        cells = []
        for variant in variants:
            x = endpoint(strategy, variant)
            cells.append(f'{x.mean():.2f} ± {x.std(ddof=1):.2f}')
        lines.append('| '+strategy+' | '+' | '.join(cells)+' |')
    lines += ['', '## 指定污染條件：絕對表現與退化量', '',
        '| 策略／條件 | 模型 | 污染 Macro-F1（%） | 退化量（百分點） |', '|---|---|---:|---:|']
    for strategy, typ, severity in [('S3', 'awgn', 10), ('combined', 'missing_consecutive', .2)]:
        for variant in variants:
            f1 = endpoint(strategy, variant, typ, severity)
            drop = endpoint(strategy, variant, typ, severity, True)
            lines.append(f'| {strategy}／{typ} {severity} | {names[variant]} | {f1.mean():.2f} ± {f1.std(ddof=1):.2f} | {drop.mean():.2f} ± {drop.std(ddof=1):.2f} |')
    lines += ['', '## 預先指定的配對檢定', '',
        '以訓練 seed 為配對單位（n=10），先平均每個 seed 的十次污染，再做雙尾精確置換 Wilcoxon；主要校正涵蓋全部 14 個比較。', '',
        '| 比較（Yu3 − 控制） | 平均差（百分點） | 原始 p | Holm p |', '|---|---:|---:|---:|']
    for row in tests:
        lines.append(f'| {row["comparison"]} | {row["mean_difference_pp"]:.3f} | {row["p_raw"]:.6f} | {row["p_holm_14"]:.6f} |')
    lines += ['', 'F1 差為正表示 Yu3 較高；退化差為負表示 Yu3 退化較少。其餘污染強度曲線為描述性結果。未顯著不能解讀為等效；乾淨起點較低也可能造成退化量較小，必須同時看絕對表現。', '',
        'Yu3 對 RecAux 同時改變傳遞次數及分類損失組成，不能把差異完全歸因於誤差訊號。單一固定資料切分的 seed 變異，也不等同跨資料集泛化證據。', '',
        '全部逐筆預測、指標、回合與 BN 更新次數已重算通過；詳見 completed_audit.json。原投稿文件未修改。', '',
        '![完整強度曲線](exp4_matched_curves.png)', '']
    (output/'exp4_matched_report_20261106.md').write_text('\n'.join(lines))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    analyze(parser.parse_args().output)
