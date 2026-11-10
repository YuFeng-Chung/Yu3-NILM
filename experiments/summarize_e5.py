"""Summarize complete, paired E5 clean runs without choosing a winning model."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads((args.output / 'config.json').read_text())
    manifest = json.loads((args.output / 'image_manifest.json').read_text())
    by_variant = {}
    for v in config['variants']:
        rows = []
        for s in config['seeds']:
            path = args.output / f'{v}_seed{s}' / 'metrics.json'
            if path.exists(): rows.append(json.loads(path.read_text()))
        by_variant[v] = rows
    reference = {r['seed']: r['macro_f1'] for r in by_variant['err_detach']}
    summary = []
    for v, rows in by_variant.items():
        f1 = np.asarray([r['macro_f1'] for r in rows]) * 100
        paired = [(r['macro_f1'] - reference[r['seed']]) * 100 for r in rows if r['seed'] in reference]
        if not rows: continue
        summary.append({'variant': v, 'n': len(rows), 'macro_f1_mean_pct': float(f1.mean()),
                        'macro_f1_sd_pct': float(f1.std(ddof=1)) if len(rows) > 1 else None,
                        'paired_delta_vs_yu3_pp': float(np.mean(paired)) if paired else None,
                        'accuracy_mean_pct': float(np.mean([r['accuracy'] for r in rows]) * 100),
                        'training_seconds_mean': float(np.mean([r['training_seconds'] for r in rows]))})
    with (args.output / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    bn_rows = []
    for seed in config['seeds']:
        path = args.output / f'err_detach_seed{seed}' / 'bn_diagnostic.json'
        if not path.exists(): continue
        for key, value in json.loads(path.read_text()).items():
            gap = np.abs(np.asarray(value['population_mean']) - np.asarray(value['running_mean']))
            scale = np.sqrt(np.asarray(value['running_variance']) + 1e-5)
            bn_rows.append({'seed': seed, 'layer_pass': key,
                'mean_absolute_mean_gap': value['mean_absolute_mean_gap'],
                'mean_absolute_variance_gap': value['mean_absolute_variance_gap'],
                'mean_absolute_standardized_mean_gap': float(np.mean(gap / scale)),
                'mode': 'eval input population moments vs frozen running statistics'})
    if bn_rows:
        with (args.output / 'bn_summary.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(bn_rows[0])); w.writeheader(); w.writerows(bn_rows)
    labels = {'err_detach': 'Yu3：誤差回饋＋detach／共用BN', 'err_no_detach': '誤差回饋不detach',
              'img_no_detach': '影像回饋不detach', 'img_detach': '影像回饋＋detach',
              'no_first_ce': '無第一階段分類loss', 'no_recon_loss': '無重建loss',
              'recon_weight_one': '重建loss權重1.0', 'rec_aux': 'RecAux：單次分類＋重建',
              'three_pass': 'Three-Pass', 'pass_specific_bn': '誤差回饋＋各pass獨立BN',
              'no_feedback': '原No-FB：單次分類、無重建'}
    lines = ['# E5 第一輪實際結果（2026-11-06）', '',
             f"固定 seed：{config['seeds']}；每組{config['epochs']}輪；clean；train 855／held-out test 219；11類。",
             '', '這是新增訓練結果，並非原CUDA結果的精確數值重現。輸入為已提供的Exp1 PNG，全部1074張已完整解碼，依原split核對群組無重疊。',
             '', '## 結果', '', '| 變體 | n | Macro-F1（%，mean ± SD） | 與同seed Yu3差值（百分點） |', '|---|---:|---:|---:|']
    for r in summary:
        sd = f" ± {r['macro_f1_sd_pct']:.2f}" if r['macro_f1_sd_pct'] is not None else ''
        lines.append(f"| {labels[r['variant']]} | {r['n']} | {r['macro_f1_mean_pct']:.2f}{sd} | {r['paired_delta_vs_yu3_pp']:+.2f} |")
    lines += ['', '## 這輪新增的判讀依據', '',
              '原Yu3共用BN、RecAux、各pass獨立BN已在同一環境與固定設定下配對重訓，可以直接查看它們的clean結果；尚不能將平均差距當作顯著、等效或改版證據。',
              '', '| 訓練seed | 原Yu3 | RecAux | 獨立BN | 無重建loss |', '|---|---:|---:|---:|---:|']
    for seed in config['seeds']:
        values = []
        for variant in ['err_detach', 'rec_aux', 'pass_specific_bn', 'no_recon_loss']:
            row = next((r for r in by_variant[variant] if r['seed'] == seed), None)
            values.append(f"{100 * row['macro_f1']:.2f}" if row else '未完成')
        lines.append(f"| {seed} | " + ' | '.join(values) + ' |')
    lines += ['', '表中均為Macro-F1（%）。請同時看每個seed；例如獨立BN並非在每個seed都優於共用BN。無重建loss若有低分run應保留並追查，不能因其拉低平均就刪除。',
              '', '完成後另用verify_e5_outputs.py重新核對每個run的19輪紀錄、219筆ID與標籤、由預測重算的指標、checkpoint中各層BN更新次數；結果見completed_audit.json。33次訓練不是33個獨立seed，配對訓練seed仍只有3個。']
    lines += ['', '## 判讀限制與下一步', '',
              '- 3個seed只供初步觀察，未做顯著性判定；不能當成正式10-seed結果。',
              '- 所有變體均依固定19輪輸出，不依test選最佳epoch／模型；本表不能用來挑選新版架構。',
              '- RecAux採與原E5相同的2-channel encoder，第二輸入為零；不是尚待E1建立的1-channel M1版本。',
              '- 尚未加入波形級AWGN／連續缺失，不據此判斷穩健性，也沒有完成E1–E4。',
              '- BN各層逐pass輸入矩與running stats已保存；差距大小本身不代表效能一定受損或一定無影響。',
              '- 原模型權重／Exp4舊pilot尚未補齊不影響這次從頭訓練；不冒充已重現投稿表格。',
              '- 正式結論需依預定家族擴至10個seed、補污染評估與配對統計；若需改版，另在inner validation驗證。',
              '', '## 可追溯產物', '',
              '同資料夾有config.json、env.json、hardware.json、requirements.lock.txt、image_manifest.json、data_audit.json、results.csv、summary.csv、bn_summary.csv及來源快照；每個variant_seed子目錄有checkpoint.pt、train_log.csv、pred_clean.csv與metrics.json。Yu3另存bn_diagnostic.json（eval模式的逐pass輸入分布與凍結running stats）。checkpoint_roundtrip_check.json記錄seed42權重重新載入後全部預測一致。',
              '', f"原切分SHA256：{manifest['split_sha256']}。",
              '', '環境：', '', '```json', (args.output / 'env.json').read_text(), '```', '']
    (args.output / 'e5_preliminary_report_20261106.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:len(summary) + 12]))


if __name__ == '__main__': main()
