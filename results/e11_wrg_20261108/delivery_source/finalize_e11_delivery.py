"""E11 presentation cleanup and delivery checks; frozen experiment sources stay unchanged."""
import csv
import hashlib
import json
import time
import zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from e11_wrg_core import OUTPUT,ROOT,OLD,SEEDS,CONDITIONS,GRIDS,assert_sources


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=OUTPUT;plan=json.loads((p/'protocol.json').read_text());assert_sources(plan)
    assert json.loads((p/'completed_audit.json').read_text())['all_checks_passed']
    # Validate all preserved co-selected outputs before visually merging equivalent labels.
    for a,b in [('S3','S4'),('augmentation','combined')]:
        for seed in SEEDS:
            for name in ['clean']+[f'{i:03}' for i in range(210)]:
                np.testing.assert_array_equal(np.load(p/'final'/f'WRG-CNN_{a}_s{seed}'/'outer_predictions'/f'{name}.npy'),
                    np.load(p/'final'/f'WRG-CNN_{b}_s{seed}'/'outer_predictions'/f'{name}.npy'))
    rows=list(csv.DictReader((p/'per_seed_averaged.csv').open()))
    values={(r['model'],r['strategy'],int(r['seed']),r['type'],float(r['severity'])):float(r['f1'])*100 for r in rows}
    contexts=[('common/'+s,s,s) for s in ['no_defense','S3','combined']]+[
        ('noise/S3/S3','S3','S3'),('noise/S3/S4','S3','S4'),
        ('missing/combined/augmentation','combined','augmentation'),('missing/combined/combined','combined','combined')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for context,ys,ws in contexts:
        types=[t for t in GRIDS if (context.startswith('common/') or context.startswith('noise/') and t in ['awgn','salt_pepper'] or context.startswith('missing/') and t.startswith('missing'))]
        fig,axes=plt.subplots(2 if len(types)==4 else 1,2,figsize=(10,7 if len(types)==4 else 3.9),layout='constrained')
        for ax,t in zip(axes.flat,types):
            sevs=GRIDS[t];xx=np.array(sevs)*(1 if t=='awgn' else 100)
            for model,strategy,color in [('Yu3',ys,'#235789'),('WRG-CNN',ws,'#d45d00')]:
                array=np.array([[values[model,strategy,k,t,float(s)] for s in sevs] for k in SEEDS])
                ax.errorbar(xx,array.mean(0),yerr=array.std(0,ddof=1),marker='o',capsize=3,
                    color=color,label=model+' / '+strategy)
            ax.set_title({'awgn':'AWGN','salt_pepper':'Salt-and-pepper','missing_random':'Random missing','missing_consecutive':'Consecutive missing'}[t])
            ax.set_xlabel('SNR (dB)' if t=='awgn' else 'Corruption (%)');ax.set_ylabel('Macro-F1 (%)')
            ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if t=='awgn':ax.invert_xaxis()
        fig.suptitle(context+' | mean +/- SD, ten training seeds',fontsize=12)
        stem=context.replace('/','_');fig.savefig(p/'figures'/f'{stem}.png',dpi=600)
        fig.savefig(p/'figures'/f'{stem}.svg');plt.close(fig)
    start=(p/'local_process.json').stat().st_mtime;end=(p/'completed_audit.json').stat().st_mtime
    zone=timezone(timedelta(hours=8));fmt=lambda t:datetime.fromtimestamp(t,zone).strftime('%Y-%m-%d %H:%M:%S %z')
    elapsed=(end-start)/60
    brief=f'''## 主要觀察與耗時

正式writer從{fmt(start)}執行至{fmt(end)}，約{elapsed:.2f}分鐘；不含前期實作／前測及後續交付QA。前測估計運算48–124分鐘、含整理1–2.5小時為保守預估；實際較快，不回寫舊估計。

依各自選定策略，雜訊使用Yu3 S3對WRG S3/S4（兩WRG標籤權重及全部預測逐值一致）：clean為72.11/66.26%，AWGN10dB為47.00/64.40%，椒鹽20%為69.76/41.40%（依序Yu3/WRG）。AWGN30dB是Yu3較高，20至0dB則WRG較高，不能概括所有AWGN強度。

缺失使用Yu3 Combined對WRG Augmentation/Combined（兩WRG標籤權重及全部預測逐值一致）：clean71.65/63.63%，連續缺失20%65.87/58.09%，50%39.95/40.55%。Yu3在連續缺失1–40%平均較高，50%則WRG平均高0.59pp，非顯著性或等效證據。

退化量揭露不同於絕對F1：Combined連續缺失20%，Yu3損失5.78pp、WRG5.53pp，Yu3雖絕對F1較高，退化略大；50%則31.70/23.08pp。S3的AWGN10dB損失25.11/1.87pp，椒鹽20%損失2.36/24.87pp。完整強度與所有同策略／選策配對見下表及CSV。

結論限於本輪歷史切分與共同評估協定：兩方法優勢依擾動類型和強度改變，不支持Yu3在所有條件普遍較佳，也不把表徵差異當作回饋機制因果證據。

'''
    report=p/'E11結果報告.md';text=report.read_text()
    if '## 主要觀察與耗時' not in text:text=text.replace('## 設計與公平性',brief+'## 設計與公平性',1);report.write_text(text)
    audit=dict(all_checks_passed=True,completed_writer_exit_checked_by_parent=True,
               exact_equivalent_prediction_pairs=20*211,exact_equivalent_prediction_rows=20*211*219,
               started_taiwan=fmt(start),compute_completed_taiwan=fmt(end),compute_minutes=elapsed,
               figure_count=7,figure_layout='common 2x2; selected 1x2, no blank subplot rows',
               report_has_both_positive_and_negative_results=True,
               source_sha256=sha(Path(__file__)),manuscript_modified=False,formal_word_created=False)
    (p/'delivery_checks.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    # Source, summaries and reproducible counts packaged; checkpoints/full old data remain at cited locations.
    selected=['README.md','E11結果報告.md','protocol.json','split_manifest.json','E3_folds.json',
        'preflight_audit.json','completed_audit.json','delivery_checks.json','equivalent_training_audit.json',
        'hpo_results.csv','cv_scores.csv','selected_setting.json','selected_strategies.json',
        'epoch_settings.json','selection_complete.json','final_run_manifest.json','evaluation_conditions.json',
        'per_seed_averaged.csv','per_class_averaged.csv','summary.csv','paired_comparisons.csv']
    files=[p/n for n in selected]
    for folder in ['figures','confusion_counts','source_snapshot']:
        files+=sorted(f for f in (p/folder).rglob('*') if f.is_file())
    for script in ['control_e11_wrg.py','finalize_e11_delivery.py']:
        dest=p/'delivery_source'/script;dest.parent.mkdir(exist_ok=True);dest.write_bytes((ROOT/'experiments'/script).read_bytes());files.append(dest)
    manifest={str(f.relative_to(p)):sha(f) for f in files}
    (p/'delivery_manifest_sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    files.append(p/'delivery_manifest_sha256.json')
    target=ROOT/'E11完整結果整理包_20261108.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,arcname='E11/'+str(f.relative_to(p)))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read('E11/'+n)).hexdigest()==h
    (p/'package_audit.json').write_text(json.dumps(dict(all_checks_passed=True,file=target.name,bytes=target.stat().st_size,
        sha256=sha(target),packaged_files=len(files)),indent=2)+'\n')
    print('E11 delivery complete',target.name,target.stat().st_size,flush=True)


if __name__=='__main__':main()
