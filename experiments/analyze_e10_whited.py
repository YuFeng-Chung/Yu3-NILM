"""Independently reconstruct all held-out metrics and write auditable results."""
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e10_core import *

def independent(y,p,k):
    # Boolean TP/FP/FN reconstruction independent of bincount confusion implementation.
    out=[];cm=np.zeros((k,k),np.int64)
    for a in range(k):
        for b in range(k):cm[a,b]=np.count_nonzero((y==a)&(p==b))
        tp=np.count_nonzero((y==a)&(p==a));fp=np.count_nonzero((y!=a)&(p==a));fn=np.count_nonzero((y==a)&(p!=a))
        pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.;f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        out.append((pr,re,f))
    return cm,np.array(out)

def sync_status(description):
    import re
    block='<!-- E10_STATE_START -->\n'+description+'\n<!-- E10_STATE_END -->'
    p=ROOT.parent.parent/'PROJECT_CONTEXT.md';s=p.read_text();pattern=r'<!-- E10_STATE_START -->[\s\S]*?<!-- E10_STATE_END -->'
    if re.search(pattern,s):s=re.sub(pattern,lambda _:block,s,count=1)
    else:s=s.replace('## 已確認決策與注意事項\n','## 已確認決策與注意事項\n\n'+block+'\n',1)
    p.write_text(s)
    p=ROOT/'實驗程式架構總覽.md';s=p.read_text()
    row='| E10 | 第二高頻V–I資料集驗證 | WHITED已知r1–r8中15類／76設備名稱群組／759筆 | 三折設備分組、15epoch pilot、M1–M5各三seed共45正式訓練 | 研究計算／獨立指標稽核／圖／ZIP完成；正式回應稿與主文圖待做 | 研究計算完成，回應稿待做 |'
    s=re.sub(r'^\| E10 \| 第二高頻V–I資料集驗證[^\n]*',lambda _:row,s,count=1,flags=re.M)
    p.write_text(s)

def analyze():
    plan=json.loads((OUTPUT/'protocol.json').read_text());assert plan['source_sha256']==source_hashes()
    cache=dict(np.load(OUTPUT/'images.npz'));k=len(cache['classes']);splits=json.loads((OUTPUT/'split.json').read_text())
    choices=json.loads((OUTPUT/'selection_complete.json').read_text())['epochs']
    assert sha(OUTPUT/'images.npz')==plan['data_sha256'] and sha(OUTPUT/'split.json')==plan['split_sha256']
    assert json.loads((OUTPUT/'training_complete.json').read_text())['runs']==45
    records=[];prf=[];counts=[];all_oof={};pred_count=0
    for name in NAMES:
        for seed in SEEDS:
            oof=np.full(len(cache['labels']),-1,np.int64)
            for s in splits:
                folder=OUTPUT/'runs'/f"fold{s['fold']}_{name}_s{seed}";a=dict(np.load(folder/'predictions.npz'));idx=np.array(s['outer_test'])
                assert np.array_equal(a['indices'],idx) and np.array_equal(a['sample_ids'],cache['sample_ids'][idx])
                assert np.array_equal(a['labels'],cache['labels'][idx]) and np.all(oof[idx]==-1)
                assert np.isin(a['predictions'],np.arange(k)).all()
                assert json.loads((folder/'complete.json').read_text())['selected_epoch']==choices[f"fold{s['fold']}_{name}"]
                assert not set(cache['groups'][idx])&set(cache['groups'][s['outer_train']])
                oof[idx]=a['predictions'];pred_count+=len(idx)
                m,cm,precision,recall,f1=metrics(a['labels'],a['predictions'],k)
                icm,ipr=independent(a['labels'],a['predictions'],k)
                assert np.array_equal(cm,icm) and np.allclose(ipr,np.array([precision,recall,f1]).T,atol=1e-14)
                assert abs(m['accuracy']-np.mean(a['labels']==a['predictions']))<1e-14
                records.append(dict(model=name,seed=seed,fold=s['fold'],samples=len(idx),**m))
            assert np.all(oof>=0);all_oof[name,seed]=oof
            m,cm,precision,recall,f1=metrics(cache['labels'],oof,k);icm,ipr=independent(cache['labels'],oof,k)
            assert np.array_equal(cm,icm) and np.allclose(ipr,np.array([precision,recall,f1]).T,atol=1e-14)
            records.append(dict(model=name,seed=seed,fold='OOF',samples=len(oof),**m))
            for j,c in enumerate(cache['classes']):prf.append(dict(model=name,seed=seed,label=str(c),support=int(cm[j].sum()),precision=float(precision[j]),recall=float(recall[j]),f1=float(f1[j])))
            for i,c in enumerate(cache['classes']):
                for j,d in enumerate(cache['classes']):counts.append(dict(model=name,seed=seed,true_label=str(c),predicted_label=str(d),count=int(cm[i,j])))
            np.savez_compressed(OUTPUT/f'oof_{name}_s{seed}.npz',sample_ids=cache['sample_ids'],groups=cache['groups'],labels=cache['labels'],predictions=oof)
    assert pred_count==len(cache['labels'])*len(NAMES)*len(SEEDS)
    summary=[]
    for name in NAMES:
        for metric in ['macro_f1','accuracy','macro_precision','macro_recall']:
            vals=np.array([r[metric] for r in records if r['model']==name and r['fold']=='OOF'])
            summary.append(dict(model=name,metric=metric,mean=float(vals.mean()),sd=float(vals.std(ddof=1)),n_training_seeds=3,
                note='OOF pooled over3folds per seed; SD is training-seed variation, not independent dataset uncertainty'))
    paired=[]
    for a,b in [('M2','M1'),('M3','M1'),('M4','M1'),('M5','M1'),('M5','M3')]:
        for seed in SEEDS:
            va=next(r['macro_f1'] for r in records if r['model']==a and r['seed']==seed and r['fold']=='OOF')
            vb=next(r['macro_f1'] for r in records if r['model']==b and r['seed']==seed and r['fold']=='OOF')
            paired.append(dict(a=a,b=b,seed=seed,difference_pp=100*(va-vb),interpretation='descriptive paired training-seed comparison, no significance claim'))
    csv_write(OUTPUT/'per_fold_seed.csv',records);csv_write(OUTPUT/'summary.csv',summary);csv_write(OUTPUT/'per_class_prf.csv',prf)
    csv_write(OUTPUT/'confusion_counts.csv',counts);csv_write(OUTPUT/'paired_differences.csv',paired)
    figdir=OUTPUT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(6.4,4))
    entries=[r for r in summary if r['metric']=='macro_f1']
    ax.bar(NAMES,[r['mean']*100 for r in entries],yerr=[r['sd']*100 for r in entries],capsize=4,color=['#788b98']*4+['#0c7c86'])
    ax.set(ylabel='Out-of-fold Macro-F1 (%)',ylim=(0,100),title='WHITED clean-data comparison: 15 eligible classes')
    ax.text(0,-.19,'Mean ± sample SD over 3 paired training seeds; 3 device-group folds.\n759 recordings / 76 filename device groups; no significance claims.',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'macro_f1.png',dpi=600,bbox_inches='tight');fig.savefig(figdir/'macro_f1.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(18,4.5),sharey=True)
    for ax,name in zip(axes,NAMES):
        cm=sum(metrics(cache['labels'],all_oof[name,seed],k)[1] for seed in SEEDS)
        norm=cm/cm.sum(1,keepdims=True);ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        ax.set_title(name);ax.set_xticks(range(k),cache['classes'],rotation=90,fontsize=6)
        ax.set_yticks(range(k),cache['classes'],fontsize=7);ax.set_xlabel('Predicted')
    axes[0].set_ylabel('True class');fig.suptitle('Row-normalized OOF confusion, pooled counts across 3 training seeds (repeated predictions)')
    fig.tight_layout();fig.savefig(figdir/'confusion.png',dpi=600,bbox_inches='tight');fig.savefig(figdir/'confusion.svg',bbox_inches='tight');plt.close(fig)
    report=['# E10 WHITED M1–M5結果報告','',f'研究完成時間：{now()}。實際執行與獨立重算已完成；正式雙語回應稿與主文整合另列。','',
        '## 資料與前處理','',
        '官方下載包SHA見prepared_audit.json。原始包頂層1339筆FLAC；README定義r1–r8的1259筆與官網數量一致，另80筆r9未定義而先排除。僅納入至少3個設備名稱群組的15類，共76群組／759筆；不代表全WHITED。完整納入／排除見sample_manifest.json及class_coverage.csv。',
        '通道0乘kit電壓校準、通道1乘電流校準；通道順序由所有納入波形的物理電壓與頻率核對，非README明文指定。以類別＋設備名稱＋地區分組，未核實機身序號。',
        '最大電流一階差分定位，前置20%的一個額定電網週期（50/60Hz由電壓頻譜判斷），polyphase反混疊重採樣至500點；V/I各自逐波形z-score、±3σ映射並clip到64×64，每點累積.1後飽和，uint8量化與原Experiment1 PNG輸入一致。不能稱所有視窗均驗證為開啟暫態或穩態。',
        '', '## 切分與訓練','',
        '三折分層設備群組CV，每個納入群組只當一次outer test。每折outer train內每類保留一個群組作inner-validation，其餘inner-train。所有集合群組互斥且全類別覆蓋。少數類別只有3個群組，內層僅1個訓練群組，必須保留此限制。',
        'M1–M5原架構、Adam lr0.0005、weight_decay0、batch32、dropout0.5；M5原detached residual兩pass與.5CE1+CE2+.5MSE。每模型每折一個seed42 epoch pilot，最多100輪，最早20輪開始選擇、patience15、min_delta.001；15組全部選定後，各模型各折seed42–44在完整outer train從頭訓練，共45次正式訓練。全部訓練完成才開outer test。',
        '此輪只有clean，無防禦選策／污染測試／PLAID→WHITED零樣本轉移。固定原optimizer，未做新HPO；三seed及三fold只作描述，不宣稱確認性顯著、等效或普遍優勢。',
        '', '## 完整模型結果','', '| 模型 | OOF Macro-F1 (%) | Accuracy (%) |','|---|---:|---:|']
    for name in NAMES:
        f=next(r for r in summary if r['model']==name and r['metric']=='macro_f1');a=next(r for r in summary if r['model']==name and r['metric']=='accuracy')
        report.append(f"| {name} | {100*f['mean']:.2f} ± {100*f['sd']:.2f} | {100*a['mean']:.2f} ± {100*a['sd']:.2f} |")
    report += ['', '每seed先合併三折out-of-fold預測再算指標，表中為3個訓練seed之平均±樣本SD，不是獨立資料集不確定性。所有逐折、逐seed、逐類、完整混淆及有利／不利差異均保留於CSV。',
        '', '## 驗證與交付界線','',f'{pred_count:,}筆保存test預測已逐筆對ID／label／fold，另以Boolean TP/FP/FN及計數重算PRF、Macro-F1與混淆一致。模型／BN於推論前後相同；設定、split與資料SHA核對通過。',
        '600dpi PNG／SVG由程式產生，後續仍需目視QA。正式Response Word、主文修訂及新前處理流程圖尚未完成；依作者要求，圖需待實驗結果與方法經作者定案，再參照主文原圖重製。學生Windows/CUDA重跑、Drive同步均未驗證。','']
    (OUTPUT/'E10結果報告.md').write_text('\n'.join(report))
    audit=dict(all_checks_passed=True,at=now(),models=5,folds=3,seeds=SEEDS,epoch_pilots=15,formal_trainings=45,
        selected_classes=k,selected_groups=len(set(cache['groups'])),selected_samples=len(cache['labels']),independently_recomputed_predictions=pred_count,
        all_prf_confusion_recomputed=True,source_hashes_verified=True,all_splits_group_disjoint=True,source_sha256=plan['source_sha256'],
        formal_response_docx_complete=False,figure_visual_inspection=False,manuscript_modified=False,main_preprocessing_figure_redrawn=False,
        summary_sha256=sha(OUTPUT/'summary.csv'),report_sha256=sha(OUTPUT/'E10結果報告.md'))
    save(OUTPUT/'independent_audit.json',audit)
    save(OUTPUT/'completed_audit.json',audit)
    # Results archive includes small source/input/predictions, excludes optimizer/weights.
    archive=ROOT/'E10_results_20261108.zip';paths=[]
    for p in OUTPUT.rglob('*'):
        if not p.is_file() or p.suffix in ['.pt','.tmp'] or p.name in ['writer.lock','queue.lock'] or p.name.endswith('.log'):continue
        paths.append(p)
    manifest={str(p.relative_to(OUTPUT)):sha(p) for p in paths};save(OUTPUT/'package_manifest.json',manifest)
    with zipfile.ZipFile(archive.with_suffix('.partial'), 'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[OUTPUT/'package_manifest.json']:z.write(p,'E10_WHITED/'+str(p.relative_to(OUTPUT)))
    os.replace(archive.with_suffix('.partial'),archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,digest in manifest.items():assert hashlib.sha256(z.read('E10_WHITED/'+rel)).hexdigest()==digest
    save(OUTPUT/'delivery_audit.json',dict(all_checks_passed=True,zip_crc_and_manifest_sha256=True,archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,at=now()))
    save(OUTPUT/'progress.json',dict(stage='research_complete',at=now(),completed_trainings=45))
    description=f'**E10 WHITED研究計算與分析完成（{now()}）**：15類／76設備名稱群組／759筆，三折group CV、15epoch pilot及M1–M5各三seed共45正式訓練，11,385筆test預測獨立重算與結果ZIP通過。僅clean資料，所有模型完整結果保留；原始包r9未定義80筆及設備不足類別先排除，非全WHITED。入口`Code/RevisedCode/results/e10_whited_20261108/README.md`。研究完成不等於正式Word、主文整合或主文前處理新圖完成；圖需待作者定案後依原圖重製。圖表目視QA及學生CUDA另列。'
    sync_status(description)
    print('E10_RESEARCH_COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':analyze()
