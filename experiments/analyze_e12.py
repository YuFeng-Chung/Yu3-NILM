"""E12 complete predeclared analysis, independent audit, figures and delivery.

Analysis plan is frozen in protocol.json before the first new test inference.
This implementation is snapshotted on entry; no strategy or endpoint selection.
"""
import csv
import hashlib
import itertools
import shutil
import time
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e12_core import *
from analyze_e5_full import paired_test,holm

def snapshot_analysis(out):
    names=['analyze_e12.py','verify_e12.py','analyze_e5_full.py']
    sources={}
    for n in names:
        p=Path(__file__).with_name(n);rel=str(p.relative_to(ROOT));sources[rel]=sha(p)
        target=out/'analysis_source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    save_json(out/'analysis_implementation.json',dict(source_sha256=sources,
        note='Analysis implementation finalized during inference; complete endpoints/contrasts/global family and bootstrap already frozen before new test inference. No outcome-dependent changes.',
        started_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()))

def metrics_and_summary(out,plan):
    averaged=[];realizations=[];class_rows=[];records={};groups={}
    for ci,c in enumerate(CONDITIONS):groups.setdefault(c['id'],[]).append(ci)
    metricdir=out/'metrics';metricdir.mkdir(exist_ok=True)
    for entry in plan['manifest']:
        key=Path(entry['path']).name;saved=np.load(out/'predictions'/f'{key}.npz');p=saved['predictions'];labels=saved['labels']
        assert p.shape==(126,219)
        values=[];cms=[];precisions=[];recalls=[];f1s=[];accuracies=[]
        for ci,(pred,cond) in enumerate(zip(p,CONDITIONS)):
            score,cm,pr,re,f1=independent_scores(labels,pred)
            values.append(score['macro_f1']);accuracies.append(score['accuracy']);cms.append(cm);precisions.append(pr);recalls.append(re);f1s.append(f1)
            realizations.append(dict(model=entry['model'],strategy=entry['strategy'],seed=entry['seed'],condition_index=ci,
                condition=cond['id'],group=cond['group'],pollution_seed=cond['pollution_seed'],source=cond['source'],**score))
        values=np.array(values);precisions=np.stack(precisions);recalls=np.stack(recalls);f1s=np.stack(f1s)
        atomic_npz(metricdir/f'{key}.npz',f1=values,accuracy=np.array(accuracies),confusion=np.stack(cms),
            precision=precisions,recall=recalls,class_f1=f1s,sample_ids=saved['sample_ids'],labels=labels)
        bycondition={}
        for condition,indices in groups.items():
            f1=float(values[indices].mean());drop=100*(values[0]-f1);bycondition[condition]=(f1*100,drop)
            cond=CONDITIONS[indices[0]]
            averaged.append(dict(model=entry['model'],strategy=entry['strategy'],seed=entry['seed'],condition=condition,
                group=cond['group'],n_realizations=len(indices),source=cond['source'],f1_pct=f1*100,
                clean_f1_pct=values[0]*100,Dabs_pp=drop,accuracy_pct=float(np.array(accuracies)[indices].mean()*100)))
            # Average PRF across realizations, not from a pooled confusion matrix.
            for cls in range(11):
                class_rows.append(dict(model=entry['model'],strategy=entry['strategy'],seed=entry['seed'],condition=condition,
                    class_index=cls,n_realizations=len(indices),precision=float(precisions[indices,cls].mean()),
                    recall=float(recalls[indices,cls].mean()),f1=float(f1s[indices,cls].mean()),
                    support=int((labels==cls).sum())))
        records[entry['model'],entry['strategy'],entry['seed']]=bycondition
    csv_write(out/'per_realization_metrics.csv',realizations);csv_write(out/'per_seed_averaged.csv',averaged)
    csv_write(out/'per_class_seed_averaged.csv',class_rows)
    summary=[];bygroup={}
    for r in averaged:bygroup.setdefault((r['model'],r['strategy'],r['condition']),[]).append(r)
    for (model,strategy,condition),rs in bygroup.items():
        assert sorted(r['seed'] for r in rs)==SEEDS
        f1=np.array([r['f1_pct'] for r in rs]);d=np.array([r['Dabs_pp'] for r in rs]);acc=np.array([r['accuracy_pct'] for r in rs])
        summary.append(dict(model=model,strategy=strategy,condition=condition,group=rs[0]['group'],source=rs[0]['source'],
            n_training_seeds=10,n_realizations_per_seed=rs[0]['n_realizations'],f1_mean_pct=float(f1.mean()),f1_sd_pct=float(f1.std(ddof=1)),
            Dabs_mean_pp=float(d.mean()),Dabs_sd_pp=float(d.std(ddof=1)),accuracy_mean_pct=float(acc.mean()),accuracy_sd_pct=float(acc.std(ddof=1))))
    csv_write(out/'summary.csv',summary)
    return records,summary

def comparisons(out,plan,records):
    boot=np.random.default_rng(20261108).integers(0,10,(10000,10));tests=[];diffs=[]
    def vec(m,s,c,endpoint):return np.array([records[m,s,seed][c][0 if endpoint=='F1' else 1] for seed in SEEDS])
    for comparison in plan['comparisons']:
        a,sa,b,sb=[comparison[k] for k in ['a','sa','b','sb']]
        for condition in ['clean']+plan['new_condition_ids']:
            for endpoint in ['F1'] if condition=='clean' else ['F1','Dabs']:
                d=vec(a,sa,condition,endpoint)-vec(b,sb,condition,endpoint)
                test_id=f'T{len(tests):04}';_,p=paired_test(d);low,high=np.quantile(d[boot].mean(1),[.025,.975])
                tests.append(dict(test_id=test_id,comparison=f'{a}/{sa} - {b}/{sb}',contexts=';'.join(comparison['contexts']),
                    a=a,strategy_a=sa,b=b,strategy_b=sb,condition=condition,endpoint=endpoint,n=10,
                    mean_difference_pp=float(d.mean()),sd_difference_pp=float(d.std(ddof=1)),CI95_mean_low_pp=float(low),CI95_mean_high_pp=float(high),
                    wins_a=int((d>1e-12).sum()) if endpoint=='F1' else int((d< -1e-12).sum()),
                    p_raw=p,interpretation='positive favors A' if endpoint=='F1' else 'negative favors A'))
                diffs.extend(dict(test_id=test_id,seed=seed,difference_pp=float(v)) for seed,v in zip(SEEDS,d))
    assert len(tests)==plan['analysis']['tests_planned']
    for r,p in zip(tests,holm([r['p_raw'] for r in tests])):r['p_holm_global']=float(p);r['holm_family_size']=len(tests)
    csv_write(out/'paired_tests.csv',tests);csv_write(out/'paired_differences.csv',diffs)
    components=[];component_diffs=[]
    for (model,strategy,seed),values in records.items():
        for combination,parts in COMBOS.items():
            for typ,severity in parts:
                single=f'{typ}_{severity:g}';delta=values[combination][0]-values[single][0]
                component_diffs.append(dict(model=model,strategy=strategy,seed=seed,combination=combination,
                    constituent=single,combined_f1_pct=values[combination][0],constituent_f1_pct=values[single][0],
                    combined_minus_constituent_pp=delta))
    grouped={}
    for r in component_diffs:grouped.setdefault((r['model'],r['strategy'],r['combination'],r['constituent']),[]).append(r)
    for (model,strategy,combination,single),rows in grouped.items():
        rows=sorted(rows,key=lambda r:r['seed']);d=np.array([r['combined_minus_constituent_pp'] for r in rows]);low,high=np.quantile(d[boot].mean(1),[.025,.975])
        components.append(dict(model=model,strategy=strategy,combination=combination,constituent=single,n=10,
            mean_combined_minus_constituent_pp=float(d.mean()),sd_difference_pp=float(d.std(ddof=1)),
            CI95_mean_low_pp=float(low),CI95_mean_high_pp=float(high),note='descriptive; no confirmatory p test; no superadditivity claim'))
    csv_write(out/'combined_vs_components.csv',components);csv_write(out/'combined_vs_components_per_seed.csv',component_diffs)
    return tests,components

def plots(out,plan,summary):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    lookup={(r['model'],r['strategy'],r['condition']):r for r in summary}
    conditions=['clean']+plan['new_condition_ids'];models=FAMILY+MAIN[4:]
    short={'clean':'Clean','awgn20_consecutive10':'AWGN20 + gap10%','awgn10_consecutive20':'AWGN10 + gap20%',
           'sp01_random05':'SP1% + random5%','added_harmonics_3pct':'Added H3/5 3%','added_harmonics_5pct':'Added H3/5 5%'}
    labels=[short.get(c,c.replace('adc_','ADC ').replace('dc_','DC ').replace('gain_','Gain ')) for c in conditions]
    for field,title,cmap,limits in [('f1_mean_pct','Macro-F1 (%)','viridis',(0,100)),('Dabs_mean_pp','Clean minus perturbed F1 (pp)','coolwarm',(-15,75))]:
        fig,axes=plt.subplots(3,1,figsize=(15,13),layout='constrained')
        for ax,strategy in zip(axes,COMMON):
            values=np.array([[lookup[m,strategy,c][field] for c in conditions] for m in models])
            im=ax.imshow(values,aspect='auto',cmap=cmap,vmin=limits[0],vmax=limits[1]);ax.set_yticks(range(len(models)),models)
            ax.set_xticks(range(len(conditions)),labels,rotation=50,ha='right',fontsize=8);ax.set_title(strategy)
            for i in range(len(models)):
                for j in range(len(conditions)):
                    ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=6,color='white' if (field=='f1_mean_pct' and values[i,j]<50) else 'black')
        fig.colorbar(im,ax=axes,label=title,shrink=.65);fig.suptitle(title+'; mean over ten training seeds; all fixed strategies')
        fig.savefig(figdir/f'{field}.png',dpi=600);plt.close(fig)
    palette=plt.get_cmap('tab10')
    for strategy in COMMON:
        fig,axes=plt.subplots(1,3,figsize=(15,5.5),sharey=True,layout='constrained')
        for ax,(combo,parts) in zip(axes,COMBOS.items()):
            cs=[f'{t}_{v:g}' for t,v in parts]+[combo]
            for mi,model in enumerate(MAIN):
                points=[lookup[model,strategy,c]['f1_mean_pct'] for c in cs]
                ax.plot(range(3),points,'o-',label=model,color=palette(mi),lw=1.4,ms=4)
            ax.set_xticks(range(3),[cs[0],cs[1],'Combined'],rotation=20,ha='right',fontsize=8);ax.set_title(short[combo]);ax.set_ylim(0,100);ax.grid(alpha=.2)
        axes[0].set_ylabel('Macro-F1 (%)');axes[-1].legend(fontsize=8,loc='lower left')
        fig.suptitle(strategy+': constituent controls and combined stress (descriptive)')
        fig.savefig(figdir/f'components_{strategy}.png',dpi=600);plt.close(fig)
    # Outcome-independent example: lexicographically smallest held-out sample ID.
    raw,tr,te=load_data();j=min(range(len(te)),key=lambda j:str(raw['sample_ids'][te[j]]))
    names=['clean','awgn20_consecutive10','awgn10_consecutive20','sp01_random05','adc_8bit','added_harmonics_5pct','dc_i_+0.05','dc_v_+0.05','gain_i_+0.05']
    fig,axes=plt.subplots(3,len(names),figsize=(18,6.5),layout='constrained')
    for col,name in enumerate(names):
        ci=next(i for i,c in enumerate(CONDITIONS) if c['id']==name);z=np.load(out/'waveform_cache'/f'{ci:03}.npz')
        axes[0,col].plot(z['current'][j],lw=.6);axes[1,col].plot(z['voltage'][j],lw=.6)
        axes[2,col].imshow(z['image'][j,0],origin='lower',cmap='gray',vmin=0,vmax=1)
        axes[0,col].set_title(short.get(name,name),fontsize=8)
        for row in range(3):axes[row,col].tick_params(labelsize=6)
    axes[0,0].set_ylabel('Current (A)');axes[1,0].set_ylabel('Voltage (V)');axes[2,0].set_ylabel('V-I image')
    fig.suptitle(f'Fixed illustrative sample ID {raw["sample_ids"][te[j]]}; first phase/noise realization; no outcome selection')
    fig.savefig(figdir/'waveform_image_example.png',dpi=600);plt.close(fig)
    save_json(figdir/'manifest.json',dict(dpi=600,files=[dict(file=p.name,sha256=sha(p)) for p in sorted(figdir.glob('*.png'))],
        representative_sample_id=str(raw['sample_ids'][te[j]]),selection='lowest lexicographic test ID; not selected by model correctness'))

def write_report(out,plan,summary,tests,components):
    lookup={(r['model'],r['strategy'],r['condition']):r for r in summary}
    inference=json.loads((out/'inference_audit.json').read_text());audit=json.loads((out/'independent_audit.json').read_text())
    lines=['# E12 組合擾動與量測誤差：完整結果報告','',
        '本輪目的為回應 R2-6，補充原論文 Experiment 4 的測試條件。全部為電腦端合成擾動，沒有新增訓練、真實現場擷取或部署實測。','',
        '## 狀態與研究範圍','',
        f'- E12 實驗、結果分析及程式獨立稽核完成；正式英中 Word 回應稿與原論文整合尚未製作。',
        f'- 330 組既有 checkpoint；九種架構（七種主要模型、ZeroFB／ShuffleFB 補充控制）；訓練 seed 42–51。',
        '- 每組 66 個新推論條件（含 clean）、60 個既有單一擾動實例，共 126 個條件；219 個測試樣本。',
        f'- 保存預測總數 {inference["predictions"]:,}，其中本輪新推論 {inference["new_inference_predictions"]:,}、重用單獨擾動預測 {inference["reused_component_predictions"]:,}。',
        f'- 程序推論起迄：{inference["started_at"]} 至 {inference["ended_at"]}；正式推論 {inference["elapsed_seconds"]/60:.2f} 分鐘，另有後續分析與稽核。',
        '- 既有 E1–E9／E11、原始資料、模型、主文、補充稿及其他回應稿均不改。','',
        '## 方法與邊界','',
        '同一 PLAID 855/219 切分與 500 點視窗；只用 outer-train 統計做固定電壓／電流正規化、64×64 occupancy V–I 成像。保留所有原 checkpoint 與 E3 並列策略；不以 E12 test 選策略或修改超參數。','',
        '| 類別 | 實作 | 解讀界線 |','|---|---|---|',
        '| 組合 | AWGN20＋連續缺失10%；AWGN10＋連續缺失20%；椒鹽1%＋隨機缺失5%。先噪聲，再缺失與線性補值 | 同一 component RNG key 與舊 E4 單獨擾動，缺失 V/I 同步；端點用最近有效值延伸 |',
        '| ADC | V/I 各依訓練 min/max、8/10/12 bit、先飽和，再等間距 half-up 量化 | 這是指定 full scale 的模擬；訓練極值可能造成較寬量程，不代表特定儀器 |',
        '| 諧波 | 僅 V 加入 3/5 次諧波，新增諧波 RMS／估計基頻 RMS 為3%或5%，各10個相位實例 | 保留原電壓失真／暫態，故不是保證最終 THD=3%/5%；I不隨電壓重新求解，非電路模擬 |',
        '| DC偏移 | I/V分開，各為訓練全範圍±1%、±5%，其他通道不變 | 正負皆報；寬範圍敏感度測試，並非已校準的真實感測器誤差 |',
        '| 電流增益 | I乘以0.95/0.98/1.02/1.05，V不變 | 不再以測試資料重新正規化 |','',
        'ADC／DC／增益為確定性條件，每個 checkpoint 只評估一次；組合與諧波各10個污染／相位 seed。先在每個訓練 seed 內平均實例 Macro-F1，再在10個訓練 seed間報 mean±sample SD；不把污染實例當成更多獨立訓練。','',
        '## 完整共同策略結果','',
        '以下為 Macro-F1（%，mean±SD，n=10）。完整九模型、各策略的退化量及原始 seed 值見 CSV；Dabs=clean F1−擾動 F1，正值代表退化。不能只憑退化量小忽略絕對F1低。','']
    cs=['clean']+plan['new_condition_ids']
    for strategy in COMMON:
        lines.extend([f'### {strategy}','','| 條件 | '+' | '.join(MAIN)+' |','|---|'+'---|'*len(MAIN)])
        for cond in cs:
            cells=[f'{lookup[m,strategy,cond]["f1_mean_pct"]:.2f}±{lookup[m,strategy,cond]["f1_sd_pct"]:.2f}' for m in MAIN]
            lines.append('| '+cond+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines.extend(['## 組合相對單獨擾動的結果','','下表逐一列出 Yu3；其餘模型完整保存在 `combined_vs_components.csv`。差值為組合F1−單獨擾動F1，負值代表組合更低；區間僅描述訓練seed波動，不宣稱擾動交互作用或超加成。','',
        '| 策略 | 組合 | 單獨對照 | F1差(pp) | 描述性95%區間 |','|---|---|---|---:|---|'])
    for r in components:
        if r['model']=='Yu3':lines.append(f'| {r["strategy"]} | {r["combination"]} | {r["constituent"]} | {r["mean_combined_minus_constituent_pp"]:.2f} | [{r["CI95_mean_low_pp"]:.2f}, {r["CI95_mean_high_pp"]:.2f}] |')
    lines.extend(['','## 公平選策與配對比較','','E3 已選定 Yu3 噪聲 S3、缺失 Combined；三基準的噪聲 S3/S4、缺失 Augmentation/Combined 並列全部保留。E12 沒有專屬 inner 選策，因此兩種既有選策軌均對所有新擾動完整報告，不將其中較好的 test 結果另稱 E12 最佳策略。','',
        f'共 {len(plan["comparisons"])} 個唯一模型／策略對比、{len(tests)} 個 clean F1／新擾動 F1及Dabs檢定；同一對比的共同／選策情境合併標籤避免重複計數，完整全域 Holm 不縮小。',
        f'全域校正 p<.05 的數量為 {sum(r["p_holm_global"]<.05 for r in tests)}。n=10 的最小雙尾精確 p=2/1024≈.001953；首步門檻 .05/{len(tests)}≈{.05/len(tests):.8f}，本設計結構上不能拒絕任何單項。**不能以未顯著作無影響或等效證據。**','',
        '所有配對差值、原始p、全域校正p與10,000次配對seed bootstrap區間保存在 `paired_tests.csv`／`paired_differences.csv`。單一歷史test、類別×住宅group代理、跨實驗已看test、Mac MPS非學生獨立CUDA重現，均限制推廣。Yu3與控制同時有推論次數及loss組成等差異，不稱純回饋因果。','',
        '## 獨立驗證與檔案','',
        f'- {audit["waveforms_verified"]:,} 組波形逐條獨立重建，{audit["images_pixel_exact"]:,} 張影像逐像素完全一致；諧波用獨立正交投影對照最小平方實作。',
        f'- {audit["predictions_recomputed"]:,} 筆預測以 sklearn 另重算，{audit["confusion_cells_verified"]:,} 個混淆計數及全部類別PRF核對。',
        f'- {audit["paired_tests_verified"]:,} 組檢定另列舉全部符號、獨立Holm與bootstrap核對。',
        '- ADC飽和、固定尺度下超過±3及實際索引clip、影像IoU/L1、諧波加入比例與前後3/5次成分均保存在 `waveform_image_diagnostics.csv`。',
        '- `protocol.json`、`preflight_audit.json`、`inference_audit.json`、`independent_audit.json`、`completed_audit.json` 分別對應凍結設計、前測、推論、獨立重算與研究交付。',
        '- `predictions/` 保存各checkpoint完整126條件預測與sample ID；`metrics/`保存每實例PRF與混淆矩陣；`waveform_cache/`保存新條件波形與影像。',
        '- `summary.csv`、`per_seed_averaged.csv`、`per_realization_metrics.csv`、`per_class_seed_averaged.csv` 為可重算數據。',
        '- `figures/` 為600 dpi圖；示例依測試sample ID字典序選第一筆，非依有利結果選圖。','',
        '## 審稿定位與待辦','','可用於 R2-6 的補充材料，提供多重合成擾動及指定量測誤差下的表現邊界。真實量測噪聲、實際擷取條件與現場資料仍未驗證，不能寫成完全解決真實場域適用性。依作者安排，13輪工作結束後再整合原稿；正式雙語回應稿尚待另外製作。',''])
    (out/'E12結果報告.md').write_text('\n'.join(lines))
    (out/'README.md').write_text('# E12 研究結果入口\n\n先讀 `E12結果報告.md` 與 `completed_audit.json`。本輪實驗／分析／獨立稽核完成；正式Word回應稿與原稿整合尚未做。\n\n全部合成條件，沒有重訓或部署／現場驗證。完整設計見 `protocol.json`；啟動與安全暫停由 `../../experiments/control_e12.py` 管理。完成後不要重啟。\n\n打包檔不重複封裝舊模型與原資料，需與原E1/E3/E4和PLAID來源搭配；`partial/`是程序中間結果，正式完整輸出以 `predictions/` 為準。\n')

def package(out):
    excluded={'partial','writer.lock','local_process.json','run.log','stderr.log','progress.json','STOP_REQUESTED','PAUSED.json','FAILED.json','delivery_manifest.json','delivery_audit.json'}
    files=[p for p in out.rglob('*') if p.is_file() and not any(part in excluded for part in p.relative_to(out).parts)]
    manifest={str(p.relative_to(out)):sha(p) for p in files}
    save_json(out/'delivery_manifest.json',dict(files=manifest,excluded='Original models/raw archive not duplicated; transient process/partial files excluded'))
    target=ROOT/'E12_results_20261108.zip'
    assert not target.exists(),'Preserve existing delivery ZIP'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[out/'delivery_manifest.json']:z.write(p,Path(out.name)/p.relative_to(out))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for rel,digest in manifest.items():assert hashlib.sha256(z.read(str(Path(out.name)/rel))).hexdigest()==digest
    save_json(out/'delivery_audit.json',dict(all_checks_passed=True,zip_crc_passed=True,zip_all_file_sha256_passed=True,
        zip=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha(target),files=len(files)+1))

def sync_shared_completion(out):
    """Re-read shared records and replace only the E12 status markers/row."""
    import re
    stamp=datetime.now(ZoneInfo('Asia/Taipei')).strftime('%Y-%m-%d %H:%M')
    description=(f'**E12研究運算與分析完成（{stamp}，台灣時間）**：330組既有checkpoint、每組66個新推論與60個單獨擾動對照完成，'
        '共9,106,020筆預測；14,454組波形／影像、全部PRF／混淆與1,353配對檢定獨立核對通過。'
        '合成組合／ADC／電壓諧波／DC及電流增益，不重訓、不依test選模型。'
        '入口`Code/RevisedCode/results/e12_measurement_20261108/README.md`；'
        '研究ZIP`Code/RevisedCode/E12_results_20261108.zip`，CRC與逐檔SHA通過。'
        '本旗標僅研究運算／程式稽核完成；圖表尚待目視、正式雙語Word與原稿整合未做。'
        'n10及完整Holm的解析度限制、非真實場域與非純回饋因果保留；不啟動E10/E13。')
    block='<!-- E12_STATE_START -->\n'+description+'\n<!-- E12_STATE_END -->'
    paths=[ROOT.parent.parent/'PROJECT_CONTEXT.md',ROOT/'session_handoff_20261106.md']
    for path in paths:
        text=path.read_text();pattern=r'<!-- E12_STATE_START -->[\s\S]*?<!-- E12_STATE_END -->'
        if len(re.findall(pattern,text))==1:
            updated=re.sub(pattern,lambda _:block,text,count=1)
            path.write_text(updated)
    path=ROOT/'實驗程式架構總覽.md';text=path.read_text()
    row='| E12 | 組合擾動及量測誤差補充 | 原330組模型／219 test／固定train尺度 | 66新推論＋60單獨對照；無重訓 | 研究分析／獨立稽核／ZIP完成；圖表目視與Word待做 | 研究計算完成，回應稿待做 |'
    text=re.sub(r'^\| E12 \| 組合擾動及量測誤差補充[^\n]*',lambda _:row,text,count=1,flags=re.M)
    text=text.replace('**目前十輪實驗／分析／回應稿完成，E12執行中、兩輪未進行：E1–E9及E11完成；E12已授權並啟動；E10、E13尚未進行。**',
        '**目前E1–E9及E11十輪實驗／分析／回應稿完成；E12研究計算與獨立稽核完成，圖表目視及回應稿待做；E10、E13尚未進行。**')
    path.write_text(text)
    save_json(out/'shared_record_sync.json',dict(synced_at=stamp,scope='E12 marked status blocks and main status row only; history preserved',
        paths=['PROJECT_CONTEXT.md','Code/RevisedCode/session_handoff_20261106.md','Code/RevisedCode/實驗程式架構總覽.md']))

def analyze(out):
    tick=time.monotonic();plan=json.loads((out/'protocol.json').read_text())
    assert json.loads((out/'inference_audit.json').read_text())['all_checks_passed']
    assert len(list((out/'predictions').glob('*.npz')))==330
    snapshot_analysis(out);records,summary=metrics_and_summary(out,plan);tests,components=comparisons(out,plan,records)
    save_json(out/'progress.json',dict(stage='independent_reconstruction_and_audit',completed_models=330,planned_models=330))
    from verify_e12 import verify
    audit=verify(out)
    plots(out,plan,summary);write_report(out,plan,summary,tests,components)
    for rel,digest in plan['source_sha256'].items():assert sha(ROOT/rel)==digest
    completed=dict(all_checks_passed=True,experiment='E12',models=330,conditions_per_model=126,training=False,
        experiment_complete=True,analysis_complete=True,independent_audit_complete=True,
        figure_visual_inspection=False,formal_response_docx_complete=False,original_manuscript_modified=False,
        completed_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),analysis_seconds=time.monotonic()-tick,
        independent_audit_sha256=sha(out/'independent_audit.json'),summary_sha256=sha(out/'summary.csv'),
        report_sha256=sha(out/'E12結果報告.md'),scope='Research calculation and programmatic audit complete; human/agent visual figure inspection pending; Word response and manuscript not included')
    save_json(out/'completed_audit.json',completed);package(out);sync_shared_completion(out)
    print('E12_RESEARCH_COMPLETE',json.dumps(completed,ensure_ascii=False),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();analyze(a.output)
