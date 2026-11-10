"""E9 complete descriptive summaries, paired factor changes and signal diagnostics."""
import pandas as pd
from e9_core import *

def diagnostic_rows(output,raw,split):
    rows=[];rms=[]
    for cell,(window,norm) in CELLS.items():
        stats=fit_stats(raw,split['outer_train'],window)
        for j in split['outer_test']:
            c0,v0=waveform(raw,j,window,stats,None);base=scatter(*standardized(c0,v0,stats,norm));mask0=base[0]>0
            rms.append(dict(cell=cell,sample_id=str(raw['sample_ids'][j]),label=int(raw['labels'][j]),current_rms=float(np.sqrt(np.mean(c0**2))),voltage_rms=float(np.sqrt(np.mean(v0**2))),current_mean=float(np.mean(c0))))
            for kind,severity,seeds in [('clean',0,[0]),('awgn',30,range(1000,1010)),('awgn',20,range(1000,1010)),('salt_pepper',.01,range(1000,1010)),('salt_pepper',.2,range(1000,1010))]:
                for seed in seeds:
                    c,v=waveform(raw,j,window,stats,(kind,severity,seed));iz,vz=standardized(c,v,stats,norm);img=scatter(iz,vz);mask=img[0]>0
                    union=int((mask|mask0).sum());inter=int((mask&mask0).sum());row=dict(cell=cell,sample_id=str(raw['sample_ids'][j]),label=int(raw['labels'][j]),kind=kind,severity=severity,pollution_seed=seed,
                        image_iou=inter/union if union else 1.,new_nonzero=int((mask&~mask0).sum()),lost_nonzero=int((mask0&~mask).sum()),
                        clean_nonzero=int(mask0.sum()),perturbed_nonzero=int(mask.sum()),image_mean_abs_change=float(np.abs(img-base).mean()))
                    for prefix,x,z,x0 in [('i',c,iz,c0),('v',v,vz,v0)]:
                        idx=((z+3)/6*63).astype(int);delta=x.astype(np.float64)-x0
                        row[prefix+'_beyond_3sigma_fraction']=float(np.mean(np.abs(z)>3))
                        row[prefix+'_index_clipped_fraction']=float(np.mean((idx<0)|(idx>63)))
                        row[prefix+'_rms']=float(np.sqrt(np.mean(x.astype(np.float64)**2)))
                        row[prefix+'_noise_mean_square']=float(np.mean(delta**2))
                        row[prefix+'_signal_variance']=float(np.var(x0.astype(np.float64)))
                        row[prefix+'_realized_snr_db']=float(10*np.log10(np.var(x0.astype(np.float64))/np.mean(delta**2))) if kind=='awgn' and np.mean(delta**2)>0 and np.var(x0)>0 else None
                    rows.append(row)
    csv_write(output/'diagnostics_per_sample.csv',rows);csv_write(output/'rms_per_sample.csv',rms)
    frame=pd.DataFrame(rows)
    columns=[c for c in frame if c not in ['cell','sample_id','label','kind','severity','pollution_seed']]
    frame.groupby(['cell','kind','severity'])[columns].mean().reset_index().to_csv(output/'diagnostics_summary.csv',index=False)
    return frame

def analyze(output=OUTPUT):
    plan=load_plan(output);raw=dict(np.load(output/'waveforms.npz'));split=json.loads((output/'split.json').read_text())['indices']
    per=[];class_rows=[];cm_rows=[]
    for seed in SEEDS:
        for cell in CELLS:
            for name in MODELS:
                folder=output/'runs'/f'{cell}_{name}_s{seed}';identity=np.load(folder/'test_identity.npz');y=identity['labels'];clean=None
                for ci,condition in enumerate(CONDITIONS):
                    pred=np.load(folder/'predictions'/f'{ci:03}.npy');m=metrics(y,pred)
                    if ci==0:clean=m['macro_f1']
                    per.append(dict(cell=cell,model=name,training_seed=seed,kind=condition[0],severity=condition[1],pollution_seed=condition[2],**m,degradation_pp=100*(clean-m['macro_f1'])))
                    cm=np.bincount(11*y+pred,minlength=121).reshape(11,11);tp=cm.diagonal().astype(float)
                    precision=np.divide(tp,cm.sum(0),out=np.zeros(11),where=cm.sum(0)>0);recall=np.divide(tp,cm.sum(1),out=np.zeros(11),where=cm.sum(1)>0)
                    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(11),where=precision+recall>0)
                    for label in range(11):class_rows.append(dict(cell=cell,model=name,training_seed=seed,condition=ci,label=label,precision=precision[label],recall=recall[label],f1=f1[label],support=int(cm[label].sum())))
                    for a in range(11):
                        for b in range(11):cm_rows.append(dict(cell=cell,model=name,training_seed=seed,condition=ci,true_label=a,predicted_label=b,count=int(cm[a,b])))
    df=pd.DataFrame(per);df.to_csv(output/'metrics_per_pollution_seed.csv',index=False)
    csv_write(output/'per_class_metrics.csv',class_rows);csv_write(output/'confusion_counts.csv',cm_rows)
    avg=df.groupby(['cell','model','training_seed','kind','severity'])[['macro_f1','accuracy','degradation_pp']].mean().reset_index();avg.to_csv(output/'metrics_per_training_seed.csv',index=False)
    summary=avg.groupby(['cell','model','kind','severity']).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),degradation_pp_mean=('degradation_pp','mean'),degradation_pp_sd=('degradation_pp','std')).reset_index();summary.to_csv(output/'summary.csv',index=False)
    paired=[]
    for name in MODELS:
        for (kind,severity),part in avg[avg.model==name].groupby(['kind','severity']):
            p=part.pivot(index='training_seed',columns='cell',values='macro_f1')
            for contrast,values in [('window_N0_B-A',p.B-p.A),('window_N1_D-C',p.D-p.C),('current_norm_W0_C-A',p.C-p.A),('current_norm_W1_D-B',p.D-p.B),('interaction_D-C-B+A',p.D-p.C-p.B+p.A),('joint_D-A',p.D-p.A)]:
                paired.append(dict(model=name,kind=kind,severity=severity,contrast=contrast,mean_delta_pp=values.mean()*100,sd_delta_pp=values.std(ddof=1)*100,positive_seeds=int((values>0).sum()),negative_seeds=int((values<0).sum()),tied_seeds=int((values==0).sum()),n=5))
    pd.DataFrame(paired).to_csv(output/'paired_factor_changes.csv',index=False)
    diagnostics=diagnostic_rows(output,raw,split)
    # Static exportable research figures. No classification result used to pick a displayed sample.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    colors={'A':'#667085','B':'#147d92','C':'#ce7d2c','D':'#8058a8'}
    for ax,name in zip(axes,MODELS):
        for cell in CELLS:
            data=summary[(summary.model==name)&(summary.cell==cell)];order=[('clean',0),('awgn',30),('awgn',20),('awgn',10),('missing_consecutive',.2)]
            r=[data[(data.kind==k)&(data.severity==s)].iloc[0] for k,s in order]
            ax.errorbar(range(5),[100*x.macro_f1_mean for x in r],yerr=[100*x.macro_f1_sd for x in r],marker='o',capsize=3,label=cell,color=colors[cell])
        ax.set_title(name);ax.set_xticks(range(5),['Clean','AWGN30','AWGN20','AWGN10','Missing20%'],rotation=20);ax.grid(alpha=.2);ax.set_ylim(0,100);ax.legend(title='Cell')
    axes[0].set_ylabel('Macro-F1 (%) / mean ± SD of 5 training seeds');fig.tight_layout();fig.savefig(figures/'f1_conditions.png',dpi=600);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for label in range(8):
        idxs=[j for j in split['outer_train'] if raw['labels'][j]==label]
        j=min(idxs,key=lambda j:int(raw['sample_ids'][j]));ax=axes.flat[label]
        for window,color in [('W0','#667085'),('W1','#147d92')]:
            ax.plot(raw[window+'_voltage'][j],raw[window+'_current'][j],lw=.8,color=color,label=window)
        ax.set_title(f'Class {label}, ID {raw["sample_ids"][j]}');ax.set_xlabel('Voltage');ax.set_ylabel('Current');ax.legend(fontsize=7)
    fig.suptitle('First numeric training ID per class (classes 0–7); no test outcome selection');fig.tight_layout();fig.savefig(figures/'fixed_training_vi_examples.png',dpi=600);plt.close(fig)
    lines=['# E9 視窗與電流正規化敏感度結果','',
      '本輪是原協定的獨立敏感度補測，不替換E1–E8/E11資料與結果；原稿及E13未修改。',
      f'共同合格樣本：outer train {len(split["outer_train"])}／test {len(split["outer_test"])}；inner train {len(split["inner_train"])}／validation {len(split["inner_validation"])}。完整原始1074筆均已核對，965筆通過固定W1規則；排除109筆的清單及逐類別coverage見來源CSV。',
      '','## 協定',
      'A=W0/N0；B=W1/N0；C=W0/N1；D=W1/N1。W0原500點；W1末段穩定性檢查通過的單週期依相位重採樣500點，不做週期平均。N0電流逐波形z-score；N1電流training-only固定z-score；電壓始終逐波形z-score。',
      'M1與Yu3，各條件5個配對training seeds42–46。8個inner-validation epoch pilots後40次正式訓練；No-Defense，沿用Yu3 E2凍結超參數至兩模型，不另HPO。每條件個別內層選epoch，選擇全部完成後才評估outer test。',
      'AWGN30/20/10 dB與連續缺失20%各十污染seeds1000–1009；每training seed先平均污染，再描述五seed差異。AWGN依去均值variance定義，V/I獨立加噪；均在取窗後施加，不測受污染訊號的取窗。',
      '','## 完整摘要','', '| Cell | Model | Condition | Macro-F1 % mean ± SD | Drop pp mean ± SD |','|---|---|---|---:|---:|']
    for r in summary.itertuples(index=False):lines.append(f'| {r.cell} | {r.model} | {r.kind} {r.severity:g} | {100*r.macro_f1_mean:.2f} ± {100*r.macro_f1_sd:.2f} | {r.degradation_pp_mean:.2f} ± {r.degradation_pp_sd:.2f} |')
    lines+=['','## 審稿對應與判讀界線',
      '- R3-6a：B−A與D−C檢查視窗差異；固定穩定性門檻、coverage與取窗記錄完整保留。單週期重採樣及末段位置均屬W1協定，不能稱只改一個物理因素。最大差分點未驗證為真實開啟事件。',
      '- R3-6b：C−A與D−B只改電流正規化，電壓一致。固定尺度在映射前保留幅值差，但clip與量化仍可損失資訊；非原Experiment4同時固定V/I正規化的重現。',
      '- R3-6c：AWGN F1、退化量及逐樣本影像IoU／新增／消失像素／L1聯合描述；無法單獨證明成像是分類脆弱性的唯一原因。',
      '- R3-6e：波形椒鹽依window-specific outer-training極值、1/20%污染；原始z超±3與整數索引實際clip分開計算，程式是截至影像邊界，不丟棄點。沒有新增像素椒鹽分類測試。舊擴充分析與兩種椒鹽的稿件處置仍屬E13，不能宣稱整題已完全解決。',
      '','## 限制與後續',
      'n=5為成本折衷，僅描述，不報確認性p值、不稱等效。共同合格子集與舊855/219不同；無合格穩態樣本不fallback，不把保留子集推廣至全部樣本。穩定性規則是可操作判準，非物理穩態保證；一個內層split與單seed pilot可能有選擇變異，超參數未為各前處理最佳化。',
      'historical test既往已看過；class×house是group代理；Mac MPS不是學生CUDA正式重現。M1/Yu3差異非純回饋因果證據。結果不作新主方法選擇；是否改主方法或補其他模型，交由作者看完整結果後決定。',
      '','## 可稽核來源',
      '`protocol.json`、`split.json`、`window_manifest.csv`、`coverage.csv`、`pilots/`、`selection_complete.json`、`runs/`、`summary.csv`、`paired_factor_changes.csv`、`diagnostics_per_sample.csv`及`independent_audit.json`。正式Word回應稿尚未製作。']
    (output/'E9結果報告.md').write_text('\n'.join(lines)+'\n')
    save_json(output/'analysis_audit.json',dict(all_checks_passed=True,prediction_rows=len(per)*len(split['outer_test']),metric_rows=len(df),per_class_rows=len(class_rows),confusion_cells=len(cm_rows),diagnostic_rows=len(diagnostics),summary_rows=len(summary),descriptive_only=True))
    print('E9 analysis PASS',len(per),'evaluations',flush=True)

if __name__=='__main__':analyze()
