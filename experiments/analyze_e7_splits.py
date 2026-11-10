"""E7 saved-prediction analysis; independent count-based PRF and paired bootstrap."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e7_split_core import *

def condition_name(c):
    return 'clean' if c is None else f'{c[0]}_{c[1]}'

def read_run(output, split, model, seed, condition):
    if split==0:return previous_predictions(model,seed,condition)
    folder=output/'runs'/f'split{split}_{model}_s{seed}'
    ids=np.load(folder/'outer_ids_labels.npz')
    fn='clean.npy' if condition is None else f'{CONDITIONS.index(tuple(condition)):03}.npy'
    return ids['sample_ids'],ids['labels'],np.load(folder/'outer_predictions'/fn)

def analyze(output=OUTPUT, part_b_only=False):
    plan=load_plan(output)
    raw=dict(np.load(DATA/'raw_windows.npz'))
    names=json.loads((DATA/'reconstructed_label_map.json').read_text());names={i:n for n,i in names.items()}
    conditions=[('clean',[None]),('awgn_10',[('awgn',10,p) for p in range(1000,1010)]),
      ('missing_consecutive_0.2',[('missing_consecutive',.2,p) for p in range(1000,1010)]),
      ('missing_consecutive_0.5',[('missing_consecutive',.5,p) for p in range(1000,1010)])]
    dest=output/'part_b' if part_b_only else output/'analysis';dest.mkdir(exist_ok=True)
    seedrows=[];classrows=[];matrices=[];values={};checked=0
    for split in ([plan['splits'][0]] if part_b_only else plan['splits']):
        s=split['split'];seeds=plan['original_seeds'] if s==0 else SEEDS
        test=np.array(split['test_indices']); truth=raw['labels'][test]
        for model in MODELS:
            for cname,cs in conditions:
                if part_b_only and cname=='missing_consecutive_0.5':continue
                cm_total=np.zeros((11,11),dtype=int);classseed=[];f1seed=[]
                for seed in seeds:
                    entries=[]
                    for c in cs:
                        ids,y,pred=read_run(output,s,model,seed,c)
                        assert np.array_equal(ids,raw['sample_ids'][test]) and np.array_equal(y,truth)
                        assert pred.shape==y.shape and np.isin(pred,np.arange(11)).all()
                        cm,prf=class_metrics(y,pred);cm_total+=cm;entries.append(prf);checked+=len(y)
                        # Independently verify stored per-condition metrics for new runs.
                        if s:
                            e=json.loads((output/'runs'/f'split{s}_{model}_s{seed}'/'evaluation.json').read_text())
                            saved=e['clean'] if c is None else e['conditions'][CONDITIONS.index(tuple(c))]
                            assert abs(float(prf[:,2].mean())-saved['macro_f1'])<1e-12
                    average=np.mean(entries,axis=0);classseed.append(average);f1seed.append(float(average[:,2].mean()))
                    seedrows.append(dict(split=s,model=model,condition=cname,seed=seed,macro_f1=f1seed[-1]))
                values[(s,model,cname)]=np.array(f1seed)
                classseed=np.array(classseed)
                for label in range(11):
                    z=classseed[:,label,:]
                    classrows.append(dict(split=s,model=model,condition=cname,label=label,class_name=names[label],
                        test_samples=int(np.sum(truth==label)),test_groups=len(set(raw['groups'][test[truth==label]])),
                        precision_mean=z[:,0].mean(),precision_sd=z[:,0].std(ddof=1),recall_mean=z[:,1].mean(),
                        recall_sd=z[:,1].std(ddof=1),f1_mean=z[:,2].mean(),f1_sd=z[:,2].std(ddof=1)))
                normalized=cm_total/cm_total.sum(1,keepdims=True)
                for i in range(11):
                    for j in range(11):matrices.append(dict(split=s,model=model,condition=cname,true_label=i,predicted_label=j,count=int(cm_total[i,j]),row_fraction=float(normalized[i,j])))
                # Individual publication figures for original main conditions, all models.
                if s==0 and cname!='missing_consecutive_0.5':
                    fig,ax=plt.subplots(figsize=(7.7,6.9))
                    im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax,shrink=.7)
                    short=['AC','CFL','Fan','Fridge','Hairdryer','Heater','ILB','Laptop','Microwave','Vacuum','Washer']
                    ax.set_xticks(range(11),short,rotation=55,ha='right');ax.set_yticks(range(11),short)
                    for i in range(11):
                        for j in range(11):
                            ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if normalized[i,j]>.55 else 'black')
                    ax.set(xlabel='Predicted class',ylabel='True class',title=f'{model} | {cname}\n10 training seeds; row-normalized pooled counts')
                    fig.tight_layout();fig.savefig(dest/f'cm_{model}_{cname}.png',dpi=600);fig.savefig(dest/f'cm_{model}_{cname}.svg');plt.close(fig)
    csv_write(dest/'seed_metrics.csv',seedrows);csv_write(dest/'per_class_metrics.csv',classrows);csv_write(dest/'confusion_matrices.csv',matrices)
    delta=[]
    for cname,cs in conditions:
        subset=[r for r in classrows if r['split']==0 and r['condition']==cname]
        if not subset:continue
        d=[dict(condition=cname,label=k,class_name=names[k],difference_pp=100*(next(r['f1_mean'] for r in subset if r['model']=='Yu3' and r['label']==k)-next(r['f1_mean'] for r in subset if r['model']=='M1' and r['label']==k))) for k in range(11)]
        order=sorted(d,key=lambda r:r['difference_pp'])
        for rank,r in enumerate(order):r.update(rank_from_lowest=rank+1,among_bottom_three=rank<3,among_top_three=rank>=8)
        delta.extend(order)
    csv_write(dest/'yu3_minus_m1_per_class.csv',delta)
    if not part_b_only:
        summaries=[];paired=[];rng=np.random.default_rng(plan['bootstrap_seed'])
        for scope,splits in [('five_new',list(range(1,6))),('all_six',list(range(6)))]:
            for cname,cs in conditions:
                for model in MODELS:
                    means=[values[(s,model,cname)].mean() for s in splits]
                    summaries.append(dict(scope=scope,model=model,condition=cname,mean_macro_f1=float(np.mean(means)),split_sd=float(np.std(means,ddof=1)),split_min=float(min(means)),split_max=float(max(means))))
                for rival in MODELS[1:]:
                    differences=[values[(s,'Yu3',cname)]-values[(s,rival,cname)] for s in splits]
                    boot=[]
                    for _ in range(plan['bootstrap_repetitions']):
                        chosen=rng.integers(0,len(splits),len(splits))
                        boot.append(np.mean([rng.choice(differences[k],size=len(differences[k]),replace=True).mean() for k in chosen]))
                    lo,hi=np.quantile(boot,[.025,.975]);d=np.array([x.mean() for x in differences])
                    paired.append(dict(scope=scope,condition=cname,contrast=f'Yu3-{rival}',mean_difference_pp=float(d.mean()*100),bootstrap_low_pp=float(lo*100),bootstrap_high_pp=float(hi*100),wins=int(np.sum(d>0)),ties=int(np.sum(d==0)),splits=len(splits)))
        csv_write(dest/'split_summary.csv',summaries);csv_write(dest/'paired_split_bootstrap.csv',paired)
    save_json(dest/'analysis_audit.json',dict(all_checks_passed=True,prediction_rows_recomputed=checked,part_b_only=part_b_only,
      training_seed_averaged_after_corruption_seeds=True,source_sha256=source_hashes(),
      limitations=plan['statistics'],original_not_a_new_split=True))
    (dest/'README.md').write_text('# E7 '+('B：既有切分類別分析' if part_b_only else 'A：多切分結果')+'\n\n'+
      'Combined策略；四模型；污染先在每個訓練seed內平均。類別指標為seed間平均／樣本標準差；混淆矩陣為所有seed與污染實現的計數加總、按列正規化。污染實現不是獨立訓練。\n\n'+
      'group是類別×住宅代理，不是已驗證的唯一實體機台；亦非整屋隔離。參見protocol.json的來源與統計限制。\n\n'+
      ('B完成不表示A訓練完成。\n' if part_b_only else '主要摘要為五個新切分；六切分敏感度包含原切分10seed與新切分各3seed。bootstrap區間僅為相依重疊切分下的條件描述，不作顯著、等效或母群95%涵蓋率主張。\n'))
    print('E7 analysis PASS',dest,'recomputed',checked)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part-b',action='store_true');a=p.parse_args();analyze(part_b_only=a.part_b)
