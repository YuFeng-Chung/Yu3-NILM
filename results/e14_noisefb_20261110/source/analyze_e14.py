"""Frozen E14 §5 inference, seed-paired tests; independent boolean F1 audit."""
import sys,json,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,pandas as pd
from scipy.stats import rankdata
from analyze_e5_full import paired_test,holm
from e14_core import metrics,save_json

def independent_f1(y,p):
    vals=[]
    for k in range(11):
        tp=np.count_nonzero((y==k)&(p==k));fp=np.count_nonzero((y!=k)&(p==k));fn=np.count_nonzero((y==k)&(p!=k))
        vals.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.)
    return float(np.mean(vals))

def exact_check(d):
    d=np.round(np.asarray(d),12);d=d[d!=0]
    if not len(d):return 1.
    r=rankdata(abs(d));observed=abs(np.sum(r*np.sign(d)))
    sums=np.array(list(itertools.product([-1,1],repeat=len(d))))@r
    return float(np.mean(abs(sums)>=observed-1e-12))

def analyze():
    rows=[];files=0;labels_checked=0
    for name in ['NoiseFB','PermFB','Yu3','ZeroFB','ShuffleFB']:
        for seed in range(42,52):
            folder=(OUT/'final' if name in ['NoiseFB','PermFB'] else OUT/'reference_evaluation')/f'{name}_no_defense_s{seed}'
            ev=json.loads((folder/'evaluation.json').read_text());data=np.load(folder/'outer_ids_labels.npz');y=data['labels']
            conds=[dict(type='clean',severity=0,pollution_seed=-1)]+ev['conditions']
            for j,c in enumerate(conds):
                key='clean' if j==0 else f'{j-1:03}'
                for endpoint,sub in [('pass1','first_pass_predictions'),('reported','outer_predictions')]:
                    p=np.load(folder/sub/f'{key}.npy');score=metrics(y,p);ind=independent_f1(y,p)
                    assert abs(score['macro_f1']-ind)<1e-14
                    saved=(ev['clean_first_pass'] if endpoint=='pass1' else ev['clean']) if j==0 else (c['first_pass'] if endpoint=='pass1' else c)
                    for metric in ['accuracy','macro_precision','macro_recall','macro_f1']:assert abs(score[metric]-saved[metric])<1e-14
                    rows.append(dict(model=name,seed=seed,type=c['type'],severity=c['severity'],pollution_seed=c['pollution_seed'],endpoint=endpoint,**{k:v for k,v in score.items() if k!='class_f1'}))
                    files+=1;labels_checked+=len(y)
    df=pd.DataFrame(rows);df.to_csv(OUT/'per_seed_condition_realization.csv',index=False)
    keys=['model','seed','type','severity','endpoint'];cols=['accuracy','macro_precision','macro_recall','macro_f1']
    avg=df.groupby(keys,sort=False)[cols].mean().reset_index();avg.to_csv(OUT/'per_seed_averaged.csv',index=False)
    summary=avg.groupby(['model','type','severity','endpoint'],sort=False)[cols].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns];summary.to_csv(OUT/'summary.csv',index=False)
    tests=[]
    for name in ['NoiseFB','PermFB']:
        for typ,sev in avg[['type','severity']].drop_duplicates().itertuples(index=False,name=None):
            for endpoint in ['pass1','reported']:
                block=avg[(avg.type==typ)&(avg.severity==sev)&(avg.endpoint==endpoint)]
                a=block[block.model==name].sort_values('seed').macro_f1.to_numpy()
                for ref in ['Yu3','ZeroFB','ShuffleFB']:
                    b=block[block.model==ref].sort_values('seed').macro_f1.to_numpy();assert len(a)==len(b)==10
                    d=a-b;stat,p=paired_test(d);assert abs(p-exact_check(d))<1e-12
                    specified=endpoint=='pass1' and typ=='salt_pepper' and sev in [.05,.1,.2] and ref in ['Yu3','ZeroFB']
                    tests.append(dict(model=name,reference=ref,type=typ,severity=sev,endpoint=endpoint,n=10,mean_difference_pp=100*d.mean(),median_difference_pp=100*np.median(d),wins=int(sum(d>0)),ties=int(sum(d==0)),statistic=stat,p_raw=p,p_holm=None,family=('primary' if name=='NoiseFB' else 'secondary')+f'_{sev}' if specified else 'exploratory_unadjusted'))
    tests=pd.DataFrame(tests)
    for fam in tests.family.unique():
        if fam=='exploratory_unadjusted':continue
        idx=tests.index[tests.family==fam];assert len(idx)==2
        tests.loc[idx,'p_holm']=holm(tests.loc[idx,'p_raw'])
    tests.to_csv(OUT/'paired_tests_all.csv',index=False)
    primary=tests[tests.family!='exploratory_unadjusted'].copy();primary.to_csv(OUT/'paired_tests_prespecified.csv',index=False)
    rules=[]
    for name in ['NoiseFB','PermFB']:
        for sev in [.05,.1,.2]:
            block=primary[(primary.model==name)&(primary.severity==sev)]
            yu=block[block.reference=='Yu3'].iloc[0];zero=block[block.reference=='ZeroFB'].iloc[0]
            if zero.mean_difference_pp>0 and zero.p_holm<.05 and abs(yu.mean_difference_pp)<=3:
                decision='varying_channel_rule_met'
            elif abs(zero.mean_difference_pp)<=3 and yu.mean_difference_pp<0 and yu.p_holm<.05:
                decision='residual_type_rule_met'
            else:decision='partial_or_intermediate'
            rules.append(dict(model=name,severity=sev,decision=decision,delta_vs_Yu3_pp=yu.mean_difference_pp,delta_vs_ZeroFB_pp=zero.mean_difference_pp,p_holm_vs_Yu3=yu.p_holm,p_holm_vs_ZeroFB=zero.p_holm))
    pd.DataFrame(rules).to_csv(OUT/'prespecified_interpretation.csv',index=False)
    save_json(OUT/'completed_audit.json',dict(all_checks_passed=True,prediction_files=files,prediction_labels=labels_checked,per_realization_rows=len(df),per_seed_rows=len(avg),prespecified_tests=len(primary),all_p_exact_sign_enumeration_verified=True,independent_boolean_F1_verified=True,conditions_per_model=211,formal_models=20,reference_models=30,interpretation=rules,limitations=['Single historical split and reused controls; new hypotheses motivated by prior diagnostics','Holm per severity two comparisons, not across severities or model families','3pp is a descriptive rule, not an equivalence test','NoiseFB preserves sample residual moments; does not establish that every varying input works','PermFB preserves sample histogram; does not uniquely identify spatial structure','Mac/MPS only; no independent student CUDA reproduction']))
    print(primary.to_string(index=False),flush=True)
if __name__=='__main__':analyze()
