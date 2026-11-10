"""E15 frozen paired comparisons; independent Boolean metrics and sign enumeration."""
import sys,json,itertools,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,pandas as pd
from scipy.stats import rankdata
from analyze_e5_full import paired_test,holm
from e15_core import metrics,save_json
NAMES=['1P-NoiseCh','1P-Mix'];REFS={'NoiseFB':'pass1','ZeroFB':'reported','Yu3':'pass1','RecAux':'reported','M1':'reported'}
E14=ROOT/'results/e14_noisefb_20261110';OLD=ROOT/'results/e1_e4_mac_20261106'
COLS=['accuracy','macro_precision','macro_recall','macro_f1']
def independent(y,p):
    pr=[];rc=[];f=[]
    for label in range(11):
        tp=np.count_nonzero((y==label)&(p==label));fp=np.count_nonzero((y!=label)&(p==label));fn=np.count_nonzero((y==label)&(p!=label))
        pr.append(tp/(tp+fp) if tp+fp else 0.);rc.append(tp/(tp+fn) if tp+fn else 0.);f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(zip(COLS,[float(np.mean(y==p)),float(np.mean(pr)),float(np.mean(rc)),float(np.mean(f))]))
def exact(d):
    d=np.round(np.asarray(d),12);d=d[d!=0]
    if not len(d):return 1.
    ranks=rankdata(np.abs(d));score=abs(np.sum(ranks*np.sign(d)))
    return float(np.mean(np.abs(np.asarray(list(itertools.product([-1,1],repeat=len(d))))@ranks)>=score-1e-12))
def analyze():
    assert (OUT/'all_training_complete.json').exists()
    conditions=json.loads((OUT/'protocol.json').read_text())['conditions'];rows=[];files=0;labels_checked=0;reference_manifest=[]
    canonical=None
    for name in NAMES+list(REFS):
        for seed in range(42,52):
            new=name in NAMES
            folder=(OUT/'final'/f'{name}_no_defense_s{seed}') if new else ((E14/'final' if name=='NoiseFB' else E14/'reference_evaluation')/f'{name}_no_defense_s{seed}' if name in ['NoiseFB','ZeroFB','Yu3'] else OLD/'final'/f'{name}_no_defense_s{seed}')
            ev=json.loads((folder/'evaluation.json').read_text());ids=np.load(folder/'outer_ids_labels.npz');y=ids['labels']
            if canonical is None:canonical=(ids['sample_ids'].copy(),y.copy())
            else:assert np.array_equal(canonical[0],ids['sample_ids']) and np.array_equal(canonical[1],y)
            assert len(ev['conditions'])==len(conditions)==210
            for c,expected in zip(ev['conditions'],conditions):assert [c['type'],c['severity'],c['pollution_seed']]==expected
            endpoints=['zero','noise'] if new else [REFS[name]]
            for endpoint in endpoints:
                sub={'zero':'outer_predictions','noise':'noise_input_predictions','pass1':'first_pass_predictions','reported':'outer_predictions'}[endpoint]
                cms=[]
                for j,c in enumerate([None]+conditions):
                    key='clean' if j==0 else f'{j-1:03}';path=folder/sub/f'{key}.npy';p=np.load(path)
                    assert p.shape==y.shape and np.isin(p,range(11)).all()
                    score=metrics(y,p);ind=independent(y,p)
                    saved=(ev['clean_noise_input'] if endpoint=='noise' else ev['clean_first_pass'] if endpoint=='pass1' else ev['clean']) if j==0 else (ev['conditions'][j-1]['noise_input'] if endpoint=='noise' else ev['conditions'][j-1]['first_pass'] if endpoint=='pass1' else ev['conditions'][j-1])
                    for k in COLS:assert abs(score[k]-ind[k])<1e-14 and abs(score[k]-saved[k])<1e-14,(name,seed,j,k)
                    rows.append(dict(model=name,seed=seed,type='clean' if c is None else c[0],severity=0 if c is None else c[1],pollution_seed=-1 if c is None else c[2],endpoint=endpoint,**{k:score[k] for k in COLS}))
                    cm=np.zeros((11,11),dtype=np.int64);np.add.at(cm,(y,p),1);cms.append(cm)
                    files+=1;labels_checked+=len(y)
                if new:
                    dest=OUT/'confusion_counts';dest.mkdir(exist_ok=True)
                    np.savez_compressed(dest/f'{name}_s{seed}_{endpoint}.npz',counts=np.asarray(cms),sample_ids=canonical[0],labels=y)
            if not new:reference_manifest.append(dict(model=name,seed=seed,endpoint=REFS[name],folder=str(folder.relative_to(ROOT)),source='saved original predictions; no new training or forward'))
    df=pd.DataFrame(rows)
    # Explicitly reconcile E14 reference scores against its previously delivered CSV.
    e14=pd.read_csv(E14/'per_seed_condition_realization.csv')
    for name in ['NoiseFB','ZeroFB','Yu3']:
        keys=['model','seed','type','severity','pollution_seed','endpoint']
        a=df[df.model==name].sort_values(keys);b=e14[(e14.model==name)&(e14.endpoint==REFS[name])].sort_values(keys)
        assert a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True))
        assert np.allclose(a[COLS],b[COLS],atol=1e-14,rtol=0)
    df.to_csv(OUT/'per_seed_condition_realization.csv',index=False)
    avg=df.groupby(['model','seed','type','severity','endpoint'],sort=False)[COLS].mean().reset_index()
    clean=avg[avg.type=='clean'][['model','seed','endpoint','macro_f1']].rename(columns={'macro_f1':'clean_macro_f1'})
    avg=avg.merge(clean,on=['model','seed','endpoint'],validate='many_to_one',sort=False);avg['drop_pp']=100*(avg.clean_macro_f1-avg.macro_f1)
    avg.to_csv(OUT/'per_seed_averaged.csv',index=False)
    summary=avg.groupby(['model','type','severity','endpoint'],sort=False)[COLS+['drop_pp']].agg(['mean','std']).reset_index();summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(OUT/'summary.csv',index=False);save_json(OUT/'reference_manifest.json',reference_manifest)
    tests=[]
    for name in NAMES:
        for typ,sev in avg[['type','severity']].drop_duplicates().itertuples(index=False,name=None):
            for endpoint in ['zero','noise']:
                a=avg[(avg.model==name)&(avg.type==typ)&(avg.severity==sev)&(avg.endpoint==endpoint)].sort_values('seed')
                for ref,refep in REFS.items():
                    b=avg[(avg.model==ref)&(avg.type==typ)&(avg.severity==sev)&(avg.endpoint==refep)].sort_values('seed')
                    assert a.seed.tolist()==b.seed.tolist()==list(range(42,52));d=a.macro_f1.to_numpy()-b.macro_f1.to_numpy()
                    stat,p=paired_test(d);assert abs(p-exact(d))<1e-12
                    specified=endpoint=='zero' and typ=='salt_pepper' and sev in [.05,.1,.2] and ref in ['NoiseFB','ZeroFB']
                    tests.append(dict(model=name,reference=ref,reference_endpoint=refep,type=typ,severity=sev,endpoint=endpoint,n=10,mean_difference_pp=100*d.mean(),median_difference_pp=100*np.median(d),wins=int(sum(d>0)),ties=int(sum(d==0)),statistic=stat,p_raw=p,p_holm=np.nan,family=('primary' if name=='1P-NoiseCh' else 'secondary')+f'_{sev}' if specified else 'exploratory_unadjusted'))
    tests=pd.DataFrame(tests)
    for fam in tests.family.unique():
        if fam=='exploratory_unadjusted':continue
        ix=tests.index[tests.family==fam];assert len(ix)==2;tests.loc[ix,'p_holm']=holm(tests.loc[ix,'p_raw'])
    tests.to_csv(OUT/'paired_tests_all.csv',index=False)
    pres=tests[tests.family!='exploratory_unadjusted'];assert len(pres)==12;pres.to_csv(OUT/'paired_tests_prespecified.csv',index=False)
    rules=[]
    for name in NAMES:
        for sev in [.05,.1,.2]:
            b=pres[(pres.model==name)&(pres.severity==sev)];noise=b[b.reference=='NoiseFB'].iloc[0];zero=b[b.reference=='ZeroFB'].iloc[0]
            if zero.mean_difference_pp>0 and zero.p_holm<.05 and abs(noise.mean_difference_pp)<=3:decision='single_pass_sufficient_rule_met'
            elif abs(zero.mean_difference_pp)<=3 and noise.mean_difference_pp<0 and noise.p_holm<.05:decision='two_pass_necessary_rule_met'
            else:decision='partial_or_intermediate'
            rules.append(dict(model=name,severity=sev,decision=decision,delta_vs_NoiseFB_pp=noise.mean_difference_pp,delta_vs_ZeroFB_pp=zero.mean_difference_pp,p_holm_vs_NoiseFB=noise.p_holm,p_holm_vs_ZeroFB=zero.p_holm))
    pd.DataFrame(rules).to_csv(OUT/'prespecified_interpretation.csv',index=False)
    save_json(OUT/'completed_audit.json',dict(all_checks_passed=True,formal_models=20,prediction_files=files,prediction_labels=labels_checked,conditions_per_model=211,per_realization_rows=len(df),per_seed_rows=len(avg),all_tests=len(tests),prespecified_tests=12,independent_boolean_four_metrics_verified=True,all_p_exact_sign_enumeration_verified=True,saved_E14_scores_identical=True,reference_models=50,new_prediction_files=8440,interpretation=rules,limitations=['Single historical outer split, hypotheses motivated by prior test diagnostics; not pristine preregistration','Calibration transferred from E14 models trained on all855, includes later inner-validation samples; not fully nested','No outer test used to calibrate scale or select epoch','Independent epochs and one versus two encoder passes and loss weights differ; not a pure causal intervention','3pp is descriptive, not equivalence; necessity only a scoped rule label for tested recipes','Reference seed pairs do not eliminate all architectural or optimization differences','Holm within each severity and model family only; other comparisons exploratory unadjusted','Mac/MPS; no independent CUDA replication']))
    print(pres.to_string(index=False));print(pd.DataFrame(rules).to_string(index=False))
if __name__=='__main__':analyze()
