"""Independent NumPy count/normalization/image audit of every E9 prediction/cache."""
import pandas as pd
from e9_core import ROOT, OUTPUT, DATA, MODELS, SEEDS, CELLS, CONDITIONS, load_plan, source_hashes, save_json, sha
import numpy as np
import json

def independent_image(c,v,stats,norm):
    if norm=='N0':
        sd=np.sqrt(np.mean((c-c.mean())**2));iz=(c-c.mean())/sd if sd>1e-5 else c.copy()
    else:iz=(c-stats['i_mean'])/stats['i_std'] if stats['i_std']>1e-5 else c-stats['i_mean']
    sd=np.sqrt(np.mean((v-v.mean())**2));vz=(v-v.mean())/sd if sd>1e-5 else v.copy()
    y=np.clip(np.trunc((iz+3)*10.5).astype(int),0,63);x=np.clip(np.trunc((vz+3)*10.5).astype(int),0,63)
    out=np.zeros((64,64),np.float32)
    for i,j in zip(y,x):out[i,j]=min(np.float32(1.),out[i,j]+np.float32(.1))
    return out[None]

def verify(output=OUTPUT):
    plan=load_plan(output);raw=dict(np.load(output/'waveforms.npz'));split=json.loads((output/'split.json').read_text())['indices']
    assert sha(output/'split.json')==plan['split_sha256'];assert source_hashes()==plan['source_sha256']
    groups={k:set(raw['groups'][v]) for k,v in split.items()}
    assert not groups['inner_train']&groups['inner_validation'] and not groups['outer_train']&groups['outer_test']
    assert set(split['inner_train'])|set(split['inner_validation'])==set(split['outer_train'])
    assert all(raw['eligible'][split[k]].all() for k in split)
    selection=json.loads((output/'selection_complete.json').read_text());assert selection['outer_predictions_opened'] is False
    df=pd.read_csv(output/'metrics_per_pollution_seed.csv');count=0;image_checks=0;norm_checks=0
    for cell,(window,norm) in CELLS.items():
        for name in MODELS:
            pilot=output/'pilots'/f'{cell}_{name}_s42';spec=json.loads((pilot/'run_spec.json').read_text())
            assert spec['train_ids']==raw['sample_ids'][split['inner_train']].tolist() and spec['validation_ids']==raw['sample_ids'][split['inner_validation']].tolist()
            log=pd.read_csv(pilot/'train_log.csv');eligible=log[log.epoch>=30];best=eligible.loc[eligible.validation_macro_f1.idxmax()]
            assert int(best.epoch)==selection['epochs'][f'{cell}_{name}']
            for seed in SEEDS:
                folder=output/'runs'/f'{cell}_{name}_s{seed}';spec=json.loads((folder/'run_spec.json').read_text());result=json.loads((folder/'result.json').read_text())
                assert spec['train_ids']==raw['sample_ids'][split['outer_train']].tolist() and not spec['validation_ids']
                for prefix,column in [('i','current'),('v','voltage')]:
                    values=raw[f'{window}_{column}'][split['outer_train']]
                    for key,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:assert spec['normalization'][prefix+'_'+key]==float(fn(values));norm_checks+=1
                assert result['selected_epoch']==selection['epochs'][f'{cell}_{name}']
                logs=pd.read_csv(folder/'train_log.csv');assert logs.epoch.tolist()==list(range(1,result['selected_epoch']+1))
                ids=np.load(folder/'test_identity.npz');y=ids['labels'];assert np.array_equal(ids['sample_ids'],raw['sample_ids'][split['outer_test']])
                for ci,(kind,severity,ps) in enumerate(CONDITIONS):
                    pred=np.load(folder/'predictions'/f'{ci:03}.npy');assert pred.shape==y.shape and np.isin(pred,np.arange(11)).all()
                    cm=np.zeros((11,11),int)
                    for actual,guess in zip(y,pred):cm[actual,guess]+=1
                    tp=cm.diagonal();f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(11,float),where=cm.sum(0)+cm.sum(1)>0).mean();accuracy=tp.sum()/cm.sum()
                    row=df[(df.cell==cell)&(df.model==name)&(df.training_seed==seed)&(df.kind==kind)&(df.severity==severity)&(df.pollution_seed==ps)].iloc[0]
                    assert np.isclose(f1,row.macro_f1,atol=1e-12) and np.isclose(accuracy,row.accuracy,atol=1e-12)
                    if ci==0:clean=f1
                    assert np.isclose(row.degradation_pp,100*(clean-f1),atol=1e-10);count+=len(y)
        # Independent loop image reconstruction on every clean test image, not just a fixture.
        stats=json.loads((output/'runs'/f'{cell}_M1_s42/run_spec.json').read_text())['normalization']
        cache=np.load(output/'evaluation_images'/cell/'000.npy')
        for k,j in enumerate(split['outer_test']):
            image=independent_image(raw[window+'_current'][j],raw[window+'_voltage'][j],stats,norm)
            assert np.array_equal(image,cache[k]),('independent clean image mismatch',cell,j);image_checks+=1
    # Reaggregate from independently verified rows: pollution seeds nested within training seed.
    avg=df.groupby(['cell','model','training_seed','kind','severity'])[['macro_f1','accuracy','degradation_pp']].mean().reset_index()
    summary=pd.read_csv(output/'summary.csv')
    for row in summary.itertuples():
        r=avg[(avg.cell==row.cell)&(avg.model==row.model)&(avg.kind==row.kind)&(avg.severity==row.severity)]
        assert len(r)==5
        for metric in ['macro_f1','accuracy','degradation_pp']:
            assert np.isclose(getattr(row,metric+'_mean'),r[metric].mean(),atol=1e-12)
            if hasattr(row,metric+'_sd'):assert np.isclose(getattr(row,metric+'_sd'),r[metric].std(ddof=1),atol=1e-12)
    diag=pd.read_csv(output/'diagnostics_per_sample.csv');assert len(diag)==4*len(split['outer_test'])*41
    assert (diag[diag.kind=='clean'].image_iou==1).all() and (diag[diag.kind=='clean'].image_mean_abs_change==0).all()
    assert diag[['i_beyond_3sigma_fraction','v_beyond_3sigma_fraction','i_index_clipped_fraction','v_index_clipped_fraction']].ge(0).all().all()
    assert diag[['i_beyond_3sigma_fraction','v_beyond_3sigma_fraction','i_index_clipped_fraction','v_index_clipped_fraction']].le(1).all().all()
    save_json(output/'independent_audit.json',dict(all_checks_passed=True,prediction_rows_recomputed=count,clean_images_independent_loop=image_checks,
        training_stat_values_checked=norm_checks,all_pilot_epoch_choices_checked=True,all_final_epochs_and_ids_checked=True,
        summary_rows_recomputed=len(summary),diagnostic_rows_checked=len(diag),source_sha256=source_hashes(),
        scope='all prediction metrics/counts and summary; all clean test images independently rendered; diagnostic bounds/clean invariants, not independent recomputation of every noisy diagnostic or all MPS predictions'))
    print('E9 independent PASS predictions',count,'clean images',image_checks,flush=True)

if __name__=='__main__':verify()
