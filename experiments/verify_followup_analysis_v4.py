"""Synthetic known-prediction integration test: no real held-out outcomes accessed."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import analyze_revision_followup_v4 as analysis
from revision_followup_core import ROOT,save_json,csv_write,fit_stats,metrics


def main():
    output=ROOT/'results/followup_preflight_20261106/synthetic_analysis_v4'
    if output.exists():shutil.rmtree(output)
    output.mkdir(parents=True)
    idx=np.arange(1074);labels=idx%11
    raw=dict(sample_ids=idx.astype(str),labels=labels,groups=idx.astype(str),
             partitions=np.where(idx<855,'train','val'),current=np.zeros((1074,2),np.float32),voltage=np.zeros((1074,2),np.float32))
    fake_data=output/'fake_data';fake_data.mkdir();np.savez(fake_data/'raw_windows.npz',**raw)
    analysis.DATA=fake_data
    plan=dict(seeds=list(range(42,52)),initial_common_strategies=['no_defense','S3','combined'],
              severity_grids={'awgn':[0],'salt_pepper':[.2],'missing_random':[.2],'missing_consecutive':[.2]})
    save_json(output/'protocol.json',plan);save_json(output/'selection_complete.json',{'all_selection_complete':True})
    choices={name:dict(noise=['S3'],missing=['combined','no_defense']) for name in analysis.CANDIDATES}
    save_json(output/'E3_selected_strategies.json',choices)
    conditions=[(t,s,ps) for t,g in plan['severity_grids'].items() for s in g for ps in range(1000,1010)]
    save_json(output/'evaluation_conditions.json',conditions)
    outer=np.arange(855);test=np.arange(855,1074);y=labels[test];stats=fit_stats(raw,outer)
    manifest=[]
    for mi,name in enumerate(analysis.FAMILY[:4]+analysis.CANDIDATES[1:]):
        for strategy in plan['initial_common_strategies']:
            for seed in plan['seeds']:
                folder=output/'final'/f'{name}_{strategy}_s{seed}';folder.mkdir(parents=True)
                manifest.append(dict(model=name,strategy=strategy,seed=seed,path=str(folder.relative_to(output))))
                spec=dict(model=name,strategy=strategy,seed=seed,train_ids=raw['sample_ids'][outer].tolist(),validation_ids=[],normalization=stats,fixed_epochs=30)
                save_json(folder/'run_spec.json',spec);save_json(folder/'result.json',{'selected_epoch':30})
                csv_write(folder/'train_log.csv',[dict(epoch=e,train_loss=.5) for e in range(1,31)])
                passes=2 if name=='Yu3' else 1
                torch.save(dict(epoch=30,state_dict={f'encoder.bn{k}.num_batches_tracked':torch.tensor(27*30*passes) for k in [1,2,3]}),folder/'checkpoint.pt')
                pred_dir=folder/'outer_predictions';pred_dir.mkdir()
                clean=y.copy();clean[:10+mi+seed%3]=(clean[:10+mi+seed%3]+1)%11
                np.save(pred_dir/'clean.npy',clean);np.savez(folder/'outer_ids_labels.npz',sample_ids=raw['sample_ids'][test],labels=y)
                rows=[]
                for i,(typ,severity,ps) in enumerate(conditions):
                    pred=clean.copy();count=30+mi+ps%3;pred[:count]=(y[:count]+1)%11
                    np.save(pred_dir/f'{i:03}.npy',pred)
                    actual=metrics(y,pred)
                    rows.append(dict(type=typ,severity=severity,pollution_seed=ps,**actual,drop_pp=100*(metrics(y,clean)['macro_f1']-actual['macro_f1'])))
                save_json(folder/'evaluation.json',dict(model=name,strategy=strategy,seed=seed,epoch=30,clean=metrics(y,clean),conditions=rows))
    save_json(output/'final_run_manifest.json',manifest)
    analysis.analyze(output)
    audit=json.loads((output/'completed_audit.json').read_text())
    assert audit['models']==210 and audit['corrupted_predictions_recomputed']==210*40*219
    rows=list(csv.DictReader((output/'per_seed_averaged.csv').open()))
    row=next(r for r in rows if r['model']=='M1' and r['strategy']=='no_defense' and r['seed']=='42' and r['type']=='awgn')
    clean=y.copy();clean[:10]=(clean[:10]+1)%11
    scores=[]
    for ps in range(1000,1010):
        pred=clean.copy();n=30+ps%3;pred[:n]=(y[:n]+1)%11;scores.append(metrics(y,pred)['macro_f1'])
    assert abs(float(row['f1'])-np.mean(scores))<1e-12
    tests=list(csv.DictReader((output/'paired_tests.csv').open()))
    assert all(0<=float(r['p_raw'])<=float(r['p_holm_family'])<=1 for r in tests)
    for baseline in analysis.CANDIDATES[1:]:
        pair_rows=[r for r in tests if r['comparison']==f'Yu3-{baseline}' and r['type']=='missing_random' and r['endpoint']=='F1' and r['family']=='E3_all_selected']
        assert len(pair_rows)==4, (baseline,len(pair_rows))
    ce=list(csv.DictReader((output/'relative_CE_raw.csv').open()))
    assert all(abs(float(r['raw_relative_CE'])-1)<1e-12 for r in ce if r['model']=='M1')
    # Corrupt a saved prediction and confirm the independent audit rejects it.
    target=output/manifest[0]['path']/'outer_predictions/000.npy';saved=np.load(target);wrong=saved.copy();wrong[0]=y[0];np.save(target,wrong)
    try:analysis.analyze(output)
    except AssertionError:rejected=True
    else:raise AssertionError('Corrupted prediction accepted')
    np.save(target,saved)
    report=dict(all_checks_passed=True,synthetic_models=210,averaging_verified=True,relative_ce_M1_exactly1=True,holm_monotone=True,corrupted_prediction_rejected=rejected,real_outer_test_evaluations=0)
    save_json(output.parent/'analysis_integration_v4_audit.json',report)
    # Keep fixtures and first valid report; note second audit intentionally failed.
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
