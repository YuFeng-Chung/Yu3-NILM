"""Sequential E1 -> E3 -> E4 queue; all selection precedes outer-test evaluation."""
import argparse
import ast
import fcntl
import gc
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from revision_followup_core import *
from run_e2_search_mps import inner_split

TRAIN_RECIPES=['no_defense','S3','S4','regularization','augmentation','combined']
NOISE=['S0','S1','S2','S3','S4']
MISSING=['no_defense','regularization','augmentation','combined']
POLLUTION=list(range(1000,1010))


def folds(raw):
    """Class-pure device groups, size-balanced within class; deterministic first tie."""
    outer=np.flatnonzero(raw['partitions']=='train'); assigned=[[] for _ in range(5)]
    rng=np.random.default_rng(42)
    for label in range(11):
        indices=outer[raw['labels'][outer]==label]
        groups, counts=np.unique(raw['groups'][indices],return_counts=True)
        assert len(groups)>=5
        counts_by_fold=np.zeros(5,dtype=int)
        # Permute first for seeded tie order, then stable largest-group-first.
        order=rng.permutation(len(groups));order=order[np.argsort(-counts[order],kind='stable')]
        for j in order:
            target=int(np.argmin(counts_by_fold));selected=indices[raw['groups'][indices]==groups[j]]
            assigned[target].extend(selected.tolist());counts_by_fold[target]+=len(selected)
    result=[]
    for validation in assigned:
        val=np.array(sorted(validation));train=np.setdiff1d(outer,val)
        assert not set(raw['groups'][train]) & set(raw['groups'][val])
        assert set(raw['labels'][val])==set(range(11))
        result.append((train,val))
    assert sorted(np.concatenate([v for _,v in result]).tolist())==outer.tolist()
    return result


def inference_filter(current, voltage, strategy):
    if strategy=='S1':
        return median_filter(current,size=3,mode='nearest'),median_filter(voltage,size=3,mode='nearest')
    if strategy=='S2':
        def bilateral(x):
            n=len(x); offsets=np.arange(-15,16); positions=np.arange(n)[:,None]+offsets
            valid=(positions>=0)&(positions<n); positions=np.clip(positions,0,n-1)
            values=x[positions].astype(np.float64)
            sigma=max(float(np.std(x)),1e-5)*.1
            sigma=max(sigma,1e-6)
            w=np.exp(-offsets.astype(float)**2/50.)[None,:]*np.exp(-(values-x[:,None].astype(np.float64))**2/(2*sigma**2))*valid
            denom=w.sum(1)
            return np.where(denom>1e-12,(w*values).sum(1)/denom,x).astype(x.dtype)
        return bilateral(current),bilateral(voltage)
    return current,voltage


def evaluation_images(raw, idx, stats, condition, strategy):
    if strategy not in ['S1','S2']: return images(raw,idx,stats,condition=condition)
    result=[]
    for k in idx:
        c,v=raw['current'][k].astype(np.float32),raw['voltage'][k].astype(np.float32)
        if condition:
            typ,severity,ps=condition
            c,v=FNS['apply_waveform_corruption'](c,v,typ,severity,stats,rng_for('test',str(raw['sample_ids'][k]),typ,severity,ps))
        c,v=inference_filter(c,v,strategy);result.append(scatter_fast(c,v,stats))
    return np.stack(result)


def cached_images(output, raw, idx, stats, condition, strategy, scope):
    from hashlib import sha256
    key=json.dumps([raw['sample_ids'][idx].tolist(),stats,condition,strategy if strategy in ['S1','S2'] else None],sort_keys=True)
    dest=output/'image_cache'/scope/(sha256(key.encode()).hexdigest()+'.npy')
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.with_suffix('.tmp').open('wb') as stream:
            np.save(stream,evaluation_images(raw,idx,stats,condition,strategy))
        os.replace(dest.with_suffix('.tmp'),dest)
    return np.load(dest,mmap_mode='r')


def cv_evaluate(output, folder, raw, val, name, strategy, device):
    dest=folder/f'cv_{strategy}.json'
    if dest.exists():return json.loads(dest.read_text())
    spec=json.loads((folder/'run_spec.json').read_text());stats=spec['normalization']
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=build(name,spec['config']['dropout']).to(device);model.load_state_dict(checkpoint['state_dict'])
    scores={}; predictions={}; labels=raw['labels'][val]
    types=['awgn','salt_pepper'] if strategy in NOISE else ['missing_random','missing_consecutive']
    for typ in types:
        severity=0 if typ=='awgn' else .2;values=[]
        for ps in POLLUTION:
            pause_check(output,f'cv_{name}_{strategy}_{typ}',checkpoint['epoch'])
            condition=(typ,severity,ps)
            array=cached_images(output,raw,val,stats,condition,strategy,'cv')
            pred=predict(model,np.array(array),name,device)
            values.append(metrics(labels,pred)['macro_f1']);predictions[f'{typ}_{ps}']=pred
        scores[typ]=float(np.mean(values))
    np.savez_compressed(folder/f'cv_{strategy}_predictions.npz',sample_ids=raw['sample_ids'][val],labels=labels,**predictions)
    save_json(dest,scores);del model;gc.collect();torch.mps.empty_cache();return scores


def select_strategies(output,name,results):
    from revision_strategy_selection_v4 import select_from_scores
    value=select_from_scores(name,results)
    save_json(output/f'E3_selection_{name}.json',value)
    return value


def freeze(output,e2):
    import run_revision_followup_v3 as prior
    base=prior.freeze(output,e2)
    amendment=json.loads((output/'tie_rule_amendment_v4.json').read_text())
    sources=[Path(__file__),Path(__file__).with_name('revision_strategy_selection_v4.py'),Path(__file__).with_name('analyze_revision_followup_v4.py')]
    plan={**base,'protocol_version':4,'active_runner':'experiments/run_revision_followup_v4.py',
          'superseded_protocol_sha256':sha(output/'protocol_v3.json'),'tie_rule_amendment':amendment,
          'strategy_selection':'Exact fold noise ties each get a win; maximize noise wins only; retain all vote leaders. Five-fold mean F1 is descriptive only. Missing: strict wins over same-fold No-Defense, maximize min-win then total-win, retain all ties. All co-selected strategies trained and evaluated; all selected pair combinations included in the E3 Holm family. No outer-test selection.',
          'source_sha256':{**base['source_sha256'],**{str(p.relative_to(ROOT)):sha(p) for p in sources}}}
    target=output/'protocol_v4.json'
    if target.exists():assert json.loads(target.read_text())==plan,'v4 protocol/source mismatch'
    else:
        assert not list((output/'final').glob('*/evaluation.json'))
        save_json(target,plan)
        for path in sources:
            dest=output/'source_snapshot_v4'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    return plan


def stage(output, name, **kwargs):
    save_json(output/'progress.json',dict(stage=name,**kwargs));print('STAGE',name,kwargs,flush=True)


def evaluate_outer(output, raw, folder, name, strategy, device, conditions):
    done=folder/'evaluation.json'
    if done.exists():return
    spec=json.loads((folder/'run_spec.json').read_text());stats=spec['normalization']
    outer=np.flatnonzero(raw['partitions']=='val');labels=raw['labels'][outer]
    model=build(name,spec['config']['dropout']).to(device)
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict'])
    clean=predict(model,np.array(cached_images(output,raw,outer,stats,None,strategy,'outer')),name,device)
    predpath=folder/'outer_predictions';predpath.mkdir(exist_ok=True)
    np.save(predpath/'clean.npy',clean)
    rows=[]
    for i,condition in enumerate(conditions):
        pause_check(output,folder.name,state['epoch'])
        dest=predpath/f'{i:03}.npy'
        if dest.exists():pred=np.load(dest)
        else:
            pred=predict(model,np.array(cached_images(output,raw,outer,stats,condition,strategy,'outer')),name,device)
            with dest.with_suffix('.tmp').open('wb') as f:np.save(f,pred)
            os.replace(dest.with_suffix('.tmp'),dest)
        score=metrics(labels,pred)
        rows.append(dict(type=condition[0],severity=condition[1],pollution_seed=condition[2],**score,
                    drop_pp=100*(metrics(labels,clean)['macro_f1']-score['macro_f1'])))
        if (i+1)%30==0:
            save_json(output/'current_run.json',dict(phase='outer_evaluation',run=folder.name,conditions_completed=i+1,planned_conditions=len(conditions)))
    # Reload checkpoint and verify deterministic clean predictions.
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
    assert np.array_equal(clean,predict(model,np.array(cached_images(output,raw,outer,stats,None,strategy,'outer')),name,device))
    np.savez_compressed(folder/'outer_ids_labels.npz',sample_ids=raw['sample_ids'][outer],labels=labels)
    save_json(done,dict(model=name,strategy=strategy,seed=spec['seed'],epoch=state['epoch'],clean=metrics(labels,clean),conditions=rows))
    print('EVALUATED',folder.name,len(conditions),flush=True)
    del model;gc.collect();torch.mps.empty_cache()


def run(output,e2):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available(),'MPS required; no fallback'
    device=torch.device('mps');plan=freeze(output,e2);selected=plan['selected_settings']
    raw=dict(np.load(DATA/'raw_windows.npz'));pt,pv,outer_test=inner_split(raw)
    outer=np.flatnonzero(raw['partitions']=='train');empty=np.array([],dtype=int)
    assert len(outer)==855 and len(outer_test)==219
    assert not set(raw['groups'][outer])&set(raw['groups'][outer_test])
    # No test images constructed until all epoch/strategy decisions are frozen.
    epoch_map={};runs={}
    def pilot(name,strategy):
        key=(name,strategy)
        if key not in epoch_map:
            recipe='no_defense' if strategy in ['S0','S1','S2'] else strategy
            folder=output/'pilots'/f'{name}_{recipe}'
            r=train(output,folder,raw,pt,pv,name,recipe,42,config_for(selected,name,recipe),device,
                    patience=20 if recipe=='no_defense' else 30)
            epoch_map[key]=r['selected_epoch']
        return epoch_map[key]
    def final(name,strategy):
        epochs=pilot(name,strategy)
        for seed in plan['seeds']:
            folder=output/'final'/f'{name}_{strategy}_s{seed}'
            train(output,folder,raw,outer,empty,name,strategy,seed,config_for(selected,name,strategy),device,fixed_epochs=epochs)
            runs[name,strategy,seed]=folder
    stage(output,'E1_strategy_epoch_pilots',planned=18)
    for strategy in STRATEGIES:
        for name in FAMILY:pilot(name,strategy)
    save_json(output/'E1_epoch_settings.json',{f'{n}/{s}':epoch_map[n,s] for s in STRATEGIES for n in FAMILY})
    stage(output,'E1_full_training',planned=180)
    for strategy in STRATEGIES:
        for name in FAMILY:final(name,strategy)
    save_json(output/'E1_training_audit.json',dict(completed=180,outer_test_evaluations=0,all_checkpoint_reload_checks_passed=True))
    stage(output,'E3_group_cv',planned_training_runs=120)
    cv=folds(raw)
    save_json(output/'E3_folds.json',[dict(fold=i,train_ids=raw['sample_ids'][t].tolist(),validation_ids=raw['sample_ids'][v].tolist(),normalization=fit_stats(raw,t)) for i,(t,v) in enumerate(cv)])
    choices={};all_scores=[]
    for name in CANDIDATES:
        results={}
        for fold,(t,v) in enumerate(cv):
            for recipe in TRAIN_RECIPES:
                folder=output/'cv'/f'{name}_fold{fold}_{recipe}'
                train(output,folder,raw,t,v,name,recipe,42,config_for(selected,name,recipe),device,
                      patience=20 if recipe=='no_defense' else 30)
            for strategy in NOISE+MISSING:
                recipe='no_defense' if strategy in ['S0','S1','S2'] else strategy
                folder=output/'cv'/f'{name}_fold{fold}_{recipe}'
                scores=cv_evaluate(output,folder,raw,v,name,strategy,device)
                results[fold,strategy]=scores
                all_scores.append(dict(model=name,fold=fold,strategy=strategy,**{k:scores.get(k) for k in ['awgn','salt_pepper','missing_random','missing_consecutive']}))
        choices[name]=select_strategies(output,name,results)
        csv_write(output/'E3_fold_scores.csv',all_scores)
        # CV caches are recomputable; retain only predictions/normalization/checkpoints.
        shutil.rmtree(output/'image_cache/cv',ignore_errors=True)
    save_json(output/'E3_selected_strategies.json',choices)
    stage(output,'E3_final_training')
    for name in CANDIDATES[1:]:
        for strategy in dict.fromkeys(STRATEGIES+choices[name]['noise']+choices[name]['missing']):final(name,strategy)
    for strategy in dict.fromkeys(choices['Yu3']['noise']+choices['Yu3']['missing']):
        for name in FAMILY:final(name,strategy)
    # Freeze full run manifest before opening held-out waveform images.
    manifest=[dict(model=n,strategy=s,seed=seed,path=str(folder.relative_to(output))) for (n,s,seed),folder in runs.items()]
    save_json(output/'final_run_manifest.json',manifest)
    save_json(output/'selection_complete.json',dict(all_selection_complete=True,outer_test_evaluations_so_far=0,planned_final_models=len(manifest),epoch_settings={f'{n}/{s}':e for (n,s),e in epoch_map.items()}))
    conditions=[(typ,severity,ps) for typ,grid in plan['severity_grids'].items() for severity in grid for ps in POLLUTION]
    save_json(output/'evaluation_conditions.json',conditions)
    stage(output,'E4_outer_evaluation',planned_models=len(manifest),conditions_per_model=len(conditions))
    for (name,strategy,seed),folder in runs.items():evaluate_outer(output,raw,folder,name,strategy,device,conditions)
    stage(output,'E4_independent_audit_and_analysis')
    from analyze_revision_followup_v4 import analyze
    analyze(output)
    stage(output,'complete',completed_models=len(manifest))
    print('E1/E3/E4 COMPLETE; audit passed. No manuscript changes.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--e2',type=Path,default=ROOT/'results/e2_mac_20261106');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'completed_audit.json').exists():print('Already complete; no retraining.');return
        if args.resume:
            (args.output/'STOP_REQUESTED').unlink(missing_ok=True);(args.output/'PAUSED.json').unlink(missing_ok=True)
        try:run(args.output,args.e2)
        except Paused:pass
        except Exception as exc:
            save_json(args.output/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise


if __name__=='__main__':main()
