"""E9 isolated 2x2 preprocessing sensitivity. Historical code/data remain immutable."""
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from revision_followup_core import build, loss, predict, atomic_torch, Paused, pause_check, csv_write
from run_e5_ablation import metrics, seed_all
from run_exp4_matched_controls import ROOT, DATA, save_json, sha, rng_for, FNS

OUTPUT=ROOT/'results/e9_preprocessing_20261108'
MODELS=['M1','Yu3']; SEEDS=[42,43,44,45,46]
CELLS={'A':('W0','N0'),'B':('W1','N0'),'C':('W0','N1'),'D':('W1','N1')}
CONDITIONS=[('clean',0,0)]+[('awgn',s,p) for s in [30,20,10] for p in range(1000,1010)]+[('missing_consecutive',.2,p) for p in range(1000,1010)]
WINDOW_RULE=dict(fs=30000,output_points=500,smooth_voltage_points=15,min_crossing_gap=400,
    min_cycle_points=450,max_cycle_points=550,tail_cycles=20,stable_cycles=3,
    rms_relative_spread_max=.05,adjacent_current_nrmse_max=.10,
    activity_fraction_of_record_cycle_rms_p95=.10,min_current_rms=1e-6,
    min_start_index=1000,selection='latest qualifying three-cycle block in last20; choose middle cycle; no averaging; no fallback')
SOURCES=['e9_core.py','prepare_e9.py','preflight_e9.py','run_e9.py','control_e9.py','analyze_e9.py','verify_e9.py',
    'revision_followup_core.py','run_e5_ablation.py','run_exp4_matched_controls.py','revision_models_mps.py','run_e2_search_mps.py']

def source_hashes():
    paths=[ROOT/'experiments'/p for p in SOURCES]+[ROOT/'實驗一/models_m1_to_m5.py',ROOT/'實驗一/plaid_loader_vi.py',ROOT/'實驗四/exp4_raw_data_V2.py',ROOT/'實驗四/exp4_defense_strategies_V2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def zscore(x):
    sd=float(np.std(x)); return (x-np.mean(x))/sd if sd>1e-5 else x.copy()

def fit_stats(raw, indices, window):
    return {f'{p}_{k}':float(fn(raw[f'{window}_{c}'][indices]))
        for p,c in [('i','current'),('v','voltage')]
        for k,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]}

def standardized(c,v,stats,norm):
    # Voltage always per-waveform; only current normalization is the experimental factor.
    iz=zscore(c) if norm=='N0' else FNS['_zscore_fixed'](c,stats['i_mean'],stats['i_std'])
    return iz,zscore(v)

def scatter(iz,vz):
    iy=np.clip(((iz+3)/6*63).astype(int),0,63);ix=np.clip(((vz+3)/6*63).astype(int),0,63)
    img=np.zeros((64,64),np.float32);np.add.at(img,(iy,ix),np.float32(.1))
    return np.clip(img,0,1)[None]

def waveform(raw,idx,window,stats,condition):
    c=raw[f'{window}_current'][idx].copy();v=raw[f'{window}_voltage'][idx].copy()
    if condition is not None and condition[0]!='clean':
        kind,severity,seed=condition
        # Keys exclude cell/model/training seed: paired across N0/N1 and both models.
        rng=rng_for('E9',str(raw['sample_ids'][idx]),kind,severity,seed)
        c,v=FNS['apply_waveform_corruption'](c,v,kind,severity,stats,rng)
    return c,v

def images(raw,indices,stats,cell,condition=None):
    window,norm=CELLS[cell]
    return np.stack([scatter(*standardized(*waveform(raw,j,window,stats,condition),stats,norm)) for j in indices])

def steady_window(c,v):
    sm=np.convolve(np.pad(v,(7,7),mode='edge'),np.ones(15)/15,mode='valid')
    hits=np.flatnonzero((sm[:-1]<=0)&(sm[1:]>0));cross=[]
    for h in hits:
        point=float(h-sm[h]/(sm[h+1]-sm[h]))
        if not cross or point-cross[-1]>=400:cross.append(point)
    records=[];cycles=[]
    for start,end in zip(cross[:-1],cross[1:]):
        if 450<=end-start<=550:
            phase=start+np.arange(500)*(end-start)/500
            ci=np.interp(phase,np.arange(len(c)),c);vi=np.interp(phase,np.arange(len(v)),v)
            cycles.append((start,end,ci,vi));records.append(float(np.sqrt(np.mean(ci**2))))
    fail=dict(eligible=False,reason='no_qualifying_tail_block',start=None,end=None,rms_spread=None,nrmse=None,
              record_cycles=len(cycles),activity_reference=float(np.percentile(records,95)) if records else 0.)
    for k in range(len(cycles)-3,max(-1,len(cycles)-20-1),-1):
        block=cycles[k:k+3]
        if len(block)!=3 or block[0][0]<1000:continue
        if any(abs(block[j][1]-block[j+1][0])>1e-6 for j in [0,1]):continue
        rms=np.array(records[k:k+3]);mean=float(rms.mean());spread=float((rms.max()-rms.min())/max(mean,1e-12))
        nrmse=max(float(np.sqrt(np.mean((block[j][2]-block[j+1][2])**2))/max(mean,1e-12)) for j in [0,1])
        if mean>=max(1e-6,.10*fail['activity_reference']) and spread<=.05 and nrmse<=.10:
            start,end,ci,vi=block[1]
            return ci,vi,dict(eligible=True,reason='pass',start=start,end=end,rms_spread=spread,nrmse=nrmse,
                record_cycles=len(cycles),activity_reference=fail['activity_reference'])
    return np.full(500,np.nan),np.full(500,np.nan),fail

def load_plan(output=OUTPUT):
    plan=json.loads((output/'protocol.json').read_text());assert plan['source_sha256']==source_hashes(),'E9 source changed after freeze'
    assert plan['data_sha256']==sha(output/'waveforms.npz')
    return plan

def train(output,folder,raw,train_idx,val_idx,cell,name,seed,config,device,fixed_epochs=None,max_epochs=150,min_epochs=30,patience=20):
    folder.mkdir(parents=True,exist_ok=True)
    stats=fit_stats(raw,train_idx,CELLS[cell][0]); spec=dict(cell=cell,model=name,seed=seed,config=config,
        train_ids=raw['sample_ids'][train_idx].tolist(),validation_ids=raw['sample_ids'][val_idx].tolist(),normalization=stats,
        fixed_epochs=fixed_epochs,max_epochs=max_epochs,min_epochs=min_epochs,patience=patience,
        voltage_normalization='per-waveform z-score in every cell',images='float32 occupancy .1 cumulative saturation; no PNG quantization')
    if (folder/'run_spec.json').exists():assert json.loads((folder/'run_spec.json').read_text())==spec
    else:save_json(folder/'run_spec.json',spec)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    pause_check(output,folder.name,None);assert not set(raw['groups'][train_idx])&set(raw['groups'][val_idx])
    seed_all(seed);model=build(name,config['dropout']).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    state=dict(epoch=0,logs=[],best=-1.,best_epoch=None,anchor=-1.,stale=0,stopped=False)
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);state=saved['state']
    x=torch.from_numpy(images(raw,train_idx,stats,cell)).to(device);y=torch.from_numpy(raw['labels'][train_idx]).long().to(device)
    val=images(raw,val_idx,stats,cell) if len(val_idx) else None
    for epoch in range(state['epoch']+1,(fixed_epochs or max_epochs)+1):
        if state['stopped']:break
        tick=time.monotonic();seed_all(seed*10000+epoch)
        order=torch.randperm(len(train_idx),generator=torch.Generator().manual_seed(seed*10000+epoch));model.train();total=0.
        for idx in order.split(32):
            idx=idx.to(device);optimizer.zero_grad(set_to_none=True);value=loss(model,x[idx],y[idx],name)
            assert torch.isfinite(value);value.backward();optimizer.step();total+=float(value.detach())*len(idx)
        score=None
        if val is not None:
            pred=predict(model,val,name,device);score=metrics(raw['labels'][val_idx],pred)['macro_f1']
            if epoch>=min_epochs and score>state['best']:
                state['best']=score;state['best_epoch']=epoch
                atomic_torch(dict(state_dict=model.state_dict(),epoch=epoch,model=name,config=config),folder/'best.pt')
                np.savez_compressed(folder/'validation_predictions.npz',sample_ids=raw['sample_ids'][val_idx],labels=raw['labels'][val_idx],predictions=pred)
            if epoch==min_epochs:state['anchor']=score
            elif epoch>min_epochs:
                if score>state['anchor']+.001:state['anchor']=score;state['stale']=0
                else:state['stale']+=1
            state['stopped']=epoch>=min_epochs and state['stale']>=patience
        state['epoch']=epoch;state['logs'].append(dict(epoch=epoch,train_loss=total/len(train_idx),validation_macro_f1=score,seconds=time.monotonic()-tick))
        atomic_torch(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state),folder/'resume.pt');csv_write(folder/'train_log.csv',state['logs'])
        save_json(output/'current_run.json',dict(run=folder.name,epoch=epoch,planned_epochs=fixed_epochs or max_epochs,seconds_last_epoch=state['logs'][-1]['seconds'],phase='final_train' if fixed_epochs else 'epoch_pilot'))
        if epoch==1 or epoch%10==0 or state['stopped']:print(folder.name,epoch,'loss',round(total/len(train_idx),5),'val',score,flush=True)
        pause_check(output,folder.name,epoch)
    if val is not None:
        cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict']);selected=state['best_epoch']
        saved=np.load(folder/'validation_predictions.npz');assert np.array_equal(predict(model,val,name,device),saved['predictions'])
    else:selected=fixed_epochs;cp=dict(state_dict=model.state_dict(),epoch=selected,model=name,config=config)
    atomic_torch(cp,folder/'checkpoint.pt')
    restored=build(name,config['dropout']).to(device);restored.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
    probe=x[:32].cpu().numpy();assert np.array_equal(predict(model,probe,name,device),predict(restored,probe,name,device))
    for layer in [1,2,3]:assert int(restored.state_dict()[f'encoder.bn{layer}.num_batches_tracked'])==int(np.ceil(len(train_idx)/32))*selected*(2 if name=='Yu3' else 1)
    result=dict(cell=cell,model=name,seed=seed,selected_epoch=selected,stopped_epoch=state['epoch'],validation_macro_f1=state['best'] if val is not None else None,
        hit_max_epochs=val is not None and state['epoch']==max_epochs and not state['stopped'],prediction_reload_audit=True,training_seconds=sum(r['seconds'] for r in state['logs']))
    save_json(folder/'result.json',result);(folder/'resume.pt').unlink(missing_ok=True);(folder/'best.pt').unlink(missing_ok=True)
    del model,restored,optimizer,x,y;gc.collect();torch.mps.empty_cache();return result
