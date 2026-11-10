"""E6 inference-only diagnostics. Frozen E1-E4 sources are imported, never edited."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from revision_followup_core import build, fit_stats
from run_exp4_matched_controls import ROOT, DATA, FNS, images, scatter_fast, rng_for, sha, save_json

PREVIOUS = ROOT/'results/e1_e4_mac_20261106'
OUTPUT = ROOT/'results/e6_mechanism_20261108'
SEEDS = list(range(42,52))
CONDITIONS = [('clean',0,-1)]+[(t,s,p) for t,s in [('awgn',10),('salt_pepper',.05),('missing_consecutive',.2)] for p in range(1000,1010)]

def key(condition):
    t,s,p=condition
    return f'{t}_{s:g}_{p}'

def atomic_npz(path, **arrays):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,**arrays)
    os.replace(path.with_suffix('.tmp'),path)

def source_hashes():
    paths=[Path(__file__),Path(__file__).with_name('run_e6_mechanism.py'),
           Path(__file__).with_name('analyze_e6_mechanism.py'),Path(__file__).with_name('control_e6_mechanism.py'),
           Path(__file__).with_name('verify_e6_mechanism.py'),
           Path(__file__).with_name('revision_followup_core.py'),Path(__file__).with_name('revision_models_mps.py'),
           Path(__file__).with_name('run_e5_ablation.py'),Path(__file__).with_name('run_exp4_matched_controls.py'),
           ROOT/'實驗一/models_m1_to_m5.py',ROOT/'實驗四/exp4_raw_data_V2.py',ROOT/'實驗四/exp4_defense_strategies_V2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def load_data():
    raw=dict(np.load(DATA/'raw_windows.npz'))
    test=np.flatnonzero(raw['partitions']=='val')
    assert len(test)==219 and set(raw['labels'][test])==set(range(11))
    # Preserve E4 order for exact saved-prediction comparison; selection uses sample ID order.
    return raw,test

def model_for(seed,device):
    folder=PREVIOUS/'final'/f'Yu3_combined_s{seed}'
    spec=json.loads((folder/'run_spec.json').read_text())
    saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=build('Yu3',spec['config']['dropout']).to(device).eval()
    model.load_state_dict(saved['state_dict'])
    return model,spec,saved['epoch']

def derangement(ids,condition):
    # Full-dataset, ID-keyed reproducible permutation, independent of inference batches.
    rng=rng_for('E6_cross_sample_v1',*condition,*map(str,ids))
    for _ in range(10000):
        order=rng.permutation(len(ids))
        if np.all(order!=np.arange(len(ids))):return order
    raise RuntimeError('Could not construct derangement')

def first_pass(model,array,device,batch=32):
    p1=[];recons=[]
    with torch.inference_mode():
        for b in torch.from_numpy(array).split(batch):
            x=b.to(device); z=model.encode(torch.cat([x,torch.zeros_like(x)],1),0)
            p1.append(model.classify(z).cpu().numpy());recons.append(model.decode(z).cpu().numpy())
    return np.concatenate(p1),np.concatenate(recons)

def second_pass(model,array,aux,device,batch=32):
    result=[]
    with torch.inference_mode():
        for start in range(0,len(array),batch):
            x=torch.from_numpy(array[start:start+batch]).to(device)
            r=torch.from_numpy(aux[start:start+batch]).to(device)
            result.append(model.classify(model.encode(torch.cat([x,r],1),1)).cpu().numpy())
    return np.concatenate(result)

def pixel_points(c,v,stats):
    c=FNS['_zscore_fixed'](c,stats['i_mean'],stats['i_std'])
    v=FNS['_zscore_fixed'](v,stats['v_mean'],stats['v_std'])
    return np.clip(((c+3)/6*63).astype(int),0,63),np.clip(((v+3)/6*63).astype(int),0,63)

def region_masks(raw,idx,stats,ps):
    masks=[];controls=[];trajectories=[];starts=[];control_starts=[];overlaps=[]
    for k in idx:
        c=raw['current'][k].astype(np.float32);v=raw['voltage'][k].astype(np.float32)
        n=len(c);length=min(n,max(1,int(round(.2*n))))
        rng=rng_for('test',str(raw['sample_ids'][k]),'missing_consecutive',.2,ps)
        start=int(rng.integers(0,max(1,n-length+1)))
        temporal=np.zeros(n,bool);temporal[start:start+length]=True
        a=FNS['_interpolate_missing'](c,temporal);b=FNS['_interpolate_missing'](v,temporal)
        ca,cb=FNS['apply_waveform_corruption'](c,v,'missing_consecutive',.2,stats,
             rng_for('test',str(raw['sample_ids'][k]),'missing_consecutive',.2,ps))
        assert np.array_equal(a,ca) and np.array_equal(b,cb)
        y,x=pixel_points(c,v,stats);yy,xx=pixel_points(a,b,stats)
        # Region includes both removed and inserted trajectory locations; square one-pixel dilation.
        def project(t):
            m=np.zeros((64,64),bool);m[y[t],x[t]]=True;m[yy[t],xx[t]]=True
            return binary_dilation(m,structure=np.ones((3,3),bool))
        candidates=[j for j in range(n-length+1) if j+length<=start or j>=start+length]
        assert candidates
        crng=rng_for('E6_control_segment_v1',str(raw['sample_ids'][k]),ps)
        cs=int(candidates[int(crng.integers(len(candidates)))])
        ct=np.zeros(n,bool);ct[cs:cs+length]=True
        assert not np.any(ct & temporal) and ct.sum()==temporal.sum()
        m=project(temporal);cm=project(ct);traj=np.zeros((64,64),bool);traj[y,x]=True;traj[yy,xx]=True
        masks.append(m);controls.append(cm);trajectories.append(traj)
        starts.append(start);control_starts.append(cs);overlaps.append(int((m&cm).sum()))
    return dict(missing_mask=np.stack(masks),control_mask=np.stack(controls),trajectory_mask=np.stack(trajectories),
                start=np.array(starts),control_start=np.array(control_starts),overlap_pixels=np.array(overlaps))

def energy_rows(residual,regions):
    e=residual[:,0].astype(np.float64)**2;total=e.sum((1,2))
    def ratio(mask):
        return np.divide((e*mask).sum((1,2)),total,out=np.full(len(e),np.nan),where=total>1e-12)
    m=regions['missing_mask'];c=regions['control_mask'];t=regions['trajectory_mask']
    return dict(total_energy=total,missing_energy_ratio=ratio(m),control_energy_ratio=ratio(c),
                missing_area_ratio=m.mean((1,2)),control_area_ratio=c.mean((1,2)),
                trajectory_coverage=np.divide((m&t).sum((1,2)),t.sum((1,2)),out=np.full(len(e),np.nan),where=t.sum((1,2))>0))

def independent_f1(labels,pred):
    cm=np.zeros((11,11),np.int64);np.add.at(cm,(labels,pred),1)
    den=cm.sum(0)+cm.sum(1)
    return float(np.divide(2*np.diag(cm),den,out=np.zeros(11,float),where=den>0).mean())
