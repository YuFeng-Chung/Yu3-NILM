"""E12 frozen-checkpoint waveform stresses. No training or test-driven selection."""
import gc
import json
import os
from pathlib import Path
import numpy as np
import torch
from revision_followup_core import ROOT, DATA, fit_stats, build, predict, csv_write
from run_exp4_matched_controls import FNS, scatter_fast, rng_for, sha, save_json

PREVIOUS = ROOT/'results/e1_e4_mac_20261106'
OUTPUT = ROOT/'results/e12_measurement_20261108'
SEEDS = list(range(42,52))
POLLUTION = list(range(1000,1010))
MAIN = ['M1','RecAux','M3','Yu3','ResNet18','MobileNetV3','ViT-Tiny']
FAMILY = ['M1','RecAux','M3','Yu3','ZeroFB','ShuffleFB']
COMMON = ['no_defense','S3','combined']
COMBOS = {
    'awgn20_consecutive10': [('awgn',20),('missing_consecutive',.1)],
    'awgn10_consecutive20': [('awgn',10),('missing_consecutive',.2)],
    'sp01_random05': [('salt_pepper',.01),('missing_random',.05)],
}

def conditions():
    out=[dict(id='clean',group='clean',kind='clean',value=0,pollution_seed=-1,source='new')]
    def add(label,group,kind,value,stochastic=False,channel='both',source='new'):
        for ps in POLLUTION if stochastic else [-1]:
            out.append(dict(id=label,group=group,kind=kind,value=value,channel=channel,pollution_seed=ps,source=source))
    for key in COMBOS: add(key,'combined','combined',key,True)
    for bits in [8,10,12]:add(f'adc_{bits}bit','adc','adc',bits)
    for thd in [.03,.05]:add(f'added_harmonics_{int(thd*100)}pct','harmonic','harmonic',thd,True,'v')
    for ch in ['i','v']:
        for v in [-.05,-.01,.01,.05]:add(f'dc_{ch}_{v:+g}','dc','dc',v,channel=ch)
    for v in [-.05,-.02,.02,.05]:add(f'gain_i_{v:+g}','gain','gain',v,channel='i')
    for kind,value in [('awgn',20),('awgn',10),('missing_consecutive',.1),('missing_consecutive',.2),('salt_pepper',.01),('missing_random',.05)]:
        add(f'{kind}_{value:g}','component_control',kind,value,True,source='E4')
    assert len(out)==126 and len([c for c in out if c['source']=='new'])==66
    return out

CONDITIONS=conditions()

def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)

def load_data():
    raw=dict(np.load(DATA/'raw_windows.npz'))
    tr=np.flatnonzero(raw['partitions']=='train');te=np.flatnonzero(raw['partitions']=='val')
    assert (len(tr),len(te))==(855,219)
    assert not set(raw['groups'][tr])&set(raw['groups'][te])
    assert raw['current'].shape==raw['voltage'].shape==(1074,500)
    assert set(raw['labels'][te])==set(range(11))
    return raw,tr,te

def component(c,v,sid,kind,value,ps,stats):
    return FNS['apply_waveform_corruption'](c,v,kind,value,stats,rng_for('test',sid,kind,value,ps))

def perturb(c,v,sid,cond,stats):
    c=c.astype(np.float32).copy();v=v.astype(np.float32).copy()
    kind,val,ps=cond['kind'],cond['value'],cond['pollution_seed']
    if kind=='clean':return c,v
    if kind=='combined':
        for k,s in COMBOS[val]:c,v=component(c,v,sid,k,s,ps,stats)
    elif kind=='adc':
        def quant(x,ch):
            lo,hi=stats[ch+'_min'],stats[ch+'_max'];levels=2**val-1
            q=np.floor((np.clip(x.astype(np.float64),lo,hi)-lo)/(hi-lo)*levels+.5)
            return (lo+q*(hi-lo)/levels).astype(np.float32)
        c,v=quant(c,'i'),quant(v,'v')
    elif kind=='harmonic':
        t=2*np.pi*np.arange(len(v))/500.
        # Nominal one-cycle 500-point grid; intercept + fundamental least squares.
        a,b,_=np.linalg.lstsq(np.column_stack([np.sin(t),np.cos(t),np.ones(len(t))]),v.astype(float),rcond=None)[0]
        amplitude=np.hypot(a,b);phase=np.arctan2(b,a)
        r=rng_for('E12_harmonics_v1',sid,ps);u3,u5=r.uniform(-np.pi,np.pi,2)
        added=amplitude*val/np.sqrt(2)*(np.sin(3*t+3*phase+u3)+np.sin(5*t+5*phase+u5))
        v=(v.astype(float)+added).astype(np.float32)
    elif kind=='dc':
        ch=cond['channel'];offset=val*(stats[ch+'_max']-stats[ch+'_min'])
        if ch=='i':c=(c.astype(float)+offset).astype(np.float32)
        else:v=(v.astype(float)+offset).astype(np.float32)
    elif kind=='gain':c=(c.astype(float)*(1+val)).astype(np.float32)
    else:c,v=component(c,v,sid,kind,val,ps,stats)
    assert np.isfinite(c).all() and np.isfinite(v).all()
    return c,v

def image_array(raw,idx,stats,cond):
    waves=[perturb(raw['current'][k],raw['voltage'][k],str(raw['sample_ids'][k]),cond,stats) for k in idx]
    return np.stack([scatter_fast(c,v,stats) for c,v in waves]),np.stack([w[0] for w in waves]),np.stack([w[1] for w in waves])

def fingerprint(state):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def load_model(entry,device):
    folder=PREVIOUS/entry['path'];spec=json.loads((folder/'run_spec.json').read_text())
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    net=build(entry['model'],spec['config']['dropout']).to(device).eval();net.load_state_dict(state['state_dict'])
    return net,spec,state

def source_hashes():
    paths=[Path(__file__),Path(__file__).with_name('preflight_e12.py'),Path(__file__).with_name('run_e12.py'),
           Path(__file__).with_name('control_e12.py'),Path(__file__).with_name('revision_followup_core.py'),
           Path(__file__).with_name('revision_models_mps.py'),Path(__file__).with_name('run_e5_ablation.py'),
           Path(__file__).with_name('run_exp4_matched_controls.py'),ROOT/'實驗一/models_m1_to_m5.py',
           ROOT/'實驗四/exp4_raw_data_V2.py',ROOT/'實驗四/exp4_defense_strategies_V2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def independent_scores(y,pred):
    cm=np.zeros((11,11),np.int64);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm).astype(float);support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(11),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(11),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(11),where=(support+predicted)>0)
    return dict(macro_f1=float(f1.mean()),accuracy=float(tp.sum()/len(y))),cm,precision,recall,f1

def comparison_specs(choices):
    out=[]
    def add(a,sa,b,sb,context):out.append(dict(a=a,sa=sa,b=b,sb=sb,context=context))
    for s in COMMON:
        for a,b in [('Yu3','M1'),('Yu3','RecAux'),('RecAux','M1'),('M3','M1'),('Yu3','ZeroFB'),('Yu3','ShuffleFB')]:add(a,s,b,s,'common/'+s)
        for b in MAIN[4:]:add('Yu3',s,b,s,'common/'+s)
    for track in ['noise','missing']:
        for b in MAIN[4:]:
            for sa in choices['Yu3'][track]:
                for sb in choices[b][track]:add('Yu3',sa,b,sb,'selected/'+track)
    # Same pair may appear in common and selected contexts: test once, keep labels.
    unique={}
    for r in out:
        key=(r['a'],r['sa'],r['b'],r['sb'])
        if key not in unique:unique[key]={**r,'contexts':[r['context']]}
        elif r['context'] not in unique[key]['contexts']:unique[key]['contexts'].append(r['context'])
    return list(unique.values())
