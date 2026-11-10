"""Supplemental independent raw-window, perturbation/image and diagnostic audit.
Does not change frozen research sources or training. Separate post-freeze QA source.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from e9_core import ROOT,DATA,OUTPUT,CELLS,save_json,sha

def render(c,v,stats,norm):
    sd=np.std(c)
    iz=(c-c.mean())/sd if norm=='N0' and sd>1e-5 else (c.copy() if norm=='N0' else ((c-stats['i_mean'])/stats['i_std'] if stats['i_std']>1e-5 else c-stats['i_mean']))
    sd=np.std(v);vz=(v-v.mean())/sd if sd>1e-5 else v.copy()
    # Independently accumulate sequentially, with same specified mapping arithmetic.
    iy=((iz+3)/6*63).astype(int);ix=((vz+3)/6*63).astype(int);out=np.zeros((64,64),np.float32)
    for y,x in zip(iy,ix):out[min(63,max(0,y)),min(63,max(0,x))]+=np.float32(.1)
    return np.minimum(out,1)[None],iz,vz,iy,ix

def perturb(c,v,sid,kind,severity,seed,stats):
    if kind=='clean':return c.copy(),v.copy()
    h=hashlib.sha256('|'.join(map(str,['E9',sid,kind,severity,seed])).encode()).digest()
    rng=np.random.default_rng(int.from_bytes(h[:8],'little'))
    out=[]
    for x,prefix in [(c,'i'),(v,'v')]:
        x=x.copy()
        if kind=='awgn':
            variance=np.var(x.astype(np.float64))
            if variance>1e-12:x=(x+rng.normal(0.,np.sqrt(variance/10**(severity/10.)),size=x.shape)).astype(np.float32)
        elif kind=='salt_pepper':
            k=round(severity*len(x));idx=rng.choice(len(x),size=k,replace=False)
            x[idx]=np.where(rng.random(k)<.5,stats[prefix+'_max'],stats[prefix+'_min'])
        elif kind=='missing_consecutive':
            k=min(len(x),max(1,round(severity*len(x))));start=int(rng.integers(0,max(1,len(x)-k+1)))
            good=np.ones(len(x),bool);good[start:start+k]=False;axis=np.arange(len(x));x=np.interp(axis,axis[good],x[good])
        else:raise ValueError(kind)
        out.append(x)
    # Missing-data mask is synchronous: use single selection for both channels.
    if kind=='missing_consecutive':
        rng=np.random.default_rng(int.from_bytes(h[:8],'little'));k=min(len(c),max(1,round(severity*len(c))));start=int(rng.integers(0,max(1,len(c)-k+1)))
        good=np.ones(len(c),bool);good[start:start+k]=False;axis=np.arange(len(c));out=[np.interp(axis,axis[good],x[good]) for x in [c,v]]
    return out

def data_audit():
    raw=dict(np.load(OUTPUT/'waveforms.npz'));manifest=pd.read_csv(OUTPUT/'window_manifest.csv',dtype={'sample_id':str});byid={r.sample_id:r for r in manifest.itertuples()}
    original={r['sample_id']:r for r in json.loads((DATA/'waveform_image_audit.json').read_text())['records']}
    outer=zipfile.ZipFile(DATA/'plaid_2014_published.zip');z=zipfile.ZipFile(io.BytesIO(outer.read('2014.zip')));files={Path(n).stem:n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX')}
    split=json.loads((OUTPUT/'split.json').read_text())['indices'];examples={}
    for label in range(11):
        examples[min((str(raw['sample_ids'][j]) for j in split['outer_train'] if raw['labels'][j]==label),key=int)]=label
    checks=0;figures=OUTPUT/'figures';figures.mkdir(exist_ok=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(11,2,figsize=(11,25))
    for j,sid in enumerate(raw['sample_ids']):
        sid=str(sid);content=z.read(files[sid]);assert hashlib.sha256(content).hexdigest()==original[sid]['csv_sha256']
        x=pd.read_csv(io.BytesIO(content),header=None).to_numpy(float);c,v=x[:,0],x[:,1];r=byid[sid]
        if r.eligible:
            phase=r.start+np.arange(500)*(r.end-r.start)/500
            assert np.allclose(np.interp(phase,np.arange(len(c)),c),raw['W1_current'][j],rtol=0,atol=1e-9)
            assert np.allclose(np.interp(phase,np.arange(len(v)),v),raw['W1_voltage'][j],rtol=0,atol=1e-7)
            assert 450<=r.end-r.start<=550 and r.start>=1000 and r.rms_spread<=.05 and r.nrmse<=.10;checks+=1
        if sid in examples:
            label=examples[sid];event=int(np.argmax(np.abs(np.diff(c))));start=max(0,event-100);end=start+500
            if end>len(c):end=len(c);start=max(0,end-500)
            ax=axes[label,0];lo=max(0,start-500);hi=min(len(c),end+500);ax.plot(np.arange(lo,hi)/30,c[lo:hi],lw=.7,color='#667085');ax.axvspan(start/30,end/30,color='#de8c41',alpha=.2);ax.axvline(event/30,color='#c44242',ls='--',lw=.8)
            ax.set_title(f'Class {label}, training ID {sid}: W0 around max difference');ax.set_xlabel('Recording time (ms)');ax.set_ylabel('Current')
            ax=axes[label,1];lo=max(0,int(r.start)-500);hi=min(len(c),int(r.end)+500);ax.plot(np.arange(lo,hi)/30,c[lo:hi],lw=.7,color='#667085');ax.axvspan(r.start/30,r.end/30,color='#147d92',alpha=.2)
            ax.set_title(f'W1: RMS spread={r.rms_spread:.3f}, NRMSE={r.nrmse:.3f}');ax.set_xlabel('Recording time (ms)');ax.set_ylabel('Current')
    fig.suptitle('Fixed first numeric eligible outer-training ID in each class; diagnostic selection, not test scores',y=1.001);fig.tight_layout();fig.savefig(figures/'all11_fixed_training_window_traces.png',dpi=300,bbox_inches='tight');plt.close(fig)
    save_json(OUTPUT/'independent_raw_window_audit.json',dict(all_checks_passed=True,raw_csv_hashes=1074,saved_W1_interpolations_checked=checks,
        fixed_training_waveform_examples=11,criteria_bounds_checked=True,scope='saved W1 interpolation/manifest bounds; not independent inference of physical activation or steady state',qa_source_sha256=sha(Path(__file__))))
    print('Independent raw window PASS',checks,flush=True)

def result_audit():
    assert (OUTPUT/'completed_audit.json').exists(),'Wait for single research writer to complete'
    raw=dict(np.load(OUTPUT/'waveforms.npz'));split=json.loads((OUTPUT/'split.json').read_text())['indices'];plan=json.loads((OUTPUT/'protocol.json').read_text());diag=pd.read_csv(OUTPUT/'diagnostics_per_sample.csv',dtype={'sample_id':str})
    lookup={(r.cell,r.sample_id,r.kind,r.severity,r.pollution_seed):r for r in diag.itertuples()};nimages=0;ndiagnostics=0
    for cell,(window,norm) in CELLS.items():
        stats=json.loads((OUTPUT/'runs'/f'{cell}_M1_s42/run_spec.json').read_text())['normalization'];clean={}
        for j in split['outer_test']:
            clean[j]=render(raw[window+'_current'][j],raw[window+'_voltage'][j],stats,norm)[0]
        for ci,condition in enumerate(plan['conditions']):
            kind,severity,seed=condition
            cache=np.load(OUTPUT/'evaluation_images'/cell/f'{ci:03}.npy')
            for k,j in enumerate(split['outer_test']):
                sid=str(raw['sample_ids'][j]);c,v=perturb(raw[window+'_current'][j],raw[window+'_voltage'][j],sid,kind,severity,seed,stats)
                img,iz,vz,iy,ix=render(c,v,stats,norm);assert np.array_equal(img,cache[k]),(cell,ci,sid);nimages+=1
        conditions=[('clean',0,0)]+[(kind,severity,seed) for kind,sevs in [('awgn',[30,20]),('salt_pepper',[.01,.2])] for severity in sevs for seed in range(1000,1010)]
        for kind,severity,seed in conditions:
            for j in split['outer_test']:
                sid=str(raw['sample_ids'][j]);c,v=perturb(raw[window+'_current'][j],raw[window+'_voltage'][j],sid,kind,severity,seed,stats);img,iz,vz,iy,ix=render(c,v,stats,norm)
                r=lookup[(cell,sid,kind,severity,seed)];base=clean[j];a=img[0]>0;b=base[0]>0;union=np.logical_or(a,b).sum();inter=np.logical_and(a,b).sum()
                expected=dict(image_iou=inter/union if union else 1.,new_nonzero=np.logical_and(a,~b).sum(),lost_nonzero=np.logical_and(~a,b).sum(),clean_nonzero=b.sum(),perturbed_nonzero=a.sum(),image_mean_abs_change=np.abs(img-base).mean(),
                    i_beyond_3sigma_fraction=np.mean(np.abs(iz)>3),v_beyond_3sigma_fraction=np.mean(np.abs(vz)>3),i_index_clipped_fraction=np.mean((iy<0)|(iy>63)),v_index_clipped_fraction=np.mean((ix<0)|(ix>63)))
                for key,value in expected.items():assert np.isclose(getattr(r,key),value,rtol=1e-8,atol=1e-10),(cell,sid,kind,severity,key)
                ndiagnostics+=1
    # Verify full exported per-class PRF and confusion counts against saved predictions.
    classes=pd.read_csv(OUTPUT/'per_class_metrics.csv')
    confusions=pd.read_csv(OUTPUT/'confusion_counts.csv')
    class_groups={k:g.sort_values('label') for k,g in classes.groupby(['cell','model','training_seed','condition'])}
    confusion_groups={k:g.sort_values(['true_label','predicted_label']) for k,g in confusions.groupby(['cell','model','training_seed','condition'])}
    checked_class_rows=0;checked_confusion_cells=0
    for folder in sorted((OUTPUT/'runs').iterdir()):
        spec=json.loads((folder/'run_spec.json').read_text());identity=np.load(folder/'test_identity.npz');y=identity['labels']
        for ci in range(41):
            pred=np.load(folder/'predictions'/f'{ci:03}.npy');cm=np.zeros((11,11),int)
            for label in range(11):
                cm[label]=np.bincount(pred[y==label],minlength=11)
            key=(spec['cell'],spec['model'],spec['seed'],ci)
            assert np.array_equal(confusion_groups[key]['count'].to_numpy().reshape(11,11),cm)
            tp=cm.diagonal().astype(float);precision=np.divide(tp,cm.sum(0),out=np.zeros(11),where=cm.sum(0)>0);recall=np.divide(tp,cm.sum(1),out=np.zeros(11),where=cm.sum(1)>0)
            f1=np.divide(2*tp,cm.sum(0)+cm.sum(1),out=np.zeros(11),where=cm.sum(0)+cm.sum(1)>0)
            actual=class_groups[key]
            for column,expected in [('precision',precision),('recall',recall),('f1',f1),('support',cm.sum(1))]:assert np.allclose(actual[column],expected,rtol=1e-10,atol=1e-12)
            checked_class_rows+=11;checked_confusion_cells+=121
    save_json(OUTPUT/'supplemental_independent_audit.json',dict(all_checks_passed=True,all_cached_test_images_independently_rebuilt=nimages,
        per_class_rows_independently_verified=checked_class_rows,confusion_cells_independently_verified=checked_confusion_cells,
        all_diagnostic_geometry_rows_independently_recomputed=ndiagnostics,independent_rng_and_waveform_corruptions=True,qa_source_sha256=sha(Path(__file__)),
        scope='all cached clean/noisy/missing images; all diagnostic overlap/L1/clip values; does not rerun every MPS classifier'))
    print('Supplemental PASS',nimages,ndiagnostics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args()
    if a.data_only:data_audit()
    else:result_audit()
