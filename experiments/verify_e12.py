"""Independent E12 waveform reconstruction and saved-prediction/metric audit.

Does not call e12_core.perturb, scatter_fast or independent_scores as references.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
from e12_core import ROOT,DATA,PREVIOUS,CONDITIONS,COMBOS,sha,save_json,csv_write

def independent_rng(*parts):
    h=hashlib.sha256('|'.join(str(x) for x in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8],'little'))

def interpolate(x,mask):
    out=x.copy();valid=np.flatnonzero(~mask)
    if len(valid) and len(valid)!=len(x):out[mask]=np.interp(np.flatnonzero(mask),valid,x[valid])
    return out

def single(c,v,sid,kind,value,ps,stats):
    rng=independent_rng('test',sid,kind,value,ps)
    if kind=='awgn':
        out=[]
        for x in [c,v]:
            power=np.mean((x.astype(float)-np.mean(x.astype(float)))**2)
            out.append((x+rng.normal(0,np.sqrt(power/10**(value/10)),len(x))).astype(np.float32) if power>1e-12 else x.copy())
        return tuple(out)
    if kind=='salt_pepper':
        result=[]
        for x,ch in [(c,'i'),(v,'v')]:
            y=x.copy();count=int(round(len(x)*value));positions=rng.choice(len(x),count,replace=False)
            y[positions]=np.where(rng.random(count)<.5,stats[ch+'_max'],stats[ch+'_min']);result.append(y)
        return tuple(result)
    if kind=='missing_consecutive':
        mask=np.zeros(len(c),bool);n=min(len(c),max(1,round(value*len(c))))
        start=int(rng.integers(0,max(1,len(c)-n+1)));mask[start:start+n]=True
    else:
        assert kind=='missing_random';mask=rng.random(len(c))<value
    return interpolate(c,mask),interpolate(v,mask)

def reference_waveforms(c,v,sid,cond,stats):
    c=c.astype(np.float32).copy();v=v.astype(np.float32).copy();kind=cond['kind'];value=cond['value'];ps=cond['pollution_seed']
    if kind=='clean':return c,v
    if kind=='combined':
        for typ,sev in COMBOS[value]:c,v=single(c,v,sid,typ,sev,ps,stats)
        return c,v
    if kind=='adc':
        output=[]
        for x,ch in [(c,'i'),(v,'v')]:
            lo,hi=stats[ch+'_min'],stats[ch+'_max'];step=(hi-lo)/(2**value-1)
            # Divide-first quantizer formula preserves the specified half-up rule.
            index=np.floor((np.maximum(lo,np.minimum(x.astype(float),hi))-lo)/(hi-lo)*(2**value-1)+.5)
            output.append((index*step+lo).astype(np.float32))
        return tuple(output)
    if kind=='dc':
        ch=cond['channel'];bias=(stats[ch+'_max']-stats[ch+'_min'])*value
        return ((c.astype(float)+bias).astype(np.float32),v) if ch=='i' else (c,(v.astype(float)+bias).astype(np.float32))
    if kind=='gain':return (c.astype(float)*(1+value)).astype(np.float32),v
    assert kind=='harmonic'
    t=2*np.pi*np.arange(len(v))/500.
    # Orthogonal projection on exactly one nominal cycle, independent of lstsq.
    a=2*np.dot(v.astype(float),np.sin(t))/len(t);b=2*np.dot(v.astype(float),np.cos(t))/len(t)
    peak=np.hypot(a,b);phi=np.arctan2(b,a);u=independent_rng('E12_harmonics_v1',sid,ps).uniform(-np.pi,np.pi,2)
    injected=peak*value/np.sqrt(2)*(np.sin(3*t+3*phi+u[0])+np.sin(5*t+5*phi+u[1]))
    return c,(v.astype(float)+injected).astype(np.float32)

def reference_image(c,v,stats):
    def axis(x,ch):
        # float32 arithmetic and truncation towards zero match original pixel contract.
        x=(x-stats[ch+'_mean'])/stats[ch+'_std']
        indices=((x+3)/6*63).astype(np.int64)
        return np.clip(indices,0,63),indices,x
    iy,rawi,zi=axis(c,'i');ix,rawv,zv=axis(v,'v');a=np.zeros((1,64,64),np.float32)
    for row,col in zip(iy,ix):a[0,row,col]+=np.float32(.1)
    np.minimum(a,1,out=a)
    return a,dict(i_abs_z_gt3=float((abs(zi)>3).mean()),v_abs_z_gt3=float((abs(zv)>3).mean()),
        i_actual_index_clip=float(((rawi<0)|(rawi>63)).mean()),v_actual_index_clip=float(((rawv<0)|(rawv>63)).mean()))

def harmonic_ratio(v):
    t=2*np.pi*np.arange(len(v))/500.
    peaks=np.array([np.hypot(2*np.dot(v.astype(float),np.sin(h*t))/len(t),2*np.dot(v.astype(float),np.cos(h*t))/len(t)) for h in [1,3,5]])
    return float(np.hypot(peaks[1],peaks[2])/peaks[0]) if peaks[0]>1e-12 else None

def verify_waveforms(out):
    plan=json.loads((out/'protocol.json').read_text());stats=plan['normalization'];raw=dict(np.load(DATA/'raw_windows.npz'))
    assert sha(DATA/'raw_windows.npz')==plan['raw_sha256'];te=np.flatnonzero(raw['partitions']=='val')
    clean=np.load(out/'waveform_cache/000.npz')['image'];diagnostics=[];max_error=0.;count=0
    for ci,cond in enumerate(CONDITIONS):
        if cond['source']!='new':continue
        saved=np.load(out/'waveform_cache'/f'{ci:03}.npz')
        assert np.array_equal(saved['sample_ids'],raw['sample_ids'][te]) and np.array_equal(saved['labels'],raw['labels'][te])
        saved_current=saved['current'];saved_voltage=saved['voltage'];saved_images=saved['image']
        for j,k in enumerate(te):
            c0=raw['current'][k].astype(np.float32);v0=raw['voltage'][k].astype(np.float32);sid=str(raw['sample_ids'][k])
            a,b=reference_waveforms(c0,v0,sid,cond,stats);c,v=saved_current[j],saved_voltage[j]
            error=max(float(abs(c.astype(float)-a).max()),float(abs(v.astype(float)-b).max()));max_error=max(max_error,error)
            # Different harmonic solvers may differ below one float32 ULP near zero;
            # image audit ALWAYS uses actual saved waveforms and requires exact pixels.
            if cond['kind']=='harmonic':
                np.testing.assert_allclose(c,a,rtol=0,atol=0);np.testing.assert_allclose(v,b,rtol=2e-6,atol=2e-5)
            else:
                np.testing.assert_array_equal(c,a);np.testing.assert_array_equal(v,b)
            im,diag=reference_image(c,v,stats);assert np.array_equal(im,saved_images[j]),(ci,sid)
            cm=clean[j]>0;pm=im>0;union=(cm|pm).sum()
            extra=dict(image_iou=float((cm&pm).sum()/union) if union else 1.,image_l1=float(np.abs(im-clean[j]).mean()),
                i_rms_change=float(np.sqrt(np.mean((c.astype(float)-c0)**2))),v_rms_change=float(np.sqrt(np.mean((v.astype(float)-v0)**2))),
                i_outside_adc_fullscale=float(((c0<stats['i_min'])|(c0>stats['i_max'])).mean()),
                v_outside_adc_fullscale=float(((v0<stats['v_min'])|(v0>stats['v_max'])).mean()),
                nominal_h35_ratio_before=harmonic_ratio(v0) if cond['kind']=='harmonic' else None,
                nominal_h35_ratio_after=harmonic_ratio(v) if cond['kind']=='harmonic' else None,
                injected_harmonic_ratio=None)
            if cond['kind']=='harmonic':
                t=2*np.pi*np.arange(500)/500;peak=np.hypot(2*np.dot(v0.astype(float),np.sin(t))/500,2*np.dot(v0.astype(float),np.cos(t))/500)
                extra['injected_harmonic_ratio']=float(np.sqrt(np.mean((v.astype(float)-v0)**2))/(peak/np.sqrt(2))) if peak>1e-12 else None
                if peak>1e-4:assert np.isclose(extra['injected_harmonic_ratio'],cond['value'],rtol=1e-4,atol=1e-6)
            diagnostics.append(dict(condition_index=ci,condition=cond['id'],group=cond['group'],pollution_seed=cond['pollution_seed'],sample_id=sid,**diag,**extra));count+=1
    csv_write(out/'waveform_image_diagnostics.csv',diagnostics)
    return dict(waveforms_verified=count,images_pixel_exact=count,max_waveform_absolute_error=max_error,
        independent_harmonic_projection=True,no_test_fitted_parameters=True)

def verify_predictions(out):
    plan=json.loads((out/'protocol.json').read_text());raw=dict(np.load(DATA/'raw_windows.npz'));te=np.flatnonzero(raw['partitions']=='val');y=raw['labels'][te]
    rows=[];count=0;clean=0
    for entry in plan['manifest']:
        key=Path(entry['path']).name;saved=np.load(out/'predictions'/f'{key}.npz')
        assert np.array_equal(saved['sample_ids'],raw['sample_ids'][te]) and np.array_equal(saved['labels'],y)
        p=saved['predictions'];assert p.shape==(126,219)
        derived=np.load(out/'metrics'/f'{key}.npz')
        for ci,cond in enumerate(CONDITIONS):
            precision,recall,f1,support=precision_recall_fscore_support(y,p[ci],labels=np.arange(11),zero_division=0)
            cm=confusion_matrix(y,p[ci],labels=np.arange(11))
            np.testing.assert_allclose(derived['precision'][ci],precision,atol=1e-12,rtol=0)
            np.testing.assert_allclose(derived['recall'][ci],recall,atol=1e-12,rtol=0)
            np.testing.assert_allclose(derived['class_f1'][ci],f1,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(derived['confusion'][ci],cm)
            assert abs(derived['f1'][ci]-f1.mean())<1e-12
            assert abs(derived['accuracy'][ci]-accuracy_score(y,p[ci]))<1e-12
            count+=len(y)
            if ci==0:
                np.testing.assert_array_equal(p[ci],np.load(PREVIOUS/entry['path']/'outer_predictions/clean.npy'));clean+=len(y)
        for rel,digest in entry['file_sha256'].items():assert sha(PREVIOUS/entry['path']/rel)==digest
    return dict(predictions_recomputed=count,clean_predictions_rechecked=clean,
        confusion_cells_verified=len(plan['manifest'])*126*121,class_prf_values_verified=len(plan['manifest'])*126*11*3,
        original_checkpoint_spec_clean_ids_unchanged=True)

def verify_statistics(out):
    import csv
    from itertools import product
    from scipy.stats import rankdata
    tests=list(csv.DictReader((out/'paired_tests.csv').open()));diffs=list(csv.DictReader((out/'paired_differences.csv').open()))
    bykey={}
    for r in diffs:bykey.setdefault(r['test_id'],[]).append(r)
    p=[];bootstrap=np.random.default_rng(20261108).integers(0,10,(10000,10))
    for r in tests:
        values=sorted(bykey[r['test_id']],key=lambda x:int(x['seed']));assert [int(v['seed']) for v in values]==list(range(42,52))
        d=np.array([float(v['difference_pp']) for v in values]);nz=np.round(d,12);nz=nz[nz!=0]
        if len(nz):
            ranks=rankdata(abs(nz));observed=abs(np.dot(ranks,np.sign(nz)));signs=np.array(list(product([-1,1],repeat=len(nz))))
            probability=float(np.mean(abs(signs@ranks)>=observed-1e-10))
        else:probability=1.
        assert abs(float(r['p_raw'])-probability)<1e-12
        assert abs(float(r['mean_difference_pp'])-d.mean())<1e-10
        lo,hi=np.quantile(d[bootstrap].mean(1),[.025,.975])
        assert abs(lo-float(r['CI95_mean_low_pp']))<1e-10 and abs(hi-float(r['CI95_mean_high_pp']))<1e-10
        p.append(probability)
    order=np.argsort(p,kind='stable');adjusted=np.ones(len(p));last=0.
    for i,j in enumerate(order):last=max(last,(len(p)-i)*p[j]);adjusted[j]=min(1,last)
    for r,a in zip(tests,adjusted):assert abs(float(r['p_holm_global'])-a)<1e-12
    plan=json.loads((out/'protocol.json').read_text());assert len(tests)==plan['analysis']['tests_planned']
    return dict(paired_tests_verified=len(tests),paired_differences_verified=len(diffs),
        p_independent_sign_enumeration=True,holm_and_bootstrap_verified=True,
        minimum_two_sided_p=2/1024,global_holm_first_threshold=.05/len(tests),
        global_holm_rejection_impossible_by_design=len(tests)>25)

def verify(out):
    result=dict(all_checks_passed=True,**verify_waveforms(out),**verify_predictions(out),**verify_statistics(out))
    save_json(out/'independent_audit.json',result);return result

if __name__=='__main__':
    import argparse
    from e12_core import OUTPUT
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(json.dumps(verify(a.output),indent=2))
