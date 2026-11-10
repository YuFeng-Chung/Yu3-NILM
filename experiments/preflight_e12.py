"""Freeze E12 before new outer-test inference; real train-only runtime estimate."""
import platform
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from e12_core import *

def main():
    out=OUTPUT;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists(),'Preserve existing frozen protocol'
    assert json.loads((PREVIOUS/'completed_audit.json').read_text())['all_checks_passed']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available();raw,tr,te=load_data();stats=fit_stats(raw,tr)
    choices=json.loads((PREVIOUS/'E3_selected_strategies.json').read_text())
    old_conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
    original=json.loads((PREVIOUS/'final_run_manifest.json').read_text());manifest=[]
    # Check every model and every reused prediction before freezing; no test scores read.
    for i,entry in enumerate(original):
        folder=PREVIOUS/entry['path'];spec=json.loads((folder/'run_spec.json').read_text())
        assert spec['normalization']==stats and spec['train_ids']==raw['sample_ids'][tr].tolist() and spec['validation_ids']==[]
        assert spec['seed']==entry['seed'] and spec['model']==entry['model']
        ids=np.load(folder/'outer_ids_labels.npz')
        assert np.array_equal(ids['sample_ids'],raw['sample_ids'][te]) and np.array_equal(ids['labels'],raw['labels'][te])
        state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert state['epoch']==spec['fixed_epochs']
        item={**entry,'file_sha256':{n:sha(folder/n) for n in ['checkpoint.pt','run_spec.json','outer_ids_labels.npz','outer_predictions/clean.npy']},
              'state_sha256':fingerprint(state['state_dict']),'component_prediction_sha256':{}}
        for ci,cond in enumerate(CONDITIONS):
            if cond['source']=='E4':
                index=old_conditions.index([cond['kind'],cond['value'],cond['pollution_seed']])
                path=folder/'outer_predictions'/f'{index:03}.npy'
                pred=np.load(path);assert pred.shape==(219,) and np.isin(pred,np.arange(11)).all()
                item['component_prediction_sha256'][str(ci)]=sha(path)
        manifest.append(item);del state
        if (i+1)%50==0:print('SOURCE_AUDIT',i+1,'/',len(original),flush=True)
    assert len(manifest)==330
    # Synthetic identity, quantizer clipping/levels, harmonic energy, gain/offset checks.
    t=np.arange(500)*2*np.pi/500;c=np.sin(t).astype(np.float32);v=(100*np.sin(t+.4)).astype(np.float32)
    synthetic=dict(i_min=-2.,i_max=2.,v_min=-200.,v_max=200.)
    checks=[]
    for cond in CONDITIONS:
        a,b=perturb(c,v,'synthetic',cond,synthetic)
        if cond['kind']=='harmonic':
            injected=np.sqrt(np.mean((b.astype(float)-v)**2))/np.sqrt(np.mean(v.astype(float)**2))
            assert np.isclose(injected,cond['value'],atol=2e-8)
        if cond['kind']=='adc':
            for x,ch in [(a,'i'),(b,'v')]:
                z=(x.astype(float)-synthetic[ch+'_min'])/(synthetic[ch+'_max']-synthetic[ch+'_min'])*(2**cond['value']-1)
                assert np.allclose(z,np.round(z),atol=.001)
        if cond['kind']=='dc':
            ch=cond['channel'];delta=cond['value']*(synthetic[ch+'_max']-synthetic[ch+'_min'])
            assert np.allclose(a if ch=='i' else b,(c if ch=='i' else v)+delta,atol=1e-5)
            assert np.array_equal(b,v) if ch=='i' else np.array_equal(a,c)
        if cond['kind']=='gain':assert np.allclose(a,c*(1+cond['value'])) and np.array_equal(b,v)
    # Bounded real training-only checks use 219 samples, including every architecture.
    idx=tr[:219];new=[c for c in CONDITIONS if c['source']=='new']
    cache_tick=time.monotonic();arrays=[];image_comparisons=0
    for cond in new:
        a,cc,vv=image_array(raw,idx[:12],stats,cond)
        for j in range(12):
            assert np.array_equal(a[j],FNS['generate_vi_image_fixed_norm'](cc[j],vv[j],stats))
            image_comparisons+=1
        arrays.append(a)
    per_condition_cache=(time.monotonic()-cache_tick)/len(new)*219/12
    clean=image_array(raw,idx,stats,CONDITIONS[0])[0]
    stressed=image_array(raw,idx,stats,CONDITIONS[1])[0]
    timings=[]
    for name in FAMILY+MAIN[4:]:
        entry=next(r for r in manifest if r['model']==name and r['strategy']=='S3' and r['seed']==42)
        net,spec,state=load_model(entry,'mps');before=fingerprint(net.state_dict())
        predict(net,clean,name,'mps');durations=[]
        for a in [clean,stressed,clean]:
            tick=time.monotonic();pred=predict(net,a,name,'mps');torch.mps.synchronize();durations.append(time.monotonic()-tick)
            assert pred.shape==(219,) and np.isin(pred,np.arange(11)).all()
        assert before==fingerprint(net.state_dict())
        timings.append(dict(model=name,median_seconds_per_219=float(np.median(durations)),all_seconds=durations))
        print('TIMING',timings[-1],flush=True)
        del net,state;gc.collect();torch.mps.empty_cache()
    cost={r['model']:r['median_seconds_per_219'] for r in timings}
    point=sum(cost[r['model']]*66+1.5 for r in manifest)+66*per_condition_cache
    comparisons=comparison_specs(choices)
    condition_ids=list(dict.fromkeys(c['id'] for c in CONDITIONS))
    stressed_ids=[k for k in condition_ids if k!='clean' and not next(c for c in CONDITIONS if c['id']==k)['source']=='E4']
    # E4-style paired tests; one complete Holm family, not reduced after seeing results.
    ntests=len(comparisons)*(1+2*len(stressed_ids))
    estimate=dict(point_inference_minutes=point/60,inference_range_minutes=[point/60*.8,point/60*1.6],
        research_with_analysis_and_independent_audit_minutes=[point/60*.8+10,point/60*1.6+25],
        excludes='Formal bilingual Word response and original-manuscript integration',timings=timings,
        real_train_only=True,benchmark_samples=219,new_conditions_per_model=66,models=330)
    plan=dict(experiment='E12 combined disturbances and synthetic measurement errors',
        authorization='2026-11-08 author: 好，我們認知一致。那開始進行這實驗吧…再告訴我預計要多久跑完',
        frozen_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),conditions=CONDITIONS,manifest=manifest,
        seeds=SEEDS,pollution_seeds=POLLUTION,choices=choices,common_strategies=COMMON,main_models=MAIN,
        supplementary_controls=['ZeroFB','ShuffleFB'],new_condition_ids=stressed_ids,
        split='Original 855 train / 219 held-out test; class x house proxy groups disjoint. Historical test previously inspected.',
        normalization=stats,normalization_rule='float32 statistics fit to all 855 training windows only; fixed V/I z-score, 64x64 occupancy; clip integer indices to image boundary',
        training='None. Reuse all 330 completed E1/E3/E4 checkpoints including all co-selected ties. No new HPO, epoch selection, strategy selection or checkpoint changes.',
        combined='Apply noise first, then synchronous V/I missing mask and linear interpolation with nearest endpoint extension. Each component uses original E4 sample-ID/type/severity/seed RNG key; constituent single-condition predictions reused byte-for-byte after hashes and clean parity.',
        adc='Both channels; per-channel train min/max full scale, saturation then 2^b equally spaced reconstruction levels including endpoints. Half-up rounding floor(z+0.5). No dither or per-test fitting; report saturation and this assumed converter full scale.',
        harmonic='Nominal 500 samples/cycle (60 Hz at 30 kHz); least-squares intercept/sin/cos fundamental of each voltage window. Add equal-amplitude third/fifth harmonic components with combined RMS/fundamental ratio 3% or 5%. Shared random relative phases per sample/pollution seed, same phases across strengths. This is injected harmonic ratio, NOT a guarantee of final waveform THD; existing harmonics/transients retained, current unchanged. Not a circuit/load response simulation.',
        dc='One channel at a time, +/-1% and +/-5% of that channel training peak-to-peak full range. Other channel unchanged. Both signs retained; broad sensitivity stresses, not calibrated device specifications.',
        gain='Current only, multiplicative (1+g), g=-5%,-2%,+2%,+5%; voltage unchanged.',
        randomness='10 noise/missing/phase realizations for stochastic conditions; deterministic ADC/DC/gain evaluated ONCE per checkpoint, never counted as 10 realizations.',
        comparisons=comparisons,
        analysis=dict(primary='All-11-class Macro-F1, clean-minus-corrupted Dabs in percentage points; average stochastic realizations within training seed, then mean +/- sample SD over seeds42-51.',
            paired='Same E4 two-sided exhaustive sign-permutation Wilcoxon on seed-paired differences, zero_method=wilcox, rounded12; all-zero p=1. One global Holm family over all predeclared unique model/strategy comparisons, clean F1 and all new stress F1/Dabs.',
            tests_planned=ntests,number_unique_comparisons=len(comparisons),number_new_stresses=len(stressed_ids),
            intervals='Fixed RNG20261108,10000 paired seed bootstraps, percentile95%; descriptive over training seeds, not population/appliance uncertainty.',
            component_analysis='For all three combinations show all constituent single controls and combined F1; paired changes combined minus each constituent, full seed CSV and descriptive CIs, no extra selective p family.',
            limitations='n10 minimum attainable two-sided p=2/1024, full Holm family makes any rejection impossible at .05 when family>25. Non-significance does not establish equivalence; retain full family without post-hoc shrinkage. Previously viewed test, single split, group proxy, Mac MPS not independent CUDA reproduction, multiple simultaneous factors prevent pure feedback causality.',
            reporting='All positive and negative outcomes; no winning strategy picked on E12 test. Supplementary target; true field/acquisition validation remains absent. No deployment or future hardware commitment.'),
        raw_sha256=sha(DATA/'raw_windows.npz'),previous_protocol_sha256=sha(PREVIOUS/'protocol_v4.json'),
        selection_sha256=sha(PREVIOUS/'E3_selected_strategies.json'),source_sha256=source_hashes(),
        environment=dict(platform=platform.platform(),host=platform.node(),torch=str(torch.__version__),device='mps',threads=4),
        runtime_estimate=estimate)
    save_json(out/'protocol.json',plan);save_json(out/'runtime_estimate.json',estimate)
    for rel in plan['source_sha256']:
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(out/'preflight_audit.json',dict(all_checks_passed=True,models_source_checked=330,reused_component_files_checked=330*60,
        real_training_image_parity=image_comparisons,synthetic_quantization_harmonic_offset_gain_checks=True,
        nine_architecture_forward_checks=timings,weights_bn_unchanged=True,no_new_outer_test_inference=True,source_sha256=plan['source_sha256']))
    print('E12_PREFLIGHT_PASS',json.dumps(estimate,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
