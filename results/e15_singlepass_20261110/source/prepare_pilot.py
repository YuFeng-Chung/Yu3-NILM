"""E15 calibration and inner-validation pilots only; no formal-training entry point."""
import sys,os,json,time,fcntl,platform,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'experiments'))
import e15_core as core
from e15_core import *
from run_e2_search_mps import inner_split
E14=ROOT/'results/e14_noisefb_20261110'
CONFIG={'lr':.0005,'weight_decay':0.,'dropout':.5}
NAMES=['1P-NoiseCh','1P-Mix']

def protected_paths():
    paths=set((ROOT/'experiments').glob('*.py'))
    paths.update((ROOT/'實驗一').glob('*.py'));paths.update((ROOT/'實驗四').glob('*.py'))
    paths.add(DATA/'raw_windows.npz')
    for p in E14.rglob('*'):
        if p.is_file() and '__pycache__' not in str(p) and p.suffix not in ['.lock']:
            paths.add(p)
    return sorted(paths)

def guard():
    for rel,digest in json.loads((OUT/'protected_sources_sha256.json').read_text()).items():
        assert sha(ROOT/rel)==digest,rel

def calibrate(raw,idx,device):
    if (OUT/'noise_scale.json').exists():
        assert sha(OUT/'noise_scale.json')==json.loads((OUT/'noise_scale_seal.json').read_text())['sha256']
        return
    means=[];stds=[];metadata=[];samples=raw['sample_ids'][idx]
    for seed in range(42,52):
        folder=E14/'final'/f'NoiseFB_no_defense_s{seed}'
        spec=json.loads((folder/'run_spec.json').read_text())
        assert spec['train_ids']==samples.tolist()
        assert spec['normalization']==fit_stats(raw,idx)
        array=images(raw,idx,spec['normalization'])
        m=core.build('NoiseFB',.5).to(device).eval()
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state_dict'])
        mu=[];sigma=[]
        with torch.inference_mode():
            for start in range(0,len(idx),32):
                x=torch.from_numpy(array[start:start+32]).to(device)
                z=m.encoder(torch.cat([x,torch.zeros_like(x)],1));r=(x-m.decode(z)).abs()
                a=r.mean((1,2,3)).cpu().numpy();b=r.std((1,2,3),correction=0).cpu().numpy()
                independent=r.cpu().numpy().astype(np.float64)
                np.testing.assert_allclose(a,independent.mean((1,2,3)),atol=1e-7,rtol=1e-5)
                np.testing.assert_allclose(b,independent.std((1,2,3)),atol=1e-7,rtol=1e-5)
                mu.extend(a.tolist());sigma.extend(b.tolist())
        means.append(mu);stds.append(sigma)
        metadata.append(dict(seed=seed,path=str((folder/'checkpoint.pt').relative_to(ROOT)),sha256=sha(folder/'checkpoint.pt')))
        print('CALIBRATED NoiseFB',seed,flush=True)
        del m;gc.collect();torch.mps.empty_cache()
    means=np.asarray(means);stds=np.asarray(stds);assert means.shape==stds.shape==(10,855)
    np.savez_compressed(OUT/'calibration_moments.npz',mean=means,std=stds,sample_ids=samples,seeds=np.arange(42,52))
    scale=dict(mu0=float(np.median(means)),sigma0=float(np.median(stds)),std_correction=0,aggregation='median of 8550 per-sample pixel moments, equal weight per sample and seed',residual='abs(X - first-pass reconstruction), eval mode, batch32',train_count=855,checkpoint_count=10,checkpoints=metadata,outer_test_used=False,independent_numpy_moments_verified=True,normalization=fit_stats(raw,idx))
    save_json(OUT/'noise_scale.json',scale)
    save_json(OUT/'noise_scale_seal.json',dict(sha256=sha(OUT/'noise_scale.json'),moments_sha256=sha(OUT/'calibration_moments.npz')))

def preflight(raw,pt,pv,device):
    x=torch.from_numpy(images(raw,pt[:32],fit_stats(raw,pt))).to(device);y=torch.tensor(raw['labels'][pt[:32]],device=device)
    checks={}
    for name in NAMES:
        seed_all(42);m=build(name,.5).to(device);m.train();m.random_context=(42,1,0)
        rng=torch.get_rng_state().clone();mrng=torch.mps.get_rng_state().clone();a=m.auxiliary(x);b=m.auxiliary(x)
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(mrng,torch.mps.get_rng_state())
        assert torch.equal(a,b) and (a>=0).all()
        g=torch.Generator().manual_seed(stable_seed(42,1,0));eps=torch.randn(x.shape,generator=g)
        expected=(m.mu0+m.sigma0*eps).clamp(min=0)
        if name=='1P-Mix':expected*=torch.rand((len(x),1,1,1),generator=g)<.5
        assert torch.equal(a.cpu(),expected)
        # Assert a single call of every major module, finite gradients and exact loss.
        calls={key:0 for key in ['encoder','classifier','dec_conv3']};hooks=[]
        for key in calls:
            def hook(mod,args,out,key=key):calls[key]+=1
            hooks.append(getattr(m,key).register_forward_hook(hook))
        out=m(x);value=loss_function(out,x,y,'rec_aux')
        assert torch.equal(value,F.cross_entropy(out[0][0],y)+.5*F.mse_loss(out[1][0],x))
        value.backward();assert torch.isfinite(value) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(v==1 for v in calls.values()),calls
        for h in hooks:h.remove()
        assert int(m.encoder.bn1.num_batches_tracked)==1
        assert {k:tuple(v.shape) for k,v in m.state_dict().items()}=={k:tuple(v.shape) for k,v in build('NoiseFB',.5).state_dict().items()}
        m.eval();m.sample_ids=raw['sample_ids'][pt[:32]];m.eval_input='noise'
        with torch.inference_mode():
            a=m(x);b=m(x);assert all(torch.equal(u,v) for aa,bb in zip(a,b) for u,v in zip(aa,bb))
            noise=m.auxiliary(x);perm=np.arange(32)[::-1].copy();m.sample_ids=m.sample_ids[perm]
            assert torch.equal(noise[torch.tensor(perm,device=device)],m.auxiliary(x))
            m.eval_input='zero';assert torch.count_nonzero(m.auxiliary(x))==0
        checks[name]=dict(single_encoder_classifier_decoder_call=True,exact_CE_plus_half_MSE=True,finite_all_gradients=True,architecture_matches_NoiseFB=True,local_rng_only=True,independent_rng_formula=True,repeat_order_stable_noise=True,zero_eval_input=True)
    # Independently compare all inherited modes to the unmodified E14 source.
    import original_e14_core as previous
    for name in ['M1','M3','RecAux','Yu3','ZeroFB','ShuffleFB','NoiseFB','PermFB']:
        seed_all(42);a=previous.build(name,.5).to(device).eval();b=build(name,.5).to(device).eval();b.load_state_dict(a.state_dict())
        a.sample_ids=b.sample_ids=raw['sample_ids'][pt[:32]]
        with torch.inference_mode():
            u=previous.logits(a,x,name);v=logits(b,x,name);assert torch.equal(u,v),name
        checks[name+'_inherited_unchanged']=True
    save_json(OUT/'preflight_audit.json',dict(all_checks_passed=True,checks=checks,torch=torch.__version__,device=str(device),platform=platform.platform(),outer_test_accessed=False))

def benchmark(raw,idx,device):
    array=images(raw,idx[:219],fit_stats(raw,idx));ids=raw['sample_ids'][idx[:219]];values={}
    for name in NAMES:
        m=build(name,.5).to(device);m.load_state_dict(torch.load(OUT/'pilots'/f'{name}_no_defense'/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
        for mode in ['zero','noise']:
            m.eval_input=mode;predict(m,array,name,device,ids);torch.mps.synchronize();tick=time.monotonic()
            for _ in range(5):predict(m,array,name,device,ids)
            torch.mps.synchronize();values[name+'_'+mode]=(time.monotonic()-tick)/5
    save_json(OUT/'evaluation_timing_train_only.json',dict(seconds_per_219=values,estimated_evaluation_seconds=sum(values.values())*10*211,outer_test_accessed=False,batch=32))

def main():
    assert not (OUT/'final').exists(),'No formal outputs allowed during preparation'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    if not (OUT/'protected_sources_sha256.json').exists():save_json(OUT/'protected_sources_sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in protected_paths()})
    guard()
    raw=dict(np.load(DATA/'raw_windows.npz'));pt,pv,_=inner_split(raw);idx=np.flatnonzero(raw['partitions']=='train');assert len(idx)==855
    device=torch.device('mps');assert torch.backends.mps.is_available()
    save_json(OUT/'local_process.json',dict(pid=os.getpid(),host=platform.node(),phase='calibration_and_pilots_only',formal_training_authorized=False))
    save_json(OUT/'split_manifest.json',dict(pilot_train_ids=raw['sample_ids'][pt].tolist(),pilot_validation_ids=raw['sample_ids'][pv].tolist(),formal_train_ids=raw['sample_ids'][idx].tolist(),outer_test_scores_accessed=False))
    calibrate(raw,idx,device);preflight(raw,pt,pv,device)
    protocol=dict(study='E15',config=CONFIG,batch=32,seeds=list(range(42,52)),pilot=dict(seed=42,min_epochs=30,patience=30,max_epochs=150,delta=.001,endpoint='clean inner-validation Macro-F1 using [X,0]',selection='best epoch>=30; strict improvement; same E14 train engine'),scale_sha256=sha(OUT/'noise_scale.json'),rng='E14 stable_seed; CPU local generator; training keys(seed,epoch,zero-based batch); epsilon drawn before per-sample Mix Bernoulli; eval keys(sampleID)',mix_probability_noise=.5,conditions=json.loads((ROOT/'results/e1_e4_mac_20261106/evaluation_conditions.json').read_text()),reference='E14 saved NoiseFB first_pass and ZeroFB native reported endpoint; no reference retraining',primary='zero-input Macro-F1 SP5/10/20; exact two-sided paired Wilcoxon n10, two contrasts per severity Holm; 1P-Mix separate secondary',interpretation='See supplied_spec.md section5; 3pp descriptive, not equivalence',calibration_boundary='Specified E14 final checkpoints used all855, including later inner validation; pilot selects epochs conditional on this transferred scale, not a nested independence claim',outer_test_accessed=False,formal_training_authorized=False)
    save_json(OUT/'protocol.json',protocol)
    epochs={};timings={}
    for name in NAMES:
        save_json(OUT/'progress.json',dict(stage='pilot',model=name,formal_training_authorized=False))
        r=train(OUT,OUT/'pilots'/f'{name}_no_defense',raw,pt,pv,name,'no_defense',42,CONFIG,device,patience=30)
        epochs[name]=r['selected_epoch'];timings[name]=r['training_seconds']/r['stopped_epoch']
    save_json(OUT/'epoch_settings.json',dict(fixed_epochs=epochs,pilot_seconds_per_epoch=timings,pilot_train_count=len(pt),pilot_validation_count=len(pv),estimated_training_seconds=sum(epochs[n]*timings[n]*855/len(pt)*10 for n in NAMES),outer_test_accessed=False))
    benchmark(raw,idx,device);guard()
    save_json(OUT/'prelaunch_source_sha256.json',{str(p.relative_to(OUT)):sha(p) for p in HERE.glob('*') if p.is_file()})
    save_json(OUT/'progress.json',dict(stage='awaiting_author_confirmation',fixed_epochs=epochs,formal_training_authorized=False,protected_sources_unchanged=True))
    print('PREPARATION COMPLETE; awaiting author confirmation; NO FORMAL TRAINING',flush=True)

if __name__=='__main__':
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            save_json(OUT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise
