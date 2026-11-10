"""Train-only preflight, inner epoch selection, final retraining then held-out CV."""
import fcntl
import platform
import shutil
import tempfile
import traceback
from e10_core import *

def smoke(device,cache):
    report=[];k=len(cache['classes']);split=json.loads((OUTPUT/'split.json').read_text())[0]
    indices=np.array(split['inner_train']);x=torch.from_numpy(cache['images'][indices[:8]].astype(np.float32)/255).to(device)
    y=torch.from_numpy(cache['labels'][indices[:8]]).long().to(device)
    for name in NAMES:
        seed_all(42);model=build(name,k).to(device);opt=torch.optim.Adam(model.parameters(),lr=.0005)
        model.train();loss(model,x,y,name).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step();model.eval()
        with torch.no_grad():before=logits(model,x,name).detach().cpu()
        assert before.shape==(len(x),k)
        with tempfile.TemporaryDirectory(prefix='yu3-e10-smoke-') as d:
            path=Path(d)/'state.pt';atomic_torch(path,dict(model=model.state_dict(),optimizer=opt.state_dict()))
            st=torch.load(path,map_location='cpu',weights_only=False);restored=build(name,k).to(device)
            restored.load_state_dict(st['model']);ropt=torch.optim.Adam(restored.parameters(),lr=.0005);ropt.load_state_dict(st['optimizer'])
            restored.eval()
            with torch.no_grad():assert torch.equal(before,logits(restored,x,name).cpu())
            for net,optimizer in [(model,opt),(restored,ropt)]:
                seed_all(123);net.train();optimizer.zero_grad();loss(net,x,y,name).backward();optimizer.step()
            assert all(torch.equal(a.cpu(),restored.state_dict()[n].cpu()) for n,a in model.state_dict().items())
        del model,opt,restored,ropt;gc.collect()
        if device.type=='mps':torch.mps.empty_cache()
        report.append(dict(model=name,device=str(device),finite_backward=True,checkpoint_forward_exact=True,optimizer_next_step_exact=True))
    return report

def prepare_plan(cache):
    audit=json.loads((OUTPUT/'prepared_audit.json').read_text());assert audit['all_checks_passed']
    assert sha(OUTPUT/'images.npz')==audit['cache_sha256'] and sha(OUTPUT/'split.json')==audit['split_sha256']
    sources=source_hashes()
    plan=dict(experiment='E10 WHITED clean M1-M5 group CV',created=now(),
        authorization='2026-11-08 author asks unattended continuation after E12 for morning results; agent chooses documented bounded protocol',
        models=NAMES,seeds=SEEDS,folds=3,training_runs=45,epoch_pilots=15,conditions=['clean'],
        source_sha256=sources,data_sha256=sha(OUTPUT/'images.npz'),split_sha256=sha(OUTPUT/'split.json'),
        dataset_audit_sha256=sha(OUTPUT/'prepared_audit.json'),
        normalization='per-waveform V and I z-score matching original Experiment1/2 image normalization, not Experiment4 global normalization',
        preprocessing='README r1-r8; device classes>=3 groups; first-channel voltage/second current with README kit factors; one nominal mains cycle around largest absolute current difference with20% preceding samples; polyphase anti-aliased resampling to500points;64x64 occupancy .1 increment saturation; original uint8 PNG-equivalent quantization',
        split='3 stratified device-group folds partition all included groups; per outer fold one remaining device group/class inner-validation, rest inner-train; group names class+device+region, not verified physical serials; not house-disjoint',
        optimizer=dict(name='Adam',lr=.0005,weight_decay=0,batch_size=32,dropout=.5),
        loss='M1-M4 CE; M5 .5CE(pass1)+CE(pass2)+.5MSE(reconstruction,input), original detached absolute residual two-pass',
        pilot=dict(seed=42,maximum_epochs=100,minimum_eligible_epoch=20,patience=15,min_delta=.001,metric='clean inner-validation Macro-F1',selection='earliest eligible epoch improving incumbent by>.001'),
        selection_boundary='all15 epoch pilots frozen before any outer-test inference; each final model retrained from scratch on entire corresponding outer-train for fixed selected epoch',
        reporting='per-fold per-seed metrics, class PRF and integer confusion; out-of-fold predictions pooled within training seed then mean/SD across3seeds; paired differences withM1 and M5-M3; no significance/equivalence claims, no post-test strategy/model selection',
        scope='only clean-data architectural replication on second dataset; no corruption-grid or cross-dataset zero-shot transfer; manuscript and final preprocessing figure remain pending author review of results',
        limitations=['included subset excludes classes<3 filename device groups and undocumentedr9','minimum3 groups yields only1 training device/class in inner pilots for rare classes',
            'single seed inner epoch selection, fixed inherited optimizer, no hyperparameter search','3fold results share training devices; folds are not independent replications',
            'group identities and channel order based on archive naming/physical checks rather than serial-number registry','event locator not verified turn-on; selected window not certified steady state',
            'Mac MPS descriptive results, not student CUDA reproduction','M5-M3 changes more than feedback signal, not pure causal effect','raw archive version not cryptographically identified by publisher; current download contains1339 root FLAC whereas official page reports1259'],
        environment=dict(platform=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),device='mps',threads=4))
    if (OUTPUT/'protocol.json').exists():
        old=json.loads((OUTPUT/'protocol.json').read_text())
        for key in ['source_sha256','data_sha256','split_sha256','dataset_audit_sha256']:assert old[key]==plan[key],key
        return old
    save(OUTPUT/'protocol.json',plan)
    for rel in sources:
        dest=OUTPUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    return plan

def run():
    assert e12_ready(),'E12 completion/exit gate not satisfied'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available(),'MPS required; no silent CPU fallback'
    device=torch.device('mps');cache=dict(np.load(OUTPUT/'images.npz'));plan=prepare_plan(cache)
    splits=json.loads((OUTPUT/'split.json').read_text())
    if not (OUTPUT/'preflight_audit.json').exists():
        checks=smoke(device,cache);timings=[]
        for name in NAMES:
            with tempfile.TemporaryDirectory(prefix='yu3-e10-epoch-') as d:
                r=train(Path(d),cache,np.array(splits[0]['inner_train']),np.array([],int),name,42,device,fixed_epochs=1)
                timings.append(dict(model=name,train_count=r['train_count'],epoch_seconds=r['training_seconds']))
        # Conservative upper schedule based on100 pilot epochs and100 final epochs.
        unit={r['model']:r['epoch_seconds']/r['train_count'] for r in timings}
        upper=sum(sum(unit[n]*(len(s['inner_train'])+3*len(s['outer_train']))*100 for n in NAMES) for s in splits)
        save(OUTPUT/'preflight_audit.json',dict(all_checks_passed=True,at=now(),checks=checks,one_epoch_timings=timings,no_outer_test_predictions=True,
            source_sha256=plan['source_sha256'],maximum_training_seconds_estimate=upper,estimate_note='100epochs each pilot and final; selected epochs/early stop usually shorter; excludes analysis and I/O'))
    choices={}
    for s in splits:
        for name in NAMES:
            folder=OUTPUT/'pilots'/f"fold{s['fold']}_{name}_s42"
            r=json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else train(folder,cache,np.array(s['inner_train']),np.array(s['inner_validation']),name,42,device)
            choices[f"fold{s['fold']}_{name}"]=r['selected_epoch']
    if (OUTPUT/'selection_complete.json').exists():assert json.loads((OUTPUT/'selection_complete.json').read_text())['epochs']==choices
    else:save(OUTPUT/'selection_complete.json',dict(at=now(),epochs=choices,outer_predictions_before_selection=0,source_sha256=plan['source_sha256']))
    for s in splits:
        for name in NAMES:
            for seed in SEEDS:
                check_stop();folder=OUTPUT/'runs'/f"fold{s['fold']}_{name}_s{seed}"
                if not (folder/'complete.json').exists():train(folder,cache,np.array(s['outer_train']),np.array([],int),name,seed,device,fixed_epochs=choices[f"fold{s['fold']}_{name}"])
    save(OUTPUT/'training_complete.json',dict(at=now(),runs=45,pilots=15,all_models_trained_before_outer_inference=True))
    # Test is opened once all configurations and final training are complete.
    for s in splits:
        te=np.array(s['outer_test'])
        for name in NAMES:
            for seed in SEEDS:
                check_stop();folder=OUTPUT/'runs'/f"fold{s['fold']}_{name}_s{seed}";dest=folder/'predictions.npz'
                if dest.exists():continue
                st=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=build(name,len(cache['classes'])).to(device);model.load_state_dict(st['model'])
                before={n:a.detach().cpu().clone() for n,a in model.state_dict().items()}
                p=predict(model,cache['images'][te].astype(np.float32)/255,name,device)
                assert all(torch.equal(a,model.state_dict()[n].cpu()) for n,a in before.items())
                tmp=dest.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,indices=te,sample_ids=cache['sample_ids'][te],labels=cache['labels'][te],predictions=p)
                os.replace(tmp,dest);del model,st,before;gc.collect();torch.mps.empty_cache()
    from analyze_e10_whited import analyze
    analyze()

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cpu-smoke',action='store_true');a=p.parse_args()
    if a.cpu_smoke:
        torch.set_num_threads(1);checks=smoke(torch.device('cpu'),dict(np.load(OUTPUT/'images.npz')))
        save(OUTPUT/'cpu_preflight_audit.json',dict(all_checks_passed=True,checks=checks,no_test_predictions=True,at=now()));return
    with (OUTPUT/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except InterruptedError:print('E10_PAUSED',flush=True)
        except Exception:save(OUTPUT/'FAILED.json',dict(at=now(),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
