import sys,json,time,fcntl,platform,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import e14_core as core
from e14_core import *
from run_e2_search_mps import inner_split
OLD=ROOT/'results/e1_e4_mac_20261106'
NAMES=['NoiseFB','PermFB']
CONFIG={'lr':.0005,'weight_decay':0.,'dropout':.5}

def preflight(raw,pt,pv,device):
    import revision_followup_core as legacy
    plan=json.loads((OLD/'protocol_v4.json').read_text())
    for rel,digest in plan['source_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    checks={}
    for name in ['ZeroFB','ShuffleFB']:
        folder=OLD/'final'/f'{name}_no_defense_s42'
        spec=json.loads((folder/'run_spec.json').read_text())
        assert spec['config']==CONFIG and spec['fixed_epochs'] is not None
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        a=legacy.build(name,.5).to(device).eval();b=build(name,.5).to(device).eval()
        a.load_state_dict(ck['state_dict']);b.load_state_dict(ck['state_dict'])
        x=torch.from_numpy(images(raw,pt[:32],fit_stats(raw,pt))).to(device)
        with torch.no_grad():
            oa=a(x);ob=b(x)
            assert all(torch.equal(u,v) for aa,bb in zip(oa,ob) for u,v in zip(aa,bb))
        checks[name+'_unchanged']=True
    for name in NAMES:
        m=build(name,.5).to(device);m.random_context=(42,1,0);m.train()
        val=loss(m,x,torch.tensor(raw['labels'][pt[:32]],device=device),name)
        val.backward();assert torch.isfinite(val)
        m.eval();m.sample_ids=raw['sample_ids'][pt[:32]]
        with torch.no_grad():
            a=m(x)[0][-1];b=m(x)[0][-1];assert torch.equal(a,b)
            perm=np.arange(32)[::-1].copy();m.sample_ids=raw['sample_ids'][pt[:32]][perm]
            c=m(x[torch.tensor(perm,device=device)])[0][-1]
            assert torch.allclose(a[torch.tensor(perm,device=device)],c,atol=1e-5,rtol=1e-5)
        checks[name+'_finite_gradient_repeat_order_invariance']=True
    save_json(OUT/'preflight_audit.json',dict(all_checks_passed=True,checks=checks,device=str(device),torch=torch.__version__,platform=platform.platform()))
    del a,b,c,m,x;gc.collect();torch.mps.empty_cache()

def pilot():
    raw=dict(np.load(DATA/'raw_windows.npz'));pt,pv,test=inner_split(raw)
    device=torch.device('mps');assert torch.backends.mps.is_available()
    preflight(raw,pt,pv,device)
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'experiments').glob('*.py')}
    protocol=dict(study='E14 NoiseFB primary, PermFB secondary',runner_basis='experiments/run_revision_followup_v4.py',engine_basis='experiments/revision_followup_core.py',config=CONFIG,seeds=list(range(42,52)),pilot=dict(seed=42,min_epochs=30,patience=30,max_epochs=150,delta=.001,selection='best clean inner-validation macro-F1 at epoch>=30'),deviation='Old v4 no_defense pilot patience=20; E14 specification requires30',rng='SHA256 UTF8 pipe-separated keys, first8 bytes little-endian modulo(2**63-1); CPU torch.Generator; train keys=(training_seed,epoch,zero_based_batch_index), eval key=(sample_ID); Noise std correction=0',conditions=json.loads((OLD/'evaluation_conditions.json').read_text()),primary='pass1 MacroF1 SP5/10/20, average 10 pollution realizations within training seed; exact two-sided signed-rank sign permutation, two contrasts per severity Holm; PermFB separate secondary family',source_sha256=source,spec_sha256=sha(ROOT.parent.parent/'nonexistent') if False else sha(ROOT.parents[1]/'Revised/Claude修訂_20261109/E14_NoiseFB_實驗規格_給Codex.md'))
    save_json(OUT/'protocol.json',protocol)
    epoch={};timings={}
    for name in NAMES:
        r=train(OUT,OUT/'pilots'/f'{name}_no_defense',raw,pt,pv,name,'no_defense',42,CONFIG,device,patience=30)
        epoch[name]=r['selected_epoch'];timings[name]=r['training_seconds']/r['stopped_epoch']
    save_json(OUT/'epoch_settings.json',dict(fixed_epochs=epoch,pilot_seconds_per_epoch=timings,estimated_training_seconds=sum(epoch[n]*timings[n]*855/len(pt)*10 for n in NAMES),outer_test_accessed=False))
    save_json(OUT/'progress.json',dict(stage='pilots_complete_pending_prelaunch_report',fixed_epochs=epoch))

def formal():
    from evaluate_e14 import evaluate
    from analyze_e14 import analyze
    epochs=json.loads((OUT/'epoch_settings.json').read_text())['fixed_epochs']
    assert json.loads((OUT/'preflight_audit.json').read_text())['all_checks_passed']
    assert (OUT/'prelaunch_report.md').exists()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for rel,digest in protocol['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    raw=dict(np.load(DATA/'raw_windows.npz'));outer=np.flatnonzero(raw['partitions']=='train');empty=np.array([],dtype=int)
    device=torch.device('mps');assert torch.backends.mps.is_available()
    save_json(OUT/'selection_complete.json',dict(fixed_epochs=epochs,outer_test_evaluations_so_far=0,config=CONFIG))
    for name in NAMES:
        for seed in range(42,52):
            save_json(OUT/'progress.json',dict(stage='formal_training',model=name,seed=seed))
            train(OUT,OUT/'final'/f'{name}_no_defense_s{seed}',raw,outer,empty,name,'no_defense',seed,CONFIG,device,fixed_epochs=epochs[name])
    for name in NAMES+['Yu3','ZeroFB','ShuffleFB']:
        for seed in range(42,52):
            save_json(OUT/'progress.json',dict(stage='evaluation',model=name,seed=seed))
            evaluate(raw,name,seed,device)
    save_json(OUT/'progress.json',dict(stage='analysis'))
    analyze()
    save_json(OUT/'progress.json',dict(stage='research_complete',report_status='pending_document_QA'))

if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if sys.argv[1]=='pilot':pilot()
            elif sys.argv[1]=='formal':formal()
            else:raise ValueError(sys.argv[1])
        except Exception as exc:
            save_json(OUT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise
