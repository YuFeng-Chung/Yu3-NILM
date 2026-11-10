"""E9 real training-only image/gradient/reload integration and immutable protocol."""
import shutil
import tempfile
import platform
from e9_core import *
from prepare_e5_waveforms import original_preprocessing

def main():
    assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    raw=dict(np.load(OUTPUT/'waveforms.npz'));split=json.loads((OUTPUT/'split.json').read_text())['indices']
    tr=np.array(split['inner_train']);va=np.array(split['inner_validation']);te=np.array(split['outer_test'])
    for a,b in [(tr,va),(tr,te),(va,te)]:assert not set(raw['groups'][a])&set(raw['groups'][b])
    selected=json.loads((ROOT/'results/e2_mac_20261106/selected_settings.json').read_text())['Yu3']
    config={k:selected[k] for k in ['lr','weight_decay','dropout']};functions,_=original_preprocessing();checks=[]
    # N0 parity to original float32 imaging, before historical PNG quantization.
    for j in tr[:64]:
        expected=functions['generate_vi_image'](raw['W0_current'][j].copy(),raw['W0_voltage'][j].copy())
        assert np.array_equal(expected,images(raw,[j],fit_stats(raw,tr,'W0'),'A')[0])
    for cell,(window,norm) in CELLS.items():
        stats=fit_stats(raw,tr,window);a=images(raw,tr[:16],stats,cell)
        assert a.shape==(16,1,64,64) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
        for name in MODELS:
            with tempfile.TemporaryDirectory(prefix='yu3-e9-smoke-') as tmp:
                folder=Path(tmp)
                result=train(folder,folder/'one_epoch',raw,tr,np.array([],int),cell,name,42,config,torch.device('mps'),fixed_epochs=1)
                checks.append(dict(cell=cell,model=name,full_training_epoch_seconds=result['training_seconds'],checkpoint_reload=True))
            seed_all(42);net=build(name,config['dropout']).to('mps');opt=torch.optim.Adam(net.parameters(),lr=config['lr'])
            x=torch.from_numpy(a[:8]).to('mps');y=torch.from_numpy(raw['labels'][tr[:8]]).long().to('mps')
            seed_all(123);l=loss(net,x,y,name);l.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None);opt.step()
            with tempfile.TemporaryDirectory(prefix='yu3-e9-reload-') as tmp:
                path=Path(tmp)/'state.pt';torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),path)
                cp=torch.load(path,map_location='cpu',weights_only=False);restored=build(name,config['dropout']).to('mps');restored.load_state_dict(cp['model'])
                ropt=torch.optim.Adam(restored.parameters(),lr=config['lr']);ropt.load_state_dict(cp['optimizer'])
                for model,optimizer in [(net,opt),(restored,ropt)]:
                    seed_all(456);model.train();optimizer.zero_grad();loss(model,x,y,name).backward();optimizer.step()
                assert all(torch.equal(v.cpu(),restored.state_dict()[k].cpu()) for k,v in net.state_dict().items()),'optimizer next-step mismatch'
            del net,restored,opt,ropt;gc.collect();torch.mps.empty_cache()
    plan=dict(experiment='E9 isolated preprocessing sensitivity',authorization='2026-11-08 author: 好的，就這樣進行吧',
        cells=CELLS,models=MODELS,seeds=SEEDS,conditions=CONDITIONS,config={n:config for n in MODELS},
        hyperparameter_source='Yu3 frozen E2 setting transferred unchanged to both M1 and Yu3; no new HPO; not per-preprocessing optimum',
        strategy='No-Defense',pilots=dict(seed=42,max_epochs=150,min_epochs=30,patience=20,min_delta=.001,
        checkpoint='highest clean inner-validation Macro-F1 eligible epoch>=30; earliest tie; patience begins after30'),
        formal_training='8 inner epoch pilots then 40 independent outer-train fixed-epoch runs; selection_complete before outer predictions',
        window_rule=WINDOW_RULE,voltage='per-waveform z-score in all cells',current='N0 per-waveform; N1 window-specific training-only fixed z-score',
        imaging='same64x64 .1 occupancy accumulation and saturation; float32 input; no uint8 PNG round-trip; N0 baseline matches original pre-PNG float image',
        normalization='one inner-training fit per window for pilots; one outer-training fit per window for final; never fit validation/test',
        split='same original inner/outer membership restricted to common W1 eligible samples; coverage and exclusions reported',
        inference='AWGN based on per-channel mean-removed variance, independent V/I noise, after window extraction; no noisy-event detection claim',
        reporting='all11-class Macro-F1, accuracy, per-class PRF/confusion; degradation clean minus corrupted in pp; average10 pollution seeds within training seed; descriptive n5, no confirmatory p-values or equivalence claims',
        diagnostics='AWGN30/20 image IoU/add/remove/L1; waveform SP1/20 pre-mapping abs(z)>3 and actual index clipping per V/I, N0/N1; current RMS; no pixel SP reanalysis or E13 deletion',
        limitations='historical test already viewed; overlapping knowledge with previous studies; group is class x house proxy; one inner split, seed42 epoch pilot; fixed transferred config; common eligibility subset; candidate stability criteria do not certify physical steady state; Mac MPS not CUDA reproduction; M1/Yu3 difference not pure feedback causality',
        source_sha256=source_hashes(),data_sha256=sha(OUTPUT/'waveforms.npz'),split_sha256=sha(OUTPUT/'split.json'),
        e2_setting_sha256=sha(ROOT/'results/e2_mac_20261106/selected_settings.json'),
        environment=dict(platform=platform.platform(),torch=str(torch.__version__),device='mps',threads=4))
    assert not (OUTPUT/'protocol.json').exists()
    save_json(OUTPUT/'protocol.json',plan)
    for rel in plan['source_sha256']:
        dest=OUTPUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(OUTPUT/'preflight_audit.json',dict(all_checks_passed=True,original_float_image_parity=64,full_epoch_checks=checks,
        finite_gradients=True,checkpoint_and_optimizer_next_step_equal=True,no_outer_predictions=True,source_sha256=plan['source_sha256']))
    print('E9 preflight PASS; one-epoch timings',checks,flush=True)

if __name__=='__main__':main()
