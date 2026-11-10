"""Real-checkpoint E6 preflight; diagnostics do not select methods or seeds."""
import argparse
import fcntl
import json
import numpy as np
import torch
from e6_mechanism_core import *

def verify(output):
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);device=torch.device('mps');raw,test=load_data()
    model,spec,epoch=model_for(42,device);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    max_delta=0.;observed=[]
    for condition in [CONDITIONS[0],CONDITIONS[1],CONDITIONS[11],CONDITIONS[21]]:
        typ,_,ps=condition
        array=images(raw,test,spec['normalization'],condition=None if typ=='clean' else condition)
        l1,recon=first_pass(model,array,device);r=np.abs(array-recon)
        normal=second_pass(model,array,r,device)
        with torch.inference_mode():
            original=model(torch.from_numpy(array[:8]).to(device))
        for a,b in [(l1[:8],original[0][0].cpu().numpy()),(normal[:8],original[0][1].cpu().numpy()),(recon[:8],original[1][0].cpu().numpy())]:
            delta=float(np.max(np.abs(a-b)));max_delta=max(max_delta,delta)
            assert np.allclose(a,b,atol=2e-5,rtol=2e-5),delta
        order=derangement(raw['sample_ids'][test],condition)
        assert sorted(order.tolist())==list(range(219)) and np.all(order!=np.arange(219))
        shuffled=second_pass(model,array,r[order],device)
        other=second_pass(model,array,r[order],device,batch=17)
        assert np.allclose(shuffled,other,atol=2e-5,rtol=2e-5)
        assert np.array_equal(shuffled.argmax(1),other.argmax(1))
        # Observe the actual second-pass encoder input, not just predictions.
        inputs=[]
        hook=model.encoder.conv1.register_forward_pre_hook(lambda m,args:inputs.append(args[0].detach().cpu().numpy()))
        second_pass(model,array[:8],np.zeros_like(r[:8]),device)
        assert np.array_equal(inputs[0][:,1:],np.zeros_like(r[:8]))
        inputs.clear();second_pass(model,array[:8],r[order][:8],device)
        assert np.array_equal(inputs[0][:,1:],r[order][:8]);hook.remove()
        observed.append(dict(condition=condition,batch_invariance_max_delta=float(np.max(np.abs(shuffled-other)))))
        if typ=='missing_consecutive':
            regions=region_masks(raw,test,spec['normalization'],ps)
            e=energy_rows(r,regions);zero=energy_rows(np.zeros_like(r),regions)
            assert np.isnan(zero['missing_energy_ratio']).all()
            assert np.all((e['missing_energy_ratio']>=0)&(e['missing_energy_ratio']<=1))
            # Pixel accumulation agrees exactly with the historical generator.
            for k in test[:10]:
                a,b=FNS['apply_waveform_corruption'](raw['current'][k].astype(np.float32),raw['voltage'][k].astype(np.float32),
                     typ,.2,spec['normalization'],rng_for('test',str(raw['sample_ids'][k]),typ,.2,ps))
                assert np.array_equal(scatter_fast(a,b,spec['normalization']),FNS['generate_vi_image_fixed_norm'](a,b,spec['normalization']))
        # Full condition normal predictions must match the previous audited run.
        prior=PREVIOUS/'final/Yu3_combined_s42/outer_predictions'
        if typ=='clean':path=prior/'clean.npy'
        else:
            conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text());path=prior/f'{conditions.index(list(condition)):03}.npy'
        assert np.array_equal(normal.argmax(1),np.load(path))
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
    # Verify atomic output replay, interrupted temporary file isolation, and duplicate writer exclusion.
    probe=output/'preflight_atomic_probe.npz';atomic_npz(probe,labels=raw['labels'][test]);digest=sha(probe)
    probe.with_suffix('.tmp').write_bytes(b'interrupted write')
    assert sha(probe)==digest and np.array_equal(np.load(probe)['labels'],raw['labels'][test])
    probe.unlink();probe.with_suffix('.tmp').unlink()
    with (output/'preflight_writer.lock').open('a+') as one,(output/'preflight_writer.lock').open('a+') as two:
        fcntl.flock(one,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:fcntl.flock(two,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise AssertionError('Duplicate lock was accepted')
    report=dict(all_checks_passed=True,device='mps',checkpoint_epoch=epoch,max_forward_delta=max_delta,
                conditions=observed,original_forward_equivalence=True,residual_formula=True,
                zero_and_cross_sample_inputs_verified=True,weights_bn_unchanged=True,
                derangement_no_self_and_batch_invariance=True,mask_interpolation_pixel_match=True,
                zero_energy_denominator_nan=True,atomic_saved_item_replay=True,duplicate_writer_rejected=True,
                source_sha256=source_hashes())
    save_json(output/'preflight_audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);verify(p.parse_args().output)
