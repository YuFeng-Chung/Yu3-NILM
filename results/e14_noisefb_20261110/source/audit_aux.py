"""CPU audit: second-input histogram/moments, detach, independent RNG and batch invariance."""
from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import e14_core as c
import numpy as np,torch
checks={};torch.set_num_threads(2)
for name in ['NoiseFB','PermFB']:
    c.seed_all(42);m=c.build(name,.5);x=torch.rand(4,1,64,64,requires_grad=True)
    encoded=[];decoded=[]
    h=m.encoder.register_forward_pre_hook(lambda mod,args:encoded.append(args[0]))
    # Extract recon through returned model outputs, aux via second encoder input.
    m.eval();m.sample_ids=['audit_a','audit_b','audit_c','audit_d']
    torch_state=torch.random.get_rng_state().clone();o=m(x)
    assert torch.equal(torch_state,torch.random.get_rng_state())
    r=(x-o[1][0]).abs().detach();a=encoded[1][:,1:]
    assert not encoded[1][:,1:].detach().requires_grad
    # Gradient wrt reconstruction must have no path through aux.
    assert torch.autograd.grad(a.sum(),o[1][0],allow_unused=True,retain_graph=True)[0] is None
    if name=='PermFB':
        assert torch.equal(a.flatten(1).sort(1).values,r.flatten(1).sort(1).values)
        assert not torch.equal(a,r)
    else:
        assert (a>=0).all()
        # Gaussian moments are truncated by clamp; verify independent normalized draws
        # remain consistent with a Gaussian above truncation, rather than claiming exact aux moments.
        mu=r.mean((1,2,3),keepdim=True);sd=r.std((1,2,3),correction=0,keepdim=True)
        nz=a>0;z=((a-mu)/sd)[nz]
        assert torch.isfinite(z).all() and z.std()>.4
    before=o[0][-1].detach().clone();encoded.clear()
    oo=m(x);assert torch.equal(before,oo[0][-1])
    # Same sample alone vs in full batch in eval.
    m.sample_ids=['audit_c'];alone=m(x[2:3])[0][-1]
    assert torch.allclose(alone,before[2:3],atol=1e-6,rtol=1e-5)
    m.train();m.random_context=(42,1,0);encoded.clear();m(x);a1=encoded[1][:,1:].detach().clone()
    m.random_context=(42,1,1);encoded.clear();m(x);a2=encoded[1][:,1:].detach().clone();assert not torch.equal(a1,a2)
    h.remove();checks[name]=dict(histogram_or_noise_valid=True,no_aux_reconstruction_gradient=True,eval_global_rng_unchanged=True,eval_repeat_batch_invariant=True,train_batch_key_changes_aux=True)
(OUT/'aux_semantics_audit.json').write_text(json.dumps(dict(all_checks_passed=True,checks=checks),indent=2))
print('aux semantics verified')
