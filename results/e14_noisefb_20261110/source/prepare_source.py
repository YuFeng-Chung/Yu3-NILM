from pathlib import Path
import hashlib,json,shutil,difflib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
src=ROOT/'experiments/revision_followup_core.py'
s=src.read_text(); old=s
s=s.replace("['ZeroFB', 'ShuffleFB']", "['ZeroFB', 'ShuffleFB', 'NoiseFB', 'PermFB']")
s=s.replace("'ZeroFB', 'ShuffleFB']", "'ZeroFB', 'ShuffleFB', 'NoiseFB', 'PermFB']")
needle="        aux = torch.zeros_like(residual) if self.mode == 'ZeroFB' else residual.roll(1, 0)"
s=s.replace(needle,'''        if self.mode in ['NoiseFB', 'PermFB']:
            if self.training:
                generator = torch.Generator(device='cpu').manual_seed(stable_seed(*self.random_context))
                if self.mode == 'NoiseFB':
                    random = torch.randn(residual.shape, generator=generator).to(residual.device)
                else:
                    random = torch.stack([torch.randperm(residual[0].numel(), generator=generator) for _ in residual]).to(residual.device)
            else:
                assert len(self.sample_ids) == len(x)
                generators = [torch.Generator(device='cpu').manual_seed(stable_seed(str(i))) for i in self.sample_ids]
                if self.mode == 'NoiseFB':
                    random = torch.stack([torch.randn(residual.shape[1:], generator=g) for g in generators]).to(residual.device)
                else:
                    random = torch.stack([torch.randperm(residual[0].numel(), generator=g) for g in generators]).to(residual.device)
            if self.mode == 'NoiseFB':
                mu = residual.mean((1,2,3), keepdim=True)
                sigma = residual.std((1,2,3), correction=0, keepdim=True)
                aux = (mu + sigma * random).clamp(min=0)
            else:
                aux = residual.flatten(1).gather(1, random).reshape_as(residual)
        else:
            aux = torch.zeros_like(residual) if self.mode == 'ZeroFB' else residual.roll(1, 0)''')
a=s.index('def predict(');b=s.index('\n\ndef fit_stats',a)
s=s[:a]+'''def stable_seed(*keys):
    from hashlib import sha256
    return int.from_bytes(sha256('|'.join(map(str, keys)).encode()).digest()[:8], 'little') % (2**63-1)


def predict(model, array, name, device, sample_ids=None):
    model.eval()
    if name in ['NoiseFB','PermFB']:
        assert sample_ids is not None and len(sample_ids) == len(array)
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(array),32):
            if sample_ids is not None: model.sample_ids=sample_ids[start:start+32]
            predictions.append(logits(model,torch.from_numpy(array[start:start+32]).to(device),name).argmax(1).cpu().numpy())
    return np.concatenate(predictions)
'''+s[b:]
s=s.replace('for idx in order.split(32):','for batch_index, idx in enumerate(order.split(32)):\n            model.random_context = (seed, epoch, batch_index)')
s=s.replace('predict(model,v,name,device)',"predict(model,v,name,device,raw['sample_ids'][val_idx])")
s=s.replace('predict(model,probe,name,device)',"predict(model,probe,name,device,raw['sample_ids'][train_idx[:32]])")
s=s.replace('predict(restored,probe,name,device)',"predict(restored,probe,name,device,raw['sample_ids'][train_idx[:32]])")
s=s.replace("['Yu3','ZeroFB','ShuffleFB']", "['Yu3','ZeroFB','ShuffleFB','NoiseFB','PermFB']")
(HERE/'e14_core.py').write_text(s)
(HERE/'core_changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original/revision_followup_core.py',tofile='e14/e14_core.py')))
shutil.copy2(src,HERE/'original_revision_followup_core.py')
shutil.copy2(ROOT/'experiments/run_revision_followup_v4.py',HERE/'original_run_revision_followup_v4.py')
