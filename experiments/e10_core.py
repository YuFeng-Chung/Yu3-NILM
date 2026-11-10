"""WHITED M1--M5 clean-data comparison, isolated from previous experiments."""
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/e10_whited_20261108'
DATA = ROOT / 'data/whited_v11'
E12 = ROOT / 'results/e12_measurement_20261108'
NAMES = ['M1', 'M2', 'M3', 'M4', 'M5']
SEEDS = [42, 43, 44]
SOURCE = ROOT / '實驗一/models_m1_to_m5.py'
spec = importlib.util.spec_from_file_location('e10_original_models', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2)); os.replace(t, p)

def csv_write(p, rows):
    rows = list(rows); p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def atomic_torch(p, state):
    t = p.with_suffix('.tmp'); torch.save(state, t); os.replace(t, p)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(s)

def build(name, classes):
    original.NUM_CLASSES = classes
    return getattr(original, name + '_Model')()

def logits(model, x, name):
    out = model(x)
    return out[1] if name == 'M5' else out

def loss(model, x, y, name):
    out = model(x)
    if name == 'M5':
        a, b, r = out
        return .5 * nn.functional.cross_entropy(a, y) + nn.functional.cross_entropy(b, y) + .5 * nn.functional.mse_loss(r, x)
    return nn.functional.cross_entropy(out, y)

def metrics(y, p, k):
    cm = np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k*k).reshape(k,k)
    tp = cm.diagonal().astype(float)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0)>0)
    re = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1)>0)
    f1 = np.divide(2*pr*re, pr+re, out=np.zeros(k), where=pr+re>0)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_f1=float(f1.mean()),
                macro_precision=float(pr.mean()), macro_recall=float(re.mean())), cm, pr, re, f1

def predict(model, x, name, device):
    model.eval(); out=[]
    with torch.inference_mode():
        for a in torch.as_tensor(x).split(32): out.append(logits(model, a.to(device), name).argmax(1).cpu().numpy())
    return np.concatenate(out)

def check_stop(folder=None, epoch=None):
    if (OUTPUT/'STOP_REQUESTED').exists():
        save(OUTPUT/'PAUSED.json', dict(at=now(), folder=str(folder) if folder else None, epoch=epoch))
        raise InterruptedError('E10 pause requested; saved epoch checkpoint retained')

def source_hashes():
    paths=[ROOT/'experiments'/n for n in ['e10_core.py','prepare_e10_whited.py','run_e10_whited.py','analyze_e10_whited.py','queue_e10_whited.py']]
    paths += [ROOT/'experiments/prepare_e5_waveforms.py', SOURCE, ROOT/'實驗一/plaid_loader_vi.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def e12_ready():
    for n in ['STOP_REQUESTED','PAUSED.json','FAILED.json']:
        if (E12/n).exists(): raise RuntimeError('E12 paused/failed; E10 cannot start: '+n)
    if not (E12/'completed_audit.json').exists(): return False
    a=json.loads((E12/'completed_audit.json').read_text())
    if not a.get('all_checks_passed'): raise RuntimeError('E12 audit failed')
    if not (E12/'independent_audit.json').exists(): return False
    if sha(E12/'independent_audit.json') != a['independent_audit_sha256']: raise RuntimeError('E12 audit hash mismatch')
    if json.loads((E12/'progress.json').read_text()).get('stage')!='research_complete': return False
    if not (ROOT/'E12_results_20261108.zip').exists(): return False
    # Check command and host, never interpret a synced PID as a local writer.
    import platform, subprocess, fcntl
    info=json.loads((E12/'local_process.json').read_text())
    if info.get('host')!=platform.node(): raise RuntimeError('E12 host mismatch: inspect actual execution host')
    cp=subprocess.run(['/bin/ps','-p',str(info['pid']),'-o','command='],capture_output=True,text=True)
    if cp.returncode==0 and 'run_e12.py' in cp.stdout and str(E12) in cp.stdout: return False
    with (E12/'writer.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
    return True

def train(folder, cache, train_idx, val_idx, name, seed, device, fixed_epochs=None):
    """Deterministic epoch RNG permits exact restart without stored device RNG blobs."""
    folder.mkdir(parents=True, exist_ok=True)
    k=len(cache['classes']); seed_all(seed); model=build(name,k).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.0005)
    x=torch.from_numpy(cache['images'][train_idx].astype(np.float32)/255).to(device)
    y=torch.from_numpy(cache['labels'][train_idx]).long().to(device)
    begin=0; best=-1.; best_epoch=0; stale=0; log=[]; seconds=0.
    if (folder/'resume.pt').exists():
        st=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        begin=st['epoch'];best=st['best'];best_epoch=st['best_epoch'];stale=st['stale'];log=st['log'];seconds=st['seconds']
    cap=fixed_epochs if fixed_epochs is not None else 100
    for epoch in range(begin+1, cap+1):
        check_stop(folder,epoch-1);tick=time.monotonic();seed_all(seed*10000+epoch)
        model.train();perm=np.random.default_rng(seed*10000+epoch).permutation(len(x));total=0.
        for start in range(0,len(x),32):
            idx=torch.as_tensor(perm[start:start+32],device=device)
            opt.zero_grad(set_to_none=True);l=loss(model,x[idx],y[idx],name)
            if not bool(torch.isfinite(l)): raise RuntimeError('Nonfinite training loss')
            l.backward();opt.step();total+=float(l.detach().cpu())*len(idx)
        row=dict(epoch=epoch,loss=total/len(x))
        if fixed_epochs is None:
            p=predict(model,cache['images'][val_idx].astype(np.float32)/255,name,device)
            f=metrics(cache['labels'][val_idx],p,k)[0]['macro_f1'];row['validation_f1']=f
            # Minimum20 applies to eligible checkpoint selection for every model.
            if epoch>=20:
                if f>best+.001: best=f;best_epoch=epoch;stale=0
                else: stale+=1
        if device.type=='mps':torch.mps.synchronize()
        seconds+=time.monotonic()-tick;row['cumulative_seconds']=seconds;log.append(row)
        atomic_torch(folder/'resume.pt',dict(model={a:b.cpu() for a,b in model.state_dict().items()},optimizer=opt.state_dict(),
            epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,log=log,seconds=seconds))
        csv_write(folder/'training_log.csv',log)
        save(OUTPUT/'progress.json',dict(stage='pilot' if fixed_epochs is None else 'training',run=folder.name,epoch=epoch,cap=cap,updated=now()))
        if fixed_epochs is None and epoch>=20 and stale>=15:break
    if fixed_epochs is not None:
        atomic_torch(folder/'model.pt',dict(model={a:b.cpu() for a,b in model.state_dict().items()},name=name,seed=seed,classes=cache['classes'].tolist(),epochs=cap))
    result=dict(model=name,seed=seed,epochs=log[-1]['epoch'],selected_epoch=best_epoch if fixed_epochs is None else cap,
        validation_f1=best if fixed_epochs is None else None,training_seconds=seconds,train_count=len(train_idx),validation_count=len(val_idx),ended_at=now())
    save(folder/'complete.json',result)
    del model,opt,x,y;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return result
