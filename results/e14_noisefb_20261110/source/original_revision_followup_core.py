"""Shared MPS E1/E3/E4 engine. Historical sources and E2 outputs stay immutable."""
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from revision_models_mps import build_model as candidate_model
from run_e5_ablation import Ablation, original, loss_function, seed_all, metrics
from run_exp4_matched_controls import ROOT, DATA, FNS, images, scatter_fast, rng_for, save_json, sha

FAMILY = ['M1', 'RecAux', 'M3', 'Yu3', 'ZeroFB', 'ShuffleFB']
CANDIDATES = ['Yu3', 'ResNet18', 'MobileNetV3', 'ViT-Tiny']
STRATEGIES = ['no_defense', 'S3', 'combined']


class RecAux(Ablation):
    def __init__(self):
        super().__init__('rec_aux')
        self.encoder = original.BaseEncoder(in_channels=1)

    def forward(self, x):
        z = self.encoder(x)
        return [self.classify(z)], [self.decode(z)]


class FeedbackControl(Ablation):
    def __init__(self, mode):
        super().__init__('err_detach')
        self.mode = mode

    def forward(self, x):
        z = self.encoder(torch.cat([x, torch.zeros_like(x)], 1))
        p1, recon = self.classify(z), self.decode(z)
        residual = (x-recon).abs().detach()
        # Fixed cyclic cross-sample replacement, no labels; batch order is saved.
        aux = torch.zeros_like(residual) if self.mode == 'ZeroFB' else residual.roll(1, 0)
        p2 = self.classify(self.encoder(torch.cat([x, aux], 1)))
        return [p1, p2], [recon]


def build(name, dropout):
    if name == 'M1': model = original.M1_Model()
    elif name == 'M3': model = original.M3_Model()
    elif name == 'RecAux': model = RecAux()
    elif name in ['ZeroFB', 'ShuffleFB']: model = FeedbackControl(name)
    else: return candidate_model(name, dropout)
    model.dropout.p = dropout
    return model


def logits(model, x, name):
    out = model(x)
    return out[0][-1] if name in ['Yu3', 'RecAux', 'ZeroFB', 'ShuffleFB'] else out


def loss(model, x, y, name):
    out = model(x)
    if name in ['Yu3', 'ZeroFB', 'ShuffleFB']: return loss_function(out, x, y, 'err_detach')
    if name == 'RecAux': return loss_function(out, x, y, 'rec_aux')
    return F.cross_entropy(out, y)


def predict(model, array, name, device):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([logits(model, b.to(device), name).argmax(1).cpu().numpy()
                               for b in torch.from_numpy(array).split(32)])


def fit_stats(raw, train):
    stats = {}
    for prefix, column in [('i', 'current'), ('v', 'voltage')]:
        values = raw[column][train].astype(np.float32)
        for suffix, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
            stats[f'{prefix}_{suffix}'] = float(fn(values))
    return stats


def training_strategy(strategy):
    if strategy in ['S3', 'S4']: return 'S3'
    if strategy in ['augmentation', 'combined']: return 'combined'
    return 'no_defense'


def config_for(selected, name, strategy):
    source = selected[name if name in CANDIDATES else 'Yu3']
    config = {k: source[k] for k in ['lr', 'weight_decay', 'dropout']}
    # Prespecified common regularization floor, disclosed relative to E2 settings.
    if strategy in ['S4', 'regularization', 'combined']:
        config['weight_decay'] = max(config['weight_decay'], 1e-4)
    return config


def atomic_torch(value, path):
    torch.save(value, path.with_suffix('.tmp'))
    os.replace(path.with_suffix('.tmp'), path)


def csv_write(path, rows):
    with path.with_suffix('.tmp').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(path.with_suffix('.tmp'), path)


class Paused(Exception): pass


def pause_check(output, run, epoch):
    if (output/'STOP_REQUESTED').exists():
        save_json(output/'PAUSED.json', {'run': run, 'saved_epoch': epoch, 'status': 'paused'})
        print(f'PAUSED {run} saved_epoch={epoch}', flush=True)
        raise Paused()


def train(output, folder, raw, train_idx, val_idx, name, strategy, seed, config, device,
          fixed_epochs=None, max_epochs=150, min_epochs=30, patience=30):
    """Pilot picks >=30 best clean val epoch; final training uses that fixed budget."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'result.json').exists():
        result = json.loads((folder/'result.json').read_text())
        assert (folder/'checkpoint.pt').exists()
        return result
    pause_check(output, folder.name, None)
    stats = fit_stats(raw, train_idx)
    assert not set(raw['groups'][train_idx]) & set(raw['groups'][val_idx])
    spec = dict(model=name, strategy=strategy, seed=seed, config=config,
                train_ids=raw['sample_ids'][train_idx].tolist(),
                validation_ids=raw['sample_ids'][val_idx].tolist(), normalization=stats,
                fixed_epochs=fixed_epochs, max_epochs=max_epochs, min_epochs=min_epochs, patience=patience)
    if (folder/'run_spec.json').exists(): assert json.loads((folder/'run_spec.json').read_text()) == spec
    else: save_json(folder/'run_spec.json', spec)
    seed_all(seed)
    model = build(name, config['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    state = dict(epoch=0, logs=[], best=-1., best_epoch=None, anchor=-1., stale=0, stopped=False)
    if (folder/'resume.pt').exists():
        saved = torch.load(folder/'resume.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); state=saved['state']
    v = images(raw, val_idx, stats) if len(val_idx) else None
    vy = raw['labels'][val_idx]
    y = torch.from_numpy(raw['labels'][train_idx]).long().to(device)
    static = images(raw, train_idx, stats) if training_strategy(strategy)=='no_defense' else None
    for epoch in range(state['epoch']+1, (fixed_epochs or max_epochs)+1):
        if state['stopped']: break
        tick = time.monotonic(); seed_all(seed*10000+epoch)
        array = static if static is not None else images(raw, train_idx, stats, training_strategy(strategy), seed, epoch)
        x = torch.from_numpy(array).to(device)
        order = torch.randperm(len(train_idx), generator=torch.Generator().manual_seed(seed*10000+epoch))
        model.train(); total=0.
        for idx in order.split(32):
            idx=idx.to(device); optimizer.zero_grad(set_to_none=True)
            value=loss(model,x[idx],y[idx],name)
            if not torch.isfinite(value): raise RuntimeError(f'Nonfinite loss: {folder}')
            value.backward(); optimizer.step(); total+=float(value.detach())*len(idx)
        score = None
        if v is not None:
            pred=predict(model,v,name,device); score=metrics(vy,pred)['macro_f1']
            if epoch>=min_epochs and score>state['best']:
                state['best']=score; state['best_epoch']=epoch
                atomic_torch({'state_dict':model.state_dict(), 'epoch':epoch, 'model':name, 'config':config},folder/'best.pt')
                np.savez_compressed(folder/'validation_predictions.npz', sample_ids=raw['sample_ids'][val_idx], labels=vy,predictions=pred)
            if epoch==min_epochs: state['anchor']=score
            elif epoch>min_epochs:
                if score>state['anchor']+.001: state['anchor']=score; state['stale']=0
                else: state['stale']+=1
            state['stopped']=epoch>=min_epochs and state['stale']>=patience
        state['epoch']=epoch
        state['logs'].append(dict(epoch=epoch,train_loss=total/len(train_idx),validation_macro_f1=score,seconds=time.monotonic()-tick))
        atomic_torch({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'state':state},folder/'resume.pt')
        csv_write(folder/'train_log.csv',state['logs'])
        save_json(output/'current_run.json', dict(run=folder.name,epoch=epoch,planned_epochs=fixed_epochs or max_epochs,
                  seconds_last_epoch=state['logs'][-1]['seconds'],phase='final_train' if fixed_epochs else 'epoch_pilot'))
        if epoch==1 or epoch%10==0 or state['stopped']:
            print(f'{folder.name} epoch{epoch}/{fixed_epochs or max_epochs} loss={total/len(train_idx):.5f} val={score} {state["logs"][-1]["seconds"]:.2f}s',flush=True)
        pause_check(output,folder.name,epoch)
    if v is not None:
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        saved=np.load(folder/'validation_predictions.npz')
        assert np.array_equal(predict(model,v,name,device),saved['predictions'])
        assert np.array_equal(saved['sample_ids'],raw['sample_ids'][val_idx])
        assert np.array_equal(saved['labels'],vy)
        assert metrics(vy,saved['predictions'])['macro_f1']==state['best']
        selected_epoch=state['best_epoch']
    else:
        selected_epoch=fixed_epochs
        checkpoint=dict(state_dict=model.state_dict(),epoch=fixed_epochs,model=name,config=config)
    atomic_torch(checkpoint,folder/'checkpoint.pt')
    # Checkpoint reload independently re-evaluates training-only reference samples.
    probe=images(raw,train_idx[:32],stats)
    before=predict(model,probe,name,device)
    restored=build(name,config['dropout']).to(device)
    restored.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
    assert np.array_equal(before,predict(restored,probe,name,device))
    passes=2 if name in ['Yu3','ZeroFB','ShuffleFB'] else 1
    if name in FAMILY:
        for layer in (1,2,3):
            assert int(restored.state_dict()[f'encoder.bn{layer}.num_batches_tracked'])==int(np.ceil(len(train_idx)/32))*selected_epoch*passes
    result=dict(model=name,strategy=strategy,seed=seed,config=config,selected_epoch=selected_epoch,
                stopped_epoch=state['epoch'],validation_macro_f1=state['best'] if v is not None else None,
                hit_max_epochs=v is not None and state['epoch']==max_epochs and not state['stopped'],
                prediction_reload_audit=True,training_seconds=sum(r['seconds'] for r in state['logs']))
    save_json(folder/'result.json',result); (folder/'resume.pt').unlink(missing_ok=True)
    (folder/'best.pt').unlink(missing_ok=True)
    print('COMPLETE',folder.name, 'selected_epoch',selected_epoch,flush=True)
    del model,restored,optimizer,x,y
    gc.collect();torch.mps.empty_cache()
    return result
