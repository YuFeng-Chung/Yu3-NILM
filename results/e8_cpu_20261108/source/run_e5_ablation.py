"""Fixed-protocol E5 clean-data ablation. No model selection using held-out scores.

Uses supplied Exp1 PNGs (855 train / 219 test), not waveform Exp4 preprocessing.
Corrupted-condition evaluation requires waveform provenance and is not implemented here.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '實驗一/models_m1_to_m5.py'
spec = importlib.util.spec_from_file_location('original_models', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
VARIANTS = ['err_detach', 'err_no_detach', 'img_no_detach', 'img_detach',
            'no_first_ce', 'no_recon_loss', 'recon_weight_one', 'rec_aux',
            'three_pass', 'pass_specific_bn', 'no_feedback']


class Ablation(original.M5_Model):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant == 'pass_specific_bn':
            self.pass2_bn = nn.ModuleList([copy.deepcopy(getattr(self.encoder, f'bn{i}')) for i in (1, 2, 3)])

    def encode(self, x, pass_index):
        for i in (1, 2, 3):
            bn = self.pass2_bn[i - 1] if self.variant == 'pass_specific_bn' and pass_index == 1 else getattr(self.encoder, f'bn{i}')
            x = getattr(self.encoder, f'pool{i}')(F.relu(bn(getattr(self.encoder, f'conv{i}')(x))))
        return x

    def classify(self, z):
        return self.classifier(self.dropout(F.relu(self.fc(z.flatten(1)))))

    def decode(self, z):
        z = F.relu(self.dec_bn1(self.dec_conv1(z)))
        z = F.relu(self.dec_bn2(self.dec_conv2(z)))
        return torch.sigmoid(self.dec_conv3(z))

    def forward(self, x):
        z = self.encode(torch.cat([x, torch.zeros_like(x)], 1), 0)
        logits, recon = [self.classify(z)], []
        if self.variant == 'no_feedback':
            return logits, recon
        recon.append(self.decode(z))
        if self.variant == 'rec_aux':
            return logits, recon
        aux = recon[0] if self.variant.startswith('img_') else (x - recon[0]).abs()
        if self.variant not in ('err_no_detach', 'img_no_detach'):
            aux = aux.detach()
        z = self.encode(torch.cat([x, aux], 1), 1)
        logits.append(self.classify(z))
        if self.variant == 'three_pass':
            recon.append(self.decode(z))
            z = self.encode(torch.cat([x, (x - recon[1]).abs().detach()], 1), 2)
            logits.append(self.classify(z))
        return logits, recon


def loss_function(outputs, x, y, variant):
    logits, recon = outputs
    if variant == 'no_feedback':
        return F.cross_entropy(logits[0], y)
    if variant == 'rec_aux':
        return F.cross_entropy(logits[0], y) + .5 * F.mse_loss(recon[0], x)
    if variant == 'three_pass':
        return sum(w * F.cross_entropy(p, y) for w, p in zip([.5, .5, 1.], logits)) + sum(.25 * F.mse_loss(r, x) for r in recon)
    loss = F.cross_entropy(logits[1], y)
    if variant != 'no_first_ce':
        loss = loss + .5 * F.cross_entropy(logits[0], y)
    if variant != 'no_recon_loss':
        loss = loss + (1. if variant == 'recon_weight_one' else .5) * F.mse_loss(recon[0], x)
    return loss


def metrics(y, pred):
    cm = np.zeros((11, 11), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(11), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(11), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(11), where=precision + recall != 0)
    return {'accuracy': float(tp.sum() / cm.sum()), 'macro_precision': float(precision.mean()),
            'macro_recall': float(recall.mean()), 'macro_f1': float(f1.mean()), 'class_f1': f1.tolist()}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def bn_diagnostic(model, x, device):
    """Record per-pass BN input population moments in eval; never update buffers."""
    accum, handles, counts = {}, [], {}
    for layer in (1, 2, 3):
        bn = getattr(model.encoder, f'bn{layer}')
        def hook(module, args, layer=layer):
            p = counts.get(layer, 0); counts[layer] = p + 1
            z = args[0].detach()
            key = f'layer{layer}_pass{p + 1}'
            count = z.shape[0] * z.shape[2] * z.shape[3]
            s = z.sum((0, 2, 3)).cpu().numpy().astype(np.float64)
            sq = z.square().sum((0, 2, 3)).cpu().numpy().astype(np.float64)
            if key not in accum: accum[key] = [0, np.zeros_like(s), np.zeros_like(sq), module]
            accum[key][0] += count; accum[key][1] += s; accum[key][2] += sq
        handles.append(bn.register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        for batch in x.split(32):
            counts.clear(); model(batch.to(device))
    for h in handles: h.remove()
    result = {}
    for k, (n, s, sq, bn) in accum.items():
        mean = s / n; var = np.maximum(0, sq / n - mean ** 2)
        rm = bn.running_mean.cpu().numpy(); rv = bn.running_var.cpu().numpy()
        result[k] = {'population_mean': mean.tolist(), 'population_variance': var.tolist(),
                     'running_mean': rm.tolist(), 'running_variance': rv.tolist(),
                     'mean_absolute_mean_gap': float(np.abs(mean - rm).mean()),
                     'mean_absolute_variance_gap': float(np.abs(var - rv).mean())}
    return result


def smoke(device):
    seed_all(42)
    a = original.M5_Model().to(device).eval()
    b = Ablation('err_detach').to(device).eval(); b.load_state_dict(a.state_dict())
    x = torch.rand(4, 1, 64, 64, device=device); y = torch.tensor([0, 1, 2, 3], device=device)
    with torch.no_grad():
        ao = a(x); bo = b(x)
        assert torch.allclose(ao[0], bo[0][0], atol=1e-6)
        assert torch.allclose(ao[1], bo[0][1], atol=1e-6)
        assert torch.allclose(ao[2], bo[1][0], atol=1e-6)
    for v in VARIANTS:
        seed_all(42); model = Ablation(v).to(device).train()
        loss = loss_function(model(x), x, y, v); loss.backward()
        assert torch.isfinite(loss)
        assert int(model.encoder.bn1.num_batches_tracked) == (3 if v == 'three_pass' else 1 if v in ('no_feedback', 'rec_aux', 'pass_specific_bn') else 2)
        if v == 'pass_specific_bn': assert int(model.pass2_bn[0].num_batches_tracked) == 1
        if v in ('no_recon_loss', 'no_feedback'):
            assert model.dec_conv1.weight.grad is None
    print('PASS smoke: original forward equivalence, all variant backward, BN update counts and detached decoder gradients.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=VARIANTS)
    parser.add_argument('--epochs', type=int, default=19)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available; explicitly choose --device cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    smoke(device)
    if args.smoke_only: return
    cache = np.load(args.output / 'e5_images.npz')
    train_x = torch.from_numpy(cache['train_images'].astype(np.float32) / 255.)
    train_y = torch.from_numpy(cache['train_labels'])
    test_x = torch.from_numpy(cache['val_images'].astype(np.float32) / 255.)
    test_y = cache['val_labels']; test_ids = cache['val_ids']
    manifest = json.loads((args.output / 'image_manifest.json').read_text())
    config = {'experiment': 'E5 clean fixed-protocol, all-11-class metrics', 'seeds': args.seeds,
              'variants': args.variants, 'epochs': args.epochs, 'batch_size': 32, 'adam_lr': .0005,
              'weight_decay': 0., 'device': str(device), 'dropout': .5, 'train_size': len(train_x),
              'test_size': len(test_y), 'source_model_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256((args.output / 'image_manifest.json').read_bytes()).hexdigest(),
              'rec_aux_note': 'Two-channel encoder with zero auxiliary channel, matched to E5; not final E1 M1-based implementation.',
              'selection': 'No architecture selection on held-out scores. Fixed final epoch, no early stopping.',
              'limits': ['New macOS/MPS environment; not numerical reproduction of original CUDA runs.',
                         'Clean PNG branch only; no waveform corruption tests yet.',
                         'BN diagnostic gaps alone do not establish performance harm or absence of harm.']}
    config_path = args.output / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Existing config differs; choose a new output directory.'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [SOURCE, Path(__file__), Path(__file__).with_name('prepare_e5_images.py')]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    env = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
           'numpy': np.__version__, 'mps_available': torch.backends.mps.is_available(), 'threads': torch.get_num_threads()}
    (args.output / 'env.json').write_text(json.dumps(env, indent=2))
    (args.output / 'requirements.lock.txt').write_text(subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze'], text=True))
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            run = args.output / f'{variant}_seed{seed}'; run.mkdir(exist_ok=True)
            if (run / 'metrics.json').exists():
                results.append(json.loads((run / 'metrics.json').read_text())); continue
            seed_all(seed)
            model = Ablation(variant).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
            generator = torch.Generator().manual_seed(seed)
            started = time.monotonic()
            with (run / 'train_log.csv').open('w', newline='') as f:
                writer = csv.writer(f); writer.writerow(['epoch', 'train_loss', 'seconds'])
                for epoch in range(1, args.epochs + 1):
                    model.train(); total = 0.
                    permutation = torch.randperm(len(train_x), generator=generator)
                    for indices in permutation.split(32):
                        x, y = train_x[indices].to(device), train_y[indices].to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(model(x), x, y, variant)
                        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
                        loss.backward(); optimizer.step(); total += float(loss.detach().cpu()) * len(indices)
                    elapsed = time.monotonic() - started
                    writer.writerow([epoch, total / len(train_x), elapsed]); f.flush()
                    print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} loss={total/len(train_x):.5f} elapsed={elapsed:.1f}s', flush=True)
            model.eval(); predictions, first = [], []
            with torch.no_grad():
                for x in test_x.split(32):
                    logits, _ = model(x.to(device))
                    predictions.extend(logits[-1].argmax(1).cpu().tolist()); first.extend(logits[0].argmax(1).cpu().tolist())
            result = {'variant': variant, 'seed': seed, 'condition': 'clean', 'epoch': args.epochs,
                      'training_seconds': elapsed, **metrics(test_y, predictions),
                      'wrong_to_correct': int(((np.array(first) != test_y) & (np.array(predictions) == test_y)).sum()),
                      'correct_to_wrong': int(((np.array(first) == test_y) & (np.array(predictions) != test_y)).sum()),
                      'first_pass_macro_f1': metrics(test_y, first)['macro_f1']}
            with (run / 'pred_clean.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['sample_id', 'y_true', 'first_pass_pred', 'y_pred'])
                w.writerows(zip(test_ids, test_y, first, predictions))
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'variant': variant, 'seed': seed, 'epoch': args.epochs}, run / 'checkpoint.pt')
            if variant == 'err_detach':
                (run / 'bn_diagnostic.json').write_text(json.dumps(bn_diagnostic(model, test_x, device), indent=2))
            (run / 'metrics.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            scalar_keys = [k for k in result if k != 'class_f1']
            with (args.output / 'results.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=scalar_keys); w.writeheader()
                w.writerows({k: r[k] for k in scalar_keys} for r in results)
            print('RESULT', json.dumps(result), flush=True)
            del model, optimizer
            if device.type == 'mps': torch.mps.empty_cache()
    print('COMPLETE', len(results), 'runs', flush=True)


if __name__ == '__main__':
    main()
