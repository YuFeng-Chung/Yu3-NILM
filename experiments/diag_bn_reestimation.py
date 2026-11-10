"""(a) Inference-only BN recalibration on No-Defense checkpoints.
Encoder BN running statistics are re-estimated on the clean TRAINING images only
(test never used), with different second-channel inputs, then pass-1 Macro-F1 is
measured on clean and impulse test sets. No weights are changed except BN buffers
of a copy of the model."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'results' / 'diagnostics'; OUTDIR.mkdir(parents=True, exist_ok=True)

import copy, json, sys
import numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from e6_mechanism_core import load_data, first_pass, independent_f1, PREVIOUS
from revision_followup_core import build
from run_exp4_matched_controls import images

torch.set_num_threads(2)
dev = torch.device('cpu')
raw, test = load_data()
train = np.flatnonzero(raw['partitions'] == 'train')
labels = raw['labels'][test]
CONDS = [('clean', 0, -1)] + [('salt_pepper', s, p) for s in (0.05, 0.1, 0.2) for p in range(1000, 1010)]
stats = json.loads((PREVIOUS / 'final' / 'Yu3_no_defense_s42' / 'run_spec.json').read_text())['normalization']
Xtr = images(raw, train, stats)
arrays = {c: images(raw, test, stats, condition=None if c[0] == 'clean' else c) for c in CONDS}


def enc_bns(m):
    return [getattr(m.encoder, f'bn{i}') for i in (1, 2, 3)]


def recal(model, second_channels):
    """Re-estimate encoder BN running stats from a list of [N,1,64,64] second-channel arrays (cumulative average)."""
    m = copy.deepcopy(model).eval()
    for bn in enc_bns(m):
        bn.reset_running_stats(); bn.momentum = None; bn.train()
    with torch.inference_mode():
        for aux in second_channels:
            for s in range(0, len(Xtr), 32):
                x = torch.from_numpy(Xtr[s:s + 32]); r = torch.from_numpy(aux[s:s + 32])
                m.encode(torch.cat([x, r], 1), 0)
    for bn in enc_bns(m):
        bn.eval()
    return m


rows = []
for name in ['Yu3', 'ShuffleFB', 'ZeroFB']:
    spec = json.loads((PREVIOUS / 'final' / f'{name}_no_defense_s42' / 'run_spec.json').read_text())
    for seed in range(42, 52):
        model = build(name, spec['config']['dropout']).eval()
        model.load_state_dict(torch.load(PREVIOUS / 'final' / f'{name}_no_defense_s{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=False)['state_dict'])
        _, rec = first_pass(model, Xtr, dev, batch=256)
        R = np.abs(Xtr - rec).astype(np.float32)
        if name == 'ShuffleFB':
            R = np.concatenate([np.roll(R[s:s + 32], 1, axis=0) for s in range(0, len(R), 32)])
        Z = np.zeros_like(R)
        variants = {'orig': model, 'x0_only': recal(model, [Z]), 'xr_only': recal(model, [R]), 'both': recal(model, [Z, R])}
        for c in CONDS:
            for k, m in variants.items():
                l1, _ = first_pass(m, arrays[c], dev, batch=256)
                rows.append(dict(model=name, seed=seed, type=c[0], severity=c[1], ps=c[2], bn=k, f1=independent_f1(labels, l1.argmax(1))))
        print(name, seed, flush=True)
        __import__('pandas').DataFrame(rows).to_csv(OUTDIR / 'bn_recal_raw.csv', index=False)
print('done')
