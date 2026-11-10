"""Inference-only diagnostic on No-Defense checkpoints (Yu3, ZeroFB, ShuffleFB).
No training, no selection. Pass1 vs pass2, and second-pass input replacement."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'results' / 'diagnostics'; OUTDIR.mkdir(parents=True, exist_ok=True)

import json, sys
import numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from e6_mechanism_core import load_data, first_pass, second_pass, derangement, independent_f1, PREVIOUS
from revision_followup_core import build, fit_stats
from run_exp4_matched_controls import images

torch.set_num_threads(2)
dev = torch.device('cpu')
raw, test = load_data()
labels = raw['labels'][test]
ids = raw['sample_ids'][test]
CONDS = [('clean', 0, -1)] + [('salt_pepper', s, p) for s in (0.05, 0.1, 0.2) for p in range(1000, 1010)] \
        
stats = json.loads((PREVIOUS / 'final' / 'Yu3_no_defense_s42' / 'run_spec.json').read_text())['normalization']
assert stats == fit_stats(raw, np.flatnonzero(raw['partitions'] == 'train'))
arrays = {c: images(raw, test, stats, condition=None if c[0] == 'clean' else c) for c in CONDS}
rows = []
for name in ['Yu3', 'ZeroFB', 'ShuffleFB']:
    spec = json.loads((PREVIOUS / 'final' / f'{name}_no_defense_s42' / 'run_spec.json').read_text())
    for seed in range(42, 52):
        saved = torch.load(PREVIOUS / 'final' / f'{name}_no_defense_s{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model = build(name, spec['config']['dropout']).eval()
        model.load_state_dict(saved['state_dict'])
        for c in CONDS:
            a = arrays[c]
            l1, recon = first_pass(model, a, dev, batch=256)
            r = np.abs(a - recon)
            order = derangement(ids, c)
            rolled = np.concatenate([np.roll(r[s:s + 32], 1, axis=0) for s in range(0, len(r), 32)])
            out = dict(model=name, seed=seed, type=c[0], severity=c[1], ps=c[2],
                       pass1=independent_f1(labels, l1.argmax(1)))
            for k, aux in [('own', r), ('zero', np.zeros_like(r)), ('cross', r[order])] + ([('rolled', rolled)] if name == 'ShuffleFB' else []):
                out[k] = independent_f1(labels, second_pass(model, a, aux, dev, batch=256).argmax(1))
            rows.append(out)
        print(name, seed, flush=True)
        __import__('pandas').DataFrame(rows).to_csv(OUTDIR / 'nodefense_diag_raw.csv', index=False)
import pandas as pd
pd.DataFrame(rows).to_csv(OUTDIR / 'nodefense_diag_raw.csv', index=False)
print('done')
