"""(b) Inference-only: Yu3 pass-1 vs pass-2 across strategies and corruption types."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'results' / 'diagnostics'; OUTDIR.mkdir(parents=True, exist_ok=True)

import json, sys
import numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from e6_mechanism_core import load_data, first_pass, second_pass, independent_f1, PREVIOUS
from revision_followup_core import build
from run_exp4_matched_controls import images
torch.set_num_threads(2); dev = torch.device('cpu')
raw, test = load_data(); labels = raw['labels'][test]
P = range(1000, 1005)
base = [('clean', 0, -1)] + [(t, s, p) for t, s in [('awgn', 20), ('awgn', 10), ('missing_consecutive', .2), ('missing_consecutive', .5)] for p in P]
extra = [('salt_pepper', s, p) for s in (.05, .2) for p in P]
plan = {'no_defense': base[1:], 'S3': base + extra, 'combined': base + extra}
rows = []
for st, conds in plan.items():
    spec = json.loads((PREVIOUS / 'final' / f'Yu3_{st}_s42' / 'run_spec.json').read_text())
    arrays = {c: images(raw, test, spec['normalization'], condition=None if c[0] == 'clean' else c) for c in conds}
    for seed in range(42, 52):
        m = build('Yu3', spec['config']['dropout']).eval()
        m.load_state_dict(torch.load(PREVIOUS / 'final' / f'Yu3_{st}_s{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=False)['state_dict'])
        for c in conds:
            a = arrays[c]; l1, rec = first_pass(m, a, dev, batch=256)
            l2 = second_pass(m, a, np.abs(a - rec), dev, batch=256)
            rows.append(dict(strategy=st, seed=seed, type=c[0], severity=c[1], ps=c[2],
                             pass1=independent_f1(labels, l1.argmax(1)), pass2=independent_f1(labels, l2.argmax(1))))
        print(st, seed, flush=True)
        __import__('pandas').DataFrame(rows).to_csv(OUTDIR / 'pass_general_raw.csv', index=False)
print('done')
