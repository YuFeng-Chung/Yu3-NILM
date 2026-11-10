"""Short CPU scheduling pilot only; timings are NOT formal E8 results.

Loads frozen seed42 S3 checkpoints, never trains or selects by test accuracy.
Run from the established research environment. No ONNX installation/export.
"""
import importlib.util
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from revision_models_mps import build_model, final_logits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/e8_schedule_20261108'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'scheduling_pilot.json'
    if target.exists():
        raise SystemExit('Pilot output exists; preserve it rather than overwrite.')
    start = time.perf_counter()
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    x = torch.rand(1, 1, 64, 64)
    models, provenance = {}, {}
    for name in ['Yu3', 'ResNet18', 'MobileNetV3', 'ViT-Tiny']:
        folder = ROOT / 'results/e1_e4_mac_20261106/final' / f'{name}_S3_s42'
        spec = json.loads((folder / 'run_spec.json').read_text())
        model = build_model(name, spec['config']['dropout']).eval()
        saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        models[name] = model
        provenance[name] = str(folder.relative_to(ROOT))
    paths = {name: (lambda name=name: final_logits(models[name], x, name)) for name in models}
    yu3 = models['Yu3']
    paths['Yu3_pass1_only'] = lambda: yu3.classify(yu3.encode(torch.cat([x, torch.zeros_like(x)], 1), 0))
    rows = []
    with torch.inference_mode():
        for threads in [1, 4]:
            torch.set_num_threads(threads)
            for fn in paths.values():
                for _ in range(5):
                    y = fn()
                    assert y.shape == (1, 11) and torch.isfinite(y).all()
            for round_id in range(3):
                order = list(paths)
                random.Random(4200 + round_id).shuffle(order)
                for name in order:
                    t = time.perf_counter()
                    for _ in range(8):
                        paths[name]()
                    rows.append(dict(threads=threads, round=round_id, path=name,
                                     calls=8, seconds=time.perf_counter() - t))
    # Draft formal workload: 30 interleaved blocks x 100 calls per path,
    # both thread settings. ORT remains an unmeasured planning allowance.
    projected = sum(np.median([r['seconds'] / r['calls'] for r in rows
                              if r['path'] == name and r['threads'] == threads]) * 3000
                    for threads in [1, 4] for name in paths)
    report = dict(status='scheduling_only_not_formal_results', date_taipei='2026-11-08',
                  torch=torch.__version__, platform=platform.platform(),
                  cpu=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  input='synthetic FP32 [1,1,64,64]; no test labels or scores',
                  checkpoint_paths=provenance, interop_threads=1, warmup_calls_per_path=5,
                  rows=rows, projected_pytorch_timed_seconds=float(projected),
                  pilot_elapsed_seconds=time.perf_counter() - start,
                  optional_packages={n: importlib.util.find_spec(n) is not None
                                     for n in ['onnx', 'onnxruntime', 'onnxscript', 'psutil']},
                  limitations=['Background load uncontrolled; not usable as paper latency.',
                               'ORT timing/export/parity not measured.',
                               'Formal workload is a proposed plan, not an existing frozen protocol.'])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'projected_pytorch_timed_seconds',
                                           'pilot_elapsed_seconds', 'optional_packages']}, indent=2))


if __name__ == '__main__':
    main()
