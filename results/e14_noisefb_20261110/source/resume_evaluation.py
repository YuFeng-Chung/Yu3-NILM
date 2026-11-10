"""Resume only inference/analysis after documented batch-size fidelity correction."""
from pathlib import Path
import sys,json,fcntl,gc
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import torch,numpy as np
from e14_core import DATA,save_json,sha
from evaluate_e14 import evaluate
from analyze_e14 import analyze

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True);assert torch.backends.mps.is_available()
    amend=json.loads((OUT/'inference_batch_amendment.json').read_text());assert sha(H/'evaluate_e14.py')==amend['revised_sha256']
    raw=dict(np.load(DATA/'raw_windows.npz'));device=torch.device('mps')
    for name in ['NoiseFB','PermFB','Yu3','ZeroFB','ShuffleFB']:
        for seed in range(42,52):
            save_json(OUT/'progress.json',dict(stage='batch32_evaluation',model=name,seed=seed,no_retraining=True))
            evaluate(raw,name,seed,device)
    save_json(OUT/'progress.json',dict(stage='analysis'))
    analyze()
    save_json(OUT/'progress.json',dict(stage='research_complete',report_status='pending_document_QA',official_evaluation_batch=32))
if __name__=='__main__':
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            save_json(OUT/'FAILED.json',dict(type=type(exc).__name__,message=str(exc)));raise
