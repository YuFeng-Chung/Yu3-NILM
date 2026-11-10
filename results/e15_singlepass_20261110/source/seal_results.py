"""Seal completed new final checkpoints, logs and predictions before analysis."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;OUT=H.parent
assert json.loads((OUT/'progress.json').read_text())['stage']=='training_and_evaluation_complete_pending_analysis'
assert not (OUT/'final_results_sha256.json').exists(),'Do not replace sealed research artifacts'
files=sorted(p for p in (OUT/'final').rglob('*') if p.is_file())
assert len(list((OUT/'final').glob('*/evaluation.json')))==20
files += [OUT/n for n in ['noise_scale.json','noise_scale_seal.json','calibration_moments.npz','selection_complete.json','epoch_settings.json']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'final_results_sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in files},indent=2,ensure_ascii=False))
print('Sealed',len(files),'completed E15 research artifacts')
