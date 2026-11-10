"""One-time preserve-first repair: original E4 eval batch32; no training or test selection."""
from pathlib import Path
import json,hashlib,shutil,subprocess,sys,socket
H=Path(__file__).resolve().parent;OUT=H.parent
AUD=OUT/'numerical_audit/batch219';AUD.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=H/'evaluate_e14.py';before=sha(p);shutil.copy2(p,AUD/'evaluate_e14_original_frozen.py')
s=p.read_text();assert "batch=32 if name=='ShuffleFB' else 219" in s
p.write_text(s.replace("batch=32 if name=='ShuffleFB' else 219","batch=32  # Match original E4 evaluation exactly for every model."))
# Preserve every initial prediction and score; only evaluation products move.
moved=[]
for root in [OUT/'final',OUT/'reference_evaluation']:
 for folder in root.glob('*'):
  if not folder.is_dir():continue
  for item in ['outer_predictions','first_pass_predictions','outer_ids_labels.npz','evaluation.json']:
   source=folder/item
   if source.exists():
    dest=AUD/root.name/folder.name/item;dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();shutil.move(str(source),str(dest));moved.append(str(source.relative_to(OUT)))
for n in ['FAILED.json','stderr.log']:
 if (OUT/n).exists():shutil.copy2(OUT/n,AUD/n)
(OUT/'FAILED.json').rename(OUT/'INITIAL_BATCH219_FAILURE.json')
# Record amendment, retaining original launch hashes unchanged.
amend=dict(reason='Native-output guard found ZeroFB seed46 condition098 differing with eval batch219 vs original32; unify ALL evaluations at32, preserve first batch; no train/model/statistical changes',original_sha256=before,revised_sha256=sha(p),original_source_copy=str((AUD/'evaluate_e14_original_frozen.py').relative_to(OUT)),moved=moved,training_repeated=False,scope='Inference numerical fidelity only; entire official evaluation recomputed at original E4 batch32',supersedes_initial_evaluation=True)
(OUT/'inference_batch_amendment.json').write_text(json.dumps(amend,ensure_ascii=False,indent=2))
print('Preserved',len(moved),'evaluation artifacts; inference batch32 enforced')
