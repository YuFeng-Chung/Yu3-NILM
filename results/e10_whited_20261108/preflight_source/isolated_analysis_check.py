import sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('Code/RevisedCode/experiments').resolve()))
import e10_core as c
import analyze_e10_whited as a
import torch
# Entire analysis/plot/archive pipeline checked using synthetic fixtures in disposable folder.
with tempfile.TemporaryDirectory(prefix='yu3-e10-analysis-check-') as d:
 root=Path(d)/'root';root.mkdir();out=root/'results';out.mkdir()
 a.ROOT=root;a.OUTPUT=out;a.sync_status=lambda s:None
 y=np.array([0,1,0,1,0,1]);g=np.array(['a','b','c','d','e','f']);classes=np.array(['a','b'])
 np.savez_compressed(out/'images.npz',labels=y,groups=g,sample_ids=g,classes=classes)
 splits=[dict(fold=j+1,outer_test=[2*j,2*j+1],outer_train=[i for i in range(6) if i not in [2*j,2*j+1]]) for j in range(3)]
 c.save(out/'split.json',splits)
 c.save(out/'protocol.json',dict(source_sha256=c.source_hashes(),data_sha256=c.sha(out/'images.npz'),split_sha256=c.sha(out/'split.json')))
 epochs={f'fold{j}_{n}':20 for j in range(1,4) for n in c.NAMES}
 c.save(out/'selection_complete.json',dict(epochs=epochs));c.save(out/'training_complete.json',dict(runs=45))
 for s in splits:
  idx=s['outer_test']
  for n in c.NAMES:
   for seed in c.SEEDS:
    folder=out/'runs'/f"fold{s['fold']}_{n}_s{seed}";folder.mkdir(parents=True)
    c.save(folder/'complete.json',dict(selected_epoch=20))
    p=y[idx].copy()
    if n=='M1':p[:]=0
    np.savez_compressed(folder/'predictions.npz',indices=idx,sample_ids=g[idx],labels=y[idx],predictions=p)
 a.analyze()
 audit=__import__('json').loads((out/'completed_audit.json').read_text())
 assert audit['all_checks_passed'] and audit['independently_recomputed_predictions']==90
 assert (root/'E10_results_20261108.zip').exists()
# Check asymmetric known confusion/PRF, not only implementation agreement.
y=np.array([0,0,1,1,1]);p=np.array([0,1,1,1,0]);m,cm,pr,re,f=c.metrics(y,p,2)
assert cm.tolist()==[[1,1],[1,2]] and np.isclose(m['accuracy'],.6) and np.isclose(m['macro_f1'],(0.5+2/3)/2)
# Gate must refuse an active E12 before any E10 GPU work.
assert c.e12_ready() is False
# Duplicate writer/queue lock exclusion is meaningful on this host.
import fcntl
with tempfile.TemporaryDirectory() as d:
 with open(Path(d)/'lock','a+') as l1,open(Path(d)/'lock','a+') as l2:
  fcntl.flock(l1,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:fcntl.flock(l2,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Lock did not exclude second writer')
c.save(c.OUTPUT/'orchestration_analysis_preflight.json',dict(all_checks_passed=True,at=c.now(),synthetic_analysis_plots_zip_independent_check=True,synthetic_data_never_research_results=True,known_asymmetric_metrics=True,active_e12_gate_refused=True,duplicate_lock_excluded=True))
print('PASS isolated synthetic analysis/figures/CRC/SHA pipeline, known metrics, live E12 gate, writer locks')
