import sys,tempfile,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path('Code/RevisedCode/experiments').resolve()))
import e10_core as c
cache=dict(np.load(c.OUTPUT/'images.npz'));split=json.loads((c.OUTPUT/'split.json').read_text())[0];tr=np.array(split['inner_train'][:8]);torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='yu3-e10-resume-check-') as d:
 out=Path(d)
 c.train(out/'full',cache,tr,np.array([],int),'M5',42,torch.device('cpu'),fixed_epochs=2)
 original=c.check_stop
 def pause(folder=None,epoch=None):
  if epoch==1:raise InterruptedError('Synthetic pause after first real training epoch')
 c.check_stop=pause
 try:c.train(out/'resumed',cache,tr,np.array([],int),'M5',42,torch.device('cpu'),fixed_epochs=2)
 except InterruptedError:pass
 else:raise AssertionError('Pause not triggered')
 c.check_stop=original
 st=torch.load(out/'resumed/resume.pt',map_location='cpu',weights_only=False);assert st['epoch']==1
 c.train(out/'resumed',cache,tr,np.array([],int),'M5',42,torch.device('cpu'),fixed_epochs=2)
 a=torch.load(out/'full/model.pt',map_location='cpu',weights_only=False);b=torch.load(out/'resumed/model.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,b['model'][n]) for n,v in a['model'].items())
c.save(c.OUTPUT/'resume_preflight_audit.json',dict(all_checks_passed=True,at=c.now(),real_training_examples=len(tr),device='cpu',model='M5',pause_after_epoch=1,resumed_through_epoch=2,uninterrupted_and_resumed_all_weights_bn_exact=True,no_test_data=True,temporary_artifacts_removed=True))
print('PASS real M5 epoch boundary pause/resume matches uninterrupted weights and BN exactly')
