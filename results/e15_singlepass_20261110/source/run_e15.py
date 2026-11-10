"""Authorized fixed-budget formal training then two-input evaluation; no selection."""
import sys,json,time,os,fcntl,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'experiments'))
from e15_core import *
from evaluate_e15 import evaluate
NAMES=['1P-NoiseCh','1P-Mix'];EPOCHS={'1P-NoiseCh':42,'1P-Mix':37}
def guard():
    for file in ['protected_sources_sha256.json','formal_protected_sources_sha256.json']:
        for rel,d in json.loads((OUT/file).read_text()).items():assert sha(ROOT/rel)==d,rel
    for rel,d in json.loads((OUT/'formal_source_sha256.json').read_text()).items():assert sha(OUT/rel)==d,rel
    assert sha(OUT/'noise_scale.json')==json.loads((OUT/'noise_scale_seal.json').read_text())['sha256']

def main():
    assert json.loads((OUT/'authorization.json').read_text())['formal_training_authorized']
    assert json.loads((OUT/'epoch_settings.json').read_text())['fixed_epochs']==EPOCHS
    assert json.loads((OUT/'prelaunch_audit.json').read_text())['all_checks_passed']
    guard();torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    raw=dict(np.load(DATA/'raw_windows.npz'));idx=np.flatnonzero(raw['partitions']=='train');assert len(idx)==855
    device=torch.device('mps');assert torch.backends.mps.is_available()
    selection=dict(fixed_epochs=EPOCHS,config={'lr':.0005,'weight_decay':0.,'dropout':.5},scale_sha256=sha(OUT/'noise_scale.json'),outer_test_evaluations_at_selection=0)
    if (OUT/'selection_complete.json').exists():assert json.loads((OUT/'selection_complete.json').read_text())==selection
    else:save_json(OUT/'selection_complete.json',selection)
    for name in NAMES:
        for seed in range(42,52):
            save_json(OUT/'progress.json',dict(stage='formal_training',model=name,seed=seed,formal_training_authorized=True))
            train(OUT,OUT/'final'/f'{name}_no_defense_s{seed}',raw,idx,np.array([],dtype=int),name,'no_defense',seed,selection['config'],device,fixed_epochs=EPOCHS[name])
    guard();save_json(OUT/'all_training_complete.json',dict(formal_models=20,fixed_epochs=EPOCHS))
    for name in NAMES:
        for seed in range(42,52):
            save_json(OUT/'progress.json',dict(stage='evaluation',model=name,seed=seed,formal_training_authorized=True))
            evaluate(raw,name,seed,device)
    guard();save_json(OUT/'progress.json',dict(stage='training_and_evaluation_complete_pending_analysis',formal_models=20,evaluations=20,formal_training_authorized=True))
    print('ALL TRAINING AND EVALUATION COMPLETE',flush=True)
if __name__=='__main__':
    with (OUT/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Paused:print('PAUSED safely',flush=True)
        except Exception as e:
            save_json(OUT/'FAILED_FORMAL.json',dict(type=type(e).__name__,message=str(e)));raise
