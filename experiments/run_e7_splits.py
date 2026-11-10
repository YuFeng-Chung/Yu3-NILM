"""Single locked E7 writer; epoch-atomic pause/resume inherited from audited engine."""
import argparse
import fcntl
import gc
import platform
import traceback
import torch
from e7_split_core import *
from revision_followup_core import train, Paused
from run_revision_followup_v4 import evaluate_outer

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();output=a.output.resolve()
    with (output/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=load_plan(output)
        assert torch.backends.mps.is_available(),'No silent CPU fallback'
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
        save_json(output/'environment.json',dict(platform=platform.platform(),torch=torch.__version__,device='mps',threads=4))
        raw=dict(np.load(DATA/'raw_windows.npz'));done=0
        try:
            for split in plan['splits'][1:]:
                s=split['split'];train_idx=np.array(split['train_indices']);test_idx=np.array(split['test_indices'])
                local=dict(raw);local['partitions']=np.full(len(raw['labels']),'train',dtype='<U5');local['partitions'][test_idx]='val'
                for seed in SEEDS:
                    for model in MODELS:
                        settings=plan['settings'][model];folder=output/'runs'/f'split{s}_{model}_s{seed}'
                        save_json(output/'progress.json',dict(stage='training_and_evaluation',completed=done,planned=60,split=s,model=model,seed=seed))
                        train(output,folder,local,train_idx,np.array([],dtype=int),model,'combined',seed,settings['config'],torch.device('mps'),fixed_epochs=settings['epochs'])
                        evaluate_outer(output,local,folder,model,'combined',torch.device('mps'),[tuple(c) for c in plan['conditions']])
                        done+=1
                        save_json(output/'progress.json',dict(stage='training_and_evaluation',completed=done,planned=60))
                        gc.collect();torch.mps.empty_cache()
            from analyze_e7_splits import analyze
            analyze(output)
            # Complete only after every run, split identity, epochs, normalization and all predictions verified.
            from revision_followup_core import fit_stats
            for split in plan['splits'][1:]:
                idx=np.array(split['train_indices']);stats=fit_stats(raw,idx)
                for seed in SEEDS:
                    for model in MODELS:
                        folder=output/'runs'/f"split{split['split']}_{model}_s{seed}"
                        spec=json.loads((folder/'run_spec.json').read_text());result=json.loads((folder/'result.json').read_text())
                        assert spec['normalization']==stats and spec['train_ids']==split['train_ids']
                        assert result['selected_epoch']==plan['settings'][model]['epochs']
                        ids=np.load(folder/'outer_ids_labels.npz');assert ids['sample_ids'].tolist()==split['test_ids']
                        assert len(json.loads((folder/'evaluation.json').read_text())['conditions'])==30
            assert len(list((output/'runs').glob('*/evaluation.json')))==60
            save_json(output/'completed_audit.json',dict(all_checks_passed=True,completed_training=60,completed_evaluation=60,
              independent_count_analysis=True,split_identity_and_training_only_normalization_verified=True,
              source_sha256=source_hashes(),formal_word_response_complete=False,manuscript_changed=False))
            save_json(output/'progress.json',dict(stage='completed',completed=60,planned=60))
        except Paused:print('E7 paused safely',flush=True)
        except Exception:
            save_json(output/'FAILED.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
