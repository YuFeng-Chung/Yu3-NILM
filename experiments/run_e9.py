"""One kernel-locked E9 writer; all epoch pilots finish before outer evaluation."""
import argparse
import fcntl
import traceback
from e9_core import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);args=p.parse_args();output=args.output.resolve()
    with (output/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=load_plan(output)
        assert torch.backends.mps.is_available();torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
        raw=dict(np.load(output/'waveforms.npz'));split=json.loads((output/'split.json').read_text())['indices'];device=torch.device('mps')
        try:
            chosen={}
            for cell in CELLS:
                for name in MODELS:
                    save_json(output/'progress.json',dict(stage='epoch_pilots',completed=len(chosen),planned=8,cell=cell,model=name))
                    r=train(output,output/'pilots'/f'{cell}_{name}_s42',raw,np.array(split['inner_train']),np.array(split['inner_validation']),cell,name,42,plan['config'][name],device)
                    chosen[f'{cell}_{name}']=r['selected_epoch']
            selection=dict(epochs=chosen,source_sha256=plan['source_sha256'],pilot_count=8,outer_predictions_opened=False)
            if (output/'selection_complete.json').exists():assert json.loads((output/'selection_complete.json').read_text())==selection
            else:save_json(output/'selection_complete.json',selection)
            cache=output/'evaluation_images';cache.mkdir(exist_ok=True)
            # Compact uint8-independent float32 arrays prepared once, reused by paired seeds.
            for cell in CELLS:
                stats=fit_stats(raw,split['outer_train'],CELLS[cell][0]);folder=cache/cell;folder.mkdir(exist_ok=True)
                for ci,condition in enumerate(CONDITIONS):
                    pause_check(output,f'cache_{cell}_{ci}',None);path=folder/f'{ci:03}.npy'
                    if not path.exists():np.save(path,images(raw,split['outer_test'],stats,cell,condition))
            done=0
            for seed in SEEDS:
                for cell in CELLS:
                    for name in MODELS:
                        folder=output/'runs'/f'{cell}_{name}_s{seed}';save_json(output/'progress.json',dict(stage='formal_training_evaluation',completed=done,planned=40,cell=cell,model=name,seed=seed))
                        train(output,folder,raw,np.array(split['outer_train']),np.array([],int),cell,name,seed,plan['config'][name],device,fixed_epochs=chosen[f'{cell}_{name}'])
                        net=build(name,plan['config'][name]['dropout']).to(device);net.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict'])
                        pred_dir=folder/'predictions';pred_dir.mkdir(exist_ok=True)
                        np.savez_compressed(folder/'test_identity.npz',sample_ids=raw['sample_ids'][split['outer_test']],labels=raw['labels'][split['outer_test']])
                        rows=[]
                        for ci,condition in enumerate(CONDITIONS):
                            pause_check(output,folder.name,f'eval_{ci}');dest=pred_dir/f'{ci:03}.npy';x=np.load(cache/cell/f'{ci:03}.npy')
                            if dest.exists():pred=np.load(dest)
                            else:pred=predict(net,x,name,device);np.save(dest,pred)
                            m=metrics(raw['labels'][split['outer_test']],pred);rows.append(dict(condition=ci,kind=condition[0],severity=condition[1],pollution_seed=condition[2],**m))
                        csv_write(folder/'evaluation.csv',rows);save_json(folder/'evaluation_complete.json',dict(conditions=41,test_n=len(split['outer_test']),selection_before_test=True))
                        del net;gc.collect();torch.mps.empty_cache();done+=1
                        save_json(output/'progress.json',dict(stage='formal_training_evaluation',completed=done,planned=40))
            from analyze_e9 import analyze
            from verify_e9 import verify
            save_json(output/'progress.json',dict(stage='analysis_diagnostics',completed=40,planned=40))
            analyze(output);verify(output)
            save_json(output/'completed_audit.json',dict(all_checks_passed=True,epoch_pilots=8,formal_training=40,evaluation_per_run=41,
                independent_audit=sha(output/'independent_audit.json'),source_sha256=source_hashes(),formal_word_response_complete=False,manuscript_changed=False))
            save_json(output/'progress.json',dict(stage='completed',completed=40,planned=40))
        except Paused:print('E9 safely paused',flush=True)
        except Exception:
            save_json(output/'FAILED.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
