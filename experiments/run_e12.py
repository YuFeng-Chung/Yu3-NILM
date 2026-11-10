"""E12 inference only; immutable protocol, data and checkpoint digest checks."""
import argparse
import fcntl
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
from e12_core import *

def run(out):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available();device=torch.device('mps')
    plan=json.loads((out/'protocol.json').read_text());pre=json.loads((out/'preflight_audit.json').read_text())
    assert pre['all_checks_passed'] and plan['source_sha256']==source_hashes()
    assert sha(DATA/'raw_windows.npz')==plan['raw_sha256']
    raw,tr,te=load_data();stats=fit_stats(raw,tr);labels=raw['labels'][te]
    (out/'PAUSED.json').unlink(missing_ok=True)
    started=time.monotonic();start_iso=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
    def pause():
        if (out/'STOP_REQUESTED').exists():
            save_json(out/'PAUSED.json',dict(time=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),atomic_items_saved=True))
            raise InterruptedError('E12 safely paused')
    cache=out/'waveform_cache';cache.mkdir(exist_ok=True)
    for ci,cond in enumerate(CONDITIONS):
        if cond['source']!='new':continue
        pause();dest=cache/f'{ci:03}.npz'
        if not dest.exists():
            array,c,v=image_array(raw,te,stats,cond)
            atomic_npz(dest,image=array,current=c,voltage=v,sample_ids=raw['sample_ids'][te],labels=labels)
        save_json(out/'progress.json',dict(stage='waveform_and_image_cache',completed_conditions=ci+1,new_conditions=66))
    old_conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
    manifest=plan['manifest'];total=len(manifest)
    for ri,entry in enumerate(manifest):
        pause();folder=PREVIOUS/entry['path'];key=Path(entry['path']).name;dest=out/'predictions'/f'{key}.npz'
        if dest.exists():continue
        for name,digest in entry['file_sha256'].items():assert sha(folder/name)==digest,(key,name)
        tick=time.monotonic();model,spec,state=load_model(entry,device)
        assert spec['normalization']==stats
        assert fingerprint(state['state_dict'])==entry['state_sha256']
        before=fingerprint(model.state_dict());predictions=[]
        for ci,cond in enumerate(CONDITIONS):
            pause()
            if cond['source']=='E4':
                i=old_conditions.index([cond['kind'],cond['value'],cond['pollution_seed']])
                path=folder/'outer_predictions'/f'{i:03}.npy'
                assert sha(path)==entry['component_prediction_sha256'][str(ci)]
                pred=np.load(path)
            else:
                cachefile=cache/f'{ci:03}.npz'
                array=np.load(cachefile)['image']
                pred=predict(model,array,entry['model'],device)
            assert pred.shape==(219,) and np.isin(pred,np.arange(11)).all()
            if ci==0:
                assert np.array_equal(pred,np.load(folder/'outer_predictions/clean.npy')),'Original clean prediction mismatch: '+key
            predictions.append(pred)
            # Save one inference item as an atomic row; allows bounded pause/resume.
            item=out/'partial'/key/f'{ci:03}.npy'
            item.parent.mkdir(parents=True,exist_ok=True)
            if cond['source']=='new' and ci>0:
                # Existing partial results are verified against fresh prediction on resume.
                if item.exists():assert np.array_equal(np.load(item),pred)
                else:
                    with item.with_suffix('.tmp').open('wb') as stream:np.save(stream,pred.astype(np.int16))
                    os.replace(item.with_suffix('.tmp'),item)
            if ci%10==0:
                save_json(out/'progress.json',dict(stage='inference',completed_models=ri,planned_models=total,current_model=key,
                    current_condition=ci,conditions_per_model=126,elapsed_seconds=time.monotonic()-started))
        assert before==fingerprint(model.state_dict()),'Weights or BatchNorm mutated'
        atomic_npz(dest,predictions=np.stack(predictions).astype(np.int16),sample_ids=raw['sample_ids'][te],labels=labels,
                   condition_index=np.arange(len(CONDITIONS)),model=np.array(entry['model']),strategy=np.array(entry['strategy']),seed=np.array(entry['seed']))
        save_json(out/'progress.json',dict(stage='inference',completed_models=ri+1,planned_models=total,current_model=key,
            seconds_last_model=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started))
        print('E12',ri+1,'/',total,key,round(time.monotonic()-tick,2),'seconds',flush=True)
        del model,state;gc.collect();torch.mps.empty_cache()
    save_json(out/'inference_audit.json',dict(all_checks_passed=True,models=total,conditions_per_model=126,
        predictions=total*126*219,new_inference_predictions=total*66*219,reused_component_predictions=total*60*219,
        clean_matches_E4=True,weights_bn_unchanged=True,started_at=start_iso,
        ended_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),elapsed_seconds=time.monotonic()-started))
    save_json(out/'progress.json',dict(stage='analysis',completed_models=total,planned_models=total))
    from analyze_e12 import analyze
    analyze(out)
    save_json(out/'progress.json',dict(stage='research_complete',completed_models=total,planned_models=total))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);args=p.parse_args();out=args.output.resolve()
    with (out/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(out)
        except InterruptedError:print('E12 safely paused.',flush=True)
        except Exception:
            save_json(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
