"""Reentrant, locked E6 inference. No training and no outcome-driven selection."""
import argparse
import fcntl
import gc
import json
import os
import shutil
import time
import traceback
import numpy as np
import torch
from e6_mechanism_core import *

class Paused(Exception):pass

def freeze(output,raw,test):
    assert json.loads((PREVIOUS/'completed_audit.json').read_text())['all_checks_passed']
    sources=source_hashes()
    checkpoints={str((PREVIOUS/'final'/f'Yu3_combined_s{s}'/'checkpoint.pt').relative_to(ROOT)):
                 sha(PREVIOUS/'final'/f'Yu3_combined_s{s}'/'checkpoint.pt') for s in SEEDS}
    plan=dict(study='E6 Yu3 Combined inference-only mechanism diagnostics',version=1,
        seeds=SEEDS,conditions=CONDITIONS,planned_items=310,test_ids=raw['sample_ids'][test].tolist(),
        previous_audit_sha256=sha(PREVIOUS/'completed_audit.json'),raw_sha256=sha(DATA/'raw_windows.npz'),
        checkpoints_sha256=checkpoints,source_sha256=sources,
        disclosure='Planned after E5 and E1-E4 results were seen. No retraining, no hyperparameter/strategy changes, no causal identification claim.',
        mask='Synchronous 100-of-500 missing segment. Union of original and interpolated segment VI locations, square one-pixel dilation. Control: fixed keyed random equal-length nonmissing temporal segment; spatial overlap retained and reported. Energy=sum squared absolute residual over all 4096 pixels; zero denominator is NaN.',
        pairing='219 test samples, 10 pollution realizations within each model seed; clean computed once. Full-test derangement independent of batching, same pairing across model seeds.',
        representatives='Lowest lexicographic sample ID per class; model seed42/pollution1000. First wrong-to-right and right-to-wrong by condition order, pollution seed, sample ID; both directions shown if available.',
        limits='Test-time sensitivity only. Pixel mask is a projected proxy with overlap; residual energy is not direct proof of recovery or attribution. No significance/causality inferred from pooled samples.',
        normalization='Checkpoint training-only fixed statistics, independently recomputed; no test fit.')
    target=output/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==json.loads(json.dumps(plan)),'E6 frozen source/input mismatch'
    else:
        save_json(target,plan)
        for relative in sources:
            dest=output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    return plan

def run(output):
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available()
    device=torch.device('mps');raw,test=load_data();plan=freeze(output,raw,test)
    report=json.loads((output/'preflight_audit.json').read_text())
    assert report['all_checks_passed'] and report['source_sha256']==plan['source_sha256']
    (output/'items').mkdir(exist_ok=True)
    (output/'STOP_REQUESTED').unlink(missing_ok=True);(output/'PAUSED.json').unlink(missing_ok=True)
    total=len(SEEDS)*len(CONDITIONS);started=time.monotonic()
    for seed in SEEDS:
        model,spec,epoch=model_for(seed,device)
        state_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        assert spec['normalization']==fit_stats(raw,np.flatnonzero(raw['partitions']=='train'))
        previous=np.load(PREVIOUS/'final'/f'Yu3_combined_s{seed}'/'outer_ids_labels.npz')
        assert np.array_equal(previous['sample_ids'],raw['sample_ids'][test])
        for condition in CONDITIONS:
            dest=output/'items'/f's{seed}_{key(condition)}.npz'
            if (output/'STOP_REQUESTED').exists():
                save_json(output/'PAUSED.json',dict(next_item=dest.name,completed=len(list((output/'items').glob('*.npz')))))
                raise Paused()
            if dest.exists():continue
            tick=time.monotonic();typ,severity,ps=condition
            array=images(raw,test,spec['normalization'],condition=None if typ=='clean' else condition)
            l1,recon=first_pass(model,array,device);residual=np.abs(array-recon)
            order=derangement(raw['sample_ids'][test],condition)
            ln=second_pass(model,array,residual,device)
            lz=second_pass(model,array,np.zeros_like(residual),device)
            ls=second_pass(model,array,residual[order],device)
            # Require matching the already-audited E4 normal-feedback predictions.
            if typ=='clean':prior=PREVIOUS/'final'/f'Yu3_combined_s{seed}'/'outer_predictions/clean.npy'
            else:
                conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
                index=conditions.index(list(condition))
                prior=PREVIOUS/'final'/f'Yu3_combined_s{seed}'/'outer_predictions'/f'{index:03}.npy'
            assert np.array_equal(ln.argmax(1),np.load(prior)),f'E4 mismatch: {dest.name}'
            arrays=dict(sample_ids=raw['sample_ids'][test],labels=raw['labels'][test],
                        logits_pass1=l1,logits_normal=ln,logits_zero=lz,logits_shuffle=ls,
                        donor_indices=order,condition=np.array([typ,str(severity),str(ps)]),model_seed=np.array(seed))
            if typ=='missing_consecutive':
                regions=region_masks(raw,test,spec['normalization'],ps)
                arrays.update(regions);arrays.update(energy_rows(residual,regions));arrays['residual']=residual
            if seed==42:arrays.update(image=array,reconstruction=recon,residual=residual)
            atomic_npz(dest,**arrays)
            completed=len(list((output/'items').glob('*.npz')))
            save_json(output/'progress.json',dict(stage='E6_inference',completed_items=completed,planned_items=total,
                      model='Yu3',strategy='combined',seed=seed,checkpoint_epoch=epoch,condition=condition,
                      seconds_last_item=time.monotonic()-tick,elapsed_seconds=time.monotonic()-started))
            print('SAVED',dest.name,completed,'/',total,flush=True)
        assert all(torch.equal(state_before[k],v.detach().cpu()) for k,v in model.state_dict().items()),'Weights/BN mutated'
        del model;gc.collect();torch.mps.empty_cache()
    save_json(output/'inference_audit.json',dict(all_checks_passed=True,items=total,normal_predictions_match_E4=True,weights_bn_unchanged=True))
    save_json(output/'progress.json',dict(stage='E6_analysis_and_figures',completed_items=total,planned_items=total))
    from analyze_e6_mechanism import analyze
    analyze(output)
    save_json(output/'progress.json',dict(stage='E6_core_complete_document_pending',completed_items=total,planned_items=total))
    print('E6 core complete; Word review and final delivery audit pending.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(output)
        except Paused:print('E6 safely paused after saved item.',flush=True)
        except Exception:
            save_json(output/'FAILED.json',dict(traceback=traceback.format_exc(),last_progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else None))
            raise

if __name__=='__main__':main()
