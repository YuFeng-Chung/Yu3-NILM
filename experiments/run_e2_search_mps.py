"""E2 full 18-setting/model search, inner group validation only; resumable CPU run."""
import argparse
import csv
import itertools
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from revision_models_mps import MODEL_NAMES,build_model,final_logits,training_loss,architecture_smoke
from run_exp4_matched_controls import ROOT,DATA,images,sha,save_json
from run_e5_ablation import metrics,seed_all


def inner_split(raw):
    """Fixed class-stratified group split, no outer-test outcomes consulted."""
    outer=np.flatnonzero(raw['partitions']=='train')
    rng=np.random.default_rng(42); validation=[]
    for label in range(11):
        indices=outer[raw['labels'][outer]==label]
        groups,counts=np.unique(raw['groups'][indices],return_counts=True)
        assert len(groups)>=2
        number=max(1,min(len(groups)-1,int(round(.2*len(groups)))))
        candidates=[rng.choice(len(groups),number,replace=False) for _ in range(2000)]
        best=min(candidates,key=lambda c:abs(counts[c].sum()-.2*len(indices)))
        validation.extend(indices[np.isin(raw['groups'][indices],groups[best])].tolist())
    val=np.array(sorted(validation)); train=np.setdiff1d(outer,val)
    test=np.flatnonzero(raw['partitions']=='val')
    for left,right in [(train,val),(train,test),(val,test)]:
        assert not set(raw['groups'][left])&set(raw['groups'][right])
    assert set(raw['labels'][train])==set(range(11))==set(raw['labels'][val])
    assert set(train)|set(val)==set(outer)
    return train,val,test


def predict(model,x,name):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([final_logits(model,b,name).argmax(1).cpu().numpy() for b in x.split(32)])


def summarize(output):
    rows=[json.loads(p.read_text()) for p in output.glob('*/result.json')]
    if rows:
        with (output/'search_results.tmp').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (output/'search_results.tmp').replace(output/'search_results.csv')
    selected={}
    for name in MODEL_NAMES:
        subset=[r for r in rows if r['model']==name]
        if len(subset)==18:
            # Tie: earlier grid index, fixed before results. No held-out scores.
            selected[name]=max(subset,key=lambda r:(r['validation_macro_f1'],-r['grid_index']))
    save_json(output/'selected_settings.json',selected)
    save_json(output/'progress.json',dict(completed=len(rows),planned=72,
                                         models_with_complete_search=list(selected)))
    lines=['# E2：公平架構與同等調參', '',f'完成 {len(rows)}/72 個搜尋run。',
           '僅使用外層訓練集內的群組驗證；未評估外層測試集。單seed搜尋不作效果顯著性推論。',
           '', '| 模型 | 已完成設定 | 最佳validation F1% | lr | weight decay | dropout | 回合 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in MODEL_NAMES:
        count=sum(r['model']==name for r in rows)
        if name in selected:
            r=selected[name];lines.append(f'| {name} | {count}/18 | {r["validation_macro_f1"]*100:.2f} | {r["lr"]} | {r["weight_decay"]} | {r["dropout"]} | {r["best_epoch"]} |')
        else: lines.append(f'| {name} | {count}/18 | 搜尋未完成 | — | — | — | — |')
    lines+=['', '此處的回合只屬No-Defense pilot；各防禦策略回合仍須在內層validation重新決定。',
            '達到最大回合的run另列hit_max_epochs，可能尚未收斂，不隱藏。選定設定不保證外層test改善。',
            '本階段不自動啟動E1/E3，避免CPU競爭；模型及設定選擇只依本protocol執行。']
    (output/'search_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-only',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-epochs',type=int,default=150)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--resume',action='store_true',help='Explicitly clear an acknowledged pause request')
    p.add_argument('--expected-split',type=Path,default=ROOT/'mac_handoff_20261106/expected_e2_split.json')
    args=p.parse_args();assert args.max_epochs>=50
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable: do not silently fall back to CPU')
    device=torch.device(args.device)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True,warn_only=True)
    raw=dict(np.load(DATA/'raw_windows.npz'));train,val,test=inner_split(raw)
    stats={}
    for prefix,column in [('i','current'),('v','voltage')]:
        values=raw[column][train].astype(np.float32)
        for suffix,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:
            stats[f'{prefix}_{suffix}']=float(fn(values))
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(split_seed=42,algorithm='class-stratified group subsets; 2000 candidates/class; closest sample count to 20%; first tie',
        train_ids=raw['sample_ids'][train].tolist(),validation_ids=raw['sample_ids'][val].tolist(),
        outer_test_ids_audit_only=raw['sample_ids'][test].tolist(),
        train_groups=sorted(set(raw['groups'][train].tolist())),
        validation_groups=sorted(set(raw['groups'][val].tolist())),
        outer_test_groups_audit_only=sorted(set(raw['groups'][test].tolist())),
        counts=dict(train=len(train),validation=len(val),outer_test_not_evaluated=len(test)),
        normalization_fit_ids=raw['sample_ids'][train].tolist(),normalization=stats)
    expected=json.loads(args.expected_split.read_text(encoding='utf-8'))
    for key in ['train_ids','validation_ids','outer_test_ids_audit_only','normalization_fit_ids']:
        assert manifest[key]==expected[key],f'Fixed split changed: {key}'
    for key,value in stats.items():
        assert np.isclose(value,expected['normalization'][key],rtol=2e-6,atol=1e-7),key
    source=[Path(__file__),Path(__file__).with_name('revision_models_mps.py'),
        Path(__file__).with_name('run_e5_ablation.py'),Path(__file__).with_name('run_exp4_matched_controls.py'),
        ROOT/'實驗一/models_m1_to_m5.py',ROOT/'實驗四/exp4_raw_data_V2.py',
        ROOT/'實驗四/exp4_defense_strategies_V2.py',DATA/'raw_windows.npz']
    grid=[dict(lr=lr,weight_decay=wd,dropout=dropout) for lr,wd,dropout in
          itertools.product([1e-4,5e-4,1e-3],[0.,1e-4,1e-3],[.2,.5])]
    plan=dict(experiment='E2',models=MODEL_NAMES,grid=grid,seed=42,batch_size=32,
        optimizer='Adam',strategy='No-Defense',device=args.device,max_epochs=args.max_epochs,min_epochs=30,
        patience=20,min_delta=.001,eligible_checkpoint='epochs>=30; highest validation F1; earliest tie',
        early_stopping='patience counter starts after epoch30; improvement over anchor by >0.001 resets; no counting before30',
        selection='all18 completed per model; highest eligible validation F1; earlier grid index tie',
        normalization='fit pilot training waveforms only, fixed for validation; no external stats reused',
        split_sha256=__import__('hashlib').sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},
        environment=dict(torch=str(torch.__version__),python=platform.python_version(),platform=platform.platform(),threads=args.threads),
        limits='single-seed HPO; one fixed inner split; outer test previously viewed in other experiments but not used by this selection; device/environment differs from original CUDA; max150 may censor convergence')
    smoke=architecture_smoke(device);save_json(args.output/'architecture_checks.json',smoke)
    save_json(args.output/'split_manifest.json',manifest)
    print(f'PASS architecture/gradient checks; pilot train{len(train)} val{len(val)}, outer test{len(test)} NOT EVALUATED',flush=True)
    if args.smoke_only:return
    if (args.output/'protocol.json').exists(): assert json.loads((args.output/'protocol.json').read_text())==json.loads(json.dumps(plan))
    else:save_json(args.output/'protocol.json',plan)
    stop_file=args.output/'STOP_REQUESTED'
    if args.resume:
        stop_file.unlink(missing_ok=True)
        (args.output/'PAUSED.json').unlink(missing_ok=True)
    if stop_file.exists():
        print('Pause request present; use --resume explicitly to continue.',flush=True);return
    for f in source:
        if f.suffix=='.npz':continue
        dest=args.output/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(f,dest)
    x=torch.from_numpy(images(raw,train,stats)).to(device);v=torch.from_numpy(images(raw,val,stats)).to(device)
    y=torch.from_numpy(raw['labels'][train]).long().to(device);vy=raw['labels'][val]
    rows=summarize(args.output)
    for name in MODEL_NAMES:
        for index,config in enumerate(grid):
            if stop_file.exists():
                save_json(args.output/'PAUSED.json',dict(status='paused',next_model=name,next_grid=index))
                print('PAUSED at run boundary; checkpoints retained.',flush=True);return
            folder=args.output/f'E2_{name}_grid{index:02}_s42';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():continue
            seed_all(42);model=build_model(name,config['dropout']).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
            state=dict(epoch=0,logs=[],best_f1=-1.,best_epoch=None,anchor=-1.,stale=0,stopped=False)
            if (folder/'resume.pt').exists():
                saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(saved.pop('model'));optimizer.load_state_dict(saved.pop('optimizer'));state=saved
            for epoch in range(state['epoch']+1,args.max_epochs+1):
                if state['stopped']:break
                seed_all(42*10000+epoch);model.train();total=0.;tick=time.monotonic()
                order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42*10000+epoch))
                for idx in order.split(32):
                    idx=idx.to(device)
                    optimizer.zero_grad(set_to_none=True);loss=training_loss(model,x[idx],y[idx],name)
                    assert torch.isfinite(loss);loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
                pred=predict(model,v,name);score=metrics(vy,pred)['macro_f1']
                state['epoch']=epoch
                state['logs'].append(dict(epoch=epoch,train_loss=total/len(train),validation_macro_f1=score,seconds=time.monotonic()-tick))
                if epoch>=30 and score>state['best_f1']:
                    state['best_f1']=score;state['best_epoch']=epoch
                    torch.save(dict(state_dict=model.state_dict(),epoch=epoch,model=name,config=config),folder/'best.tmp')
                    os.replace(folder/'best.tmp',folder/'best.pt')
                    np.savez_compressed(folder/'best_validation_predictions.npz',sample_ids=raw['sample_ids'][val],labels=vy,predictions=pred)
                if epoch==30:state['anchor']=score;state['stale']=0
                elif epoch>30:
                    if score>state['anchor']+.001:state['anchor']=score;state['stale']=0
                    else:state['stale']+=1
                state['stopped']=epoch>=30 and state['stale']>=20
                torch.save(dict(**state,model=model.state_dict(),optimizer=optimizer.state_dict()),folder/'resume.tmp')
                os.replace(folder/'resume.tmp',folder/'resume.pt')
                if stop_file.exists():
                    save_json(args.output/'PAUSED.json',dict(status='paused',model=name,grid=index,epoch=epoch))
                    print(f'PAUSED after saved epoch {epoch}; safe to close lid.',flush=True);return
                with (folder/'train_log.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(state['logs'][0]));writer.writeheader();writer.writerows(state['logs'])
                if epoch==1 or epoch%10==0 or state['stopped']:
                    print(f'{name} grid{index:02} {config} epoch{epoch}/{args.max_epochs} valF1={score:.4f} patience={state["stale"]}/20 {state["logs"][-1]["seconds"]:.2f}s',flush=True)
            checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'])
            saved=np.load(folder/'best_validation_predictions.npz')
            assert np.array_equal(saved['sample_ids'],raw['sample_ids'][val])
            assert np.array_equal(saved['labels'],vy)
            assert np.array_equal(predict(model,v,name),saved['predictions'])
            assert metrics(vy,saved['predictions'])['macro_f1']==state['best_f1']
            result=dict(model=name,grid_index=index,seed=42,**config,best_epoch=state['best_epoch'],
                validation_macro_f1=state['best_f1'],stopped_epoch=state['epoch'],
                hit_max_epochs=state['epoch']==args.max_epochs and not state['stopped'],prediction_audit_passed=True)
            save_json(folder/'result.json',result);(folder/'resume.pt').unlink()
            rows=summarize(args.output)
            if args.device=='mps':
                del model,optimizer
                import gc
                gc.collect();torch.mps.empty_cache()
            print(f'COMPLETE {len(rows)}/72 {name} grid{index:02} best={state["best_f1"]:.4f} epoch{state["best_epoch"]}',flush=True)
    save_json(args.output/'completed_audit.json',dict(all_checks_passed=True,search_runs=len(rows),
        held_out_evaluations=0,checks='group split disjoint; training-only norm; all logs/checkpoint predictions recomputed per run'))
    print('E2 SEARCH COMPLETE; E1/E3 require separate strategy-specific epoch pilots',flush=True)


if __name__=='__main__':main()
