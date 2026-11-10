"""Resumable exclusive E11 writer: HPO -> five-fold strategy choice -> final -> audit."""
import argparse
import fcntl
import shutil
import traceback
from e11_wrg_core import *


def stage(output,name,**kw):
    engine.save_json(output/'progress.json',dict(stage=name,**kw));print('STAGE',name,kw,flush=True)


def run(output):
    plan=json.loads((output/'protocol.json').read_text());assert_sources(plan)
    assert plan['preflight']['all_checks_passed']
    install();torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available();device=torch.device('mps')
    raw=load_raw();t,v,test=inner_split(raw);outer=np.flatnonzero(raw['partitions']=='train');empty=np.array([],dtype=int)
    rows=[]
    stage(output,'HPO',completed=0,planned=18)
    for i,cfg in enumerate(GRID):
        result=run_train(output,output/'hpo'/f'grid{i:02}',raw,t,v,'no_defense',42,cfg,device,patience=20)
        rows.append(dict(grid_index=i,**result,**cfg))
        engine.csv_write(output/'hpo_results.csv',rows);stage(output,'HPO',completed=len(rows),planned=18)
    selected=max(rows,key=lambda r:(r['validation_macro_f1'],-r['grid_index']))
    engine.save_json(output/'selected_setting.json',selected)
    # A complete old Yu3 search is reused, not reranked using WRG or held-out outcomes.
    cfgbase={k:selected[k] for k in ['lr','weight_decay','dropout']}
    results={};all_scores=[]
    for fold,(ct,cv) in enumerate(previous.folds(raw)):
        for recipe in RECIPES:
            stage(output,'strategy_cv_training',fold=fold,recipe=recipe,completed_training_runs=fold*6+RECIPES.index(recipe),planned=30)
            run_train(output,output/'cv'/f'fold{fold}_{recipe}',raw,ct,cv,recipe,42,
                      config_for(cfgbase,recipe),device,patience=20 if recipe=='no_defense' else 30)
        for strategy in NOISE+MISSING:
            scores=previous.cv_evaluate(output,output/'cv'/f'fold{fold}_{recipe_for(strategy)}',raw,cv,'WRG-CNN',strategy,device)
            results[fold,strategy]=scores
            all_scores.append(dict(model='WRG-CNN',fold=fold,strategy=strategy,
                  **{k:scores.get(k) for k in GRIDS}))
            engine.csv_write(output/'cv_scores.csv',all_scores)
    choice=select_from_scores('WRG-CNN',results)
    engine.save_json(output/'selected_strategies.json',choice)
    shutil.rmtree(output/'image_cache/cv',ignore_errors=True)
    labels=list(dict.fromkeys(['no_defense','S3','combined']+choice['noise']+choice['missing']))
    recipes=list(dict.fromkeys(recipe_for(s) for s in labels));epoch_map={}
    for recipe in recipes:
        stage(output,'epoch_pilots',recipe=recipe,completed=len(epoch_map),planned=len(recipes))
        r=run_train(output,output/'pilots'/recipe,raw,t,v,recipe,42,config_for(cfgbase,recipe),device,
                    patience=20 if recipe=='no_defense' else 30)
        epoch_map[recipe]=r['selected_epoch']
    engine.save_json(output/'epoch_settings.json',epoch_map)
    completed=0
    for recipe in recipes:
        for seed in SEEDS:
            stage(output,'final_training',recipe=recipe,seed=seed,completed=completed,planned=len(recipes)*10)
            run_train(output,output/'final'/f'WRG-CNN_{recipe}_s{seed}',raw,outer,empty,recipe,seed,
                      config_for(cfgbase,recipe),device,fixed_epochs=epoch_map[recipe])
            completed+=1
    manifest=[]
    for strategy in labels:
        recipe=recipe_for(strategy)
        for seed in SEEDS:
            folder=output/'final'/f'WRG-CNN_{strategy}_s{seed}'
            if strategy!=recipe:
                # Share immutable no-defense weights; keep every strategy's own evaluation output.
                folder.mkdir(parents=True,exist_ok=True)
                source=output/'final'/f'WRG-CNN_{recipe}_s{seed}'
                for name in ['run_spec.json','result.json','train_log.csv','checkpoint.pt']:
                    dest=folder/name
                    if not dest.exists():shutil.copy2(source/name,dest)
                    else:assert engine.sha(dest)==engine.sha(source/name)
                engine.save_json(folder/'shared_training.json',dict(source=str(source.relative_to(output)),recipe=recipe,strategy=strategy))
            manifest.append(dict(model='WRG-CNN',strategy=strategy,recipe=recipe,seed=seed,path=str(folder.relative_to(output))))
    engine.save_json(output/'final_run_manifest.json',manifest)
    # This gate is written only after ALL selection and training are complete.
    gate=dict(all_selection_complete=True,outer_test_evaluations_so_far=0,selected_setting=selected,
              choice=choice,epochs=epoch_map,planned_strategy_labels=len(manifest),trained_checkpoints=completed)
    target=output/'selection_complete.json'
    if target.exists():assert json.loads(target.read_text())==gate
    else:engine.save_json(target,gate)
    engine.save_json(output/'evaluation_conditions.json',CONDITIONS)
    for i,entry in enumerate(manifest):
        stage(output,'outer_evaluation',completed=i,planned=len(manifest),strategy=entry['strategy'],seed=entry['seed'])
        previous.evaluate_outer(output,raw,output/entry['path'],'WRG-CNN',entry['strategy'],device,CONDITIONS)
    stage(output,'analysis_and_independent_audit')
    from analyze_e11_wrg import analyze
    analyze(output)
    stage(output,'complete',trained_checkpoints=completed,evaluated_strategy_labels=len(manifest))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--resume',action='store_true');a=p.parse_args()
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'completed_audit.json').exists():print('Already complete; not restarting.');return
        if a.resume:
            (output/'STOP_REQUESTED').unlink(missing_ok=True);(output/'PAUSED.json').unlink(missing_ok=True)
            (output/'FAILED.json').unlink(missing_ok=True)
        engine.pause_check(output,'start',None)
        try:run(output)
        except engine.Paused:pass
        except Exception as exc:
            engine.save_json(output/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
