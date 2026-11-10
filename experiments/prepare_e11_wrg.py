"""Freeze E11 before search; train-only correctness, integration and schedule preflight."""
import argparse
import platform
import shutil
import tempfile
import time
from e11_wrg_core import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    assert not (output/'protocol.json').exists(),'Protocol already frozen; do not overwrite.'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=True)
    assert torch.backends.mps.is_available();device=torch.device('mps')
    raw=load_raw();t,v,test=inner_split(raw);outer=np.flatnonzero(raw['partitions']=='train')
    saved_split=json.loads((E2/'split_manifest.json').read_text())
    assert raw['sample_ids'][t].tolist()==saved_split['train_ids']
    assert raw['sample_ids'][v].tolist()==saved_split['validation_ids']
    assert raw['sample_ids'][test].tolist()==saved_split['outer_test_ids_audit_only']
    cv=previous.folds(raw);saved_cv=json.loads((OLD/'E3_folds.json').read_text())
    for i,(ct,cvv) in enumerate(cv):
        assert raw['sample_ids'][ct].tolist()==saved_cv[i]['train_ids']
        assert raw['sample_ids'][cvv].tolist()==saved_cv[i]['validation_ids']
    reuse=[E2/'selected_settings.json',E2/'protocol.json',E2/'split_manifest.json',E2/'completed_audit.json',
           OLD/'E3_selected_strategies.json',OLD/'E3_folds.json',OLD/'protocol_v4.json',
           OLD/'completed_audit.json',OLD/'evaluation_conditions.json']
    assert json.loads((OLD/'completed_audit.json').read_text())['all_checks_passed']
    assert json.loads((E2/'completed_audit.json').read_text())['all_checks_passed']
    oldplan=json.loads((OLD/'protocol_v4.json').read_text())
    assert oldplan['seeds']==SEEDS
    assert json.loads((OLD/'evaluation_conditions.json').read_text())==[list(c) for c in CONDITIONS]
    oldrawkey='data/plaid_2014_published/raw_windows.npz'
    assert oldplan['source_sha256'][oldrawkey]==engine.sha(engine.DATA/'raw_windows.npz')
    # Freeze all old Yu3 E2/CV/pilot/final evidence actually reused, including every prediction.
    for pat in ['E2_Yu3_*/result.json','E2_Yu3_*/best_validation_predictions.npz']:
        reuse+=sorted(E2.glob(pat))
    for parent,pat in [('cv','Yu3_*'),('pilots','Yu3_*'),('final','Yu3_*')]:
        for folder in sorted((OLD/parent).glob(pat)):
            reuse += [f for f in sorted(folder.rglob('*')) if f.is_file()]
    reuse=list(dict.fromkeys(reuse))
    # Audit source equivalence for imported historical implementation and models.
    for f in source_paths():
        rel=str(f.relative_to(ROOT))
        if rel in oldplan['source_sha256']:
            assert engine.sha(f)==oldplan['source_sha256'][rel],rel
    stats=engine.fit_stats(raw,t);checks=0
    # No outer-test values evaluated here; compare entire native feature to author code.
    sample=t[np.linspace(0,len(t)-1,33,dtype=int)]
    for condition in [None,('awgn',0,1000),('salt_pepper',.2,1000),('missing_consecutive',.5,1000)]:
        for c,w in waveforms(raw,sample,stats,condition=condition):
            expected=author_feature.get_weighted_reccurrence_graph(
                author_feature.create_distance_similarity_matrix(c,50,1),10,20).numpy()
            np.testing.assert_array_equal(native_wrg(c),expected);checks+=1
    # Shared corruption/augmentation parity: reconstruct historical VI images from exact same waves.
    for recipe in ['no_defense','S3','combined']:
        rebuilt=np.stack([engine.scatter_fast(c,w,stats) for c,w in waveforms(raw,sample,stats,recipe,42,1)])
        np.testing.assert_array_equal(rebuilt,ORIGINAL_IMAGES(raw,sample,stats,recipe,42,1))
    for strategy in ['S0','S1','S2']:
        condition=('awgn',0,1000)
        rebuilt=np.stack([engine.scatter_fast(c,w,stats) for c,w in waveforms(raw,sample,stats,condition=condition,filter_strategy=strategy)])
        np.testing.assert_array_equal(rebuilt,previous.evaluation_images(raw,sample,stats,condition,strategy))
    install();timings=[]
    with tempfile.TemporaryDirectory(prefix='yu3-e11-schedule-') as tmp:
        temp=Path(tmp)
        # Full outer-training size, two epochs per recipe; no held-out data and no parameter choice.
        for recipe in ['no_defense','S3','combined']:
            cfg=dict(lr=.001,weight_decay=0.0001 if recipe=='combined' else 0.,dropout=.5)
            start=time.monotonic()
            r=run_train(temp,temp/recipe,raw,outer,np.array([],dtype=int),recipe,42,cfg,device,fixed_epochs=2)
            timings.append(dict(recipe=recipe,samples=855,epochs=2,epoch_seconds=r['training_seconds']/2,
                                wall_seconds=time.monotonic()-start,reload=r['prediction_reload_audit']))
    # Evaluation timing uses 219 TRAIN samples as a shape proxy, not held-out test.
    model=build('WRG-CNN',.5).to(device);evaluation_timings=[]
    for strategy in ['S0','S1','S2']:
        start=time.monotonic()
        for condition in [None,('awgn',0,1000),('missing_consecutive',.5,1000)]:
            x=evaluation_images(raw,t[:219],stats,condition,strategy)
            pred=engine.predict(model,x,'WRG-CNN',device);assert len(pred)==219
        evaluation_timings.append(dict(strategy=strategy,seconds_per_219=time.monotonic().__sub__(start)/3))
    del model;torch.mps.empty_cache()
    # Transparent estimate, not a model/performance decision: 54 selecting fits + 30-60 finals.
    epoch=float(np.mean([r['epoch_seconds'] for r in timings]))
    ev=next(r['seconds_per_219'] for r in evaluation_timings if r['strategy']=='S0')
    central=54*85*epoch+40*65*epoch+(30*90+60*211)*ev
    worst=(54+60)*150*epoch+(30*90+80*211)*max(r['seconds_per_219'] for r in evaluation_timings)
    preflight=dict(all_checks_passed=True,author_feature_exact_comparisons=checks,
        waveform_augmentation_and_filter_parity=True,full_size_timings=timings,evaluation_timings=evaluation_timings,
        estimated_central_compute_hours=central/3600,estimated_conservative_compute_hours=worst/3600,
        assumption='Central: 54 selecting fits x85 epochs, 40 finals x65; max54+60 fits x150. Central evaluation assumes 60 strategy labels including shared weights. Filesystem/checkpoint/caching variation and report QA extra; no finish-time guarantee.',
        held_out_evaluations=0)
    engine.save_json(output/'preflight_audit.json',preflight)
    plan=dict(experiment='E11 Yu3 vs WRG-CNN',upstream_commit='3e5879d9d1a9c4c1ef93c557a650454b1581ac87',
        grid=GRID,seeds=SEEDS,pollution_seeds=POLLUTION,severity_grids=GRIDS,
        hpo=dict(runs=18,seed=42,optimizer='Adam',batch_size=32,min_epochs=30,max_epochs=150,patience=20,min_delta=.001,
                 selection='highest eligible inner-validation Macro-F1; earliest grid-index tie'),
        strategy_cv=dict(folds=5,recipes=RECIPES,seed=42,min_epochs=30,max_epochs=150,patience_no_defense=20,patience_other=30,
                         rule='existing author-confirmed v4, retain all ties',noise_objective='AWGN0dB/SP20%, ten pollution seeds',
                         missing_objective='random20%/consecutive20%, ten pollution seeds'),
        finals='All 3 common recipes No-Defense/S3/Combined plus all CV co-selected strategies; ten seeds each; identical no-defense weights reused for S0/S1/S2 filters, every label retained.',
        epochs='each unique training recipe gets inner-split pilot at selected HPO, min30/max150, patience20 no-defense else30; fixed pilot-best budget on all855',
        comparison='native VI Yu3 vs native current-only WRG+Cnn under common 500-point waveform protocol; system comparison, not architecture-only or isolated feedback effect',
        adaptations=['WRG original multi-cycle extraction replaced by shared500-point windows',
          'native author mean-PAA/current-only50x50/eps10/delta20 retained; paper median discrepancy disclosed',
          'CNN original shape retained; dropout follows same18-grid .2/.5 rather than author .25',
          'Adam/batch32/epoch rules match Yu3 rather than original SGD/batch16',
          'no extra representation-HPO, optimizer candidate, or test-selected settings for either model'],
        analysis=dict(primary='descriptive paired-seed Macro-F1 and Dabs/Drel across all strengths; all co-selected pairs; no confirmatory significance claim',
          seed_unit='training seed; average10 pollution realizations inside each seed; test samples fixed and seeds do not add independent households',
          descriptive_intervals='paired training-seed bootstrap only, conditional on this historical split; not population confidence'),
        limits=['original test previously viewed; this extension is not a fresh held-out confirmation',
                'groups are class x household proxies, not verified unique devices or whole-house isolation',
                'single inner-split single-seed HPO and non-nested strategy selection',
                'WRG2020 alone does not resolve recent-method reviewer request',
                'not original-paper numerical replication; MacMPS, not studentCUDA'],
        source_sha256={str(f.relative_to(ROOT)):engine.sha(f) for f in source_paths()},
        reuse_sha256={str(f.relative_to(ROOT)):engine.sha(f) for f in reuse},
        raw_sha256=engine.sha(engine.DATA/'raw_windows.npz'),
        environment=dict(torch=str(torch.__version__),platform=platform.platform(),device='mps',threads=4),
        frozen_before='all WRG HPO/CV/epoch selection/outer evaluation; preflight train-only',preflight=preflight)
    engine.save_json(output/'protocol.json',plan)
    engine.save_json(output/'split_manifest.json',saved_split)
    engine.save_json(output/'E3_folds.json',saved_cv)
    for f in source_paths():
        dest=output/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    print(json.dumps(preflight,indent=2),flush=True)


if __name__=='__main__':main()
