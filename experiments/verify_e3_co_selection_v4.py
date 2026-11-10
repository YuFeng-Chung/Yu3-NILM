"""Co-selection amendment checks; never opens current outer-test predictions."""
import copy
import inspect
import json
from pathlib import Path
import numpy as np
import torch
import revision_strategy_selection_v4 as rule
import revision_followup_core as core
import run_revision_followup_v4 as runner


def verify(output):
    assert not list((output/'final').glob('*/evaluation.json'))
    assert runner.train is core.train
    fixture={(f,s):dict(awgn=.5,salt_pepper=.5,missing_random=.6,missing_consecutive=.6) for f in range(5) for s in rule.NOISE+rule.MISSING}
    # Full ties: noise candidates all retained; equality to baseline is not a missing win.
    got=rule.select_from_scores('fixture',fixture)
    assert got['noise']==rule.NOISE and got['missing']==rule.MISSING[1:]
    assert got['noise_votes']==dict.fromkeys(rule.NOISE,5)
    assert set(got['random_wins'].values())=={0}
    # A unique fold winner stays unique; min-win takes priority over larger total-win.
    fixture2=copy.deepcopy(fixture)
    for f in range(5):
        fixture2[f,'S3']['awgn']=.9
        if f<3:fixture2[f,'augmentation'].update(missing_random=.8,missing_consecutive=.8)
        fixture2[f,'combined']['missing_random']=.8
        if f<2:fixture2[f,'combined']['missing_consecutive']=.8
    got2=rule.select_from_scores('fixture2',fixture2)
    assert got2['noise']==['S3'] and got2['missing']==['augmentation']
    # Equal votes stay tied despite unequal means; no rounding tolerance introduced.
    fixture3=copy.deepcopy(fixture)
    for f in range(5):
        fixture3[f,'S3']['awgn']=.9 if f<2 else .1
        fixture3[f,'S4']['awgn']=.8 if 2<=f<4 else .1
        fixture3[f,'S0']['awgn']=.6 if f==4 else .1
    assert rule.select_from_scores('fixture3',fixture3)['noise']==['S3','S4']
    # Reverse all candidate orders: membership unaffected.
    oldn,oldm=rule.NOISE,rule.MISSING
    try:
        rule.NOISE=list(reversed(oldn));rule.MISSING=['no_defense']+list(reversed(oldm[1:]))
        rev=rule.select_from_scores('fixture',fixture)
        assert set(rev['noise'])==set(got['noise']) and set(rev['missing'])==set(got['missing'])
    finally:rule.NOISE,rule.MISSING=oldn,oldm
    recomputed=0;choices={}
    for name in core.CANDIDATES:
        results={};complete=True
        for fold in range(5):
            for s in rule.NOISE+rule.MISSING:
                recipe='no_defense' if s in ['S0','S1','S2'] else s
                folder=output/'cv'/f'{name}_fold{fold}_{recipe}';f=folder/f'cv_{s}.json'
                if not f.exists():complete=False;continue
                scores=json.loads(f.read_text());saved=np.load(folder/f'cv_{s}_predictions.npz')
                for typ,value in scores.items():
                    mean=float(np.mean([core.metrics(saved['labels'],saved[f'{typ}_{ps}'])['macro_f1'] for ps in range(1000,1010)]))
                    assert abs(mean-value)<1e-12
                    recomputed+=len(saved['labels'])*10
                results[fold,s]=scores
        if complete:choices[name]=rule.select_from_scores(name,results)
    # Every saved artifact is unchanged across amendment creation.
    inventory=json.loads((output/'tie_rule_vote_only_20261107/artifact_sha256.json').read_text())
    for relative,digest in inventory.items():assert core.sha(output/relative)==digest,relative
    paused=json.loads((output/'PAUSED.json').read_text())
    if paused['run'].startswith('cv_'):
        current=json.loads((output/'current_run.json').read_text())
        checkpoint=torch.load(output/'cv'/current['run']/'checkpoint.pt',map_location='cpu',weights_only=False)
        paused_epoch=checkpoint['epoch']
    else:
        resume=torch.load(output/'cv'/paused['run']/'resume.pt',map_location='cpu',weights_only=False)
        paused_epoch=resume['state']['epoch']
    assert paused_epoch==paused['saved_epoch']
    report=dict(all_checks_passed=True,complete_model_choices=choices,prediction_rows_recomputed=recomputed,
        retained_artifacts_checked=len(inventory),training_function_identity=True,paused_epoch=paused_epoch,
        real_outer_test_evaluations=0,synthetic_cases=4,
        source_sha256={str(f.relative_to(core.ROOT)):core.sha(f) for f in [Path(__file__),Path(runner.__file__),Path(rule.__file__),Path(__file__).with_name('analyze_revision_followup_v4.py')]})
    core.save_json(output/'tie_rule_vote_only_20261107/tie_rule_audit.json',report)
    for name,value in choices.items():core.save_json(output/f'E3_selection_{name}.json',value)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':verify(core.ROOT/'results/e1_e4_mac_20261106')
