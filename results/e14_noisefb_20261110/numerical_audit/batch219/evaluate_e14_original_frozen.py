"""E4 identical image construction and conditions, pass1 and reported pass saved separately."""
import sys,json,gc,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,torch
import e14_core as core
from e14_core import build,fit_stats,metrics,save_json,pause_check
from run_revision_followup_v4 import cached_images
OLD=ROOT/'results/e1_e4_mac_20261106'

def predictions(model,array,name,device,ids):
    first=[];reported=[];model.eval()
    with torch.inference_mode():
        batch=32 if name=='ShuffleFB' else 219
        for start in range(0,len(array),batch):
            x=torch.from_numpy(np.array(array[start:start+batch])).to(device)
            if name in ['NoiseFB','PermFB']:model.sample_ids=ids[start:start+batch]
            out=model(x)[0]
            first.extend(out[0].argmax(1).cpu().tolist());reported.extend(out[-1].argmax(1).cpu().tolist())
    return np.array(first),np.array(reported)

def evaluate(raw,name,seed,device):
    isnew=name in ['NoiseFB','PermFB']
    source=(OUT if isnew else OLD)/'final'/f'{name}_no_defense_s{seed}'
    dest=source if isnew else OUT/'reference_evaluation'/f'{name}_no_defense_s{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'evaluation.json').exists():return
    spec=json.loads((source/'run_spec.json').read_text());stats=spec['normalization']
    outer=np.flatnonzero(raw['partitions']=='val');ids=raw['sample_ids'][outer];labels=raw['labels'][outer]
    assert stats==fit_stats(raw,np.flatnonzero(raw['partitions']=='train'))
    state=torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=build(name,spec['config']['dropout']).to(device);model.load_state_dict(state['state_dict'])
    conditions=json.loads((OLD/'evaluation_conditions.json').read_text())
    predpath=dest/'outer_predictions';firstpath=dest/'first_pass_predictions'
    predpath.mkdir(exist_ok=True);firstpath.mkdir(exist_ok=True);rows=[]
    tick=time.monotonic()
    for j,condition in enumerate([None]+conditions):
        pause_check(OUT,dest.name,j)
        key='clean' if condition is None else f'{j-1:03}'
        if (predpath/f'{key}.npy').exists() and (firstpath/f'{key}.npy').exists():
            p1=np.load(firstpath/f'{key}.npy');p2=np.load(predpath/f'{key}.npy')
        else:
            array=cached_images(OUT,raw,outer,stats,condition,'no_defense','outer')
            p1,p2=predictions(model,array,name,device,ids)
            np.save(firstpath/f'{key}.npy',p1);np.save(predpath/f'{key}.npy',p2)
        if not isnew:
            assert np.array_equal(p2,np.load(source/'outer_predictions'/f'{key}.npy')),f'Old native output mismatch: {name}/{seed}/{key}'
        if j==0:clean=metrics(labels,p2);clean1=metrics(labels,p1)
        else:
            rows.append(dict(type=condition[0],severity=condition[1],pollution_seed=condition[2],**metrics(labels,p2),drop_pp=100*(clean['macro_f1']-metrics(labels,p2)['macro_f1']),first_pass=metrics(labels,p1)))
        if j%30==0:save_json(OUT/'current_run.json',dict(phase='outer_evaluation',run=dest.name,conditions_completed=j,planned_conditions=len(conditions)))
    np.savez_compressed(dest/'outer_ids_labels.npz',sample_ids=ids,labels=labels)
    save_json(dest/'evaluation.json',dict(model=name,strategy='no_defense',seed=seed,epoch=state['epoch'],clean=clean,clean_first_pass=clean1,conditions=rows,seconds=time.monotonic()-tick,old_reported_predictions_identical=True if not isnew else None))
    print('EVALUATED',name,seed,round(time.monotonic()-tick,2),flush=True)
    del model;gc.collect();torch.mps.empty_cache()
