"""Two frozen E15 inputs, all original E4 conditions; batch32 always."""
import sys,json,time,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'experiments'))
import e15_core as core
from e15_core import *
from run_revision_followup_v4 import cached_images

def predictions(model,array,device,ids):
    result={}
    for mode in ['zero','noise']:
        model.eval_input=mode
        result[mode]=predict(model,array,model.mode,device,ids)
    return result

def evaluate(raw,name,seed,device):
    folder=OUT/'final'/f'{name}_no_defense_s{seed}'
    if (folder/'evaluation.json').exists():return
    spec=json.loads((folder/'run_spec.json').read_text());stats=spec['normalization']
    idx=np.flatnonzero(raw['partitions']=='val');ids=raw['sample_ids'][idx];y=raw['labels'][idx]
    assert len(idx)==219 and stats==fit_stats(raw,np.flatnonzero(raw['partitions']=='train'))
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    m=build(name,.5).to(device);m.load_state_dict(ck['state_dict']);m.eval()
    bn={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
    conditions=json.loads((OUT/'protocol.json').read_text())['conditions'];rows=[];tick=time.monotonic()
    for sub in ['outer_predictions','noise_input_predictions']:(folder/sub).mkdir(exist_ok=True)
    for j,c in enumerate([None]+conditions):
        pause_check(OUT,folder.name,j);key='clean' if j==0 else f'{j-1:03}'
        paths={'zero':folder/'outer_predictions'/f'{key}.npy','noise':folder/'noise_input_predictions'/f'{key}.npy'}
        if all(p.exists() for p in paths.values()):p={mode:np.load(path) for mode,path in paths.items()}
        else:
            array=cached_images(OUT,raw,idx,stats,c,'no_defense','outer')
            # E14 stores the identical canonical E4 image; byte equality, no rewrite.
            from hashlib import sha256
            cachekey=json.dumps([ids.tolist(),stats,c,None],sort_keys=True)
            e14=ROOT/'results/e14_noisefb_20261110/image_cache/outer'/(sha256(cachekey.encode()).hexdigest()+'.npy')
            assert e14.exists() and np.array_equal(array,np.load(e14,mmap_mode='r')),str(e14)
            p=predictions(m,array,device,ids)
            for mode,path in paths.items():np.save(path,p[mode])
        assert all(v.shape==(219,) and np.isin(v,np.arange(11)).all() for v in p.values())
        scores={mode:metrics(y,v) for mode,v in p.items()}
        if j==0:clean=scores
        else:rows.append(dict(type=c[0],severity=c[1],pollution_seed=c[2],**scores['zero'],noise_input=scores['noise'],drop_pp=100*(clean['zero']['macro_f1']-scores['zero']['macro_f1'])))
        if j%30==0:save_json(OUT/'current_run.json',dict(phase='outer_evaluation',run=folder.name,conditions_completed=j,planned_conditions=211))
    assert all(torch.equal(m.state_dict()[k],v) for k,v in bn.items())
    np.savez_compressed(folder/'outer_ids_labels.npz',sample_ids=ids,labels=y)
    save_json(folder/'evaluation.json',dict(model=name,strategy='no_defense',seed=seed,epoch=ck['epoch'],clean=clean['zero'],clean_noise_input=clean['noise'],conditions=rows,seconds=time.monotonic()-tick,batch=32,BN_unchanged=True,canonical_E14_images_identical=True))
    print('EVALUATED',name,seed,round(time.monotonic()-tick,2),flush=True)
    del m;gc.collect();torch.mps.empty_cache()
