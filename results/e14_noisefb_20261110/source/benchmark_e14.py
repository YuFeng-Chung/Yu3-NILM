import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
from e14_core import *
from evaluate_e14 import predictions
raw=dict(np.load(DATA/'raw_windows.npz'));train_idx=np.flatnonzero(raw['partitions']=='train');ids=raw['sample_ids'][train_idx[:219]]
array=images(raw,train_idx[:219],fit_stats(raw,train_idx));device=torch.device('mps');torch.set_num_threads(4)
values={}
for name in ['NoiseFB','PermFB','Yu3','ZeroFB','ShuffleFB']:
    model=build(name,.5).to(device)
    predictions(model,array,name,device,ids)
    if name in ['NoiseFB','PermFB']:
        state=torch.load(OUT/'pilots'/f'{name}_no_defense'/'checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict'])
        small=predict(model,array,name,device,ids)
        large=predictions(model,array,name,device,ids)[1]
        assert np.array_equal(small,large)
    torch.mps.synchronize();tick=time.monotonic()
    for _ in range(10):predictions(model,array,name,device,ids)
    if name in ['NoiseFB','PermFB']:
        state=torch.load(OUT/'pilots'/f'{name}_no_defense'/'checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['state_dict'])
        small=predict(model,array,name,device,ids)
        large=predictions(model,array,name,device,ids)[1]
        assert np.array_equal(small,large)
    torch.mps.synchronize();values[name]=(time.monotonic()-tick)/10
save_json(OUT/'evaluation_timing_train_only.json',dict(seconds_per_219=values,estimated_evaluation_seconds=sum(values.values())*10*211,outer_test_accessed=False))
print(values)
