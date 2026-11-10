from pathlib import Path
import sys,json,math
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,pandas as pd,torch
from e14_core import fit_stats,DATA

def main():
    checks={}
    raw=dict(np.load(DATA/'raw_windows.npz'));tr=np.flatnonzero(raw['partitions']=='train');te=np.flatnonzero(raw['partitions']=='val')
    assert len(tr)==855 and len(te)==219
    assert not set(raw['groups'][tr]) & set(raw['groups'][te])
    conditions=json.loads((ROOT/'results/e1_e4_mac_20261106/evaluation_conditions.json').read_text())
    expected=json.loads((OUT/'epoch_settings.json').read_text())['fixed_epochs']
    for name in ['NoiseFB','PermFB']:
        for seed in range(42,52):
            folder=OUT/'final'/f'{name}_no_defense_s{seed}'
            spec=json.loads((folder/'run_spec.json').read_text());res=json.loads((folder/'result.json').read_text());ev=json.loads((folder/'evaluation.json').read_text())
            assert spec['config']=={'lr':.0005,'weight_decay':0.,'dropout':.5}
            assert spec['seed']==seed and spec['fixed_epochs']==expected[name]==res['selected_epoch']==ev['epoch']
            assert spec['train_ids']==raw['sample_ids'][tr].tolist() and spec['validation_ids']==[]
            assert spec['normalization']==fit_stats(raw,tr)
            log=pd.read_csv(folder/'train_log.csv');assert log.epoch.tolist()==list(range(1,expected[name]+1))
            assert log.validation_macro_f1.isna().all() and np.isfinite(log.train_loss).all()
            saved=np.load(folder/'outer_ids_labels.npz');assert np.array_equal(saved['sample_ids'],raw['sample_ids'][te]) and np.array_equal(saved['labels'],raw['labels'][te])
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==expected[name]
            for k in [1,2,3]:assert ck['state_dict'][f'encoder.bn{k}.num_batches_tracked'].item()==math.ceil(855/32)*2*expected[name]
            assert len(ev['conditions'])==210
            assert [[c['type'],c['severity'],c['pollution_seed']] for c in ev['conditions']]==conditions
            assert len(list((folder/'outer_predictions').glob('*.npy')))==len(list((folder/'first_pass_predictions').glob('*.npy')))==211
    checks['twenty_formal_models_config_epochs_split_norm_BN_complete']=True
    (OUT/'qa/training_audit.json').write_text(json.dumps(dict(all_checks_passed=True,checks=checks),indent=2))
    print('training audit passed')
if __name__=='__main__':main()
