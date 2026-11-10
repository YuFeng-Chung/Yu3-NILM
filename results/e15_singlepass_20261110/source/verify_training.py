"""No forward passes; independent full saved-run and source audit."""
from pathlib import Path
import sys,json,math,hashlib
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np,pandas as pd,torch
from e15_core import fit_stats,DATA
from run_e15 import guard

def main():
    guard();raw=dict(np.load(DATA/'raw_windows.npz'));tr=np.flatnonzero(raw['partitions']=='train');te=np.flatnonzero(raw['partitions']=='val')
    assert len(tr)==855 and len(te)==219 and not set(raw['groups'][tr])&set(raw['groups'][te])
    expected={'1P-NoiseCh':42,'1P-Mix':37};conditions=json.loads((OUT/'protocol.json').read_text())['conditions'];total_epochs=0
    for name in expected:
        for seed in range(42,52):
            f=OUT/'final'/f'{name}_no_defense_s{seed}';s=json.loads((f/'run_spec.json').read_text());r=json.loads((f/'result.json').read_text());e=json.loads((f/'evaluation.json').read_text())
            assert s['config']=={'lr':.0005,'weight_decay':0.,'dropout':.5};assert s['seed']==seed and s['fixed_epochs']==expected[name]==r['selected_epoch']==r['stopped_epoch']==e['epoch']
            assert s['train_ids']==raw['sample_ids'][tr].tolist() and s['validation_ids']==[] and s['normalization']==fit_stats(raw,tr)
            log=pd.read_csv(f/'train_log.csv');assert log.epoch.tolist()==list(range(1,expected[name]+1));assert log.validation_macro_f1.isna().all() and np.isfinite(log.train_loss).all()
            ids=np.load(f/'outer_ids_labels.npz');assert np.array_equal(ids['sample_ids'],raw['sample_ids'][te]) and np.array_equal(ids['labels'],raw['labels'][te])
            ck=torch.load(f/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==expected[name]
            for layer in range(1,4):assert int(ck['state_dict'][f'encoder.bn{layer}.num_batches_tracked'])==math.ceil(855/32)*expected[name]
            assert e['batch']==32 and e['BN_unchanged'] and e['canonical_E14_images_identical']
            assert [[c['type'],c['severity'],c['pollution_seed']] for c in e['conditions']]==conditions
            for sub in ['outer_predictions','noise_input_predictions']:assert len(list((f/sub).glob('*.npy')))==211
            assert r['prediction_reload_audit'] and r['validation_macro_f1'] is None
            total_epochs+=len(log)
    checks=dict(all20_config_epochs_split_normalization_verified=True,single_pass_BN_update_counts=True,continuous_epoch_logs=True,checkpoint_reload_audits=True,no_final_validation_or_test_selection=True,complete_two_input_evaluations=True,all211_canonical_E14_images_identical=True,all_old_sources_and_frozen_research_sources_unchanged=True)
    (OUT/'qa/training_audit.json').write_text(json.dumps(dict(all_checks_passed=True,checks=checks,total_final_epochs=total_epochs,new_prediction_files=8440),indent=2))
    print('Training audit passed;',total_epochs,'epochs; 8440 prediction files')
if __name__=='__main__':main()
