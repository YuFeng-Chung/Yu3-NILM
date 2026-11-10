"""Freeze E7 before new training/test evaluation; derive E7-B from saved predictions."""
import shutil
import numpy as np
from e7_split_core import *

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    raw=dict(np.load(DATA/'raw_windows.npz'))
    metadata=json.loads((DATA/'meta_2014.json').read_text())
    byid={str(r['id']):r['meta'] for r in metadata}
    names=json.loads((DATA/'reconstructed_label_map.json').read_text())
    names={i:n for n,i in names.items()}
    for i,sid in enumerate(raw['sample_ids']):
        m=byid[str(sid)]
        assert raw['groups'][i]==f"{m['appliance']['type']}_house{m['location']}"
        assert names[int(raw['labels'][i])]==m['appliance']['type']
    selected=json.loads((ROOT/'results/e2_mac_20261106/selected_settings.json').read_text())
    baseline=max(['ResNet18','MobileNetV3','ViT-Tiny'],key=lambda n:selected[n]['validation_macro_f1'])
    assert baseline=='ResNet18'
    assert json.loads((PREVIOUS/'completed_audit.json').read_text())['all_checks_passed']
    settings={}
    for model in MODELS:
        r=json.loads((PREVIOUS/'final'/f'{model}_combined_s42/result.json').read_text())
        settings[model]={'config':r['config'],'epochs':r['selected_epoch'],'strategy':'combined'}
    splits=[]; counts=[]
    for seed in [0,1,2,3,4,5]:
        if seed:
            train,test=split_indices(raw,seed)
            train2,test2=split_indices(raw,seed)
            assert np.array_equal(test,test2) and np.array_equal(train,train2)
        else:
            train=np.flatnonzero(raw['partitions']=='train');test=np.flatnonzero(raw['partitions']=='val')
        assert not set(raw['groups'][train])&set(raw['groups'][test])
        split={'split':seed,'split_seed':seed if seed else 42,'original':seed==0,
          'train_indices':train.tolist(),'test_indices':test.tolist(),
          'train_ids':raw['sample_ids'][train].tolist(),'test_ids':raw['sample_ids'][test].tolist(),
          'train_groups':sorted(set(raw['groups'][train])),'test_groups':sorted(set(raw['groups'][test]))}
        splits.append(split)
        for label in range(11):
            ti=train[raw['labels'][train]==label];vi=test[raw['labels'][test]==label]
            counts.append(dict(split=seed,label=label,class_name=names[label],train_samples=len(ti),test_samples=len(vi),
              train_groups=len(set(raw['groups'][ti])),test_groups=len(set(raw['groups'][vi]))))
    assert len({tuple(s['test_ids']) for s in splits})==6
    overlap=[]
    for a in splits:
        for b in splits:
            x,y=set(a['test_groups']),set(b['test_groups'])
            overlap.append(dict(split_a=a['split'],split_b=b['split'],shared_test_groups=len(x&y),jaccard=len(x&y)/len(x|y)))
    plan=dict(models=MODELS,training_seeds=SEEDS,settings=settings,splits=splits,conditions=CONDITIONS,
      planned_new_training=60,original_seeds=list(range(42,52)),bootstrap_repetitions=10000,bootstrap_seed=20261108,
      baseline_selection='Highest E2 inner-validation Macro-F1 among three baselines, no E7 outer score selection',
      strategy='Original E3 missing strategy transferred; M1/RecAux inherit Yu3 Combined; ResNet18 Augmentation=Combined effective settings and epochs verified',
      split_algorithm='Historical per-class 2000 random subsets, nearest 20% samples, first strict optimum; Python random seeds 1-5, explicit lexical group order replacing historical os.listdir-dependent order. Original split reused, not reconstructed.',
      group_definition='appliance type x location (house): device proxy, not verified unique appliance ID; not house-disjoint across classes',
      inference='Descriptive split sensitivity conditional on fixed original HPO/strategy/epoch. Original outcomes already seen. New split test groups may have contributed to original training/tuning. No fresh external validation claim.',
      statistics='Average corruption seeds within training seed; paired training seeds within each split. Report five-new-split estimates as primary and all-six as sensitivity. Hierarchical paired split/seed bootstrap 10000, pointwise descriptive percentile intervals, original has 10 seeds versus new 3. Overlapping splits are dependent: intervals are conditional resampling summaries, not calibrated population coverage or multiplicity-controlled significance. No Wilcoxon/Holm claims.',
      raw_sha256=sha(DATA/'raw_windows.npz'),source_sha256=source_hashes(),
      evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'results/e2_mac_20261106/selected_settings.json', PREVIOUS/'E3_selection_Yu3.json', PREVIOUS/'E3_selection_ResNet18.json',ROOT/'實驗一/make_dataset_split.py',ROOT/'實驗一/plaid_loader_vi.py']})
    for seed in range(42,52):
        for model in MODELS:
            r=json.loads((PREVIOUS/'final'/f'{model}_combined_s{seed}/result.json').read_text())
            assert r['config']==settings[model]['config'] and r['selected_epoch']==settings[model]['epochs']
        # Preserve co-selected equivalent strategy labels; verify predictions, not test-based choice.
        a=PREVIOUS/'final'/f'ResNet18_augmentation_s{seed}'
        b=PREVIOUS/'final'/f'ResNet18_combined_s{seed}'
        for condition in [None]+CONDITIONS:
            fn='clean.npy' if condition is None else f"{json.loads((PREVIOUS/'evaluation_conditions.json').read_text()).index(list(condition)):03}.npy"
            assert np.array_equal(np.load(a/'outer_predictions'/fn),np.load(b/'outer_predictions'/fn))
    if (OUTPUT/'protocol.json').exists():assert json.loads((OUTPUT/'protocol.json').read_text())==json.loads(json.dumps(plan))
    else:save_json(OUTPUT/'protocol.json',plan)
    csv_write(OUTPUT/'split_class_counts.csv',counts);csv_write(OUTPUT/'split_overlap.csv',overlap)
    for rel in plan['source_sha256']:
        dest=OUTPUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(OUTPUT/'preparation_audit.json',dict(all_checks_passed=True,raw_samples=len(raw['labels']),groups=len(set(raw['groups'])),unique_splits=6,
      metadata_group_match=True,split_reproducible=True,all_classes_and_disjoint_groups=True,baseline=baseline,
      frozen_before_new_training=True,source_sha256=plan['source_sha256']))
    print('E7 preparation PASS: 6 distinct splits, 60 planned runs; frozen protocol')

if __name__=='__main__':main()
