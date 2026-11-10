"""Freeze separately authorized E7 S3 supplement without touching Combined writer."""
import shutil
from e7_s3_split_core import *

def main():
    output=OUTPUT;output.mkdir(exist_ok=True)
    combined=ROOT/'results/e7_splits_20261108'
    original=json.loads((combined/'protocol.json').read_text())
    assert sha(DATA/'raw_windows.npz')==original['raw_sha256']
    raw=dict(np.load(DATA/'raw_windows.npz'))
    settings={};evidence={}
    for model in MODELS:
        path=PREVIOUS/'final'/f'{model}_S3_s42/result.json'
        r=json.loads(path.read_text());settings[model]=dict(config=r['config'],epochs=r['selected_epoch'],strategy='S3')
        for seed in range(42,52):
            rp=PREVIOUS/'final'/f'{model}_S3_s{seed}/result.json';row=json.loads(rp.read_text())
            assert row['config']==r['config'] and row['selected_epoch']==r['selected_epoch']
            evidence[str(rp.relative_to(ROOT))]=sha(rp)
    selected=json.loads((ROOT/'results/e2_mac_20261106/selected_settings.json').read_text())
    assert max(['ResNet18','MobileNetV3','ViT-Tiny'],key=lambda n:selected[n]['validation_macro_f1'])=='ResNet18'
    for split in original['splits']:
        train=np.array(split['train_indices']);test=np.array(split['test_indices'])
        assert raw['sample_ids'][test].tolist()==split['test_ids']
        assert not set(raw['groups'][train])&set(raw['groups'][test])
        assert set(raw['labels'][test])==set(raw['labels'][train])==set(range(11))
        if split['split']:
            a,b=split_indices(raw,split['split']);assert np.array_equal(train,a) and np.array_equal(test,b)
    conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
    for seed in range(42,52):
        a=PREVIOUS/'final'/f'ResNet18_S3_s{seed}';b=PREVIOUS/'final'/f'ResNet18_S4_s{seed}'
        for c in [None]+CONDITIONS:
            fn='clean.npy' if c is None else f'{conditions.index(list(c)):03}.npy'
            assert np.array_equal(np.load(a/'outer_predictions'/fn),np.load(b/'outer_predictions'/fn))
    plan={**original,'settings':settings,'conditions':CONDITIONS,'source_sha256':source_hashes(),
      'study':'E7 S3 authorized supplement, separate from Combined and retaining all original results',
      'authorization_source':'e7_s3_handoff_20261108.md',
      'addition_reason':'Cover original S3 noise main comparisons for R3-8(b) and minor-9; not replacing unfavorable Combined results',
      'strategy':'Fixed S3 for all four models; M1/RecAux transfer Yu3 recipe; ResNet18 S3/S4 effective equivalence retained',
      'noise_grid':{'awgn':[30,20,15,10,5,0],'salt_pepper':[.01,.05,.1,.2]},
      'analysis':'All grid conditions retained. Macro-F1 and clean-minus-corrupt degradation in pp; per-class PRF; pooled and row-normalized count matrices. Original split ten seeds separate; five new splits primary and all six sensitivity. All comparisons descriptive; paired split/seed bootstrap 10000, no multiple-testing or independent-split population coverage claim.',
      'combined_protocol_sha256':sha(combined/'protocol.json'),'evidence_sha256':evidence,
      'training_gate':'Combined completed_audit all_checks_passed, progress completed, process exited, kernel writer lock free; then S3 real MPS preflight, then independent writer. Later pause overrides queue.'}
    if (output/'protocol.json').exists():assert json.loads((output/'protocol.json').read_text())==json.loads(json.dumps(plan))
    else:save_json(output/'protocol.json',plan)
    for n in ['split_class_counts.csv','split_overlap.csv']:shutil.copy2(combined/n,output/n)
    for rel in plan['source_sha256']:
        dest=output/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    save_json(output/'preparation_audit.json',dict(all_checks_passed=True,identical_combined_splits=True,original_s3_settings_verified_all_40_runs=True,
      resnet_s3_s4_predictions_identical=True,planned_training=60,conditions_per_model=101,source_sha256=plan['source_sha256']))
    print('S3 supplement frozen:',{k:v['epochs'] for k,v in settings.items()},'60 runs, 101 predictions/run')

if __name__=='__main__':main()
