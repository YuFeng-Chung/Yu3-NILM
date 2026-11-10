"""Independent direct Boolean PRF/count audit of both complete E7 strategies."""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'e7_full_review_20261108'

def read(p):return list(csv.DictReader(p.open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    from e7_split_core import load_plan as combined_plan
    from e7_s3_split_core import load_plan as s3_plan
    combined_plan();s3_plan()
    raw=dict(np.load(ROOT/'data/plaid_2014_published/raw_windows.npz'))
    previous=ROOT/'results/e1_e4_mac_20261106'
    old_conditions=json.loads((previous/'evaluation_conditions.json').read_text())
    total=0;checks={};hashes={}
    for strategy,dirname in [('Combined','e7_splits_20261108'),('S3','e7_s3_splits_20261108')]:
        output=ROOT/'results'/dirname;plan=json.loads((output/'protocol.json').read_text())
        assert json.loads((output/'completed_audit.json').read_text())['all_checks_passed']
        src=read(output/'analysis/seed_metrics.csv')
        seed_table={(int(r['split']),r['model'],int(r['seed']),r['condition']):float(r['macro_f1']) for r in src}
        classes={(int(r['split']),r['model'],r['condition'],int(r['label'])):r for r in read(output/'analysis/per_class_metrics.csv')}
        matrices=read(output/'analysis/confusion_matrices.csv')
        cms={}
        for r in matrices:
            key=(int(r['split']),r['model'],r['condition'])
            if key not in cms:cms[key]=np.zeros((11,11),dtype=np.int64)
            cms[key][int(r['true_label']),int(r['predicted_label'])]=int(r['count'])
        verified=0
        for split in plan['splits']:
            s=split['split'];test=np.array(split['test_indices']);train=np.array(split['train_indices'])
            assert not set(raw['groups'][train])&set(raw['groups'][test])
            for model in plan['models']:
                class_seed={};counts={}
                seeds=plan['original_seeds'] if s==0 else plan['training_seeds']
                for seed in seeds:
                    old_strategy='combined' if strategy=='Combined' else 'S3'
                    folder=previous/'final'/f'{model}_{old_strategy}_s{seed}' if s==0 else output/'runs'/f'split{s}_{model}_s{seed}'
                    identity=np.load(folder/'outer_ids_labels.npz');y=identity['labels']
                    assert identity['sample_ids'].tolist()==split['test_ids'] and np.array_equal(y,raw['labels'][test])
                    spec=json.loads((folder/'run_spec.json').read_text());res=json.loads((folder/'result.json').read_text())
                    assert res['selected_epoch']==plan['settings'][model]['epochs']
                    if s:
                        assert spec['train_ids']==split['train_ids'] and spec['config']==plan['settings'][model]['config']
                        for prefix,column in [('i','current'),('v','voltage')]:
                            values=raw[column][train].astype(np.float32)
                            for suffix,fn in [('mean',np.mean),('std',np.std),('min',np.min),('max',np.max)]:
                                assert spec['normalization'][prefix+'_'+suffix]==float(fn(values))
                    result=json.loads((folder/'evaluation.json').read_text());per_condition={}
                    for i,c in enumerate([None]+plan['conditions']):
                        j=None if c is None else (old_conditions.index(c) if s==0 else i-1)
                        filename='clean.npy' if c is None else f'{j:03}.npy'
                        pred=np.load(folder/'outer_predictions'/filename)
                        assert pred.shape==y.shape and np.isin(pred,np.arange(11)).all()
                        actual=np.array([(y==k) for k in range(11)])
                        predicted=np.array([(pred==k) for k in range(11)])
                        tp=np.sum(actual&predicted,axis=1);ny=actual.sum(1);npred=predicted.sum(1)
                        precision=np.divide(tp,npred,out=np.zeros(11),where=npred>0)
                        recall=np.divide(tp,ny,out=np.zeros(11),where=ny>0)
                        f1=np.divide(2*tp,ny+npred,out=np.zeros(11),where=ny+npred>0)
                        prf=np.stack([precision,recall,f1],axis=1)
                        saved=result['clean'] if c is None else result['conditions'][j]
                        assert abs(prf[:,2].mean()-saved['macro_f1'])<1e-12
                        cname='clean' if c is None else f'{c[0]}_{c[1]}'
                        per_condition.setdefault(cname,[]).append(prf)
                        count=actual.astype(np.int64)@predicted.astype(np.int64).T
                        counts[cname]=counts.get(cname,np.zeros((11,11),dtype=np.int64))+count
                        total+=len(y);verified+=len(y)
                    for cname,prfs in per_condition.items():
                        averaged=np.mean(prfs,axis=0);class_seed.setdefault(cname,[]).append(averaged)
                        assert abs(averaged[:,2].mean()-seed_table[s,model,seed,cname])<1e-12
                for cname,values in class_seed.items():
                    values=np.array(values)
                    for label in range(11):
                        r=classes[s,model,cname,label]
                        assert int(r['test_groups'])==len(set(raw['groups'][test[raw['labels'][test]==label]]))
                        for j,metric in enumerate(['precision','recall','f1']):
                            assert abs(values[:,label,j].mean()-float(r[metric+'_mean']))<1e-12
                            assert abs(values[:,label,j].std(ddof=1)-float(r[metric+'_sd']))<1e-12
                    assert np.array_equal(counts[cname],cms[s,model,cname])
        # Recompute split summaries and paired win counts independently from verified seed table.
        summary=read(output/'analysis/split_summary.csv');paired=read(output/'analysis/paired_split_bootstrap.csv')
        for r in summary:
            ss=range(1,6) if r['scope']=='five_new' else range(6)
            means=[np.mean([v for (s,m,seed,c),v in seed_table.items() if s==split and m==r['model'] and c==r['condition']]) for split in ss]
            assert abs(np.mean(means)-float(r['mean_macro_f1']))<1e-12
            assert abs(np.std(means,ddof=1)-float(r['split_sd']))<1e-12
        for r in paired:
            a,b=r['contrast'].split('-');ss=range(1,6) if r['scope']=='five_new' else range(6)
            ds=[np.mean([v for (s,m,seed,c),v in seed_table.items() if s==split and m==a and c==r['condition']])-np.mean([v for (s,m,seed,c),v in seed_table.items() if s==split and m==b and c==r['condition']]) for split in ss]
            assert abs(100*np.mean(ds)-float(r['mean_difference_pp']))<1e-10
            assert sum(d>0 for d in ds)==int(r['wins'])
        checks[strategy]=dict(all_checks_passed=True,prediction_rows=verified,seed_metric_rows=len(src),class_rows=len(classes),matrix_count_rows=len(matrices),summary_and_pair_means_verified=True)
        for f in output.glob('analysis/*.csv'):hashes[str(f.relative_to(ROOT))]=sha(f)
    report=dict(all_checks_passed=True,total_prediction_rows=total,checks=checks,source_sha256=hashes,method='Independent direct Boolean TP/FP/FN, indicator-matrix multiplication for pooled counts, saved prediction identity, training-only normalization, seed/split means and wins. No model training or inference rerun.',bootstrap_limitations='Source intervals are descriptive. No independent population coverage, equivalence or multiplicity-controlled inference claim.')
    (OUT/'independent_data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
