"""E7 frozen group-split sensitivity; no model or strategy selection on new tests."""
import json
import random
from pathlib import Path
import numpy as np
from revision_followup_core import ROOT, DATA, sha, save_json, csv_write

PREVIOUS = ROOT/'results/e1_e4_mac_20261106'
OUTPUT = ROOT/'results/e7_splits_20261108'
MODELS = ['Yu3', 'M1', 'RecAux', 'ResNet18']
SEEDS = [42, 123, 2026]
CONDITIONS = [('awgn', 10, ps) for ps in range(1000,1010)] + [( 'missing_consecutive', s, ps) for s in [.2,.5] for ps in range(1000,1010)]
SOURCES = ['e7_split_core.py','prepare_e7_splits.py','preflight_e7_splits.py','run_e7_splits.py','analyze_e7_splits.py','control_e7_splits.py',
 'revision_followup_core.py','run_revision_followup_v4.py','run_e2_search_mps.py',
 'run_exp4_matched_controls.py','run_e5_ablation.py','revision_models_mps.py']

def source_hashes():
    paths = [ROOT/'experiments'/n for n in SOURCES] + [ROOT/'實驗一/models_m1_to_m5.py',
        ROOT/'實驗四/exp4_raw_data_V2.py',ROOT/'實驗四/exp4_defense_strategies_V2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def split_indices(raw, seed):
    # Same 2000-subset search as historical splitter; explicit sorted machine order
    # replaces its unrecoverable os.listdir insertion order. First strict optimum.
    rng = random.Random(seed); test=[]
    for label in range(11):
        idx=np.flatnonzero(raw['labels']==label)
        groups=sorted(set(raw['groups'][idx])); assert len(groups)>=2
        counts={g:int(np.sum(raw['groups'][idx]==g)) for g in groups}
        best=None; difference=float('inf')
        for _ in range(2000):
            selected=rng.sample(groups,rng.randint(1,len(groups)-1))
            diff=abs(sum(counts[g] for g in selected)-len(idx)*.2)
            if diff<difference: best=selected; difference=diff
        test.extend(idx[np.isin(raw['groups'][idx],best)].tolist())
    test=np.array(sorted(test)); train=np.setdiff1d(np.arange(len(raw['labels'])),test)
    assert not set(raw['groups'][test])&set(raw['groups'][train])
    assert set(raw['labels'][test])==set(raw['labels'][train])==set(range(11))
    return train,test

def class_metrics(labels, pred):
    cm=np.bincount(11*np.asarray(labels,dtype=int)+np.asarray(pred,dtype=int),minlength=121).reshape(11,11)
    tp=cm.diagonal().astype(float)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(11),where=cm.sum(0)!=0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(11),where=cm.sum(1)!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(11),where=precision+recall!=0)
    return cm,np.stack([precision,recall,f1],axis=1)

def previous_predictions(model, seed, condition):
    folder=PREVIOUS/'final'/f'{model}_combined_s{seed}'
    ids=np.load(folder/'outer_ids_labels.npz')
    conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
    filename='clean.npy' if condition is None else f'{conditions.index(list(condition)):03}.npy'
    pred=np.load(folder/'outer_predictions'/filename)
    assert pred.shape==ids['labels'].shape
    return ids['sample_ids'],ids['labels'],pred

def load_plan(output=OUTPUT):
    plan=json.loads((output/'protocol.json').read_text())
    assert plan['source_sha256']==source_hashes(),'E7 source changed after freeze'
    assert plan['raw_sha256']==sha(DATA/'raw_windows.npz')
    return plan
