"""E11 adapter: native author WRG/CNN, shared waveform recipes and training engine."""
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import revision_followup_core as engine
import run_revision_followup_v4 as previous
from run_e2_search_mps import inner_split
from revision_strategy_selection_v4 import select_from_scores

ROOT = engine.ROOT
OUTPUT = ROOT / 'results/e11_wrg_20261108'
UP = ROOT / 'e11_prototype_20261108/upstream'
OLD = ROOT / 'results/e1_e4_mac_20261106'
E2 = ROOT / 'results/e2_mac_20261106'
RECIPES = ['no_defense','S3','S4','regularization','augmentation','combined']
NOISE = previous.NOISE
MISSING = previous.MISSING
SEEDS = list(range(42,52))
POLLUTION = list(range(1000,1010))
GRID = [dict(lr=lr,weight_decay=wd,dropout=d) for lr,wd,d in
        itertools.product([1e-4,5e-4,1e-3],[0.,1e-4,1e-3],[.2,.5])]
GRIDS = json.loads((OLD/'protocol_v4.json').read_text())['severity_grids']
CONDITIONS = [(t,s,p) for t,ss in GRIDS.items() for s in ss for p in POLLUTION]


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


author_model=load_module('e11_author_cnn',UP/'src/net/model.py')
author_feature=load_module('e11_author_features',UP/'src/utils/feature_representation.py')
ORIGINAL_BUILD=engine.build
ORIGINAL_IMAGES=engine.images


def build(name, dropout):
    if name=='WRG-CNN':
        return author_model.CNN2DModel(n_channels=1,n_kernels=64,n_layers=3,
                emb_size=50,dropout=dropout,output_size=11)
    return ORIGINAL_BUILD(name,dropout)


def native_wrg(current):
    """Author mean-PAA, float32 L1 distance, floor*eps and clamp; no extra normalization."""
    c=np.asarray(current)
    means=(c.reshape(50,10).sum(axis=1,dtype=np.float64)/10.).astype(np.float32)
    return np.minimum(np.floor(np.abs(means[:,None]-means[None,:])*np.float32(10)),np.float32(20))[None]


def waveforms(raw, idx, stats, strategy='no_defense', seed=0, epoch=0, condition=None, filter_strategy=None):
    for k in idx:
        c=raw['current'][k].astype(np.float32);v=raw['voltage'][k].astype(np.float32)
        sid=str(raw['sample_ids'][k])
        if condition:
            typ,severity,ps=condition
            c,v=engine.FNS['apply_waveform_corruption'](c,v,typ,severity,stats,
                                                       engine.rng_for('test',sid,typ,severity,ps))
        elif strategy!='no_defense':
            fn='dynamic_noise_augment' if strategy=='S3' else 'dynamic_missing_augment'
            c,v=engine.FNS[fn](c,v,stats,engine.rng_for('train',sid,strategy,seed,epoch))
        if filter_strategy in ['S1','S2']:
            c,v=previous.inference_filter(c,v,filter_strategy)
        yield c,v


def images(raw, idx, stats, strategy='no_defense', seed=0, epoch=0, condition=None):
    return np.stack([native_wrg(c) for c,v in waveforms(raw,idx,stats,strategy,seed,epoch,condition)])


def evaluation_images(raw,idx,stats,condition,strategy):
    return np.stack([native_wrg(c) for c,v in waveforms(raw,idx,stats,condition=condition,filter_strategy=strategy)])


def install():
    # These globals are local to the new process; no historical source file is edited.
    engine.build=build;engine.images=images
    previous.build=build;previous.evaluation_images=evaluation_images


def config_for(selected, recipe):
    config={k:selected[k] for k in ['lr','weight_decay','dropout']}
    if recipe in ['S4','regularization','combined']:
        config['weight_decay']=max(config['weight_decay'],1e-4)
    return config


def recipe_for(strategy):
    return 'no_defense' if strategy in ['S0','S1','S2'] else strategy


def source_paths():
    names=['e11_wrg_core.py','prepare_e11_wrg.py','run_e11_wrg.py','analyze_e11_wrg.py',
      'revision_followup_core.py','run_revision_followup_v4.py','run_e2_search_mps.py',
      'revision_strategy_selection_v4.py','revision_models_mps.py','run_e5_ablation.py',
      'run_exp4_matched_controls.py']
    return [Path(__file__).with_name(n) for n in names]+[
      ROOT/'實驗四/exp4_raw_data_V2.py',ROOT/'實驗四/exp4_defense_strategies_V2.py',
      ROOT/'實驗一/models_m1_to_m5.py',UP/'src/net/model.py',UP/'src/utils/feature_representation.py']


def assert_sources(plan):
    for p,h in plan['source_sha256'].items():
        assert engine.sha(ROOT/p)==h, f'Source changed: {p}'
    assert engine.sha(engine.DATA/'raw_windows.npz')==plan['raw_sha256']
    for p,h in plan['reuse_sha256'].items():
        assert engine.sha(ROOT/p)==h, f'Reused evidence changed: {p}'


def run_train(output,folder,raw,t,v,recipe,seed,config,device,fixed_epochs=None,patience=30):
    return engine.train(output,folder,raw,t,v,'WRG-CNN',recipe,seed,config,device,
                        fixed_epochs=fixed_epochs,min_epochs=30,max_epochs=150,patience=patience)


def load_raw():
    return dict(np.load(engine.DATA/'raw_windows.npz'))
