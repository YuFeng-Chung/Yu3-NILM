"""Independent saved-artifact checks and prelaunch report, no model training."""
import os,sys,json,ast,hashlib,math,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'experiments'))
from run_e5_ablation import metrics
from run_exp4_matched_controls import DATA,sha,save_json
from run_e2_search_mps import inner_split
import torch
raw=dict(np.load(DATA/'raw_windows.npz'));pt,pv,_=inner_split(raw)
assert json.loads((OUT/'progress.json').read_text())['stage']=='awaiting_author_confirmation'
scale=json.loads((OUT/'noise_scale.json').read_text());seal=json.loads((OUT/'noise_scale_seal.json').read_text())
assert sha(OUT/'noise_scale.json')==seal['sha256']
a=np.load(OUT/'calibration_moments.npz');assert a['mean'].shape==a['std'].shape==(10,855)
assert np.array_equal(a['sample_ids'],raw['sample_ids'][raw['partitions']=='train'])
assert float(np.median(a['mean']))==scale['mu0'];assert float(np.median(a['std']))==scale['sigma0']
assert sha(OUT/'calibration_moments.npz')==seal['moments_sha256']
new=ast.parse((HERE/'e15_core.py').read_text());old=ast.parse((HERE/'original_e14_core.py').read_text())
for name in ['FeedbackControl','RecAux','train','predict','stable_seed','fit_stats','training_strategy','config_for','atomic_torch','csv_write','pause_check']:
    f=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)
    assert f(new)==f(old),name
settings=json.loads((OUT/'epoch_settings.json').read_text());summaries={}
for name in ['1P-NoiseCh','1P-Mix']:
    p=OUT/'pilots'/f'{name}_no_defense';r=json.loads((p/'result.json').read_text());s=json.loads((p/'run_spec.json').read_text())
    assert s['train_ids']==raw['sample_ids'][pt].tolist() and s['validation_ids']==raw['sample_ids'][pv].tolist()
    assert s['config']=={'lr':.0005,'weight_decay':0.,'dropout':.5}
    rows=list(csv.DictReader((p/'train_log.csv').open()));assert [int(x['epoch']) for x in rows]==list(range(1,r['stopped_epoch']+1))
    best=max((row for row in rows if int(row['epoch'])>=30),key=lambda row:float(row['validation_macro_f1']))
    assert int(best['epoch'])==r['selected_epoch']==settings['fixed_epochs'][name]
    anchor=float(rows[29]['validation_macro_f1']);stale=0
    for row in rows[30:]:
        if float(row['validation_macro_f1'])>anchor+.001:anchor=float(row['validation_macro_f1']);stale=0
        else:stale+=1
        assert stale<30 or int(row['epoch'])==r['stopped_epoch']
    assert stale>=30 or r['stopped_epoch']==150
    pred=np.load(p/'validation_predictions.npz');assert np.array_equal(pred['sample_ids'],raw['sample_ids'][pv]) and np.array_equal(pred['labels'],raw['labels'][pv])
    y=pred['labels'];yp=pred['predictions'];f=[]
    for label in range(11):
        tp=np.count_nonzero((y==label)&(yp==label));fp=np.count_nonzero((y!=label)&(yp==label));fn=np.count_nonzero((y==label)&(yp!=label));f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    assert abs(np.mean(f)-r['validation_macro_f1'])<1e-14
    state=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==r['selected_epoch']
    for layer in range(1,4):assert int(state['state_dict'][f'encoder.bn{layer}.num_batches_tracked'])==math.ceil(len(pt)/32)*r['selected_epoch']
    summaries[name]=dict(fixed_epochs=r['selected_epoch'],stopped_epoch=r['stopped_epoch'],validation_macro_f1=r['validation_macro_f1'],seconds_per_epoch=settings['pilot_seconds_per_epoch'][name])
protected=json.loads((OUT/'protected_sources_sha256.json').read_text())
for rel,digest in protected.items():assert sha(ROOT/rel)==digest,rel
assert not (OUT/'final').exists()
assert not (OUT/'image_cache').exists()
timing=json.loads((OUT/'evaluation_timing_train_only.json').read_text());train=settings['estimated_training_seconds'];evaluation=timing['estimated_evaluation_seconds']
report=f'''# E15 開跑前回報

日期：2026-11-09（Asia/Taipei）。目前僅完成校準、前測與兩個pilot，等待作者確認；正式20模型尚未訓練，未開外層test評分。

## 固定雜訊尺度

- μ0 = {scale['mu0']:.17g}
- σ0 = {scale['sigma0']:.17g}
- 十個E14 NoiseFB final checkpoint seed42–51，各855筆train、batch32、eval模式第一次推論絕對殘差；逐樣本像素mean／population std（correction=0），8550值各自取中位數。
- 尺度SHA256：`{seal['sha256']}`；`noise_scale.json`與`noise_scale_seal.json`已封存，不更動。`calibration_moments.npz`保存原始逐樣本統計並以獨立NumPy公式核對。

## Pilot 選輪

兩模型seed42，inner train {len(pt)}／validation {len(pv)}，共同group split；min30／patience30／max150／delta.001，從第30輪起最高clean validation Macro-F1（[X,0]）選fixed epoch，同分取最早。均保留原E14引擎選輪／早停／checkpoint邏輯。

| 模型 | fixed_epochs | 停止輪次 | validation F1 | 實測秒／epoch |
|---|---:|---:|---:|---:|
'''
for name,r in summaries.items():report+=f"| {name} | {r['fixed_epochs']} | {r['stopped_epoch']} | {r['validation_macro_f1']:.10f} | {r['seconds_per_epoch']:.3f} |\n"
report+=f'''
## 正式規畫與估時

正式兩模型×seed42–51，共20模型；855train／219test；Adam lr.0005／WD0／dropout.5／batch32。全部211條件（clean＋210污染），兩種輸入[X,0]／[X,N]，推論batch32。NoiseFB／ZeroFB重用E14保存數值，不重訓。

依pilot訓練秒數按855/{len(pt)}外推：純正式訓練 **{train/60:.1f}分鐘**。同train的219筆實測、兩種輸入推論外推 **{evaluation/60:.1f}分鐘**。估時不使用test影像或分數；不含首次完整污染成像／IO／分析。保守估正式研究 **35–55分鐘**，加完整Word／圖表／文件QA約 **60–90分鐘**；為目前估計，非完成承諾。

## 邊界與稽核

- 單次encoder／classifier／decoder、CE＋0.5MSE、所有梯度、兩通道參數結構、局部CPU Generator、Mix逐樣本Bernoulli、固定樣本ID推論噪聲均通過前測。
- Mix先產生整批Gaussian，再取每樣本獨立Bernoulli（<.5為noise）；同E14 SHA256 key與映射。驗證固定[X,0]，次要[X,N]不參與選輪。
- 按規格以E14 full855訓練之模型校準，包含後續inner validation樣本。pilot為既定轉移尺度下的選輪，不宣稱完整nested獨立選模。外層test未參與校準／選輪／估時。
- 主要比較兩模型各自對NoiseFB第一次推論／ZeroFB原生輸出；SP5/10/20每強度內兩項精確雙尾Wilcoxon n10＋Holm；1P-Mix獨立次要家族。逐強度3pp判讀保留規格原文，非等效或普遍必要性證明。
- {len(protected)}個既有來源／E14產物SHA256在pilot前後一致； inherited train、predict、FeedbackControl與選輪函式AST完全相同；舊八模式前向值一致。
- E15程式／產物全部在本目錄；`source/prepare_pilot.py`僅有校準與pilot，沒有正式訓練入口。writer排他鎖、STOP_REQUESTED沿用E14。正式執行須等待本次作者明確確認，不能依舊對話授權自動啟動。
'''
(OUT/'prelaunch_report.md').write_text(report)
save_json(OUT/'prelaunch_audit.json',dict(all_checks_passed=True,scale_sealed=True,independent_median=True,independent_saved_validation_F1=True,epoch_selection_and_stop_verified=True,one_pass_BN_counts=True,inherited_AST_unchanged=True,protected_source_count=len(protected),protected_sources_unchanged=True,outer_test_scores_accessed=False,formal_model_count=0,summaries=summaries))
print(json.dumps(dict(scale={k:scale[k] for k in ['mu0','sigma0']},pilots=summaries,training_minutes=train/60,evaluation_minutes=evaluation/60,protected_count=len(protected)),indent=2))
