"""Independent prediction/epoch audit and paired E1/E3/E4 analysis."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata
from revision_followup_core import FAMILY,CANDIDATES,csv_write,save_json,metrics,DATA,fit_stats,ROOT,sha
from analyze_e5_full import paired_test,holm


def relative_ratio(numerator, denominator):
    raw=None if abs(denominator)<=1e-6 else numerator/denominator
    return raw, denominator>1e-6


def analyze(output):
    (output/'completed_audit.json').unlink(missing_ok=True)
    plan_path=next((output/n for n in ['protocol_v4.json','protocol_v3.json','protocol_v2.json','protocol.json'] if (output/n).exists()))
    plan=json.loads(plan_path.read_text());manifest=json.loads((output/'final_run_manifest.json').read_text())
    assert json.loads((output/'selection_complete.json').read_text())['all_selection_complete']
    for relative,digest in plan.get('source_sha256',{}).items():assert sha(ROOT/relative)==digest,relative
    conditions=json.loads((output/'evaluation_conditions.json').read_text());seeds=plan['seeds']
    assert seeds==list(range(42,52));raw=dict(np.load(DATA/'raw_windows.npz'))
    outer=np.flatnonzero(raw['partitions']=='train');test=np.flatnonzero(raw['partitions']=='val')
    records={};averaged=[];audited=0
    class_keys=['model','strategy','seed','type','severity','pollution_seed','class_index','f1']
    cm_keys=['model','strategy','seed','type','severity','pollution_seed','true_class','predicted_class','count']
    for filename,keys in [('per_class_metrics.csv',class_keys),('confusion_matrices.csv',cm_keys)]:
        with (output/filename).open('w',newline='') as stream:csv.DictWriter(stream,fieldnames=keys).writeheader()
    for entry in manifest:
        name,strategy,seed=entry['model'],entry['strategy'],entry['seed'];folder=output/entry['path']
        perclass=[];confusions=[]
        spec=json.loads((folder/'run_spec.json').read_text());result=json.loads((folder/'result.json').read_text())
        ev=json.loads((folder/'evaluation.json').read_text());ids=np.load(folder/'outer_ids_labels.npz')
        assert (ev['model'],ev['strategy'],ev['seed'])==(name,strategy,seed)
        assert spec['train_ids']==raw['sample_ids'][outer].tolist() and spec['validation_ids']==[]
        assert spec['normalization']==fit_stats(raw,outer)
        assert np.array_equal(ids['sample_ids'],raw['sample_ids'][test]) and np.array_equal(ids['labels'],raw['labels'][test])
        labels=ids['labels'];logs=list(csv.DictReader((folder/'train_log.csv').open()))
        epochs=spec['fixed_epochs'];assert [int(r['epoch']) for r in logs]==list(range(1,epochs+1))
        assert all(np.isfinite(float(r['train_loss'])) for r in logs)
        checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==ev['epoch']==result['selected_epoch']==epochs
        if name in FAMILY:
            for layer in (1,2,3):
                assert int(checkpoint['state_dict'][f'encoder.bn{layer}.num_batches_tracked'])==27*epochs*(2 if name in ['Yu3','ZeroFB','ShuffleFB'] else 1)
        clean=np.load(folder/'outer_predictions/clean.npy');clean_score=metrics(labels,clean)
        for k in clean_score:assert np.allclose(clean_score[k],ev['clean'][k],rtol=0,atol=1e-12)
        perclass.extend(dict(model=name,strategy=strategy,seed=seed,type='clean',severity=0,pollution_seed=-1,class_index=cls,f1=clean_score['class_f1'][cls]) for cls in range(11))
        clean_cm=np.zeros((11,11),dtype=int);np.add.at(clean_cm,(labels,clean),1)
        confusions.extend(dict(model=name,strategy=strategy,seed=seed,type='clean',severity=0,pollution_seed=-1,true_class=cls,predicted_class=target,count=int(clean_cm[cls,target])) for cls in range(11) for target in range(11))
        grouped={}
        assert len(ev['conditions'])==len(conditions)
        for index,condition in enumerate(conditions):
            c=ev['conditions'][index];assert [c['type'],c['severity'],c['pollution_seed']]==condition
            pred=np.load(folder/f'outer_predictions/{index:03}.npy');actual=metrics(labels,pred)
            for k in actual:assert np.allclose(actual[k],c[k],rtol=0,atol=1e-12)
            assert np.isclose(c['drop_pp'],100*(clean_score['macro_f1']-actual['macro_f1']),atol=1e-10)
            grouped.setdefault((c['type'],c['severity']),[]).append(actual)
            audited+=len(pred)
            # All class-wise scores and confusion matrices retained per realization.
            cm=np.zeros((11,11),dtype=int);np.add.at(cm,(labels,pred),1)
            for cls in range(11):
                perclass.append(dict(model=name,strategy=strategy,seed=seed,type=c['type'],severity=c['severity'],pollution_seed=c['pollution_seed'],class_index=cls,f1=actual['class_f1'][cls]))
                for target in range(11):
                    confusions.append(dict(model=name,strategy=strategy,seed=seed,type=c['type'],severity=c['severity'],pollution_seed=c['pollution_seed'],true_class=cls,predicted_class=target,count=int(cm[cls,target])))
        values={('clean',0):clean_score['macro_f1']}
        for key,rs in grouped.items():
            assert len(rs)==10
            values[key]=float(np.mean([r['macro_f1'] for r in rs]))
        records[name,strategy,seed]=values
        for (typ,severity),f1 in values.items():
            d=values['clean',0]-f1
            averaged.append(dict(model=name,strategy=strategy,seed=seed,type=typ,severity=severity,clean_f1=values['clean',0],f1=f1,Dabs_pp=100*d,Drel=d/values['clean',0] if values['clean',0]>1e-6 else None))
        for filename,keys,rows in [('per_class_metrics.csv',class_keys,perclass),('confusion_matrices.csv',cm_keys,confusions)]:
            with (output/filename).open('a',newline='') as stream:csv.DictWriter(stream,fieldnames=keys).writerows(rows)
        del checkpoint,perclass,confusions
    csv_write(output/'per_seed_averaged.csv',averaged)
    groups={}
    for r in averaged:groups.setdefault((r['model'],r['strategy'],r['type'],r['severity']),[]).append(r)
    summary=[]
    for (name,strategy,typ,severity),rows in groups.items():
        assert len(rows)==10
        f1=np.array([r['f1']*100 for r in rows]);d=np.array([r['Dabs_pp'] for r in rows]);dr=[r['Drel'] for r in rows if r['Drel'] is not None]
        summary.append(dict(model=name,strategy=strategy,type=typ,severity=severity,n=10,f1_mean_pct=f1.mean(),f1_sd_pct=f1.std(ddof=1),Dabs_mean_pp=d.mean(),Dabs_sd_pp=d.std(ddof=1),Drel_mean=np.mean(dr) if len(dr)==10 else None,Drel_sd=np.std(dr,ddof=1) if len(dr)==10 else None))
    csv_write(output/'summary.csv',summary)
    choices=json.loads((output/'E3_selected_strategies.json').read_text())
    from itertools import product
    selected_contexts={}
    for track in ['noise','missing']:
        contexts=[]
        for strategies in product(*(choices[n][track] for n in CANDIDATES)):
            context='selected/'+track+'/'+','.join(f'{n}={v}' for n,v in zip(CANDIDATES,strategies))
            contexts.append(context)
        selected_contexts[track]=contexts
    def strategy_for(name,context):
        if context.startswith('common/'):return context.split('/')[1]
        assignment=dict(v.split('=') for v in context.split('/')[2].split(','))
        return assignment[name if name in CANDIDATES else 'Yu3']
    def vector(name,context,key,drop=False):
        s=strategy_for(name,context)
        return np.array([(records[name,s,seed]['clean',0]-records[name,s,seed][key]) if drop else records[name,s,seed][key] for seed in seeds])*100
    tests=[];paired=[];seen_tests=set();rng=np.random.default_rng(20261106);bootstrap=rng.integers(0,10,(10000,10))
    def test(family,a,b,context,key,drop):
        signature=(family,a,b,strategy_for(a,context),strategy_for(b,context),key,drop)
        if signature in seen_tests:return
        seen_tests.add(signature)
        d=vector(a,context,key,drop)-vector(b,context,key,drop)
        _,p=paired_test(d);round_d=np.round(d,12);ranks=rankdata(abs(round_d[round_d!=0]));signs=np.sign(round_d[round_d!=0])
        effect=float((ranks*signs).sum()/ranks.sum()) if len(ranks) else 0.
        lo,hi=np.quantile(d[bootstrap].mean(1),[.025,.975])
        row=dict(family=family,comparison=f'{a}-{b}',context=context,type=key[0],severity=key[1],endpoint='Dabs' if drop else 'F1',n=10,mean_difference_pp=d.mean(),sd_difference_pp=d.std(ddof=1),CI95_mean_low_pp=lo,CI95_mean_high_pp=hi,rank_biserial=effect,p_raw=p)
        tests.append(row)
        paired.extend(dict(comparison=f'{a}-{b}',context=context,type=key[0],severity=key[1],endpoint=row['endpoint'],seed=seed,difference_pp=float(v)) for seed,v in zip(seeds,d))
    all_keys=[(typ,severity) for typ,grid in plan['severity_grids'].items() for severity in grid]
    for strategy in plan['initial_common_strategies']:
        context=f'common/{strategy}'
        for a,b in [('Yu3','RecAux'),('Yu3','M1'),('RecAux','M1'),('M3','M1')]:
            test('E1_all_common',a,b,context,('clean',0),False)
            for key in all_keys:
                for drop in [False,True]:test('E1_all_common',a,b,context,key,drop)
    for baseline in CANDIDATES[1:]:
        test('E3_all_selected','Yu3',baseline,'common/no_defense',('clean',0),False)
        for context in selected_contexts['noise']+selected_contexts['missing']:
            test('E3_all_selected','Yu3',baseline,context,('clean',0),False)
            for key in all_keys:
                if (context.split('/')[1]=='noise')!=(key[0] in ['awgn','salt_pepper']):continue
                for drop in [False,True]:test('E3_all_selected','Yu3',baseline,context,key,drop)
    for family in ['E1_all_common','E3_all_selected']:
        rows=[r for r in tests if r['family']==family]
        for row,p in zip(rows,holm([r['p_raw'] for r in rows])):row['p_holm_family']=float(p);row['family_n_tests']=len(rows)
    csv_write(output/'paired_tests.csv',tests);csv_write(output/'paired_differences.csv',paired)
    # Relative CE from Word equation, training-seed level, explicit unstable denominators.
    ce=[];mce=[]
    for context in ['common/no_defense','common/S3','common/combined']+selected_contexts['noise']+selected_contexts['missing']:
        for name in FAMILY+CANDIDATES[1:]:
            for seed in seeds:
                s=strategy_for(name,context);ref_s=strategy_for('M1',context)
                if (name,s,seed) not in records:continue
                vals=records[name,s,seed];ref=records['M1',ref_s,seed]
                valid_ce=[]
                for typ,levels in plan['severity_grids'].items():
                    native=[v for v in levels if typ!='missing_consecutive' or v<=.2]
                    grids=[('native',native)]+([('extended',levels)] if typ=='missing_consecutive' else [])
                    for scope,grid in grids:
                        numerator=sum(vals['clean',0]-vals[typ,v] for v in grid)
                        denominator=sum(ref['clean',0]-ref[typ,v] for v in grid)
                        ratio,stable=relative_ratio(numerator,denominator)
                        ce.append(dict(context=context,model=name,strategy=s,M1_strategy=ref_s,seed=seed,type=typ,grid=scope,numerator=numerator,denominator=denominator,raw_relative_CE=ratio,denominator_stable=stable,interpretation='do_not_interpret_random' if typ=='missing_random' else 'eligible' if stable else 'unstable_or_negative_denominator'))
                        if typ!='missing_random' and scope=='native':valid_ce.append(ratio if stable else None)
                mce.append(dict(context=context,model=name,seed=seed,mCE_native=float(np.mean(valid_ce)) if len(valid_ce)==3 and all(v is not None for v in valid_ce) else None))
    csv_write(output/'relative_CE_raw.csv',ce);csv_write(output/'mCE_native.csv',mce)
    ce_summary=[]
    for key in dict.fromkeys((r['context'],r['model'],r['type'],r['grid']) for r in ce):
        rows=[r for r in ce if (r['context'],r['model'],r['type'],r['grid'])==key]
        values=[r['raw_relative_CE'] for r in rows if r['denominator_stable']]
        ce_summary.append(dict(context=key[0],model=key[1],type=key[2],grid=key[3],n_valid=len(values),n_planned=10,
            mean_if_all10_valid=float(np.mean(values)) if len(values)==10 else None,
            sd_if_all10_valid=float(np.std(values,ddof=1)) if len(values)==10 else None,
            interpretation='do_not_interpret_random' if key[2]=='missing_random' else 'complete' if len(values)==10 else 'incomplete_denominator_flags'))
    csv_write(output/'relative_CE_summary.csv',ce_summary)
    mce_summary=[]
    for key in dict.fromkeys((r['context'],r['model']) for r in mce):
        values=[r['mCE_native'] for r in mce if (r['context'],r['model'])==key and r['mCE_native'] is not None]
        mce_summary.append(dict(context=key[0],model=key[1],n_valid=len(values),
            mean_if_all10_valid=float(np.mean(values)) if len(values)==10 else None,
            sd_if_all10_valid=float(np.std(values,ddof=1)) if len(values)==10 else None))
    csv_write(output/'mCE_summary.csv',mce_summary)
    gains=[]
    for strategy in plan['initial_common_strategies']:
        for seed in seeds:
            v=lambda n:records[n,strategy,seed]['clean',0]*100
            gains.append(dict(strategy=strategy,seed=seed,Yu3_minus_RecAux_pp=v('Yu3')-v('RecAux'),RecAux_minus_M1_pp=v('RecAux')-v('M1'),M3_minus_M1_pp=v('M3')-v('M1'),note='descriptive component contrasts, not causal decomposition'))
    csv_write(output/'component_contrasts_clean.csv',gains)
    decomposition=[]
    for baseline in CANDIDATES[1:]:
        for seed in seeds:
            value=lambda n:records[n,'no_defense',seed]['clean',0]*100
            decomposition.append(dict(seed=seed,baseline=baseline,Yu3_minus_RecAux_pp=value('Yu3')-value('RecAux'),
                RecAux_minus_M1_pp=value('RecAux')-value('M1'),M1_minus_baseline_pp=value('M1')-value(baseline),
                Yu3_minus_baseline_pp=value('Yu3')-value(baseline),interpretation='algebraic descriptive contrasts; not causal decomposition'))
    csv_write(output/'NoDefense_component_table.csv',decomposition)
    # Compact evidence plot; all severity data remain in CSV.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,strategy in zip(axes,plan['initial_common_strategies']):
        for name in FAMILY[:4]:
            rows=sorted([r for r in summary if r['model']==name and r['strategy']==strategy and r['type']=='missing_consecutive'],key=lambda r:r['severity'])
            ax.errorbar([0]+[r['severity']*100 for r in rows],[0]+[r['Dabs_mean_pp'] for r in rows],yerr=[0]+[r['Dabs_sd_pp'] for r in rows],marker='.',label=name,capsize=2)
        ax.set(title=strategy,xlabel='Consecutive missing (%)',ylabel='Clean - corrupted Macro-F1 (pp)');ax.grid(alpha=.2)
    axes[0].legend()
    for suffix in ['png','pdf','svg']:fig.savefig(output/f'consecutive_degradation.{suffix}',dpi=200)
    plt.close(fig)
    # CV and pilot training-only normalization audit.
    for specfile in list((output/'pilots').glob('*/run_spec.json'))+list((output/'cv').glob('*/run_spec.json')):
        spec=json.loads(specfile.read_text());index={str(s):i for i,s in enumerate(raw['sample_ids'])}
        t=np.array([index[str(s)] for s in spec['train_ids']]);v=np.array([index[str(s)] for s in spec['validation_ids']])
        assert set(t)|set(v)==set(outer)
        assert not set(raw['groups'][t])&set(raw['groups'][v])
        assert spec['normalization']==fit_stats(raw,t)
        result=json.loads((specfile.parent/'result.json').read_text())
        logs=list(csv.DictReader((specfile.parent/'train_log.csv').open()))
        assert [int(r['epoch']) for r in logs]==list(range(1,result['stopped_epoch']+1))
        eligible=[r for r in logs if int(r['epoch'])>=30]
        best=max(eligible,key=lambda r:(float(r['validation_macro_f1']),-int(r['epoch'])))
        assert result['selected_epoch']==int(best['epoch'])
        assert result['validation_macro_f1']==float(best['validation_macro_f1'])
        saved=np.load(specfile.parent/'validation_predictions.npz')
        assert np.array_equal(saved['sample_ids'],raw['sample_ids'][v])
        assert np.array_equal(saved['labels'],raw['labels'][v])
        assert metrics(saved['labels'],saved['predictions'])['macro_f1']==result['validation_macro_f1']
    # Independently recompute all strategy votes from saved fold predictions.
    if (output/'cv').exists():
        noise_order=plan['noise_order'];missing_order=plan['missing_order']
        for name in CANDIDATES:
            noise_votes=np.zeros(len(noise_order),int);rw=np.zeros(len(missing_order),int);cw=rw.copy()
            noise_fold_scores=[];noise_fold_winners=[]
            for fold in range(5):
                fold_scores={}
                for strategy in noise_order+missing_order:
                    recipe='no_defense' if strategy in ['S0','S1','S2'] else strategy
                    folder=output/'cv'/f'{name}_fold{fold}_{recipe}'
                    saved=np.load(folder/f'cv_{strategy}_predictions.npz')
                    scores=json.loads((folder/f'cv_{strategy}.json').read_text())
                    for typ in scores:
                        recomputed=np.mean([metrics(saved['labels'],saved[f'{typ}_{ps}'])['macro_f1'] for ps in plan['pollution_seeds']])
                        assert np.isclose(recomputed,scores[typ],rtol=0,atol=1e-12)
                    fold_scores[strategy]=scores
                noise_scores=np.array([np.mean([fold_scores[s]['awgn'],fold_scores[s]['salt_pepper']]) for s in noise_order])
                winners=np.flatnonzero(noise_scores==noise_scores.max());noise_votes[winners]+=1
                noise_fold_scores.append(noise_scores);noise_fold_winners.append(winners.tolist())
                for j,strategy in enumerate(missing_order):
                    if strategy=='no_defense':continue
                    rw[j]+=int(fold_scores[strategy]['missing_random']>fold_scores['no_defense']['missing_random'])
                    cw[j]+=int(fold_scores[strategy]['missing_consecutive']>fold_scores['no_defense']['missing_consecutive'])
            chosen=choices[name]
            tied=np.flatnonzero(noise_votes==noise_votes.max()).tolist()
            mean_scores=np.mean(noise_fold_scores,axis=0)
            expected_noise=[noise_order[j] for j in tied]
            keys={j:(min(rw[j],cw[j]),int(rw[j]+cw[j])) for j,s in enumerate(missing_order) if s!='no_defense'}
            expected_missing=[missing_order[j] for j in keys if keys[j]==max(keys.values())]
            assert chosen['noise']==expected_noise
            assert chosen['missing']==expected_missing
            assert chosen['noise_fold_winners']==[[noise_order[j] for j in row] for row in noise_fold_winners]
            assert chosen['protocol_version']==4
            assert chosen['noise_votes']==dict(zip(noise_order,noise_votes.tolist()))
            assert chosen['random_wins']==dict(zip(missing_order,rw.tolist()))
            assert chosen['consecutive_wins']==dict(zip(missing_order,cw.tolist()))
    audit=dict(protocol_version=4,co_selection_all_pairs=True,all_checks_passed=True,models=len(manifest),conditions_per_model=len(conditions),corrupted_predictions_recomputed=audited,clean_predictions_recomputed=len(manifest)*219,outer_selection=False,checks='all raw predictions/metrics, IDs/labels,10-realization averages, epochs/BN counts, train-only normalization, disjoint groups, complete paired seeds',manuscript_modified=False)
    save_json(output/'E3_co_selection_report.json',choices)
    lines=['# E1／E3／E4 新增受控補證完成','',f'完成 {len(manifest)} 個正式模型，每模型 {len(conditions)} 次污染評估；全部預測重算與訓練紀錄稽核通過。','',
           'E3策略計數程式於外層測試前修正，恢復原Stage1相對No-Defense勝場定義；修正前已查看部分內層fold結果，完整揭露於selection_rule_amendment.json，不稱完全事前註冊。',
           '作者於已查看部分inner scores後、外層測試前核准v4：單折雜訊完全平手各計勝票，最終勝票平手並列，跨折平均F1僅描述、不參與平手處理；缺失min-win及total-win均平手並列。所有並列策略均評估，所有不重複策略配對納入同一E3 Holm family，不依test挑選。詳tie_rule_amendment_v4.json及E3_co_selection_report.json。',
           '單一固定外層切分，10 個訓練 seed；每個 seed 先平均 10 次污染。不混用舊 CPU／MPS 結果。所有 epoch 與策略決定先於本輪外層評估。',
           'E3 使用可重建的新五折群組切分，不聲稱重現歷史 folds。Combined 使用明載的 weight decay 下限，部分基線 E2 已有較高 weight decay，正則化候選可能重複。',
           'RecAux 採一通道編碼器。Yu3−RecAux 同時涉及 pass 次數、分類 loss 與輸入介面；不能視為純殘差因果效果。ZeroFB／ShuffleFB 是另列的敏感度控制。',
           '', '乾淨及全部污染絕對 F1、Dabs、Drel 見 summary.csv；10seed逐筆平均見 per_seed_averaged.csv；配對差、效果量、區間與兩個完整 Holm families 見 paired_tests.csv。大量檢定的解析度有限，未顯著不能當作等效。',
           '相對 CE 逐 seed 依原 Word 公式計算；分母不穩定或非正值明列。隨機缺失 CE 不予解讀。原強度與延伸連續缺失網格分列；30–50%僅測試、不訓練。',
           '', '| 策略 | 模型 | Clean F1% | 連續缺失20% F1% | 連續缺失20% Dabs pp |','|---|---|---:|---:|---:|']
    lines.extend(['','## E3並列策略','| 模型 | 雜訊 | 缺失 |','|---|---|---|'])
    lines.extend(f"| {n} | {' / '.join(choices[n]['noise'])} | {' / '.join(choices[n]['missing'])} |" for n in CANDIDATES)
    lines.extend(['','## 共同策略結果','| 策略 | 模型 | Clean F1% | 連續缺失20% F1% | 連續缺失20% Dabs pp |','|---|---|---:|---:|---:|'])
    for strategy in plan['initial_common_strategies']:
        for name in FAMILY[:4]:
            clean=next(r for r in summary if (r['model'],r['strategy'],r['type'])==(name,strategy,'clean'))
            missing=next(r for r in summary if (r['model'],r['strategy'],r['type'],r['severity'])==(name,strategy,'missing_consecutive',.2))
            lines.append(f'| {strategy} | {name} | {clean["f1_mean_pct"]:.2f}±{clean["f1_sd_pct"]:.2f} | {missing["f1_mean_pct"]:.2f}±{missing["f1_sd_pct"]:.2f} | {missing["Dabs_mean_pp"]:.2f}±{missing["Dabs_sd_pp"]:.2f} |')
    lines+=['','![連續缺失退化](consecutive_degradation.png)','','結果不自動改稿或決定投稿；需連同 absolute F1、paired差、校正及限制解讀。']
    (output/'followup_report.md').write_text('\n'.join(lines))
    save_json(output/'completed_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);analyze(p.parse_args().output)
