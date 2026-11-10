"""Recompute saved predictions independently and report every frozen E11 comparison."""
import csv
from collections import defaultdict
from e11_wrg_core import *


def independent_scores(y,p):
    assert y.shape==p.shape and np.all((p>=0)&(p<11))
    # Boolean membership computation independent of the training engine's confusion accumulation.
    tp=np.array([np.count_nonzero((y==c)&(p==c)) for c in range(11)])
    predicted=np.array([np.count_nonzero(p==c) for c in range(11)])
    actual=np.array([np.count_nonzero(y==c) for c in range(11)])
    precision=np.divide(tp,predicted,out=np.zeros(11),where=predicted!=0)
    recall=np.divide(tp,actual,out=np.zeros(11),where=actual!=0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(11),where=actual+predicted!=0)
    return dict(accuracy=float(np.mean(y==p)),macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()),macro_f1=float(f1.mean()),class_f1=f1.tolist()),precision,recall


def check_saved(y,p,saved):
    actual,precision,recall=independent_scores(y,p)
    for k,value in actual.items():np.testing.assert_allclose(value,saved[k],rtol=0,atol=2e-12)
    return actual,precision,recall


def analyze(output=OUTPUT):
    plan=json.loads((output/'protocol.json').read_text());assert_sources(plan)
    gate=json.loads((output/'selection_complete.json').read_text());assert gate['all_selection_complete']
    raw=load_raw();train,val,test=inner_split(raw);outer=np.flatnonzero(raw['partitions']=='train')
    ids=raw['sample_ids'][test];y=raw['labels'][test]
    # Verify every HPO checkpoint's saved validation score and deterministic grid selection.
    hpo=[]
    for i in range(18):
        f=output/'hpo'/f'grid{i:02}';r=json.loads((f/'result.json').read_text())
        p=np.load(f/'validation_predictions.npz')
        assert np.array_equal(p['sample_ids'],raw['sample_ids'][val])
        assert np.array_equal(p['labels'],raw['labels'][val])
        score=independent_scores(p['labels'],p['predictions'])[0]['macro_f1']
        assert abs(score-r['validation_macro_f1'])<2e-12
        hpo.append((score,-i,r))
    selected=json.loads((output/'selected_setting.json').read_text())
    assert selected['grid_index']==-max(hpo,key=lambda q:q[:2])[1]
    cv_scores={}
    for fold,(ct,cv) in enumerate(previous.folds(raw)):
        for strategy in NOISE+MISSING:
            f=output/'cv'/f'fold{fold}_{recipe_for(strategy)}'
            p=np.load(f/f'cv_{strategy}_predictions.npz');saved=json.loads((f/f'cv_{strategy}.json').read_text())
            assert np.array_equal(p['sample_ids'],raw['sample_ids'][cv])
            assert np.array_equal(p['labels'],raw['labels'][cv])
            keys=['awgn','salt_pepper'] if strategy in NOISE else ['missing_random','missing_consecutive']
            scores={t:float(np.mean([independent_scores(p['labels'],p[f'{t}_{ps}'])[0]['macro_f1'] for ps in POLLUTION])) for t in keys}
            for k in keys:assert abs(scores[k]-saved[k])<2e-12
            # Retain author-code rounded scores for exact-tie reproduction; independently checked above.
            cv_scores[fold,strategy]=saved
    choice=json.loads((output/'selected_strategies.json').read_text())
    assert choice==select_from_scores('WRG-CNN',cv_scores)
    manifest=json.loads((output/'final_run_manifest.json').read_text())
    wrglabels=list(dict.fromkeys(e['strategy'] for e in manifest))
    jobs=[('WRG-CNN',s,k,output/'final'/f'WRG-CNN_{s}_s{k}') for s in wrglabels for k in SEEDS]
    jobs += [('Yu3',s,k,OLD/'final'/f'Yu3_{s}_s{k}') for s in ['no_defense','S3','combined'] for k in SEEDS]
    records={};averaged=[];perclass=[];count=0
    countsdir=output/'confusion_counts';countsdir.mkdir(exist_ok=True)
    for model,strategy,seed,f in jobs:
        ev=json.loads((f/'evaluation.json').read_text());spec=json.loads((f/'run_spec.json').read_text())
        result=json.loads((f/'result.json').read_text())
        assert spec['train_ids']==raw['sample_ids'][outer].tolist() and spec['validation_ids']==[]
        assert spec['normalization']==engine.fit_stats(raw,outer)
        savedids=np.load(f/'outer_ids_labels.npz')
        assert np.array_equal(savedids['sample_ids'],ids) and np.array_equal(savedids['labels'],y)
        logs=list(csv.DictReader((f/'train_log.csv').open()))
        assert [int(r['epoch']) for r in logs]==list(range(1,spec['fixed_epochs']+1))
        assert all(np.isfinite(float(r['train_loss'])) for r in logs)
        checkpoint=torch.load(f/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==spec['fixed_epochs']==result['selected_epoch']==ev['epoch']
        if model=='WRG-CNN':
            assert spec['config']==config_for(selected,recipe_for(strategy))
            assert spec['fixed_epochs']==gate['epochs'][recipe_for(strategy)]
        pred=np.load(f/'outer_predictions/clean.npy')
        clean,cp,cr=check_saved(y,pred,ev['clean']);count+=len(y)
        means={('clean',0):clean['macro_f1']};classes={('clean',0):[(clean,cp,cr)]};grouped=defaultdict(list)
        cms=[]
        def cm(p):return np.bincount(y*11+p,minlength=121).reshape(11,11)
        cms.append(cm(pred))
        assert len(ev['conditions'])==len(CONDITIONS)==210
        for i,condition in enumerate(CONDITIONS):
            saved=ev['conditions'][i]
            assert [saved['type'],saved['severity'],saved['pollution_seed']]==list(condition)
            pred=np.load(f/f'outer_predictions/{i:03}.npy')
            actual,pr,re=check_saved(y,pred,saved);count+=len(y);cms.append(cm(pred))
            assert abs(saved['drop_pp']-100*(clean['macro_f1']-actual['macro_f1']))<2e-10
            key=condition[:2];grouped[key].append(actual['macro_f1']);classes.setdefault(key,[]).append((actual,pr,re))
        for key,values in grouped.items():
            assert len(values)==10;means[key]=float(np.mean(values))
        records[model,strategy,seed]=means
        for (typ,severity),f1 in means.items():
            averaged.append(dict(model=model,strategy=strategy,seed=seed,type=typ,severity=severity,
                    clean_f1=clean['macro_f1'],f1=f1,Dabs_pp=100*(clean['macro_f1']-f1),
                    Drel=(clean['macro_f1']-f1)/clean['macro_f1'] if clean['macro_f1'] else None))
            rs=classes[typ,severity]
            for c in range(11):
                perclass.append(dict(model=model,strategy=strategy,seed=seed,type=typ,severity=severity,
                    class_index=c,precision=float(np.mean([r[1][c] for r in rs])),
                    recall=float(np.mean([r[2][c] for r in rs])),f1=float(np.mean([r[0]['class_f1'][c] for r in rs]))))
        np.savez_compressed(countsdir/f'{model}_{strategy}_s{seed}.npz',counts=np.stack(cms),sample_ids=ids,
                conditions=np.array([['clean','0','-1']]+[[str(x) for x in c] for c in CONDITIONS]))
        del checkpoint
    engine.csv_write(output/'per_seed_averaged.csv',averaged);engine.csv_write(output/'per_class_averaged.csv',perclass)
    grouped=defaultdict(list)
    for r in averaged:grouped[r['model'],r['strategy'],r['type'],r['severity']].append(r)
    summary=[]
    for (m,s,t,sev),rows in grouped.items():
        assert len(rows)==10
        summary.append(dict(model=m,strategy=s,type=t,severity=sev,n=10,
            f1_mean_pct=float(np.mean([r['f1']*100 for r in rows])),
            f1_sd_pct=float(np.std([r['f1']*100 for r in rows],ddof=1)),
            Dabs_mean_pp=float(np.mean([r['Dabs_pp'] for r in rows])),
            Dabs_sd_pp=float(np.std([r['Dabs_pp'] for r in rows],ddof=1))))
    engine.csv_write(output/'summary.csv',summary)
    yu3choice=json.loads((OLD/'E3_selected_strategies.json').read_text())['Yu3']
    contexts=[('common/'+s,s,s) for s in ['no_defense','S3','combined']]
    contexts += [(track+'/'+ys+'/'+ws,ys,ws) for track in ['noise','missing']
                 for ys in yu3choice[track] for ws in choice[track]]
    comparisons=[]
    for context,ys,ws in contexts:
        for typ,sev in records['Yu3',ys,42]:
            if context.startswith('noise/') and typ not in ['clean','awgn','salt_pepper']:continue
            if context.startswith('missing/') and typ not in ['clean','missing_random','missing_consecutive']:continue
            ya=np.array([records['Yu3',ys,k][typ,sev] for k in SEEDS])*100
            wa=np.array([records['WRG-CNN',ws,k][typ,sev] for k in SEEDS])*100
            yc=np.array([records['Yu3',ys,k]['clean',0] for k in SEEDS])*100
            wc=np.array([records['WRG-CNN',ws,k]['clean',0] for k in SEEDS])*100
            diff=ya-wa;dropdiff=(yc-ya)-(wc-wa)
            rng=np.random.default_rng(20261108);idx=rng.integers(0,10,size=(10000,10))
            lo,hi=np.quantile(diff[idx].mean(1),[.025,.975])
            comparisons.append(dict(context=context,Yu3_strategy=ys,WRG_strategy=ws,type=typ,severity=sev,n=10,
                Yu3_mean_pct=float(ya.mean()),WRG_mean_pct=float(wa.mean()),Yu3_minus_WRG_pp=float(diff.mean()),
                paired_sd_pp=float(diff.std(ddof=1)),Yu3_winning_seeds=int(np.sum(diff>0)),
                tied_seeds=int(np.sum(diff==0)),bootstrap_lo_pp=float(lo),bootstrap_hi_pp=float(hi),
                Yu3_minus_WRG_Dabs_pp=float(dropdiff.mean())))
    engine.csv_write(output/'paired_comparisons.csv',comparisons)
    # All conditions plotted; no favorable-strength filtering.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    for context,ys,ws in contexts:
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for ax,(typ,sevs) in zip(axes.flat,GRIDS.items()):
            if context.startswith('noise/') and not typ in ['awgn','salt_pepper']:
                ax.axis('off');continue
            if context.startswith('missing/') and not typ.startswith('missing'):
                ax.axis('off');continue
            for model,strategy,color in [('Yu3',ys,'#235789'),('WRG-CNN',ws,'#d45d00')]:
                values=np.array([[records[model,strategy,k][typ,sev]*100 for sev in sevs] for k in SEEDS])
                ax.errorbar(sevs,values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,
                            label=model+' / '+strategy,color=color)
            ax.set_title(typ);ax.set_ylabel('Macro-F1 (%)');ax.set_xlabel('SNR (dB)' if typ=='awgn' else 'Fraction')
            ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(fontsize=8)
            if typ=='awgn':ax.invert_xaxis()
        fig.suptitle(context+' | mean +/- training-seed SD; fixed historical split')
        stem=context.replace('/','_');fig.savefig(figures/(stem+'.png'),dpi=600);fig.savefig(figures/(stem+'.svg'));plt.close(fig)
    lines=['# E11 Yu3 vs WRG-CNN 結果與重現說明','',
      '2026-11-08（台灣）。本輪為共同500點波形協定下的原生表徵整套方法比較；未修改原稿。','',
      '## 設計與公平性','',
      'WRG與Yu3同用18組Adam超參數搜尋、相同inner split、五折group選策v4、epoch選擇程序及42–51十個訓練seed。WRG固定作者CNN與50×50、eps10/delta20平均PAA；共同波形先污染，再各自成像。Adam/batch32/共同500點視窗及可調dropout為相對原WRG協定的明示調整。',
      'Yu3使用已完成且來源hash凍結的E2/E3/E4產物，未看WRG test結果重選。WRG所有選擇及訓練完成後才開外層評估。','',
      'WRG最佳inner設定：`'+json.dumps({k:selected[k] for k in ['grid_index','lr','weight_decay','dropout','selected_epoch']})+'`。',
      'WRG noise並列策略：'+', '.join(choice['noise'])+'；missing並列策略：'+', '.join(choice['missing'])+'。Yu3 noise=S3、missing=combined。',
      '所有並列配對完整列出，不依test挑出單一贏家。','',
      '## 完整配對比較','',
      '下表差值為Yu3−WRG，正F1差代表Yu3較高；正退化差代表Yu3損失較大。污染先在每訓練seed內平均10個實例，再跨10個訓練seed彙整。','',
      '| 比較 | 條件 | 強度 | Yu3 F1% | WRG F1% | F1差pp | Yu3勝出seed | 退化差pp |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['context']} | {r['type']} | {r['severity']} | {r['Yu3_mean_pct']:.2f} | {r['WRG_mean_pct']:.2f} | {r['Yu3_minus_WRG_pp']:+.2f} | {r['Yu3_winning_seeds']}/10 | {r['Yu3_minus_WRG_Dabs_pp']:+.2f} |")
    lines+=['','## 推論限制','',
      '本輪以描述性配對結果為主，不作確認性顯著性、等效或普遍最優主張。CSV中的bootstrap範圍只描述固定歷史切分下的訓練隨機性，不能作母群／新住宅的信賴區間；沒有以縮減Holm family補救前輪設計。',
      '原test曾在既有研究查看，本輪不是全新未見資料。group是類別×住宅代理；HPO為單inner split單seed，選策非完整nested；MacMPS不等於學生CUDA重現。',
      'WRG主要使用電流，Yu3使用V–I；結果歸屬整套表徵與分類方法，不能歸因於回饋本身。保留作者程式mean-PAA及初始化行為，不稱原論文數值重現；2020WRG比較也不能單獨宣稱已回答近期方法要求。','',
      '## 程式架構與交付','',
      '| 入口 | 目的 | 產出 |','|---|---|---|',
      '| prepare_e11_wrg.py | 來源／切分核對、train-only前測、凍結規格 | protocol.json、preflight_audit.json |',
      '| e11_wrg_core.py | 作者CNN、原生WRG及共用波形策略接合 | train／eval張量 |',
      '| run_e11_wrg.py | HPO→CV→pilots→final→外層評估 | checkpoint、逐筆預測、log及選擇紀錄 |',
      '| analyze_e11_wrg.py | Boolean逐筆重算、選策稽核、配對與繪圖 | CSV、confusion_counts、圖及本報告 |','',
      f'獨立重算正式預測共{count:,}筆；所有HPO/CV保存預測、ID、source hash、epoch與正式摘要核對通過。',
      '原始checkpoint／prediction與完整來源保留本結果目錄；source_snapshot為執行時凍結版本。']
    for context,_,_ in contexts:
        stem=context.replace('/','_');lines+=['',f'![{context}](figures/{stem}.png)']
    (output/'E11結果報告.md').write_text('\n'.join(lines)+'\n')
    assert_sources(plan)
    engine.save_json(output/'completed_audit.json',dict(all_checks_passed=True,
        prediction_rows_independently_recomputed=count,hpo_runs=18,cv_training_runs=30,
        final_unique_training_runs=gate['trained_checkpoints'],final_strategy_evaluations=len(manifest),
        conditions_per_model=211,paired_comparisons=len(comparisons),source_reuse_unchanged=True,
        independent_metric_method='boolean TP/actual/predicted denominators; all saved scores matched',
        manuscript_modified=False,formal_word_created=False,confirmatory_inference=False))
    print('E11 complete; independent audit PASS',count,flush=True)


if __name__=='__main__':analyze()
