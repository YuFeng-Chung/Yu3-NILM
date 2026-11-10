"""Independent saved-array audit, seed-level summaries, and fixed-choice E6 figures."""
import csv
import html
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e6_mechanism_core import *

def write_csv(path,rows):
    assert rows
    with path.with_suffix('.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    os.replace(path.with_suffix('.tmp'),path)

def label_map():
    d=json.loads((DATA/'reconstructed_label_map.json').read_text())
    return d

def analyze(output):
    raw,test=load_data();ids=raw['sample_ids'][test];truth=raw['labels'][test]
    plan=json.loads((output/'protocol.json').read_text())
    assert plan['source_sha256']==source_hashes()
    for relative,digest in plan['checkpoints_sha256'].items():assert sha(ROOT/relative)==digest
    assert sha(DATA/'raw_windows.npz')==plan['raw_sha256']
    predictions=[];realizations=[];energies=[];manifest=[];item_hashes={}
    for seed in SEEDS:
        spec=json.loads((PREVIOUS/'final'/f'Yu3_combined_s{seed}'/'run_spec.json').read_text())
        for condition in CONDITIONS:
            path=output/'items'/f's{seed}_{key(condition)}.npz';item_hashes[path.name]=sha(path)
            with np.load(path) as d:
                assert np.array_equal(d['sample_ids'],ids) and np.array_equal(d['labels'],truth)
                assert int(d['model_seed'])==seed
                assert d['condition'].tolist()==[condition[0],str(condition[1]),str(condition[2])]
                order=d['donor_indices'];assert np.array_equal(order,derangement(ids,condition))
                pred={m:d[f'logits_{m}'].argmax(1) for m in ['pass1','normal','zero','shuffle']}
                for mode in pred:
                    assert d[f'logits_{mode}'].shape==(219,11) and np.isfinite(d[f'logits_{mode}']).all()
                # Independently reproduce normal prediction match against E4 for all 310 items.
                prior=PREVIOUS/'final'/f'Yu3_combined_s{seed}'/'outer_predictions'
                if condition[0]=='clean':reference=np.load(prior/'clean.npy')
                else:
                    previous_conditions=json.loads((PREVIOUS/'evaluation_conditions.json').read_text())
                    reference=np.load(prior/f'{previous_conditions.index(list(condition)):03}.npy')
                assert np.array_equal(pred['normal'],reference)
                a=pred['pass1']==truth;b=pred['normal']==truth
                transitions=np.select([~a&b,a&~b,a&b],['wrong_to_right','right_to_wrong','both_correct'],default='both_wrong')
                row=dict(seed=seed,type=condition[0],severity=condition[1],pollution_seed=condition[2])
                row.update({f'f1_{m}_pct':100*independent_f1(truth,p) for m,p in pred.items()})
                row.update({t:int((transitions==t).sum()) for t in ['wrong_to_right','right_to_wrong','both_correct','both_wrong']})
                assert sum(row[t] for t in ['wrong_to_right','right_to_wrong','both_correct','both_wrong'])==219
                realizations.append(row)
                for j,sid in enumerate(ids):
                    predictions.append(dict(**{k:row[k] for k in ['seed','type','severity','pollution_seed']},sample_id=str(sid),
                        truth=int(truth[j]),pass1=int(pred['pass1'][j]),normal=int(pred['normal'][j]),zero=int(pred['zero'][j]),
                        shuffle=int(pred['shuffle'][j]),donor_sample_id=str(ids[order[j]]),transition=str(transitions[j])))
                if seed==42:
                    assert np.allclose(d['residual'],np.abs(d['image']-d['reconstruction']),atol=0,rtol=0)
                if condition[0]=='missing_consecutive':
                    regions=region_masks(raw,test,spec['normalization'],condition[2])
                    for k in regions:assert np.array_equal(d[k],regions[k]),k
                    computed=energy_rows(d['residual'],regions)
                    # Recalculate sum-of-squares directly, separately from saved metric arrays.
                    for j,sid in enumerate(ids):
                        e=np.square(d['residual'][j,0].astype(np.float64));total=float(e.sum())
                        ratio=float(e[regions['missing_mask'][j]].sum()/total) if total>1e-12 else np.nan
                        assert np.isclose(ratio,d['missing_energy_ratio'][j],equal_nan=True)
                        energies.append(dict(seed=seed,pollution_seed=condition[2],sample_id=str(sid),
                            start=int(regions['start'][j]),control_start=int(regions['control_start'][j]),length=100,
                            overlap_pixels=int(regions['overlap_pixels'][j]),
                            **{k:float(v[j]) for k,v in computed.items()}))
                    for k,v in computed.items():assert np.allclose(d[k],v,equal_nan=True),k
    write_csv(output/'predictions.csv',predictions);write_csv(output/'per_realization.csv',realizations)
    write_csv(output/'missing_energy_per_sample.csv',energies)
    by_seed=[]
    for seed in SEEDS:
        for typ in ['clean','awgn','salt_pepper','missing_consecutive']:
            rows=[r for r in realizations if r['seed']==seed and r['type']==typ]
            assert len(rows)==(1 if typ=='clean' else 10)
            row=dict(seed=seed,type=typ)
            for k in rows[0]:
                if k not in ['seed','type','severity','pollution_seed']:row[k]=float(np.mean([r[k] for r in rows]))
            row['pass2_minus_pass1_pp']=row['f1_normal_pct']-row['f1_pass1_pct']
            row['normal_minus_zero_pp']=row['f1_normal_pct']-row['f1_zero_pct']
            row['normal_minus_shuffle_pp']=row['f1_normal_pct']-row['f1_shuffle_pct']
            by_seed.append(row)
    write_csv(output/'per_seed_summary.csv',by_seed)
    summary=[]
    for typ in ['clean','awgn','salt_pepper','missing_consecutive']:
        rows=[r for r in by_seed if r['type']==typ];row=dict(type=typ,n_model_seeds=10)
        for k in rows[0]:
            if k not in ['seed','type']:
                vals=np.array([r[k] for r in rows]);row[k+'_mean']=float(vals.mean());row[k+'_sd']=float(vals.std(ddof=1))
        summary.append(row)
    write_csv(output/'summary.csv',summary)
    energy_seed=[]
    for seed in SEEDS:
        rows=[r for r in energies if r['seed']==seed];row=dict(seed=seed,n_samples_realizations=len(rows))
        for k in ['missing_energy_ratio','control_energy_ratio','missing_area_ratio','control_area_ratio','trajectory_coverage','overlap_pixels']:
            a=np.array([r[k] for r in rows]);q=np.nanpercentile(a,[25,50,75]);row.update({k+'_q25':q[0],k+'_median':q[1],k+'_q75':q[2]})
        row['zero_energy_count']=sum(r['total_energy']<=1e-12 for r in rows);energy_seed.append(row)
    write_csv(output/'missing_energy_per_seed.csv',energy_seed)
    figures(output,raw,test,predictions,summary,energy_seed,manifest)
    save_json(output/'figure_manifest.json',manifest)
    lines=['# E6 Yu3回饋機制推論報告','',
        '使用Yu3 Combined seed42–51既有模型，不重訓。每模型四條件：clean一次，AWGN10dB、椒鹽5%、連續缺失20%各10污染seed；共310個推論項目。',
        '污染先在模型seed內平均，再以10個模型seed計算平均與標準差。表內F1為百分比、差值為百分點。',
        '', '| 條件 | Pass1 F1 | 正常回饋Pass2 F1 | Pass2−Pass1 | 正常−置零 | 正常−跨樣本 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('| '+r['type']+' | '+' | '.join(f"{r[k+'_mean']:.2f}±{r[k+'_sd']:.2f}" for k in ['f1_pass1_pct','f1_normal_pct','pass2_minus_pass1_pp','normal_minus_zero_pp','normal_minus_shuffle_pp'])+' |')
    lines+=['','決策轉換的兩個方向均保留：錯轉對不等於整體改善，須同看對轉錯與Macro-F1。所有逐筆資料見predictions.csv。',
       '', '## 缺失區域誤差',
       '區域為缺失100點在原始與內插後V–I位置的聯集，向外以3×3方形膨脹一像素；未缺失對照段按固定亂數選取等長且時間上不重疊，但投影可能重疊，另列overlap_pixels。能量為R²占全圖R²總和，零分母列NaN。',
       'missing_energy_per_sample.csv保留全部21900筆；missing_energy_per_seed.csv為各模型seed內中位數與IQR，不把樣本或污染亂數當獨立模型重複。',
       '', '## 限制與回覆可用範圍',
       '這是固定權重下的test-time敏感度診斷。差值為負或很小亦完整呈現；不能把置零／跨樣本結果當嚴格因果證明，也不能把誤差集中視為確已恢復缺失資訊。',
       'E6實作時已看過E1–E4外層結果。圖依固定sample_id選擇，兩方向決策案例按固定順序取第一筆，不依有利程度挑圖。',
       '可選Grad-CAM未執行。原投稿主文未修改。核心稽核與Word／最終交付稽核是不同完成邊界。',
       '', '## Supplementary建議置入內容（待作者審閱）',
       '建議呈現上表、十一類X／Xhat／R圖、雙向決策轉換案例與缺失能量分布，並保留上述限制；正式章節、頁碼及圖號待實際改稿後填寫。']
    (output/'e6_report.md').write_text('\n'.join(lines))
    body='\n'.join('<p>'+html.escape(line)+'</p>' for line in lines)
    (output/'e6_report.html').write_text('<!doctype html><meta charset="utf-8"><title>E6 report</title>'+body)
    save_json(output/'core_audit.json',dict(all_checks_passed=True,items=310,prediction_rows=len(predictions),
         energy_rows=len(energies),models=10,model_condition_summaries=len(by_seed),
         item_sha256=item_hashes,source_sha256=source_hashes(),figures=len(manifest),
         checks='Saved IDs/labels, all logits, normal predictions vs E4, transitions, fixed derangement, region masks and independent squared-energy recomputation; 10 pollution realizations averaged within model seed.',
         final_delivery_complete=False,word_review_pending=True))

def figures(output,raw,test,predictions,summary,energy_seed,manifest):
    folder=output/'figures';folder.mkdir(exist_ok=True)
    ids=raw['sample_ids'][test];truth=raw['labels'][test]
    fixed=[min(np.flatnonzero(truth==c),key=lambda j:str(ids[j])) for c in range(11)]
    conditions=[CONDITIONS[i] for i in [0,1,11,21]]
    for cls,j in enumerate(fixed):
        fig,axes=plt.subplots(4,3,figsize=(6.6,8.6))
        for row,condition in enumerate(conditions):
            with np.load(output/'items'/f's42_{key(condition)}.npz') as d:
                p1=int(d['logits_pass1'][j].argmax());p2=int(d['logits_normal'][j].argmax())
                for col,(name,cmap) in enumerate([('image','gray'),('reconstruction','gray'),('residual','magma')]):
                    a=d[name][j,0];ax=axes[row,col];ax.imshow(a,cmap=cmap,vmin=0,vmax=1,origin='upper');ax.axis('off')
                    ax.set_title(f'{condition[0]} / {name}\ntrue {cls}, P1 {p1}, P2 {p2}',fontsize=7)
                    if condition[0]=='missing_consecutive':ax.contour(d['missing_mask'][j],levels=[.5],colors='red',linewidths=.35)
                    single,sax=plt.subplots(figsize=(2.2,2.2));sax.imshow(a,cmap=cmap,vmin=0,vmax=1,origin='upper');sax.axis('off')
                    if condition[0]=='missing_consecutive':sax.contour(d['missing_mask'][j],levels=[.5],colors='red',linewidths=.4)
                    sax.set_title(f'class {cls} | {condition[0]} | {name}\ntrue {cls}; P1 {p1}; P2 {p2}',fontsize=7)
                    single.tight_layout();path=folder/f'class{cls}_{condition[0]}_{name}.png';single.savefig(path,dpi=600);plt.close(single)
                    manifest.append(dict(file=str(path.relative_to(output)),sample_id=str(ids[j]),model_seed=42,condition=condition,kind=name,dpi=600))
        fig.suptitle(f'Class {cls} | sample {ids[j]} | model seed42',fontsize=10);fig.tight_layout()
        path=folder/f'class{cls}_four_conditions.png';fig.savefig(path,dpi=600);plt.close(fig)
        manifest.append(dict(file=str(path.relative_to(output)),sample_id=str(ids[j]),model_seed=42,kind='four_conditions',dpi=600))
    # Both conversion directions, fixed order; use seed42 to match saved reconstruction maps.
    cases=[]
    ordered=sorted([r for r in predictions if r['seed']==42],key=lambda r:(CONDITIONS.index((r['type'],r['severity'],r['pollution_seed'])),r['sample_id']))
    for direction in ['wrong_to_right','right_to_wrong']:
        candidates=[r for r in ordered if r['transition']==direction]
        if not candidates:cases.append(dict(direction=direction,available=False));continue
        r=candidates[0];cases.append(dict(direction=direction,available=True,**r));j=list(map(str,ids)).index(r['sample_id'])
        condition=(r['type'],r['severity'],r['pollution_seed'])
        with np.load(output/'items'/f's42_{key(condition)}.npz') as d:
            fig,axes=plt.subplots(1,3,figsize=(7,2.8))
            for ax,name in zip(axes,['image','reconstruction','residual']):
                ax.imshow(d[name][j,0],cmap='magma' if name=='residual' else 'gray',vmin=0,vmax=1);ax.axis('off');ax.set_title(name)
            fig.suptitle(f"{direction} | ID {r['sample_id']} | true {r['truth']} | P1 {r['pass1']} -> P2 {r['normal']}",fontsize=9)
            fig.tight_layout();path=folder/f'{direction}.png';fig.savefig(path,dpi=600);plt.close(fig)
            manifest.append(dict(file=str(path.relative_to(output)),sample_id=r['sample_id'],model_seed=42,condition=condition,kind=direction,dpi=600))
    save_json(output/'decision_cases.json',cases)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6));x=np.arange(4)
    for k,label in [('f1_pass1_pct','Pass1'),('f1_normal_pct','Normal R'),('f1_zero_pct','Zero R'),('f1_shuffle_pct','Cross-sample R')]:
        axes[0].errorbar(x,[r[k+'_mean'] for r in summary],yerr=[r[k+'_sd'] for r in summary],marker='o',capsize=2,label=label)
    axes[0].set_xticks(x,['Clean','AWGN10','SP5%','Missing20%'],rotation=15);axes[0].set_ylabel('Macro-F1 (%)');axes[0].legend(fontsize=7)
    for k,label in [('missing_energy_ratio','Missing region'),('control_energy_ratio','Control region')]:
        axes[1].plot(range(42,52),[r[k+'_median'] for r in energy_seed],marker='o',label=label)
    axes[1].set_xlabel('Model seed');axes[1].set_ylabel('Within-seed median residual energy fraction');axes[1].legend(fontsize=7)
    fig.tight_layout();path=folder/'mechanism_summary.png';fig.savefig(path,dpi=600);plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)),kind='summary',dpi=600))

if __name__=='__main__':analyze(OUTPUT)
