"""Post-hoc paired device-group bootstrap of fixed E10 OOF predictions, no retraining."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/e10_whited_20261108';OUT=ROOT/'e10_statistics_20261109'
METRICS=['accuracy','macro_precision','macro_recall','macro_f1']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def write(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scores(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);den=cm.sum(-2)
 pr=np.divide(tp,den,out=np.zeros_like(tp,dtype=float),where=den>0);re=tp/support
 f=np.divide(2*pr*re,pr+re,out=np.zeros_like(pr),where=pr+re>0)
 return np.stack([tp.sum(-1)/cm.sum((-1,-2)),pr.mean(-1),re.mean(-1),f.mean(-1)],axis=-1)*100

def main():
 plan=json.loads((OUT/'analysis_plan.json').read_text());names=plan['models'];seeds=plan['seeds'];B=plan['n_resamples']
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h
 raw=dict(np.load(DATA/'images.npz'));y=raw['labels'];groups=raw['groups'];k=len(raw['classes']);unique=sorted(set(groups));G=len(unique);assert G==76
 ix=[np.flatnonzero(groups==g) for g in unique];group_class=np.array([y[a[0]] for a in ix]);assert all(len(set(y[a]))==1 for a in ix)
 pred=np.empty((5,3,len(y)),int);cms=np.zeros((G,5,3,k,k),np.int64)
 for m,n in enumerate(names):
  for s,seed in enumerate(seeds):
   a=dict(np.load(DATA/f'oof_{n}_s{seed}.npz'));assert np.array_equal(a['sample_ids'],raw['sample_ids']) and np.array_equal(a['labels'],y)
   pred[m,s]=a['predictions']
   for g,idx in enumerate(ix):cms[g,m,s]=np.bincount(y[idx]*k+pred[m,s,idx],minlength=k*k).reshape(k,k)
 point=scores(cms.sum(0)).mean(1)
 target=list(csv.DictReader((DATA/'summary.csv').open()))
 for m,n in enumerate(names):
  for j,metric in enumerate(METRICS):assert abs(point[m,j]-100*float(next(r for r in target if r['model']==n and r['metric']==metric)['mean']))<1e-12
 rng=np.random.default_rng(plan['rng_seed']);weights=np.zeros((B,G),np.int16)
 for c in range(k):
  subset=np.flatnonzero(group_class==c);n=len(subset);weights[:,subset]=rng.multinomial(n,np.full(n,1/n),size=B)
  assert np.all(weights[:,subset].sum(1)==n)
 assert np.all(weights.sum(1)==G)
 distribution=np.empty((B,5,4))
 for start in range(0,B,250):
  cm=np.einsum('bg,gmsij->bmsij',weights[start:start+250],cms,optimize=True)
  distribution[start:start+250]=scores(cm).mean(2)
 # Direct row replication and Boolean TP/FP/FN verification of50 fixed resamples.
 checks=np.linspace(0,B-1,50,dtype=int)
 for b in checks:
  idx=np.concatenate([np.tile(a,int(w)) for a,w in zip(ix,weights[b]) if w])
  for m in range(5):
   vals=[]
   for s in range(3):
    yy=y[idx];pp=pred[m,s,idx];pr=[];re=[];ff=[]
    for c in range(k):
     tp=np.sum((yy==c)&(pp==c));fp=np.sum((yy!=c)&(pp==c));fn=np.sum((yy==c)&(pp!=c))
     pr.append(tp/(tp+fp) if tp+fp else 0);re.append(tp/(tp+fn));ff.append(2*tp/(2*tp+fp+fn))
    vals.append([np.mean(yy==pp),np.mean(pr),np.mean(re),np.mean(ff)])
   np.testing.assert_allclose(np.mean(vals,axis=0)*100,distribution[b,m],atol=1e-12,rtol=0)
 rows=[]
 for m,n in enumerate(names):
  for j,metric in enumerate(METRICS):
   lo,hi=np.percentile(distribution[:,m,j],[2.5,97.5]);rows.append(dict(model=n,metric=metric,point_pct=float(point[m,j]),bootstrap_low_pct=float(lo),bootstrap_high_pct=float(hi),resamples=B,interval_type='descriptive conditional unadjusted central95percentile; no retraining coverage claim'))
 pairs=[]
 for a,b in plan['original_contrasts']+plan['additional_reference']:
  ai=names.index(a);bi=names.index(b)
  for j,metric in enumerate(METRICS):
   dd=distribution[:,ai,j]-distribution[:,bi,j];lo,hi=np.percentile(dd,[2.5,97.5]);pairs.append(dict(a=a,b=b,metric=metric,point_pp=float(point[ai,j]-point[bi,j]),bootstrap_low_pp=float(lo),bootstrap_high_pp=float(hi),resamples=B,role='originalcontrast' if [a,b] in plan['original_contrasts'] else 'additionalreference',interval_type='descriptive conditional unadjusted central95percentile; not a significance test'))
 write(OUT/'model_intervals.csv',rows);write(OUT/'paired_intervals.csv',pairs)
 np.savez_compressed(OUT/'bootstrap_replicates.npz',group_weights=weights,model_metric_pct=distribution,group_names=np.array(unique),model_names=np.array(names),metric_names=np.array(METRICS))
 report=['# E10補充統計分析','', '本次為看過研究結果後新增的描述性分析，不重訓、不重選模型或前處理。','',
  '## 原稿既有統計','',
  '原Experiment1正文II-E與III-A／TableIV使用30個配對訓練run：四指標Friedman整體檢定及五組Wilcoxon架構對比。TableIV為未校正p，原文以0.05–0.10描述趨勢，修稿須依審稿意見移除過強trend說法。原Macro-F1 Friedman χ²(4)=1.84、p=.765，M5−M3原Macro-F1未校正p=.088，均不是確立優勢。',
  '送審Supplementary S5為切分敏感度，列Mann–Whitney U p<.001且註探索性；S6為Experiment1延伸的M3/M5污染比較，n10、按scenario×perturbation type作Holm校正。S7–S9則屬Experiment3，不能混稱Experiment1。原FigS1有run分布，不等於bootstrap區間。','',
  '## 為何E10不直接複製檢定','',
  'E10只有3個配對訓練seed；n3、非零且無tie時雙尾精確Wilcoxon最小p=2/8=.25，五對比Holm第一步=.01。任何seed-level對比都無法達.05。三fold有訓練重疊，不把三fold×三seed當n9。本文未新做Friedman近似檢定或確認性p檢定，也不由未顯著推論等效。','',
  '## 追加方法','',
  '固定原75組模型（15pilot＋45正式訓練之正式45組用於推論；不把pilot當結果），固定既有OOF預測與3seed。76個設備名稱group按15類分層，各類保留相同group數，有放回抽10000次（rng20261109），每group的所有量測一起抽；同一抽樣權重套用所有模型及seed。每次各seed重算四指標，再平均三個固定seed，配對差也在指標重算後相減。',
  '報告2.5／97.5百分位之中央95%描述性bootstrap區間，未做多重比較區間校正。此為固定OOF預測下的設備組成敏感度，沒有重訓、沒有修復fold相依、沒有可靠評估seed母群變異；不能當作跨資料重訓泛化效果具95%涵蓋率的信賴區間，也不能把不含零當確認性顯著或等效證據。','',
  '## Macro F1配對差','', '| 對比 | 平均差pp | 描述性bootstrap區間pp |','|---|---:|---:|']
 for r in pairs:
  if r['metric']=='macro_f1':report.append(f"| {r['a']} − {r['b']} | {r['point_pp']:+.2f} | [{r['bootstrap_low_pp']:+.2f}, {r['bootstrap_high_pp']:+.2f}] |")
 report+=['','完整四指標模型區間及配對區間見CSV；原seed SD仍分開保留。區間是描述性補強，不能宣稱修復原REDD統計檢定力或已完整證實架構優勢。','',
  '來源：SciPy bootstrap官方文件 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html ；Bengio與Grandvalet 2004 https://www.jmlr.org/papers/v5/grandvalet04a.html 。後者指出CV誤差相關性需特別處理，並非替本次bootstrap涵蓋率背書。','']
 # Exact model count correction in prose; only45 trained final models are fixed.
 report=[s.replace('固定原75組模型（15pilot＋45正式訓練之正式45組用於推論；不把pilot當結果）','固定45組正式模型（15pilot僅供選epoch，不當成結果模型）') for s in report]
 (OUT/'統計分析說明.md').write_text('\n'.join(report))
 for rel,h in plan['source_sha256'].items():assert sha(ROOT/rel)==h
 save(OUT/'analysis_audit.json',dict(all_checks_passed=True,n_resamples=B,paired=True,class_stratified_groups=76,all_class_group_counts_preserved=True,point_estimates_match_original=True,
  independent_direct_replications_checked=50,model_metric_checks=50*5*4,source_files_unchanged=True,model_rows=20,paired_rows=24,
  original_source_main='投稿/Yu3_Manuscript_0922Final.docx',original_source_supplement='投稿/Yu3_Supplementary_0922Final.pdf',
  no_retraining=True,no_confirmatory_inference=True,script_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'analysis_plan.json')))
 print('\n'.join(report[report.index('## Macro F1配對差'):]))
if __name__=='__main__':main()
