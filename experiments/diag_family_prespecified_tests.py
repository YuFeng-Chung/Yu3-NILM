import pandas as pd, numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'e1_e4_mac_20261106'
OUTDIR = ROOT / 'results' / 'diagnostics'; OUTDIR.mkdir(parents=True, exist_ok=True)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run,min(1,(m-i)*p[j])); adj[j]=run
    return adj
d=pd.read_csv(RES / 'per_seed_averaged.csv'); d=d[d.strategy.isin(['no_defense','S3','combined'])]
d['F1']=d.f1*100; d['Dabs']=d.Dabs_pp
C=[('Yu3','RecAux'),('RecAux','M1'),('Yu3','M1'),('Yu3','M3')]
rows=[]
for (st,ty,sv),g in d.groupby(['strategy','type','severity']):
  for ep in ['F1','Dabs']:
    if ty=='clean' and ep=='Dabs': continue
    ps=[];tmp=[]
    for a,b in C:
      x=g[g.model==a].sort_values('seed')[ep].values; y=g[g.model==b].sort_values('seed')[ep].values
      dd=x-y
      p=1.0 if np.allclose(dd,0) else wilcoxon(x,y,method='exact').pvalue
      tmp.append(dict(strategy=st,type=ty,severity=sv,endpoint=ep,contrast=f'{a}-{b}',mean=dd.mean(),sd=dd.std(ddof=1),pos=(dd>0).sum(),p_raw=p))
    adj=holm([t['p_raw'] for t in tmp])
    for t,a in zip(tmp,adj): t['p_holm4']=a; rows.append(t)
R=pd.DataFrame(rows); R.to_csv(OUTDIR / 'family_prespecified.csv',index=False)
print(len(R),'tests; sig:',(R.p_holm4<.05).sum())
print(R[R.p_holm4<.05].round(4).to_string())
# check against Codex raw p
old=pd.read_csv(RES / 'paired_tests.csv'); old=old[old.family=='E1_all_common']
m=old.merge(R,left_on=['comparison','type','severity','endpoint'],right_on=['contrast','type','severity','endpoint'])
m=m[m.context=='common/'+m.strategy]
print('max raw p diff vs Codex:',(m.p_raw_x-m.p_raw_y).abs().max(), len(m))
