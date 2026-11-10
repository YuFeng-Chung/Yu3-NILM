"""Reporting-only original architecture contrasts and device-fold plot, no model selection."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results/e10_whited_20261108';OUT=ROOT/'e10_full_review_20261109'
def main():
 scores=list(csv.DictReader((DATA/'per_fold_seed.csv').open()))
 def score(n,s,f,m='macro_f1'):return float(next(r for r in scores if r['model']==n and int(r['seed'])==s and r['fold']==str(f))[m])
 rows=[]
 for a,b in [('M2','M1'),('M3','M1'),('M4','M2'),('M4','M3'),('M5','M3'),('M5','M1')]:
  for m in ['accuracy','macro_precision','macro_recall','macro_f1']:
   for s in [42,43,44]:rows.append(dict(a=a,b=b,metric=m,seed=s,difference_pp=100*(score(a,s,'OOF',m)-score(b,s,'OOF',m)),scope='descriptive report extension of original contrasts, no confirmatory test'))
 with (OUT/'report_contrasts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(figsize=(8,4));x=np.arange(1,4);colors=['#555555','#9a6b35','#5c8b66','#79638d','#007f8b']
 for j,(name,col) in enumerate(zip(['M1','M2','M3','M4','M5'],colors)):
  a=np.array([[100*score(name,s,f) for s in [42,43,44]] for f in [1,2,3]])
  ax.errorbar(x+(j-2)*.045,a.mean(1),yerr=a.std(1,ddof=1),marker='o',capsize=3,lw=1.2,color=col,label=name)
 ax.set(xticks=x,xticklabels=['Fold 1  n=309','Fold 2  n=230','Fold 3  n=220'],ylabel='Macro-F1 (%)',title='WHITED clean data by device-group fold',ylim=(20,55));ax.legend(ncol=5,loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.18)
 fig.tight_layout();fig.savefig(OUT/'figures/E10_F2_fold_comparison.png',dpi=600);fig.savefig(OUT/'figures/E10_F2_fold_comparison.svg');plt.close(fig)
if __name__=='__main__':main()
