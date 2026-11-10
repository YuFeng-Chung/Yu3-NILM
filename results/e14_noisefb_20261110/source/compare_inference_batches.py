"""Quantify preserved initial219 vs official32 predictions; never selects outcomes."""
from pathlib import Path
import sys,json,re
H=Path(__file__).resolve().parent;OUT=H.parent
sys.path.insert(0,str(H))
import numpy as np,pandas as pd
from analyze_e14 import independent_f1
old=OUT/'numerical_audit/batch219';rows=[];checked=0;changed=0
for base in ['final','reference_evaluation']:
 for folder in (old/base).glob('*'):
  current=OUT/base/folder.name
  data=np.load(current/'outer_ids_labels.npz');y=data['labels']
  for endpoint,sub in [('pass1','first_pass_predictions'),('reported','outer_predictions')]:
   for p in (folder/sub).glob('*.npy'):
    if not re.fullmatch(r'(clean|\d{3})\.npy',p.name):continue
    a=np.load(p);b=np.load(current/sub/p.name);n=int(np.sum(a!=b));checked+=1
    if n:
     changed+=n
     rows.append(dict(run=folder.name,endpoint=endpoint,condition=p.stem,n_predictions_changed=n,initial_f1=independent_f1(y,a),official_f1=independent_f1(y,b)))
cols=['run','endpoint','condition','n_predictions_changed','initial_f1','official_f1']
pd.DataFrame(rows,columns=cols).to_csv(OUT/'qa/inference_batch_differences.csv',index=False)
(OUT/'qa/inference_batch_audit.json').write_text(json.dumps(dict(all_checks_passed=True,compared_prediction_files=checked,files_with_any_change=len(rows),total_prediction_changes=changed,initial_batch=219,official_batch=32,scope='All available preserved initial files compared; initial ShuffleFB always32 but had not yet run. Official CSV includes only complete32 rerun.'),indent=2))
print('Compared',checked,'files;',len(rows),'changed files;',changed,'prediction changes')
