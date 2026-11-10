from pathlib import Path
import pandas as pd,numpy as np,json
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
a=pd.read_csv(ROOT/'claude_diagnostics_20261109/nodefense_diag_raw.csv')
b=pd.read_csv(OUT/'per_seed_condition_realization.csv');b=b[b.endpoint=='pass1']
joined=a.merge(b,left_on=['model','seed','type','severity','ps'],right_on=['model','seed','type','severity','pollution_seed'],validate='one_to_one')
assert len(joined)==len(a)==930
err=np.abs(joined.pass1-joined.macro_f1).max()
assert err<1e-12,err
(OUT/'qa/reference_diagnostics_audit.json').write_text(json.dumps(dict(all_checks_passed=True,matched_rows=len(joined),max_abs_f1_difference=err,scope='Old Yu3/ZeroFB/ShuffleFB pass1 primary conditions vs saved Claude CPU diagnostics'),indent=2))
print('Reference diagnostic matched',len(joined),'rows')
