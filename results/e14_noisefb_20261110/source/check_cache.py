from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;OUT=H.parent;ROOT=H.parents[2]
old=ROOT/'results/e1_e4_mac_20261106/image_cache/outer';new=OUT/'image_cache/outer'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=list(new.glob('*.npy'));assert len(paths)==211
for p in paths:
    assert (old/p.name).exists(),p.name
    assert sha(p)==sha(old/p.name),p.name
(OUT/'qa/image_cache_audit.json').write_text(json.dumps(dict(all_checks_passed=True,matching_cached_arrays=len(paths),scope='Byte-identical clean and210 E4 contaminated219-sample image arrays under identical normalization'),indent=2))
print('All211 image arrays byte-identical to E4')
