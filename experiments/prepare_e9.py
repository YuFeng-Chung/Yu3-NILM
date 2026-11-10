"""Read full public CSVs, freeze label-independent steady rule, preserve coverage."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from e9_core import *
from prepare_e5_waveforms import original_preprocessing
from run_e2_search_mps import inner_split

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    assert not (OUTPUT/'protocol.json').exists(),'Frozen E9 exists; do not overwrite'
    save_json(OUTPUT/'window_rule_before_data.json',WINDOW_RULE)
    old=dict(np.load(DATA/'raw_windows.npz'));audit=json.loads((DATA/'waveform_image_audit.json').read_text());records={r['sample_id']:r for r in audit['records']}
    assert hashlib.md5((DATA/'plaid_2014_published.zip').read_bytes()).hexdigest()=='5997e0b22ad44e3f46c9ac74bb57fcda'
    outer=zipfile.ZipFile(DATA/'plaid_2014_published.zip');archive=zipfile.ZipFile(io.BytesIO(outer.read('2014.zip')))
    files={Path(p).stem:p for p in archive.namelist() if p.endswith('.csv') and not p.startswith('__MACOSX')};assert len(files)==1074
    functions,_=original_preprocessing();crop=functions['get_activation_window'];rows=[]
    def process(j):
        sid=str(old['sample_ids'][j]);content=archive.read(files[sid]);assert hashlib.sha256(content).hexdigest()==records[sid]['csv_sha256']
        x=pd.read_csv(io.BytesIO(content),header=None).to_numpy(dtype=np.float64);assert x.shape[1]==2 and np.isfinite(x).all()
        c,v=x[:,0],x[:,1];a,b=crop(c,v,window_len=500)
        assert np.array_equal(a,old['current'][j]) and np.array_equal(b,old['voltage'][j])
        ci,vi,info=steady_window(c,v)
        info.update(sample_id=sid,label=int(old['labels'][j]),group=str(old['groups'][j]),partition=str(old['partitions'][j]),raw_rows=len(c),
            largest_difference_index=int(np.argmax(np.abs(np.diff(c)))),csv_sha256=hashlib.sha256(content).hexdigest())
        return ci,vi,info
    currents=[];voltages=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k,(c,v,info) in enumerate(pool.map(process,range(len(old['labels']))),1):
            currents.append(c);voltages.append(v);rows.append(info)
            if k%100==0 or k==1074:print('E9 full CSV',k,'eligible',sum(r['eligible'] for r in rows),flush=True)
    raw={k:old[k] for k in ['sample_ids','labels','groups','partitions','original_png_paths']}
    raw.update(W0_current=old['current'],W0_voltage=old['voltage'],W1_current=np.stack(currents),W1_voltage=np.stack(voltages),eligible=np.array([r['eligible'] for r in rows]))
    np.savez_compressed(OUTPUT/'waveforms.npz',**raw);csv_write(OUTPUT/'window_manifest.csv',rows)
    train,val,test=inner_split(old);eligible=np.flatnonzero(raw['eligible'])
    split={k:np.intersect1d(v,eligible).tolist() for k,v in [('inner_train',train),('inner_validation',val),('outer_train',np.flatnonzero(old['partitions']=='train')),('outer_test',test)]}
    for idx in split.values():assert set(raw['labels'][idx])==set(range(11)),'Missing class after coverage restriction'
    save_json(OUTPUT/'split.json',dict(indices=split,sample_ids={k:raw['sample_ids'][v].tolist() for k,v in split.items()},
        note='Original E2 inner and original855/219 outer membership intersected with fixed-rule W1 eligibility for all four cells; no outcome-driven exclusions.'))
    coverage=[]
    for part in ['train','val']:
        for label in range(11):
            m=(raw['partitions']==part)&(raw['labels']==label)
            coverage.append(dict(partition=part,label=label,total=int(m.sum()),eligible=int((m&raw['eligible']).sum())))
    csv_write(OUTPUT/'coverage.csv',coverage)
    save_json(OUTPUT/'data_audit.json',dict(all_checks_passed=True,read_full_csv=1074,raw_sha256_match=1074,W0_exact_crop_match=1074,
        eligible=int(raw['eligible'].sum()),excluded_ids=raw['sample_ids'][~raw['eligible']].tolist(),original_counts=dict(train=855,test=219),
        common_counts={k:len(v) for k,v in split.items()},window_rule=WINDOW_RULE,raw_archive_sha256=sha(DATA/'plaid_2014_published.zip')))
    print('DATA PASS', {k:len(v) for k,v in split.items()},flush=True)

if __name__=='__main__':main()
