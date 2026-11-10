"""Independent reconstruction of frozen E8 raw timing, quality gates and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/e8_cpu_20261108'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    summary=list(csv.DictReader((OUT/'latency_summary.csv').open()))
    checks={};total=0;attempt_count=0
    for runtime,threads in protocol['setting_order']:
        data=json.loads((OUT/f'timing_{runtime}_{threads}.json').read_text())
        good=[]
        for attempt in data['attempts']:
            attempt_count+=1
            ext=[r['external_cpu_percent_machine'] for r in attempt['monitor']]
            valid=bool(ext and np.mean(ext)<=15 and max(ext)<=35 and attempt['ac_start'] and attempt['ac_end'])
            assert valid==attempt['accepted']
            assert len(attempt['blocks'])==5
            assert [b['path'] for b in attempt['blocks']]==protocol['path_orders'][attempt['round_index']]
            for block in attempt['blocks']:
                assert len(block['ns'])==100 and min(block['ns'])>0
            if valid:good.append(attempt)
        assert len(good)==30 and data['accepted_rounds']==30 and data['status']=='measured'
        assert [a['round_index'] for a in good]==list(range(30))
        for name in protocol['models']:
            raw=np.array([next(b['ns'] for b in a['blocks'] if b['path']==name) for a in good])/1e6
            row=next(r for r in summary if r['runtime']==runtime and int(r['threads'])==threads and r['path']==name)
            med=np.median(raw,axis=1)
            expected=dict(mean_ms=raw.mean(),std_ms=raw.std(ddof=1),median_ms=np.median(raw),
                p95_ms=np.percentile(raw,95),block_median_mean_ms=med.mean(),block_median_sd_ms=med.std(ddof=1),
                last10_first10_ratio=np.median(med[-10:])/np.median(med[:10]))
            for key,value in expected.items():assert np.isclose(float(row[key]),value,rtol=1e-12,atol=1e-12)
            total+=raw.size
        checks[f'{runtime}_{threads}_raw_reconstruction']=True
    assert total==60000 and len(summary)==20
    checks['all_raw_calls_and_summary_reconstructed']=True
    for name,model in protocol['models'].items():
        assert sha(ROOT/model['checkpoint'])==model['checkpoint_sha256']
        assert sha(OUT/f'{name}.onnx')==model['onnx_sha256']
        assert model['argmax_equal'] and model['parity_cases']==16
    checks['checkpoint_graph_hashes_and_parity']=True
    for rel,digest in protocol['source_hashes'].items():assert sha(ROOT/rel)==digest
    checks['frozen_source_hashes']=True
    for runtime,threads in protocol['setting_order']:
        for name in protocol['models']:
            row=json.loads((OUT/f'memory_{runtime}_{threads}_{name}.json').read_text())
            assert row['runtime']==runtime and row['threads']==threads and row['path']==name
            assert row['load_delta_rss_bytes']==row['loaded_rss_bytes']-row['baseline_rss_bytes']
            assert row['warmup_delta_rss_bytes']==row['after_warmup_rss_bytes']-row['loaded_rss_bytes']
            assert row['peak_lifetime_rss_bytes']>0
    checks['twenty_separate_memory_records']=True
    amendment=json.loads((OUT/'rerun_amendment.json').read_text())
    assert sha(OUT/'protocol.json')==amendment['original_protocol_sha256']
    initial=json.loads((OUT/amendment['preserved_initial_file']).read_text())
    assert initial['status']=='incomplete' and len(initial['attempts'])==40
    assert sum(a['accepted'] for a in initial['attempts'])==amendment['initial_accepted_rounds']==23
    for attempt in initial['attempts']:
        ext=[r['external_cpu_percent_machine'] for r in attempt['monitor']]
        assert attempt['accepted']==bool(ext and np.mean(ext)<=15 and max(ext)<=35 and attempt['ac_start'] and attempt['ac_end'])
        assert [b['path'] for b in attempt['blocks']]==protocol['path_orders'][attempt['round_index']]
        assert all(len(b['ns'])==100 and min(b['ns'])>0 for b in attempt['blocks'])
    quiet=json.loads((OUT/'rerun_quiet_gate.json').read_text())
    assert len(quiet)>=30 and all(r['cpu_percent']<=15 for r in quiet[-30:])
    assert quiet[-1]['epoch']-quiet[-30]['epoch']>=28.9
    parity4=json.loads((OUT/'additional_parity_4threads.json').read_text())
    assert len(parity4)==5 and all(r['threads']==4 and r['cases']==4 for r in parity4)
    checks['initial_incomplete_session_retained_and_criteria_verified']=True
    checks['whole_setting_rerun_amendment_and_30second_gate']=True
    checks['additional_four_thread_parity_present']=True
    result=dict(all_checks_passed=all(checks.values()),checks=checks,accepted_calls=total,
                total_round_attempts=attempt_count,note='Independent integrity audit; consult measurement_audit drift flags for stability limitations.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
