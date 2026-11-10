"""Read official archive; metadata-first inclusion, no classifier/test selection."""
import io
import re
import shutil
from collections import defaultdict
from zipfile import ZipFile
import soundfile as sf
from scipy.signal import resample_poly
from e10_core import *

FACTORS={'MK1':(1033.64,61.4835),'MK2':(861.15,60.200),'MK3':(988.926,60.9562)}

def image(current,voltage):
    def z(x): return (x-x.mean())/x.std() if x.std()>1e-5 else x.copy()
    i,v=z(current),z(voltage)
    iy=np.clip(((i+3)/6*63).astype(int),0,63);ix=np.clip(((v+3)/6*63).astype(int),0,63)
    a=np.zeros((64,64),np.float32);np.add.at(a,(iy,ix),np.float32(.1))
    return (np.clip(a,0,1)*255).astype(np.uint8)[None]

def window(x,fs,kit):
    # Channel0 behaves as line voltage with README voltFactor, channel1 as load current.
    v=x[:,0]*FACTORS[kit][0];c=x[:,1]*FACTORS[kit][1]
    assert np.isfinite(x).all() and np.std(v)>1e-5 and np.std(c)>1e-5
    spectrum=np.abs(np.fft.rfft((v-v.mean())*np.hanning(len(v))))**2
    freq=np.fft.rfftfreq(len(v),1/fs);mask=(freq>=45)&(freq<=65)
    peak=float(freq[np.flatnonzero(mask)[np.argmax(spectrum[mask])]])
    nominal=50 if abs(peak-50)<abs(peak-60) else 60
    assert abs(peak-nominal)<1,peak
    n=round(fs/nominal);event=int(np.argmax(np.abs(np.diff(c))))
    start=max(0,min(len(c)-n,event-round(.2*n)))
    # Anti-aliased rational resampling of exactly one nominal mains-cycle duration.
    cc=resample_poly(c[start:start+n],500,n);vv=resample_poly(v[start:start+n],500,n)
    assert len(cc)==len(vv)==500
    assert 60<float(np.sqrt(np.mean(v*v)))<300, 'Unexpected calibrated voltage: channel mapping needs review'
    return cc,vv,dict(fs=fs,raw_points=len(c),voltage_rms=float(np.sqrt(np.mean(v*v))),current_rms=float(np.sqrt(np.mean(c*c))),
        frequency_peak_hz=peak,nominal_frequency_hz=nominal,native_window_points=n,event_index=event,start_index=start,
        normalized_raw_saturation_fraction=float(np.mean(np.abs(x)>=.999)),
        event_note='largest absolute current first difference; not verified turn-on or steady state')

def main():
    torch.set_num_threads(1);OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'prepared_audit.json').exists():
        a=json.loads((OUTPUT/'prepared_audit.json').read_text())
        assert a['all_checks_passed'] and a['cache_sha256']==sha(OUTPUT/'images.npz');return
    archive=DATA/'WhiteD.zip';records=[];groups=defaultdict(set)
    with ZipFile(archive) as z:
        readme=z.read('DATEN/_readme.txt').decode();(OUTPUT/'official_archive_readme.txt').write_text(readme)
        for n in sorted(z.namelist()):
            if n.count('/')!=1 or not n.startswith('DATEN/') or not n.lower().endswith('.flac'):continue
            stem=Path(n).stem;body,region,kit,stamp=stem.rsplit('_',3);label,device=body.split('_',1)
            assert re.fullmatch(r'\d{14}',stamp) and kit in FACTORS
            region=region.lower();label=label.casefold()
            group=label+'|'+device.casefold()+'|'+region
            # README documents r1..r8; r9 is unidentified and omitted before scores.
            known_region=region in ['r'+str(i) for i in range(1,9)]
            records.append(dict(path=n,label=label,device=device,region=region,kit=kit,timestamp=stamp,group=group,
                included=False,reason='pending_device_count' if known_region else 'region_not_documented_in_archive_readme'))
            if known_region:groups[label].add(group)
        classes=sorted(c for c,g in groups.items() if len(g)>=3)
        assert len(classes)>=5, 'Insufficient eligible classes; no automatic fallback'
        arrays=[];curr=[];volt=[];included=[];diagnostics=[];hash_groups={}
        for rec in records:
            if rec['reason'].startswith('region_'):continue
            if rec['label'] not in classes:rec['reason']='fewer_than_three_device_groups';continue
            data=z.read(rec['path']);digest=hashlib.sha256(data).hexdigest()
            if digest in hash_groups:assert hash_groups[digest]==rec['group'],'Byte-identical files across device groups'
            hash_groups[digest]=rec['group']
            x,fs=sf.read(io.BytesIO(data),dtype='float64',always_2d=True)
            assert x.shape[1]==2 and fs==44100
            c,v,d=window(x,fs,rec['kit']);a=image(c,v)
            # Independent original PLAID pre-PNG imaging parity for every selected sample.
            from prepare_e5_waveforms import original_preprocessing
            if not arrays:fns,_=original_preprocessing()
            expected=(fns['generate_vi_image'](c.copy(),v.copy())*255).astype(np.uint8)
            assert np.array_equal(a,expected),rec['path']
            rec.update(included=True,reason='eligible',sha256=digest,label_index=classes.index(rec['label']),index=len(arrays))
            arrays.append(a);curr.append(c);volt.append(v);included.append(rec)
            diagnostics.append(dict(path=rec['path'],**d))
        group_array=np.array([r['group'] for r in included]);labels=np.array([r['label_index'] for r in included])
        # Deterministic, stratified threefold group CV; test folds partition included devices.
        rng=np.random.default_rng(20261108);class_folds={}
        for c in classes:
            g=np.array(sorted(groups[c]));rng.shuffle(g);class_folds[c]=[a.tolist() for a in np.array_split(g,3)]
        splits=[]
        for fold in range(3):
            test=set();val=set()
            for c in classes:
                parts=class_folds[c];test.update(parts[fold]);remaining=[g for j,part in enumerate(parts) if j!=fold for g in part]
                val.add(sorted(remaining)[0])
            te=np.flatnonzero(np.isin(group_array,list(test)));va=np.flatnonzero(np.isin(group_array,list(val)))
            tr=np.flatnonzero(~np.isin(group_array,list(test|val)));outer=np.flatnonzero(~np.isin(group_array,list(test)))
            for a,b in [(tr,va),(tr,te),(va,te)]:assert not set(group_array[a])&set(group_array[b])
            for a in [tr,va,te,outer]:assert set(labels[a])==set(range(len(classes)))
            splits.append(dict(fold=fold+1,inner_train=tr.tolist(),inner_validation=va.tolist(),outer_train=outer.tolist(),outer_test=te.tolist()))
        all_test=[j for s in splits for j in s['outer_test']];assert sorted(all_test)==list(range(len(arrays)))
        np.savez_compressed(OUTPUT/'images.npz',images=np.stack(arrays),labels=labels,groups=group_array,
            sample_ids=np.array([r['path'] for r in included]),classes=np.array(classes),current=np.stack(curr),voltage=np.stack(volt))
        save(OUTPUT/'split.json',splits);save(OUTPUT/'sample_manifest.json',records)
        csv_write(OUTPUT/'waveform_diagnostics.csv',diagnostics)
        counts=[dict(label=c,device_groups=len(groups[c]),recordings=sum(r['label']==c and not r['reason'].startswith('region_') for r in records),
            included=c in classes,criterion='at least3 groups for distinct inner-train/validation/outer-test') for c in sorted(groups)]
        csv_write(OUTPUT/'class_coverage.csv',counts)
        audit=dict(all_checks_passed=True,at=now(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
            official_page_measurements=1259,root_flac_count=len(records),readme_regions_flac_count=sum(not r['reason'].startswith('region_') for r in records),
            undocumented_region_excluded=sum(r['reason'].startswith('region_') for r in records),classes=classes,selected_samples=len(arrays),
            selected_device_groups=len(set(group_array)),all_original_imaging_parity=True,all_test_groups_partitioned_once=True,
            group_identity='casefold(class, device-name, region), recording kit and timestamp excluded; conservative filename identity, not independently verified physical serial number',
            channel_mapping='0=voltage with voltFactor,1=current with currFactor; checked calibrated line voltage60..300V and45..65Hz for every selected waveform; README does not explicitly label channel order',
            cache_sha256=sha(OUTPUT/'images.npz'),split_sha256=sha(OUTPUT/'split.json'),manifest_sha256=sha(OUTPUT/'sample_manifest.json'))
        save(OUTPUT/'prepared_audit.json',audit)
        print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
