"""Reconstruct published PLAID windows and require exact clean-PNG pixel matching."""
import argparse
import ast
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def original_preprocessing():
    path = ROOT / '實驗一/plaid_loader_vi.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    names = {'get_activation_window', 'generate_vi_image'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in functions} == names
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    outer = zipfile.ZipFile(args.archive); archives = [outer]
    for name in outer.namelist():
        if name.lower().endswith('.zip') and not name.startswith('__MACOSX'):
            archives.append(zipfile.ZipFile(io.BytesIO(outer.read(name))))
    source = next(z for z in archives if len([n for n in z.namelist() if n.lower().endswith('.csv')]) == 1074)
    json_candidates = [(z, n) for z in archives for n in z.namelist()
                       if Path(n).name in ('meta_2014.json', 'meta1.json') and not n.startswith('__MACOSX')]
    assert len(json_candidates) == 1, [(n) for _, n in json_candidates]
    meta_bytes = json_candidates[0][0].read(json_candidates[0][1])
    metadata = json.loads(meta_bytes); assert len(metadata) == 1074
    by_id = {str(r['id']): r.get('meta', r) for r in metadata}
    csv_by_id = {Path(n).stem: n for n in source.namelist() if n.lower().endswith('.csv')}
    assert set(csv_by_id) == set(by_id) == {str(i) for i in range(1, 1075)}
    cache = np.load(args.images / 'e5_images.npz')
    manifest = json.loads((args.images / 'image_manifest.json').read_text())
    path_metadata = {r['path']: r for r in manifest['samples']}
    lookup = defaultdict(deque)
    for partition in ('train', 'val'):
        for image, path in zip(cache[partition + '_images'], cache[partition + '_ids']):
            group = path_metadata[str(path)]['group']
            lookup[(group, hashlib.sha256(image.squeeze().tobytes()).hexdigest())].append((partition, str(path)))
    ambiguous_keys = sum(len(q) > 1 for q in lookup.values())
    functions, loader_hash = original_preprocessing()
    crop, build_image = functions['get_activation_window'], functions['generate_vi_image']

    def read(sid):
        data = source.read(csv_by_id[sid])
        frame = pd.read_csv(io.BytesIO(data), header=None)
        assert frame.shape[1] == 2 and len(frame) >= 2, (sid, frame.shape)
        values = frame.to_numpy(dtype=np.float64)
        assert np.isfinite(values).all(), f'Nonfinite raw data: {sid}'
        current, voltage = crop(values[:, 0], values[:, 1], window_len=500)
        assert len(current) == len(voltage) == 500
        current, voltage = current.copy(), voltage.copy()
        image = (build_image(current, voltage).squeeze() * 255).astype(np.uint8)
        meta = by_id[sid]; label = meta['appliance']['type']; group = f"{label}_house{meta.get('location', 'unknown')}"
        return current, voltage, image, {'sample_id': sid, 'group': group, 'label': label,
            'csv_sha256': hashlib.sha256(data).hexdigest(), 'csv_bytes': len(data), 'raw_rows': len(frame)}

    currents, voltages, records, missing = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (current, voltage, image, record) in enumerate(pool.map(read, sorted(by_id, key=int)), 1):
            key = (record['group'], hashlib.sha256(image.tobytes()).hexdigest())
            if not lookup[key]:
                missing.append(record['sample_id']); continue
            partition, path = lookup[key].popleft()
            record.update({'partition': partition, 'original_png_path': path})
            currents.append(current); voltages.append(voltage); records.append(record)
            if index % 100 == 0 or index == 1074: print(f'Read {index}/1074 CSV; unmatched={len(missing)}', flush=True)
    unmatched_png = [path for q in lookup.values() for _, path in q]
    audit = {'source': 'https://doi.org/10.6084/m9.figshare.11605074.v1',
             'published_archive_md5': hashlib.md5(args.archive.read_bytes()).hexdigest(),
             'original_loader_sha256': loader_hash, 'metadata_sha256': hashlib.sha256(meta_bytes).hexdigest(),
             'raw_sample_count': len(metadata), 'matched_clean_images': len(records),
             'unmatched_raw_ids': missing, 'unmatched_png_paths': unmatched_png,
             'duplicate_pixel_group_keys': ambiguous_keys,
             'mapping_note': 'Exact pixel multiset and group matching. Duplicate clean images within a group cannot identify unique historical raw-to-PNG correspondence.',
             'records': records}
    (args.output / 'waveform_image_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    assert not missing and not unmatched_png and len(records) == 1074, 'Clean image mismatch: do not perform corrupted evaluation.'
    assert sum(r['partition'] == 'train' for r in records) == 855
    assert sum(r['partition'] == 'val' for r in records) == 219
    np.savez_compressed(args.output / 'raw_windows.npz', current=np.stack(currents), voltage=np.stack(voltages),
        sample_ids=np.asarray([r['sample_id'] for r in records]), groups=np.asarray([r['group'] for r in records]),
        labels=np.asarray([manifest['classes'].index(r['label']) for r in records]),
        partitions=np.asarray([r['partition'] for r in records]), original_png_paths=np.asarray([r['original_png_path'] for r in records]))
    (args.output / 'meta_2014.json').write_bytes(meta_bytes)
    print('PASS all 1074 published waveforms reconstruct supplied clean PNGs exactly.', flush=True)


if __name__ == '__main__': main()
