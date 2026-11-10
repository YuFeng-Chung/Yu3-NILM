"""Validate the supplied PNG split and cache its exact uint8 images for E5."""
import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in (args.dataset / 'train').iterdir() if p.is_dir())
    record_bytes = (args.dataset / 'split_record.json').read_bytes()
    record = json.loads(record_bytes)
    manifest = {'classes': classes, 'split_sha256': hashlib.sha256(record_bytes).hexdigest(),
                'note': 'Supplied preprocessed PNGs; source-waveform equivalence not established.', 'samples': []}
    arrays = {}
    sets = {}
    for partition, expected, key in [('train', 855, 'train_appliance_ids'), ('val', 219, 'val_appliance_ids')]:
        files = sorted((args.dataset / partition).rglob('*.png'))
        assert len(files) == expected, (partition, len(files))
        aliases = {group.replace(' ', '_'): group for group in record[key]}
        assert len(aliases) == len(record[key]), 'Ambiguous filename group aliases'

        def read(path):
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                assert im.mode == 'L' and im.size == (64, 64), (str(path), im.mode, im.size)
                a = np.asarray(im).copy()
            filename_group = path.stem.rsplit('__', 1)[0]
            group = filename_group if filename_group in record[key] else aliases.get(filename_group)
            assert group in record[key], (str(path), group)
            return a, classes.index(path.parent.name), {'path': str(path.relative_to(args.dataset)),
                'group': group, 'label': path.parent.name, 'sha256': hashlib.sha256(data).hexdigest()}

        images, labels, ids = [], [], []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, (a, y, item) in enumerate(pool.map(read, files), 1):
                images.append(a); labels.append(y); ids.append(item['path'])
                manifest['samples'].append(item)
                if n % 50 == 0 or n == len(files):
                    print(f'{partition}: fully read {n}/{len(files)} PNGs', flush=True)
        arrays[partition + '_images'] = np.stack(images)[:, None]
        arrays[partition + '_labels'] = np.asarray(labels)
        arrays[partition + '_ids'] = np.asarray(ids)
        sets[partition] = {p.stem.rsplit('__', 1)[0] for p in files}
    assert not sets['train'] & sets['val']
    np.savez_compressed(args.output / 'e5_images.npz', **arrays)
    (args.output / 'image_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('PASS: all 1074 PNGs decoded; groups disjoint; fixed split preserved.', flush=True)


if __name__ == '__main__':
    main()
