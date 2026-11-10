"""Build E5 waveform-corrupted tests after exact clean-image provenance validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_e5_waveforms import original_preprocessing

CONDITIONS = [('awgn_10db', 10.), ('consecutive_missing_20pct', .2)]
CORRUPTION_SEEDS = list(range(1000, 1010))


def keyed_seed(sid, condition, seed):
    text = f'yu3_e5_waveform_v1|{sid}|{condition}|{seed}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def corrupt(current, voltage, sid, condition, seed):
    rng = np.random.default_rng(keyed_seed(sid, condition, seed))
    if condition == 'awgn_10db':
        variances = [float(np.var(x.astype(np.float64))) / 10. for x in [current, voltage]]
        c = current + rng.normal(0, np.sqrt(variances[0]), len(current))
        v = voltage + rng.normal(0, np.sqrt(variances[1]), len(voltage))
        return c, v, {'current_noise_variance': variances[0], 'voltage_noise_variance': variances[1]}
    assert condition == 'consecutive_missing_20pct'
    n = len(current); length = int(round(.2 * n)); start = int(rng.integers(0, n - length + 1))
    mask = np.zeros(n, dtype=bool); mask[start:start + length] = True
    index = np.arange(n); valid = ~mask
    def interpolate(x):
        out = x.copy(); out[mask] = np.interp(index[mask], index[valid], x[valid]); return out
    return interpolate(current), interpolate(voltage), {'missing_start': start, 'missing_length': length}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--waveforms', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.waveforms / 'waveform_image_audit.json').read_text())
    assert audit['matched_clean_images'] == 1074 and not audit['unmatched_raw_ids'] and not audit['unmatched_png_paths']
    raw = np.load(args.waveforms / 'raw_windows.npz'); selected = np.flatnonzero(raw['partitions'] == 'val')
    assert len(selected) == 219
    functions, source_hash = original_preprocessing(); build = functions['generate_vi_image']
    supplied = np.load(args.images / 'e5_images.npz')
    by_path = {str(path): image for path, image in zip(supplied['val_ids'], supplied['val_images'])}
    test_paths = raw['original_png_paths'][selected]; clean = np.stack([by_path[str(p)] for p in test_paths])
    for index, image in zip(selected, clean):
        regenerated = (build(raw['current'][index], raw['voltage'][index]).squeeze() * 255).astype(np.uint8)
        assert np.array_equal(regenerated, image.squeeze())
    arrays = {'clean_images': clean, 'sample_ids': raw['sample_ids'][selected],
              'original_png_paths': test_paths, 'labels': raw['labels'][selected]}
    records = []
    for condition, _ in CONDITIONS:
        for seed in CORRUPTION_SEEDS:
            images = []
            for index in selected:
                sid = str(raw['sample_ids'][index])
                c, v, details = corrupt(raw['current'][index], raw['voltage'][index], sid, condition, seed)
                images.append((build(c, v).squeeze() * 255).astype(np.uint8)[None])
                records.append({'sample_id': sid, 'condition': condition, 'corruption_seed': seed, **details})
            arrays[f'{condition}_seed{seed}'] = np.stack(images)
            print(f'Built {condition} seed{seed}: {len(images)} samples', flush=True)
    np.savez_compressed(args.output / 'corrupted_images.npz', **arrays)
    config = {'conditions': CONDITIONS, 'corruption_seeds': CORRUPTION_SEEDS, 'sample_count': 219,
        'preprocessing': 'E5 original per-waveform z-score, then identical VI grid construction and uint8 quantization. Not Exp4 fixed normalization.',
        'awgn': 'Independent current/voltage Gaussian draws, noise variance = centered waveform population variance / 10^(10/10).',
        'missing': 'One shared contiguous mask for both channels; length100 of500; valid start0..400 inclusive; linear interpolation with nearest boundary extension.',
        'randomness': 'Sample-ID/condition/corruption-seed keyed SHA256; independent of iteration order and model.',
        'loader_sha256': source_hash,
        'waveform_audit_sha256': hashlib.sha256((args.waveforms / 'waveform_image_audit.json').read_bytes()).hexdigest(),
        'clean_images_exactly_match_supplied_test': True,
        'averaging': 'Compute each corruption-seed metric then average within each training seed; inferential n=10 training seeds, not100.'}
    (args.output / 'corruption_config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'corruption_manifest.json').write_text(json.dumps(records, indent=2))
    print('PASS clean equivalence and 20 reproducible waveform-corrupted test sets.', flush=True)


if __name__ == '__main__': main()
