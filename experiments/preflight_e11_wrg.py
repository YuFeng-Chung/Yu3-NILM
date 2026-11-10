"""Bounded WRG integration smoke; train-only, no benchmark/model selection."""
import ast
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'e11_prototype_20261108'
UP = OUT / 'upstream'
COMMIT = '3e5879d9d1a9c4c1ef93c557a650454b1581ac87'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    tick = time.monotonic()
    results = OUT / 'smoke'
    results.mkdir(exist_ok=True)
    if (results / 'audit.json').exists():
        raise RuntimeError('Completed smoke exists; preserve evidence and use a new version for reruns.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model_source = module('wrg_author_model', UP / 'src/net/model.py')
    feature = module('wrg_author_feature', UP / 'src/utils/feature_representation.py')
    raw_path = ROOT / 'data/plaid_2014_published/raw_windows.npz'
    split_path = ROOT / 'e2_full_review_20261107/original_split_manifest.json'
    split = json.loads(split_path.read_text())
    # Read shared archive, then access waveform/label values ONLY for manifest train IDs.
    with np.load(raw_path) as z:
        ids = z['sample_ids']
        outer_train = np.flatnonzero(np.isin(ids, split['train_ids']))
        assert len(outer_train) == len(split['train_ids']) == 682
        train_i = z['current'][outer_train].copy()
        train_v = z['voltage'][outer_train].copy()
        train_y = z['labels'][outer_train].copy()
        train_ids = ids[outer_train].copy()
    classes = np.unique(train_y)
    selected = np.concatenate([np.flatnonzero(train_y == c)[:8] for c in classes])
    selected_ids = train_ids[selected].tolist()
    assert not set(selected_ids) & (set(split['outer_test_ids_audit_only']) | set(split['validation_ids']))
    assert len(classes) == 11
    current, voltage, labels = train_i[selected], train_v[selected], train_y[selected]
    stats = {}
    for prefix, array in [('i', train_i), ('v', train_v)]:
        for suffix, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
            stats[prefix + '_' + suffix] = float(fn(array))

    # AST selects only reviewed, pure corruption functions; original script is not run.
    pollution_path = ROOT / '實驗四/exp4_raw_data_V2.py'
    pollution_names = {'add_awgn_waveform', 'add_salt_pepper_waveform',
                       '_interpolate_missing', 'add_missing_random_waveform',
                       'add_missing_consecutive_waveform', 'apply_waveform_corruption'}
    tree = ast.parse(pollution_path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in pollution_names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CORRUPTION_FUNCS' for t in n.targets)]
    ns = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(pollution_path), 'exec'), ns)

    def image(c):
        d = feature.create_distance_similarity_matrix(c, 50, 1)
        return feature.get_weighted_reccurrence_graph(d, 10., 20.)

    # Independent NumPy calculation, including original float32 distance rounding.
    def reference(c):
        p = c.reshape(50, 10).mean(axis=1).astype(np.float32)
        return np.minimum(np.floor(np.abs(p[:, None] - p[None, :]) * np.float32(10)), 20)[None]

    image_rows = []
    all_images = {}
    for condition, severity in [('clean', 0), ('awgn', 10), ('salt_pepper', .05), ('missing_consecutive', .2)]:
        xs = []
        for j, (c, v) in enumerate(zip(current, voltage)):
            seed_bytes = hashlib.sha256(('E11-smoke|' + selected_ids[j] + '|' + condition).encode()).digest()
            seed = int.from_bytes(seed_bytes[:8], 'little')
            c2, v2 = ns['apply_waveform_corruption'](c, v, condition, severity, stats, np.random.default_rng(seed))
            actual = image(c2)
            np.testing.assert_array_equal(actual.numpy(), reference(c2))
            assert torch.isfinite(actual).all() and actual.shape == (1, 50, 50)
            assert torch.equal(actual, actual.transpose(-1, -2))
            assert torch.count_nonzero(actual.diagonal(dim1=-2, dim2=-1)) == 0
            xs.append(actual)
        all_images[condition] = torch.stack(xs)
        image_rows.append(dict(condition=condition, severity=severity, samples=len(xs),
                               min=float(all_images[condition].min()), max=float(all_images[condition].max()),
                               reference_exact=True))
    # Explicit zero/constant and extreme finite waveforms, independent of real data.
    for c in [np.zeros(500), np.full(500, 3.), np.linspace(-100, 100, 500)]:
        np.testing.assert_array_equal(image(c).numpy(), reference(c))

    config = dict(n_channels=1, n_kernels=64, n_layers=3, emb_size=50, dropout=.25, output_size=11)
    x, y = all_images['clean'], torch.tensor(labels, dtype=torch.long)
    device_rows = []
    device_names = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
    for device_name in device_names:
        torch.manual_seed(42)
        model = model_source.CNN2DModel(**config).to(device_name)
        opt = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, nesterov=True)
        start = time.monotonic()
        losses = []
        # Three fixed smoke epochs, no validation, scoring, strategy choice or early stop.
        for epoch in range(3):
            model.train()
            order = torch.randperm(len(y), generator=torch.Generator().manual_seed(4200 + epoch))
            total = 0.
            for batch in order.split(16):
                opt.zero_grad(set_to_none=True)
                pred = model(x[batch].to(device_name))
                assert pred.shape == (len(batch), 11)
                value = torch.nn.functional.cross_entropy(pred, y[batch].to(device_name))
                assert torch.isfinite(value)
                value.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in gradients)
                assert sum(float(g.abs().sum()) for g in gradients) > 0
                opt.step()
                total += float(value.detach()) * len(batch)
            losses.append(total / len(y))
        if device_name == 'mps':
            torch.mps.synchronize()
        duration = time.monotonic() - start
        model.eval()
        with torch.no_grad():
            before = model(x[:16].to(device_name)).cpu()
            for data in all_images.values():
                assert torch.isfinite(model(data[:16].to(device_name))).all()
        saved = results / ('checkpoint_' + device_name + '.pt')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=3, config=config), saved)
        checkpoint = torch.load(saved, map_location=device_name, weights_only=True)
        fresh = model_source.CNN2DModel(**config).to(device_name)
        fresh.load_state_dict(checkpoint['model'])
        opt2 = torch.optim.SGD(fresh.parameters(), lr=.001, momentum=.9, nesterov=True)
        opt2.load_state_dict(checkpoint['optimizer'])
        fresh.eval()
        with torch.no_grad():
            after = fresh(x[:16].to(device_name)).cpu()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        # Same next step with dropout disabled isolates model+momentum state restoration.
        for net, optimizer in [(model, opt), (fresh, opt2)]:
            optimizer.zero_grad(set_to_none=True)
            val = torch.nn.functional.cross_entropy(net(x[:16].to(device_name)), y[:16].to(device_name))
            val.backward()
            optimizer.step()
        for p1, p2 in zip(model.parameters(), fresh.parameters()):
            torch.testing.assert_close(p1, p2, rtol=0, atol=0)
        device_rows.append(dict(device=device_name, epochs=3, samples=len(y), loss=losses,
                                training_seconds=duration, seconds_per_smoke_epoch=duration / 3,
                                parameters=sum(p.numel() for p in model.parameters()),
                                reload_logits_exact=True, optimizer_next_step_exact=True,
                                gradients_finite=True, all_four_condition_forward_finite=True,
                                checkpoint_sha256=digest(saved)))
        del model, fresh, opt, opt2
        if device_name == 'mps':
            torch.mps.empty_cache()
    np.savez_compressed(results / 'train_only_images.npz', sample_ids=np.array(selected_ids), labels=labels,
                        **{k: v.numpy() for k, v in all_images.items()})
    audit = dict(all_checks_passed=True, upstream_commit=COMMIT, torch=torch.__version__,
                 platform=platform.platform(), threads=4, config=config, selected_train_ids=selected_ids,
                 inner_train_count=682, validation_waveforms_accessed=False, test_waveforms_accessed=False, test_scoring=False,
                 selection='first 8 manifest training samples per class; fixed before training',
                 normalization='native WRG current amplitudes; no VI zscore; salt-pepper bounds inner-train only',
                 conditions=image_rows, devices=device_rows,
                 source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(UP.rglob('*.py'))},
                 local_script_sha256=digest(Path(__file__)), raw_archive_sha256=digest(raw_path),
                 split_manifest_sha256=digest(split_path), pollution_source_sha256=digest(pollution_path),
                 total_seconds=time.monotonic() - tick,
                 scope='Integration smoke only; not original numerical replication, accuracy benchmark, HPO, or formal E11')
    (results / 'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(all_checks_passed=True, samples=len(y), devices=device_rows,
                          total_seconds=audit['total_seconds']), indent=2), flush=True)


if __name__ == '__main__':
    main()
