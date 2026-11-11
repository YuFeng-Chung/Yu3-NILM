# Yu3: Two-pass training with a varying auxiliary channel for robust V–I appliance identification

Code, data splits, protocols, and result summaries for the manuscript

> **Yu3: Improving the Robustness of V–I Appliance Identification to Unseen Waveform Perturbations through Two-Pass Training with a Varying Auxiliary Channel**  
> Yu-Feng Chung, Tsang-Yu Chen, Yu-Hsiang Tseng, Cheng-Ying Chung. *IEEE Access* (under review).

Yu3 runs a shared CNN encoder twice. In pass 1 the input is `[X, 0]` (the V–I image `X` plus a zero channel); a decoder reconstructs `X̂`. In pass 2 the input is `[X, R]` with `R = stopgrad(|X − X̂|)`. The loss is `0.5·CE₁ + CE₂ + 0.5·MSE(X̂, X)`.  
**Main finding:** the robustness to unseen waveform impulses is created during training. It needs a non-zero, varying second channel, but not the content of the residual. At deployment, running only pass 1 keeps this robustness at 0.40–0.45 times the latency of the full model.

## Repository layout

| Folder | Content |
|---|---|
| `實驗一/` | Experiment 1 (PLAID, M1–M5, 30 seeds, CUDA): group split, epoch pilot, training/evaluation |
| `實驗三/` | Original Experiment 3 ablation runner |
| `實驗四/` | Shared models and data modules: raw-waveform loader, defense strategies, M1–M5 and baseline models, label map, fixed normalization statistics |
| `實驗五/` | Earlier efficiency script (superseded by `experiments/run_e8_efficiency.py`) |
| `experiments/` | Revision pipeline and inference diagnostics (internal run IDs E1–E12; see `docs/EXPERIMENT_MAP.md`) |
| `data/` | PLAID metadata, label map, normalization statistics and extracted 500-sample windows (derived from PLAID, CC BY 4.0) |
| `results/` | For every experiment: pre-specified protocol (`protocol*.json`), rule amendments, selection records, per-run specs and metrics, per-seed summaries, paired tests and audits. The E14/E15 mechanism-control code is in `results/e14_*/source/` and `results/e15_*/source/`. Checkpoints are archived separately (see below). |
| `docs/` | Manuscript ↔ code ↔ results mapping; reproduction notes |

## Mapping to the manuscript
See **[`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md)** for the table/figure → script → result-folder mapping (Experiments 1–5 and Supplementary Tables S1–S41).

## Environment
Experiments 2–5: Python 3.12 and PyTorch 2.15.0 (MPS backend on Apple M2 for Experiments 2–4), with ONNX Runtime 1.31.1 for Experiment 5. Experiment 1 (including the random-split control): Python 3.9.25, PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124 on an NVIDIA RTX 5060 Laptop GPU. Exact package versions: `requirements.txt` and `results/*/environment_lock.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data
- **PLAID 2014** (CC BY 4.0): https://doi.org/10.6084/m9.figshare.11605074.v1 — download with `experiments/download_plaid_2014.py`. `實驗一/dataset/split_record.json` is the fixed group-disjoint split (855/219 records, 183/47 groups) used by all PLAID experiments.
- **WHITED v1.1**: obtain it from the dataset authors (Kahl et al., 2016); it is not redistributed here. The 15-class, 76-group, three-fold split used in Experiment 2 is in `results/e10_whited_20261108/split.json` and `sample_manifest.json`.

See `data/README.md` for details.

## Reproduction
`docs/REPRODUCE.md` lists the run order and dependencies. All model selection (hyperparameters, defense strategy, epochs) uses training data only and was fixed before test evaluation; the selection records and the tie-rule amendments are kept in `results/e1_e4_mac_20261106/`.

## Archived artifacts (Zenodo)
Trained checkpoints (about 16 GB of `.pt` files), ONNX exports, per-sample predictions and full source snapshots are archived at **Zenodo: DOI to be added**. Image caches are not archived because they are regenerated deterministically from the raw windows.

## Third-party code
The WRG-CNN comparison uses the authors' public implementation (Faustine & Pereira, https://github.com/sambaiga/WRG-NILM, commit `3e5879d`); clone it as described in `docs/REPRODUCE.md`; it is not redistributed here.

## Citation
See `CITATION.cff`. Please also cite PLAID and WHITED when using the data.

## License
Code: MIT (see `LICENSE`). Derived PLAID files keep the original CC BY 4.0 license.
