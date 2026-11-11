# Reproduction notes

## Folder layout
Scripts resolve paths relative to the repository root (the parent of `experiments/`). Keep the folder names unchanged, including the Chinese folder names `實驗一` to `實驗五` (Experiment 1–5 modules).

## 1. Data
```bash
python experiments/download_plaid_2014.py        # PLAID 2014 from figshare (CC BY 4.0), MD5-verified
```
`data/plaid_2014_published/raw_windows.npz` already contains the 500-sample windows used by Experiments 3–5, so the download is only needed to regenerate them. For Experiment 2, place WHITED v1.1 in `data/whited_v11/` (obtain it from the dataset authors).
The scripts in `實驗一/` and `實驗三/` were run on Windows; edit the `DATA_DIR` constants at the top of each script to your local path.

## 2. Run order (Experiment 4 depends on this order)
1. **Experiment 1** (CUDA): `實驗一/make_dataset_split.py` → `pilot_epoch_search.py` → `plaid_m1_to_m5_30run.py`; random-split control: `make_random_split_datasets.py` → `run_random_split_check.py`.
2. **Experiment 3** (E5): `experiments/run_e5_ablation.py` → `analyze_e5_full.py`.
3. **Experiment 4 tuning** (E2): `experiments/run_e2_search_mps.py`.
4. **Experiment 4 selection and final models** (E3/E1/E4): `experiments/run_revision_followup_v4.py` (inner-CV strategy selection, epoch pilots, 330 final models; the test partition is opened only after `selection_complete.json` is written) → `analyze_revision_followup_v4.py` → `verify_followup_analysis_v4.py`.
5. **Mechanism controls**: `results/e14_noisefb_20261110/source/run_e14.py`, then `results/e15_singlepass_20261110/source/run_e15.py`.
6. **Inference diagnostics** (read the checkpoints from step 4): `experiments/diag_*.py`, `experiments/run_e6_mechanism.py`.
7. **Extensions**: E7 (extra splits), E9 (preprocessing), E11 (WRG-CNN), E12 (combined perturbations).
8. **Experiment 5** (E8): `experiments/run_e8_efficiency.py` on an idle machine on AC power.

Experiments 2–5 were run on Apple M2 (PyTorch MPS); Experiment 1 on an NVIDIA RTX 5060 Laptop GPU (Python 3.9.25, PyTorch 2.6.0+cu124). Results on other hardware may differ slightly.

## 3. WRG-CNN upstream
Clone the authors' repository (https://github.com/sambaiga/WRG-NILM) and check out commit `3e5879d9d1a9c4c1ef93c557a650454b1581ac87` into `e11_prototype_20261108/upstream/` before running `experiments/prepare_e11_wrg.py`.

## 4. Checkpoints
To skip training, download the archived checkpoints (Zenodo DOI in the README) and place them under `results/<run>/final/<model>_<strategy>_s<seed>/checkpoint.pt`. The analysis and diagnostic scripts then run without retraining.
