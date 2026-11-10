> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E7: additional group splits and per-class analysis, Combined strategy (Supp. Tables S31–S34, Figs. S2–S4)

## Contents
- **E7-B (`part_b/`)**: per-class results on the main split (132 rows), confusion matrices, count CSVs and Yu3−M1 class differences for four models × three conditions × 10 training seeds; 183,960 predictions recomputed, with `analysis_audit.json` and an independent per-class TP/FP/FN audit (`independent_prf_audit.json`). RecAux is included in addition to the original plan; no new training.
- **E7-A (`analysis/`, `runs/`)**: 5 new splits × 3 seeds (42/123/2026) × Yu3/M1/RecAux/ResNet18 = 60 final trainings.

## Frozen design (`protocol.json`)
| Model | Strategy | Fixed epochs | LR | Weight decay | Dropout |
| --- | --- | ---: | ---: | ---: | ---: |
| Yu3 | Combined | 36 | 0.0005 | 0.0001 | 0.5 |
| M1 | Combined | 42 | 0.0005 | 0.0001 | 0.5 |
| RecAux | Combined | 35 | 0.0005 | 0.0001 | 0.5 |
| ResNet18 | dynamic masking / Combined (equivalent) | 48 | 0.0001 | 0.001 | 0.2 |

ResNet18 was the baseline chosen only by its E2 inner-validation score (0.65947), not by E7 test results. M1 and RecAux reuse Yu3's E3 missing-data strategy. (Weight decay 0.0001 is the Combined floor.)

Split generation follows the original program: for each class, 2,000 candidate device subsets, the subset closest to 20% of the records, replaced only on strict improvement; new split seeds 1–5. Because the original loader used `os.listdir` and the historical insertion order cannot be recovered, the new splits sort groups lexicographically, so they are not a bit-exact rerun of the original generator with another seed. The main split reuses the verified 855/219 mapping. The six test-ID lists differ, all classes are present, and train/test groups never overlap; `split_class_counts.csv` and `split_overlap.csv` give counts and overlaps.

A group is the original loader's "class × house location", not a verified unique device; different appliances in the same house can still be split across train and test, so this is not whole-house isolation.

Each new split fits the global normalization on its own training set only, and keeps the tuning, strategy and epochs of the main split; nothing is retuned, and no epoch or strategy is chosen on a new test set. Because those settings were derived on the main split, test records of a new split may have been used for training or tuning before. This is therefore a sensitivity analysis of fixed settings to the split, not new external validation or nested CV.

Testing: clean once; AWGN 10 dB and 20%/50% consecutive missing with 10 perturbation seeds (1000–1009). Realizations are averaged within each training seed, then over the seeds of the same split.

The five new splits form the primary summary and the six-split set is a sensitivity analysis; the main split has 10 seeds and each new split 3, reported separately. Paired bootstrap over splits and shared seed indices (10,000 draws, seed 20261108) gives mean differences, pointwise percentile intervals and the number of splits won; no Wilcoxon/Holm significance claims are made. Because test groups overlap between splits, the intervals are conditional descriptions only, not 95% coverage of a population of splits, confirmatory inference, equivalence or multiplicity control.

## Code (relative to the repository root)
| Script | Purpose |
| --- | --- |
| `experiments/e7_split_core.py` | fixed models/conditions, 2,000-subset split algorithm, metrics, source checks |
| `experiments/prepare_e7_splits.py` | metadata/split/setting checks; freezes the protocol and sources |
| `experiments/preflight_e7_splits.py` | one real MPS epoch per model; value check of the 31 original conditions |
| `experiments/run_e7_splits.py` | single-writer training with fixed epochs, 31-condition evaluation, automatic analysis/audit |
| `experiments/analyze_e7_splits.py` | recomputation, per-class/confusion analysis, paired bootstrap |
| `experiments/control_e7_splits.py` | start/status/pause/resume |
