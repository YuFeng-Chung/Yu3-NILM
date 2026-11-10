> English translation of the original working note `followup_report.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E1/E3/E4: feedback family, baselines and strategy selection (Experiment 4)

330 final models were trained, each evaluated under 210 perturbation conditions; all predictions were recomputed and all training records audited.

- The E3 strategy-counting code was corrected before outer-test evaluation to restore the original definition (wins relative to No-Defense). Some inner-fold results had been seen before the correction; this is fully disclosed in `selection_rule_amendment.json`, and the procedure is not described as fully pre-registered.
- After seeing some inner scores and before the outer test, the authors approved rule v4 (see `tie_rule_vote_only_20261107/README_EN.md`, `tie_rule_amendment_v4.json`, `E3_co_selection_report.json`).
- One fixed outer split and 10 training seeds; each seed first averages 10 perturbation realizations. Earlier CPU/MPS results are not mixed in. All epochs and strategies were fixed before this outer evaluation.
- E3 uses reconstructible new five-fold group splits and does not claim to reproduce historical folds. Combined uses an explicit weight-decay floor; some baselines already had a higher weight decay from E2, so the regularization candidate can coincide with another candidate.
- RecAux uses a one-channel encoder. Yu3−RecAux therefore changes the number of passes, the classification loss and the input interface together and is not a pure residual effect. ZeroFB and ShuffleFB are separate sensitivity controls.

Absolute F1, Dabs and Drel for clean data and all perturbations are in `summary.csv`; per-seed averages in `per_seed_averaged.csv`; paired differences, effect sizes, intervals and both Holm families in `paired_tests.csv`. With many tests the resolution is limited, and non-significance is not evidence of equivalence. Relative CE follows the original formula per seed; unstable or non-positive denominators are listed, and CE under random missing data is not interpreted. Consecutive missing at 30–50% is evaluated only (not used in training).

## Strategies selected by inner CV
| Model | Noise | Missing |
|---|---|---|
| Yu3 | S3 | Combined |
| ResNet18 | S3 / S4 | dynamic masking / Combined |
| MobileNetV3-Small | S3 / S4 | dynamic masking / Combined |
| ViT-Tiny | S3 / S4 | dynamic masking / Combined |

## Family results under common strategies
| Strategy | Model | Clean F1 (%) | Consecutive missing 20% F1 (%) | Consecutive missing 20% Dabs (pp) |
|---|---|---:|---:|---:|
| No-Defense | M1 | 70.14 ± 4.06 | 66.72 ± 3.61 | 3.42 ± 1.38 |
| No-Defense | RecAux | 71.11 ± 5.40 | 66.44 ± 3.26 | 4.67 ± 2.96 |
| No-Defense | M3 | 69.79 ± 4.70 | 64.31 ± 4.23 | 5.48 ± 2.01 |
| No-Defense | Yu3 | 71.63 ± 3.60 | 66.61 ± 2.74 | 5.02 ± 2.22 |
| S3 | M1 | 71.35 ± 4.47 | 66.21 ± 3.09 | 5.14 ± 1.68 |
| S3 | RecAux | 71.53 ± 5.33 | 66.58 ± 4.29 | 4.94 ± 2.33 |
| S3 | M3 | 70.21 ± 4.57 | 64.72 ± 3.97 | 5.49 ± 1.84 |
| S3 | Yu3 | 72.11 ± 5.15 | 67.28 ± 3.35 | 4.83 ± 2.74 |
| Combined | M1 | 71.41 ± 4.67 | 67.89 ± 3.95 | 3.52 ± 2.84 |
| Combined | RecAux | 70.43 ± 4.15 | 65.92 ± 3.97 | 4.52 ± 2.02 |
| Combined | M3 | 65.41 ± 5.79 | 62.37 ± 5.82 | 3.04 ± 2.56 |
| Combined | Yu3 | 71.65 ± 4.30 | 65.87 ± 4.47 | 5.78 ± 2.78 |

Results must be read together with absolute F1, paired differences, correction and the limitations above.
