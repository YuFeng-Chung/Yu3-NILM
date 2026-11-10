> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E7-S3: additional group splits, S3 strategy (noise conditions; Supp. Tables S31–S32, Fig. S2)

Complements E7 (Combined) with the main noise strategy S3; the Combined results are kept and no strategy is reselected. Protocol, source hashes, the same six split lists, the four models' original S3 settings and epochs are frozen in `protocol.json`. New training: the same five new splits × seeds 42/123/2026 × four models = 60 runs.

## Fixed settings
| Model | Strategy | Epochs | LR | Weight decay | Dropout |
|---|---|---:|---:|---:|---:|
| Yu3 | S3 | 52 | 0.0005 | 0 | 0.5 |
| M1 | S3 | 42 | 0.0005 | 0 | 0.5 |
| RecAux | S3 | 42 | 0.0005 | 0 | 0.5 |
| ResNet18 | S3 / S4 (equivalent) | 66 | 0.0001 | 0.001 | 0.2 |

Settings and epochs were checked against the 40 original S3 final runs of E1/E3/E4. ResNet18 was chosen by E2 inner validation; M1/RecAux reuse Yu3's strategy. ResNet18's S3 and S4 predictions are identical across all ten seeds; no strategy was picked on test results.

Conditions: clean; AWGN 30/20/15/10/5/0 dB and impulses 1/5/10/20%, each with 10 perturbation seeds (1000–1009), i.e. 101 per model. The complete noise grid is used, not only favorable levels.

## Main split (`part_b/`)
884,760 predictions recomputed from the saved main-split S3 predictions: four models × 11 conditions × 10 training seeds; 484 rows of per-class P/R/F1 (mean ± SD and number of test groups), 44 confusion matrices and count CSVs, and Yu3−M1 class differences, with `analysis_audit.json` and an independent TP/FP/FN audit.

## Analysis
Realizations are averaged within each training seed, then over the seeds of a split. Outputs: F1, clean-minus-perturbed degradation, per-class PRF, confusion counts, and paired Yu3 versus M1/RecAux/ResNet18 differences with a descriptive split/seed bootstrap (10,000 draws). The five new splits are the primary summary and the six-split set is a sensitivity analysis.

Groups are class × house-location proxies; splits overlap, and the new test sets may have been used for training or tuning on another split. The bootstrap intervals are conditional descriptions and are not used for confirmatory significance, equivalence or multiplicity control. No seeds, levels or corrections were added to reach significance.

## Code
`experiments/e7_s3_split_core.py`, `prepare_e7_s3_splits.py`, `queue_e7_s3.py` (starts only after the Combined run has finished and passed its audit), `preflight_e7_s3_splits.py`, `run_e7_s3_splits.py`, `analyze_e7_s3_splits.py`, `control_e7_s3_splits.py`.
