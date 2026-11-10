> English translation of the original working note `search_report.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E2: architecture adaptation and equal-budget hyperparameter search (Experiment 4, Table II; Supp. Tables S9a–S9d)

All 72 search runs completed (4 candidates × 18 configurations).
Only the group-disjoint validation split inside the outer training set was used; the outer test set was not evaluated. Single-seed search results are not used for significance claims.

| Model | Configurations completed | Best validation Macro-F1 (%) | LR | Weight decay | Dropout | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| Yu3 | 18/18 | 71.62 | 0.0005 | 0.0 | 0.5 | 32 |
| ResNet18 | 18/18 | 65.95 | 0.0001 | 0.001 | 0.2 | 64 |
| MobileNetV3-Small | 18/18 | 57.99 | 0.001 | 0.0001 | 0.5 | 40 |
| ViT-Tiny | 18/18 | 57.82 | 0.0005 | 0.0001 | 0.2 | 51 |

The epochs above belong to the No-Defense pilot only; the epochs of each defense strategy were re-determined on the inner validation split.
Runs that reached the maximum number of epochs are flagged (`hit_max_epochs`) and may not have converged; they are not hidden. The selected configuration does not guarantee better outer-test performance.
