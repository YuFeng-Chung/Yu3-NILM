# Experiment 1 results (PLAID, NVIDIA RTX 5060 Laptop GPU, CUDA)

| File | Content |
|---|---|
| `exp1_per_seed.csv` | M1–M5, 30 paired seeds (42–71), group split: accuracy, macro-precision, macro-recall, Macro-F1 (%), fixed epochs (14/34/13/21/19). Parsed from `exp1_console_log.txt`. Source of Table III. |
| `exp1_console_log.txt` | Full console output of `實驗一/plaid_m1_to_m5_30run.py`, including the Friedman and Wilcoxon tables |
| `random_split_results.json` | Random sample-level split control (Supp. Table S5): M1 and Yu3 (M5), seeds 42–51, accuracy and Macro-F1 (%) |
| `leakage_stats.json` | Per split: 859/215 records and the share of test samples whose group also appears in training (96.7–98.6%, mean 97.9%) |

Code: `實驗一/make_random_split_datasets.py` (stratified 80/20 `StratifiedShuffleSplit`, seeds 42–51) and `實驗一/run_random_split_check.py` (same models, epochs M1 14 / M5 19, Adam 0.0005, batch 32 as in the group split).
