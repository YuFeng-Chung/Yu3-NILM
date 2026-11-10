> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E9: preprocessing sensitivity (Supp. Tables S35–S37)

A separate 2 × 2 study of the capture window (W0: window around the event; W1: steady single cycle near the end of the record) and current normalization (N0: per-waveform; N1: training-set fixed), for M1 and Yu3 with five seeds: 40 final trainings plus 8 inner-validation epoch pilots. It does not replace the preprocessing used in the other experiments. The frozen design is `protocol.json`; the window rule was fixed before data were inspected (`window_rule_before_data.json`).

Records eligible under both windows: 965 of 1,074. Intersected with the main split this gives 765 training / 200 test records (inner split 616/149). The 109 excluded records are concentrated in the Laptop class; `coverage.csv` and `window_manifest.csv` list them. There is no fallback rule, and no threshold was tuned on classification results.

Results: `summary.csv`, `paired_factor_changes.csv`, `diagnostics_summary.csv` (image-level diagnostics such as occupied-pixel IoU), `per_class_metrics.csv`; audits: `independent_audit.json`, `completed_audit.json`.
