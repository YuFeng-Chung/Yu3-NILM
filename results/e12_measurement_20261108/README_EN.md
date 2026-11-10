> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E12: combined perturbations and synthetic measurement errors (Table XI, Fig. 4; Supp. Tables S38–S41)

Inference-only evaluation of the 330 trained Experiment 4 models; no retraining and no field validation. All conditions are synthetic. The full design is in `protocol.json`.

Main outputs: `summary.csv`, `per_seed_averaged.csv`, `combined_vs_components.csv` (each combination versus its single components), `paired_differences.csv`, `paired_tests.csv` (uncorrected, descriptive) and `waveform_image_diagnostics.csv`. Audits: `independent_audit.json`, `completed_audit.json`.
