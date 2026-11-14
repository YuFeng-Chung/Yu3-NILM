# Manuscript ↔ code ↔ results

Internal run IDs (E1–E15) were assigned during the revision and do not match the manuscript's experiment numbers. This table is the authoritative mapping. Table and figure numbers refer to the resubmitted manuscript (main text: Tables I–VIII, Figs. 1–4) and its Supplementary Material (Tables S1–S41, Figs. S1–S8). All paths are relative to the repository root.

## Experiment 1 – PLAID architecture comparison (Table II; Supp. Tables S1, S5)
| Item | Code | Results |
|---|---|---|
| Group-disjoint split (855/219 records, 183/47 groups) | `實驗一/make_dataset_split.py` | `實驗一/dataset/split_record.json` |
| Epoch pilot (patience 15, max 100) | `實驗一/pilot_epoch_search.py` | `實驗一/best_epochs.json` |
| M1–M5, 30 paired seeds, CUDA | `實驗一/plaid_m1_to_m5_30run.py`, `實驗一/models_m1_to_m5.py`, `實驗一/plaid_loader_vi.py` | `results/exp1_plaid_cuda/` |
| Per-seed results (Table II) | console output of the script above | `results/exp1_plaid_cuda/exp1_per_seed.csv`, `exp1_console_log.txt` |
| Random sample-level split control (10 splits, seeds 42–51; Supp. Table S5) | `實驗一/make_random_split_datasets.py`, `實驗一/run_random_split_check.py` | `results/exp1_plaid_cuda/random_split_results.json`, `leakage_stats.json` |

## Experiment 2 – WHITED architecture comparison (Table III; S2–S4) — run ID E10
`experiments/prepare_e10_whited.py`, `run_e10_whited.py`, `e10_core.py`, `analyze_e10_whited.py`, `analyze_e10_statistics.py` (group bootstrap) → `results/e10_whited_20261108/` (`split.json`, `sample_manifest.json`, `summary.csv`, `per_fold_seed.csv`, `paired_differences.csv`).

## Experiment 3 – Component ablation (Section III-C; S6–S8, S25) — run ID E5
`experiments/prepare_e5_images.py`, `prepare_e5_waveforms.py`, `run_e5_ablation.py`, `summarize_e5.py`, `analyze_e5_full.py`, `prepare_e5_corruptions.py`, `evaluate_e5_corruptions.py`, `analyze_e5_corruptions.py` → `results/e5_full_20261106/`. The original runner is `實驗三/yu3_ablation_runner.py`.

## Experiment 4 – Perturbation robustness
| Manuscript item | Run ID | Code | Results |
|---|---|---|---|
| Candidate tuning, 18-config grid (S9a–S9e, Fig. S1) | E2 | `experiments/run_e2_search_mps.py`, `revision_models_mps.py` | `results/e2_mac_20261106/` |
| Defense-strategy selection, 5-fold inner CV (S10–S11) | E3 | `experiments/revision_strategy_selection_v4.py`, `verify_e3_co_selection_v4.py` | `results/e1_e4_mac_20261106/E3_*`, `selection_rule_amendment.json`, `tie_rule_amendment_v3.json`, `tie_rule_amendment_v4.json` |
| Feedback family, ZeroFB/ShuffleFB, baselines: 330 final models (Tables IV, VI; Fig. 4; S15, S16, S16a, S19–S21) | E1/E4 | `experiments/run_revision_followup_v4.py`, `revision_followup_core.py`, `analyze_revision_followup_v4.py`, `verify_followup_analysis_v4.py`; models in `實驗四/` | `results/e1_e4_mac_20261106/` (`protocol_v4.json`, `final/*/run_spec.json`, `final/*/result.json`, `per_seed_averaged.csv`, `paired_tests.csv`) |
| Prespecified family tests (4 contrasts, Holm within condition) | – | `experiments/diag_family_prespecified_tests.py` | `results/diagnostics/family_prespecified.csv` |
| Baseline contrasts (3 contrasts, Holm within condition; S16) | – | derived from `results/e1_e4_mac_20261106/paired_tests.csv` | `results/diagnostics/baseline_contrasts_table_S16.csv` |
| Mechanism controls NoiseFB, PermFB (Fig. 3; S17, S18a) | E14 | `results/e14_noisefb_20261110/source/` | `results/e14_noisefb_20261110/` |
| Single-pass controls 1P-NoiseCh, 1P-Mix (Fig. 3; S18, S18a) | E15 | `results/e15_singlepass_20261110/source/` | `results/e15_singlepass_20261110/` |
| Pass 1 vs pass 2, channel replacement, BN re-estimation (Table V; S22–S24) | – | `experiments/diag_pass1_vs_pass2.py`, `diag_nodefense_pass_replacement.py`, `diag_bn_reestimation.py` | `results/diagnostics/*.csv` |
| Residual maps and decision examples (Figs. S2, S6), decision transitions, residual-energy location (S26–S27) | E6 | `experiments/run_e6_mechanism.py`, `e6_mechanism_core.py`, `analyze_e6_mechanism.py` | `results/e6_mechanism_20261108/` |
| WRG-CNN comparison (Table VII; S28–S30) | E11 | `experiments/prepare_e11_wrg.py`, `run_e11_wrg.py`, `e11_wrg_core.py`, `analyze_e11_wrg.py` | `results/e11_wrg_20261108/` |
| Five extra group splits (S31–S32, Fig. S3) | E7 | `experiments/prepare_e7_splits.py`, `run_e7_splits.py`, `run_e7_s3_splits.py`, `analyze_e7_*` | `results/e7_splits_20261108/`, `results/e7_s3_splits_20261108/` |
| Per-class results, confusion matrices (S33–S34, Figs. S4–S5) | E4 | `experiments/analyze_revision_followup_v4.py` | `results/e1_e4_mac_20261106/` |
| Preprocessing sensitivity, 2 × 2 (S35–S37) | E9 | `experiments/prepare_e9.py`, `run_e9.py`, `e9_core.py`, `analyze_e9.py` | `results/e9_preprocessing_20261108/` |
| Combined perturbations and synthetic measurement errors (S38–S41, Fig. S7) | E12 | `experiments/preflight_e12.py`, `run_e12.py`, `e12_core.py`, `analyze_e12.py`, `verify_e12.py` | `results/e12_measurement_20261108/` |

## Experiment 5 – Computational cost (Table VIII; S12–S13, Fig. S8) — run ID E8
`experiments/preflight_e8_schedule.py`, `run_e8_efficiency.py`, `execute_e8_batch.py`, `supplement_e8_batch.py`, `verify_e8_measurements.py`, `plot_e8_results.py` → `results/e8_cpu_20261108/` (`latency_summary.csv`, `raw_accepted_timings.csv`, `memory_summary.csv`, `rerun_quiet_gate.json`, `initial_incomplete_*` for the first incomplete PyTorch 4-thread run).

## Notes
- `protocol*.json` files record each experiment's pre-specified design; `*_amendment*.json` files record every rule change and when it was made.
- `*_audit.json` and `*_sha256.json` files are integrity checks written by the pipeline.
- Original run notes and some audit fields are in Chinese (the working language of the team); English translations of every run note are listed below. Numbers and file formats are language-independent.

## Run notes in English
Each result folder keeps the original working note written during the run (in Chinese, unchanged because it is covered by the run's SHA-256 records). An English translation sits next to it with the suffix `_EN`:

| Run | English note |
|---|---|
| Exp. 1 | [`results/exp1_plaid_cuda/README.md`](../results/exp1_plaid_cuda/README.md) |
| E10 (Exp. 2, WHITED) | [`results/e10_whited_20261108/README_EN.md`](../results/e10_whited_20261108/README_EN.md) |
| E5 (Exp. 3, ablation) | [`results/e5_full_20261106/e5_supporting_evidence_EN.md`](../results/e5_full_20261106/e5_supporting_evidence_EN.md) |
| E2 (tuning) | [`results/e2_mac_20261106/search_report_EN.md`](../results/e2_mac_20261106/search_report_EN.md) |
| E1/E3/E4 (family, baselines, selection) | [`results/e1_e4_mac_20261106/followup_report_EN.md`](../results/e1_e4_mac_20261106/followup_report_EN.md), [selection rule v4](../results/e1_e4_mac_20261106/tie_rule_vote_only_20261107/README_EN.md) |
| E14 (NoiseFB, PermFB) | [`README_EN.md`](../results/e14_noisefb_20261110/README_EN.md), [`prelaunch_report_EN.md`](../results/e14_noisefb_20261110/prelaunch_report_EN.md) |
| E15 (single-pass controls) | [`README_EN.md`](../results/e15_singlepass_20261110/README_EN.md), [`supplied_spec_EN.md`](../results/e15_singlepass_20261110/supplied_spec_EN.md), [`prelaunch_report_EN.md`](../results/e15_singlepass_20261110/prelaunch_report_EN.md) |
| E6 (Fig. 3 diagnostics) | [`results/e6_mechanism_20261108/e6_report_EN.md`](../results/e6_mechanism_20261108/e6_report_EN.md) |
| E11 (WRG-CNN) | [`results/e11_wrg_20261108/README_EN.md`](../results/e11_wrg_20261108/README_EN.md) |
| E7 (extra splits) | [Combined](../results/e7_splits_20261108/README_EN.md) ([A](../results/e7_splits_20261108/analysis/README_EN.md), [B](../results/e7_splits_20261108/part_b/README_EN.md)), [S3](../results/e7_s3_splits_20261108/README_EN.md) |
| E9 (preprocessing) | [`results/e9_preprocessing_20261108/README_EN.md`](../results/e9_preprocessing_20261108/README_EN.md) |
| E12 (combined perturbations) | [`results/e12_measurement_20261108/README_EN.md`](../results/e12_measurement_20261108/README_EN.md) |
| E8 (Exp. 5, cost) | [`results/e8_cpu_20261108/README_EN.md`](../results/e8_cpu_20261108/README_EN.md) |
| Inference diagnostics | `results/diagnostics/` (scripts `experiments/diag_*.py`; outputs are plain CSV) |
