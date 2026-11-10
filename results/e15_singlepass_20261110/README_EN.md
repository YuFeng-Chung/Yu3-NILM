> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E15: single pass with a varying second channel (Table VII; Supp. Table S18)

20 final models, 790 training epochs, two test inputs × 211 conditions per model, and the complete analysis. Research completeness is recorded in `completed_audit.json` and `qa/training_audit.json`.

- `supplied_spec.md` (English: `supplied_spec_EN.md`): the specification provided by the authors, including the pre-specified interpretation rules.
- `noise_scale.json`, `noise_scale_seal.json`: the fixed scale from the ten E14 NoiseFB final models on the 855 training records, with its SHA-256; never changed. `calibration_moments.npz`: per-seed, per-sample mean / population SD and IDs.
- `source/e15_core.py`: isolated copy of the E14 core that only adds two single-pass modes (`source/core_changes.diff`, `source/original_e14_core.py`).
- `source/prepare_pilot.py`: calibration, preflight and epoch pilots only (no final-training entry point).
- `pilots/`, `split_manifest.json`, `epoch_settings.json`: inner epoch selection (validated with the primary input [X, 0]).
- `preflight_audit.json`, `prelaunch_audit.json`: forward/loss/gradient/RNG checks and equivalence of the old modes.
- `protected_sources_sha256.json`: the pre-existing code, raw data and E14 artifacts, verified before and after.
- `prelaunch_report.md` (English: `prelaunch_report_EN.md`): fixed scale, epochs and time estimate reported before launch.

Randomness: SHA-256, UTF-8, pipe-separated, first 8 bytes little-endian, mod (2^63 − 1); training key (seed, epoch, zero-based batch), evaluation key (sample ID); local CPU `torch.Generator`. 1P-Mix draws the Gaussian for the whole batch first, then an independent Bernoulli per sample (< 0.5 → noise); the inference input [X, N] uses a fixed Gaussian per sample. Evaluation batch size 32.

Calibration used the E14 models trained on all 855 training records, as specified; these include the later inner-validation records. The pilot is therefore epoch selection under a transferred scale, not a fully nested model selection. The outer test was not used for calibration, epoch selection or timing.

## Main results
Primary input [X, 0], impulses 5/10/20%, mean Macro-F1 (%):

| Model / endpoint | 5% | 10% | 20% |
|---|---:|---:|---:|
| 1P-NoiseCh, [X, 0] | 56.27 | 48.88 | 39.22 |
| 1P-Mix, [X, 0] | 59.75 | 50.16 | 41.11 |
| E14 NoiseFB, pass 1 | 64.21 | 57.17 | 48.61 |
| E14 ZeroFB, reported output | 50.49 | 41.15 | 32.27 |

At all three levels both models fall into the "partial effect" branch of the specification: neither "a single pass suffices" nor "two passes are necessary". Both are significantly below NoiseFB in all six conditions. 1P-Mix is significantly above ZeroFB at every level (Holm p = 0.0039); 1P-NoiseCh versus ZeroFB is not significant at 5% (p = 0.105) and significant at 10% and 20% (p = 0.0098 and 0.0039). Holm is applied over the two comparisons within each model and level, not across levels or models. The 3-pp threshold is not an equivalence margin and does not support a general claim that two passes are necessary.

## Code
- `source/run_e15.py`: final runner after the scale and epochs were approved; `source/evaluate_e15.py`: inference with both inputs at batch 32 (images identical to E14).
- `source/analyze_e15.py`: recomputes the four metrics from 18,990 prediction files (4,158,810 labels) of the new models and the 50 saved references; all 440 exact p values are checked by sign permutation; the 12 pre-specified contrasts and the per-level interpretation are listed separately.
- `source/verify_training.py`, `source/verify_delivery.py`: training, ID, normalization, BN-count, checkpoint-reload and output-integrity checks.
- `confusion_counts/`: integer 11 × 11 confusion counts per condition and input. `reference_manifest.json`: the saved E14/E1–E4 sources used as references. `analysis_plan.json`: comparisons frozen before the new outer evaluation. `formal_source_sha256.json`, `final_results_sha256.json`: sealed sources and outputs.
