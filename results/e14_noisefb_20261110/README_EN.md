> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E14: NoiseFB and PermFB (mechanism controls; Table VII; Supp. Table S17)

## Protocol and sources
No-Defense only; NoiseFB is the primary and PermFB the secondary control. The specification is `protocol.json`. The old control checkpoints (Yu3, ZeroFB, ShuffleFB) were trained on 2026-11-06 by the initial `run_revision_followup.py` with the shared `revision_followup_core.py`; later runs continued with v4, whose pilot/final code is AST-identical (`runner_provenance.json`). Existing sources, models and results were not modified.

All E14 code is in `source/`. `e14_core.py` is an isolated copy that adds a FeedbackControl mode, build/logits/loss names and the passing of random-number context and sample IDs, without changing any existing mode (`core_changes.diff`). Epochs were chosen on the original inner group split: NoiseFB 42, PermFB 49. The new specification uses pilot patience 30, while the old controls used 20, so pilot patience and compute budget are not identical. Seeds 42–51 per model (20 final models), the main 855/219 split, training-only scaling statistics and the E2 Yu3 hyperparameters.

## Running
In this folder: `python source/run_e14.py pilot` (epoch pilots), then `python source/run_e14.py formal` (final training, inference and analysis). A writer lock prevents concurrent runs, and existing results are never retrained.

## Results and limits of inference
- `final/`: checkpoints, run_spec, training logs, evaluation, outer IDs/labels, predictions and pass-1 predictions of the new models.
- `reference_evaluation/` (archived with the checkpoints on Zenodo): inference-only copies of the old Yu3/ZeroFB/ShuffleFB models; native outputs were checked value-by-value against the saved E4 predictions.
- `per_seed_condition_realization.csv` (per model, seed, realization and endpoint), `per_seed_averaged.csv` (10 realizations averaged), `summary.csv` (mean and sample SD over 10 seeds). Accuracy/PRF columns are 0–1; difference columns in pp are percentage points.
- `paired_tests_prespecified.csv`: NoiseFB (primary) and PermFB (secondary), 3 levels × 2 references = 12 tests; Holm over the two references within each model and level, exact two-sided Wilcoxon, n = 10; no correction across levels or across primary/secondary.
- `paired_tests_all.csv`: also ShuffleFB, other levels and native outputs; tests not pre-specified are marked `exploratory_unadjusted` with an empty `p_holm`.
- `prespecified_interpretation.csv`: section 5 of the specification applied mechanically per level; 3 pp is descriptive, not an equivalence test.

NoiseFB keeps each sample's residual mean/SD, and PermFB keeps each sample's residual value distribution, so the results do not show that no sample information at all is involved, nor that any varying channel suffices. The test partition had been seen before and the hypothesis came from earlier diagnostics; single split, different epochs and the Mac/MPS platform are limitations.

## Randomness and audits
Training seeds are SHA-256(seed, epoch, zero-based batch index); evaluation seeds are SHA-256(sample ID): UTF-8, pipe-separated, first 8 bytes little-endian, mod (2^63 − 1), drawn with a CPU `torch.Generator` without touching the global RNG. Noise uses population SD (correction = 0), per-pixel Gaussian then clamp(min = 0), so the clamped moments need not equal those of R exactly. PermFB permutes each sample independently. At inference the same sample ID reuses the same noise map across perturbation conditions; perturbations still follow the E4 formulas and seeds 1000–1009.
`completed_audit.json` contains an independent Boolean F1 recomputation and a sign-permutation check of every p value. `protected_sources_sha256.json` freezes 6,694 related existing sources and results; `e14_source_sha256.json` records the E14 sources at launch.

## Inference batch-size correction
All final results use the original E4 batch size of 32. A first pass used batch 219; a guard stopped it when 1 of 219 predictions differed (ZeroFB seed 46, impulses 20%, perturbation seed 1008). That first pass is kept in `numerical_audit/batch219/` and documented in `inference_batch_amendment.json`. `source/resume_evaluation.py` redoes inference and analysis only; none of the 20 models was retrained. The only change to the evaluator is the batch size of 32.
