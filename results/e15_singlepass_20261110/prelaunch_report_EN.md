> English translation of the original working note `prelaunch_report.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E15 pre-launch report (2026-11-09; before final training and before any outer-test scoring)

## Fixed noise scale
- μ0 = 0.013518345542252064, σ0 = 0.021488790400326252.
- Computed from the ten E14 NoiseFB final checkpoints (seeds 42–51) on the 855 training images (batch 32, eval mode): absolute pass-1 residuals, per-sample pixel mean and population SD (correction = 0); the median of the 8,550 values of each.
- SHA-256 of the scale: `5321bebe21c36dff9407115103ecf4052e1afc175208c5e3bb4270c4a2b8f6a6`. `noise_scale.json` and `noise_scale_seal.json` are sealed; `calibration_moments.npz` stores the per-sample statistics.

## Epoch pilots
Seed 42, inner split 682 train / 173 validation (common group split); min 30 / patience 30 / max 150 / delta 0.001; highest clean validation Macro-F1 with input [X, 0] from epoch 30, earliest epoch on ties.

| Model | fixed_epochs | Stopped at | Validation F1 |
|---|---:|---:|---:|
| 1P-NoiseCh | 42 | 72 | 0.6773 |
| 1P-Mix | 37 | 67 | 0.6940 |

## Final plan
Two models × seeds 42–51 (20 models); 855 train / 219 test; Adam, LR 0.0005, WD 0, dropout 0.5, batch 32. All 211 conditions (clean + 210 perturbed), two test inputs ([X, 0] and [X, N]), inference batch 32. NoiseFB and ZeroFB values are reused from E14 without retraining.

## Boundaries and checks
- Single pass of encoder/classifier/decoder, loss CE + 0.5 MSE, gradients, parameter structure, local CPU generator, per-sample Bernoulli in 1P-Mix and fixed per-sample inference noise all passed the preflight.
- As specified, the scale was calibrated on E14 models trained on all 855 training records, which include the later inner-validation records. The pilot therefore selects epochs under a transferred, fixed scale and is not a fully nested model selection. The outer test was not used for calibration, epoch selection or timing.
- Primary comparisons: each model versus NoiseFB pass 1 and versus ZeroFB, at impulses 5/10/20%; two exact two-sided Wilcoxon tests (n = 10) with Holm within each level; 1P-Mix is a separate secondary family. The 3-pp rule is descriptive and is not an equivalence test.
- 36,379 existing source/E14 artifacts had identical SHA-256 before and after the pilot; inherited training, prediction, FeedbackControl and epoch-selection functions are AST-identical; the eight old modes give identical forward values.
