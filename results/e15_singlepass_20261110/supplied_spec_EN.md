> English translation of the original working note `supplied_spec.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. This specification was frozen before any E15 result existed. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E15 specification: single pass with a varying second channel (1P-NoiseCh, 1P-Mix)

Provided by the authors on 2026-11-09. The rules below were frozen before results were produced.

## 1. Aim
E14 showed that when the second channel is non-zero and varying during training (NoiseFB, PermFB), the robustness of pass 1 to unseen impulses matches Yu3, whereas an all-zero channel (ZeroFB) does not. E15 asks whether two passes are needed at all, or whether a single pass with noise in the second channel suffices.

## 2. Two new models
Isolated copy of the E14 `e14_core.py`; no existing mode is changed. Same two-channel encoder, decoder and classifier as Yu3/NoiseFB; one forward pass only; loss L = CE + 0.5·MSE(X̂, X).
- **1P-NoiseCh (primary)**: training input is always [X, N].
- **1P-Mix (secondary)**: each training sample uses [X, 0] with probability 0.5 and [X, N] with probability 0.5, decided by a stable random draw.

N = clamp(μ0 + σ0·ε, min = 0), ε ~ N(0, 1) independently per pixel. μ0 and σ0 are fixed scalars: from the ten E14 NoiseFB final checkpoints on the 855 training images, take the per-sample pixel mean and SD of the pass-1 absolute residual R, then the median over all samples × seeds. They are written to `noise_scale.json` before launch and never changed; no test data are used.
Random numbers as in E14: SHA-256 stable seeds (training: seed, epoch, batch index; inference: sample ID), CPU `torch.Generator`, no consumption of the global RNG.

## 3. Training protocol
No-Defense only; LR 5e-4, weight decay 0, dropout 0.5, batch 32. Epochs are chosen per model by an inner-validation pilot (min 30, patience 30, max 150) without looking at the outer test. Seeds 42–51; the main 855/219 split; training-set fixed V/I normalization; same Mac/MPS; inference batch 32.

## 4. Evaluation
All perturbation conditions and realizations of E4/E14 (seeds 1000–1009). Each model is evaluated with [X, 0] (primary; corresponds to deploying Yu3 pass 1) and [X, N] (secondary; inference noise fixed by sample ID).

## 5. Pre-specified comparisons and interpretation
Primary endpoint: No-Defense, impulses 5%, 10%, 20%, Macro-F1 with [X, 0].
Primary comparisons: 1P-NoiseCh versus NoiseFB pass 1 and versus ZeroFB; within each level two exact two-sided paired Wilcoxon tests (n = 10) with Holm correction. 1P-Mix is analysed the same way as a secondary analysis. NoiseFB/ZeroFB values come from E14 without retraining.
Applied per impulse level (3 pp is a descriptive threshold, not an equivalence test):
- Significantly above ZeroFB and within 3 pp of NoiseFB → two passes are not needed; the robustness comes from the varying auxiliary channel during training.
- Within 3 pp of ZeroFB and significantly below NoiseFB → two passes are necessary.
- Anything in between → reported as a partial effect. If levels disagree, each level is reported; no favorable level is selected.
Exploratory (uncorrected): Yu3 pass 1, RecAux, M1; all AWGN levels. The results are reported in the paper, the supplement and the response letter whatever their direction.
