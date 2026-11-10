> English translation of the original working note `e5_supporting_evidence_20261106.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E5: component ablation with 10 paired seeds (Experiment 3; Table V; Supp. Tables S6–S8, S25)

The Yu3 architecture is kept, and controlled ablations are added as supporting evidence; the design was not changed on held-out results. 11 configurations × 10 training seeds = 110 trainings; each with 19 fixed epochs, Adam 0.0005, batch 32, dropout 0.5; the main 855/219 split.

The 33 runs of seeds 42–44 from the first round are reused unchanged, and seeds 45–51 (77 runs) were added in the same environment with the same code and input hashes. No low-scoring run was removed. The first three seeds had been seen, and the analysis plan was fixed before the remaining seeds finished; this is not a fully blind pre-registration.

## Clean data
| Configuration | Macro-F1 (%, mean ± SD) | Difference from Yu3 (pp) | Paired p (Holm over all 12 contrasts) |
|---|---:|---:|---:|
| Yu3 (Err-D, shared BN) | 68.46 ± 3.68 | +0.00 | — |
| Err-ND (residual, no stop-gradient) | 65.02 ± 5.10 | −3.45 | 1.0000 |
| Img-ND (image feedback, no stop-gradient) | 65.44 ± 3.84 | −3.03 | 1.0000 |
| Img-D (image feedback, stop-gradient) | 69.30 ± 3.80 | +0.84 | 1.0000 |
| No CE1 | 67.21 ± 3.12 | −1.25 | 1.0000 |
| No Rec | 59.26 ± 14.01 | −9.21 | 0.1172 |
| Rec w = 1 | 68.11 ± 3.70 | −0.36 | 1.0000 |
| RecAux (two-channel encoder, zero second channel) | 69.94 ± 2.51 | +1.48 | 1.0000 |
| Three-Pass | 67.26 ± 4.20 | −1.20 | 1.0000 |
| Pass BN (separate BN per pass) | 69.85 ± 3.45 | +1.39 | 0.4082 |
| No-FB (no reconstruction) | 69.39 ± 2.51 | +0.93 | 1.0000 |

n = 10 is the number of training seeds, not 110 independent seeds. Non-significance does not mean equivalence. All 12 contrasts (including the interaction contrast and RecAux vs No-FB), with uncorrected, all-contrast and within-family p values, are in `paired_tests_clean.csv`.

## BatchNorm and feedback
Pass BN: uncorrected p = 0.0371, family D p = 0.0371, all-12 p = 0.4082. Removing the reconstruction loss: uncorrected p = 0.0098, family B p = 0.0293, all-12 p = 0.1172. The original plan used within-family correction; the more conservative all-12 correction was added, and both are reported in full.
No clean advantage of Yu3 over RecAux was detected. Within the same Yu3 checkpoints, the mean Macro-F1 of pass 1 / pass 2 is 67.72% / 68.46%, with on average 4.4 wrong-to-right and 3.3 right-to-wrong changes (a description within one trained model, not an independent control).
BN input statistics are saved per Yu3 run (`bn_diagnostic.json`, archived with the checkpoints) and summarized in `bn_input_summary.csv` and `bn_pass_distribution_difference.csv` (also in `corrupted_evaluation/`): in eval mode, the input distribution of each layer in each pass is compared with the frozen running statistics; the size of the gap alone does not show an effect on performance. Pass BN separates both the affine parameters and the running statistics (convolutions remain shared), so its effect cannot be attributed to the statistics alone.
Yu3 and RecAux also differ in the number of passes and in the loss (0.5 CE1 + CE2 + 0.5 MSE versus CE1 + 0.5 MSE), so their comparison is a system-level contrast; signal type / stop-gradient and loss composition are controlled separately in families A and B.

## Waveform-level perturbations (secondary)
PLAID 2014 (https://doi.org/10.6084/m9.figshare.11605074.v1) was verified by the publisher's MD5, and the 1,074 clean images rebuilt from the raw waveforms matched the existing PNGs pixel by pixel and by group before evaluation. E5 keeps per-waveform z-score normalization; AWGN noise power is set from the de-meaned variance of each channel; consecutive missing removes the same 100/500-sample segment from both channels and fills it by linear interpolation. Perturbation seeds 1000–1009 are fixed by sample ID and identical for all models. The event window is extracted from the clean waveform first, then perturbed, normalized and imaged; event detection under noise is not tested.
Per-condition results (Macro-F1 and degradation from each configuration's own clean score) are in `corrupted_evaluation/summary_corrupted.csv` and `results_averaged.csv`; per-seed results, per-class results, confusion counts and training losses are in the same folder. Each training seed averages its 10 perturbation seeds, so n remains 10. The 24 secondary degradation contrasts (Holm) and the 36-contrast sensitivity correction are in `corrupted_evaluation/paired_tests_degradation.csv`.
Resolution: with 10 non-zero paired differences the minimum two-sided exact p is 2/1024 ≈ 0.00195; with Holm over 36 contrasts the smallest first-step adjusted p is at least 36 × 2/1024 ≈ 0.0703, so that sensitivity correction cannot reach 0.05 at this sample size.
Yu3 − RecAux degradation: AWGN 10 dB −3.74 pp (uncorrected p = 0.0273, Holm over 24 = 0.6289); 20% consecutive missing −0.52 pp (p = 0.6953, Holm 1.0000). Neither is claimed as a robustness gain. All four main controls have low Macro-F1 under AWGN 10 dB in this setting.

## Verification
The 19-epoch logs of all 110 runs, the 219 clean predictions, metric recomputation and BN update counts were checked by `verify_e5_outputs.py`. Source snapshots, environment versions, split and data hashes and paired-difference CSVs are kept. Runs were on Apple M2 (MPS); exact numerical reproduction of earlier CUDA results is not claimed. Wilcoxon: two-sided exact sign permutation; zero differences handled by the `wilcox` method (all-zero → p = 1); differences rounded to 12 decimals before ranking to avoid spurious floating-point ties.
