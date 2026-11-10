> English translation of the original working note `e6_report.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E6: inference diagnostics on Combined Yu3 checkpoints (Fig. 3; Supp. Tables S26–S27)

Existing Yu3 Combined models (seeds 42–51), no retraining. Four conditions per model: clean once; AWGN 10 dB, 5% impulses and 20% consecutive missing with 10 perturbation seeds each (310 inference items). Realizations are averaged within each model seed, then mean and SD are taken over the 10 seeds. F1 in %, differences in percentage points.

| Condition | Pass 1 F1 | Pass 2 F1 (normal feedback) | Pass 2 − Pass 1 | Normal − zeroed | Normal − cross-sample |
|---|---:|---:|---:|---:|---:|
| Clean | 71.85 ± 3.56 | 71.65 ± 4.30 | −0.20 ± 2.90 | −0.20 ± 2.90 | 4.78 ± 3.43 |
| AWGN 10 dB | 28.53 ± 3.41 | 25.06 ± 4.89 | −3.48 ± 2.41 | −3.48 ± 2.41 | 3.25 ± 1.28 |
| Impulse 5% | 62.21 ± 7.36 | 57.40 ± 7.94 | −4.80 ± 4.51 | −4.80 ± 4.51 | 3.59 ± 3.11 |
| Consecutive missing 20% | 66.29 ± 3.38 | 65.87 ± 4.47 | −0.42 ± 2.88 | −0.42 ± 2.88 | 4.78 ± 2.35 |

Decision changes in both directions are kept: wrong-to-right changes do not by themselves mean an overall improvement and must be read with right-to-wrong changes and Macro-F1. Per-sample data are in `predictions.csv`.

## Residual energy in the missing region
The region is the union of the original and interpolated V–I positions of the 100 missing samples, dilated by one pixel (3 × 3). An equally long, non-overlapping-in-time unmissed control segment is drawn with a fixed random seed (its projection may overlap; see `overlap_pixels`). Energy is the share of R² in the region relative to the whole image (NaN for zero denominators). `missing_energy_per_sample.csv` keeps all 21,900 values; `missing_energy_per_seed.csv` gives per-seed medians and IQRs, so samples and perturbation draws are not treated as independent model replicates.

## Limitations
These are test-time sensitivity diagnostics with fixed weights. Zeroed or cross-sample results are not strict causal proof, and concentration of residual energy is not evidence that missing information was recovered. E6 was implemented after the E1–E4 outer results had been seen. The examples in Fig. 3 were chosen by fixed sample IDs (the first case in each direction in a fixed order), not by how favorable they look.
