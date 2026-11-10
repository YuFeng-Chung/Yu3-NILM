> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E11: Yu3 versus WRG-CNN under a common protocol (Table X; Supp. Tables S28–S30)

18 tuning runs, 30 inner-CV runs, 5 epoch pilots, 50 final trainings and 50 × 211 complete evaluations were completed; an independent recomputation of 3,696,720 predictions passed (`completed_audit.json`).

## Design
- WRG-CNN uses the same 18-configuration Adam search as E2: batch 32, the 682/173 inner group split, seed 42, min 30 / max 150 epochs, patience 20. No extra optimizer or representation search is given.
- Strategy selection (E3) uses the same five group folds and six training recipes, with the v4 vote rules for noise and missing data; all ties are kept.
- Final WRG-CNN models: No-Defense, S3, Combined and any other selected strategy; each trained recipe has its epoch fixed by an inner pilot, then is trained on the 855 records with seeds 42–51. S0/S1/S2 share the No-Defense weights, with separate inference for each strategy.
- Yu3 reuses its frozen E2/E3/E4 search, CV and No-Defense/S3/Combined ten-seed results; it is neither retrained nor reselected.
- Shared 500-point waveforms, perturbation functions and perturbation seeds 1000–1009; each pipeline images the perturbed waveform itself. 211 conditions: clean + 21 levels × 10 realizations (AWGN 30/20/15/10/5/0 dB; impulses 1/5/10/20%; random missing 1/5/10/20%; consecutive missing 1/5/10/20/30/40/50%).
- The outer test was evaluated only after all WRG-CNN search, strategy selection, epoch selection and final training were done. All pairings are reported, including unfavorable ones.

## Representation and fairness adjustments
WRG-CNN keeps the authors' CNN with current-only mean-PAA, a 50 × 50 distance (recurrence) graph and ε = 10, δ = 20; the WRG CNN is not applied to V–I images. The common 500-point window replaces the original multi-cycle extraction, and Adam / batch 32 / tuned dropout follow the same search procedure as the other models (instead of the original SGD / batch 16 / dropout 0.25). This is a reimplementation under a common evaluation protocol, not a reproduction of the original paper's numbers. Differences between the paper (median) and the authors' code (mean), and the exception handling of the upstream MLP initialization, are kept and disclosed. Upstream code: https://github.com/sambaiga/WRG-NILM at commit `3e5879d9d1a9c4c1ef93c557a650454b1581ac87`.

This compares two complete pipelines; differences in input information and representation are not attributed to the CNN or to the feedback mechanism alone.

## Checks
`preflight_audit.json`: 132 value-by-value checks against the authors' graph construction, value checks of the shared waveform → image path for augmentation and the S0/S1/S2 filters, and real MPS training/reload for 3 recipes × 855 records × 2 epochs. The inference preflight used 219 training records as a shape proxy, not the test set. All weights and 4,220 prediction files of S3/S4 and dynamic masking/Combined were verified identical across the ten seeds.

## Scope of inference
Descriptive Macro-F1, Dabs and Drel; realizations are averaged within each training seed, then ten seeds are paired. Bootstrap intervals describe training randomness on the fixed split only and are not population confidence intervals. The test partition had been seen in earlier analyses; groups are class × house proxies; tuning is single-seed and not nested; runs were on a Mac, not CUDA.
Summary of the 112 paired comparisons: Yu3 (S3) is lower than WRG-CNN (S3/S4) from AWGN 20 dB to 0 dB and higher under impulses; under Combined, WRG-CNN is slightly higher on average at 50% consecutive missing. Files: `summary.csv`, `per_seed_averaged.csv`, `paired_comparisons.csv`, `per_class_averaged.csv`.
