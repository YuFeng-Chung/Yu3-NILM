> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E10: WHITED architecture comparison (Experiment 2; Table IV; Supp. Tables S2–S4)

An independent study; existing models, experiments and the manuscript were not modified.

## Data
Official download `WhiteD.zip` (2,061,689,222 bytes; SHA-256 `88e6d5409a550645f7775e2470910b6620f2857fbe6ea7b56e0430919770a242`). It contains 1,339 FLAC files at the top level; the 1,259 recordings of regions r1–r8 documented in the archive README match the official count, and 80 undocumented r9 recordings are excluded. No official checksum exists for the phase-corrected release, so the actual downloaded archive is frozen; the release version is not claimed to be verified file by file. WHITED itself is not redistributed here (see `official_archive_readme.txt`).

## Split
A class is kept only if it has at least three device-name groups, so that inner-train, validation and outer-test can be separated: 15 classes, 76 groups, 759 records (`class_coverage.csv`, `sample_manifest.json`). Groups are class + device name + region, with kit and timestamp removed; physical serial numbers are not verified, and the split is not house-disjoint. Each class is divided into three mutually exclusive test folds, so every included device is tested once; within each fold one group per class is held out for validation (`split.json`).

## Training
15 single-seed epoch pilots, then M1–M5 × 3 folds × seeds 42/43/44 = 45 final trainings, clean data only. The outer test was opened only after all pilots had finished, epochs were frozen and all models were trained. Adam LR 0.0005, batch 32, dropout 0.5, no weight decay; max 100 epochs, epoch selectable from 20, patience 15, min_delta 0.001. No new hyperparameter search; the architectures and losses of the original manuscript are kept (M5 = detached-residual two-pass Yu3). `protocol.json` is the executed design.

## Preprocessing
One nominal mains cycle around the largest first difference of the current, with 20% pre-trigger; V and I resampled together to 500 points with a polyphase anti-aliasing filter; per-waveform z-score; 64 × 64 occupancy image saturated at 0.1 and quantized to uint8. For the same extracted waveforms, all 759 images are identical to those of the PLAID imaging function; this does not show that the windows capture every turn-on event or steady state. Channel 0 = voltage and 1 = current were inferred from the kit calibration factors (60–300 V rms and 45–65 Hz were checked for the included data), because the dataset README does not state the channel order explicitly.

## Outputs
`summary.csv`, `per_fold_seed.csv`, `per_class_prf.csv`, `confusion_counts.csv`, `paired_differences.csv`; `independent_audit.json` recomputes the metrics, PRF and integer confusion counts from all 11,385 saved test predictions and verifies split, IDs and sources. Code: `experiments/prepare_e10_whited.py`, `e10_core.py`, `run_e10_whited.py`, `analyze_e10_whited.py`, `analyze_e10_statistics.py`.

Means ± SD over three seeds describe training variability only. The three folds share training data, class inclusion is restricted, and all unfavorable results are kept; no confirmatory significance or equivalence claims are made.
