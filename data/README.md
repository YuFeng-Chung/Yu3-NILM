# Data

## PLAID 2014 (`data/plaid_2014_published/`)
Source: https://doi.org/10.6084/m9.figshare.11605074.v1 (Gao et al.; Medico et al.), license CC BY 4.0.
- `meta_2014.json`: record metadata (appliance type, house, record ID).
- `raw_windows.npz`: 500-sample voltage/current windows extracted around the event locator (derived data, CC BY 4.0).
- `reconstructed_label_map.json`, `reconstructed_exp4_norm_stats.json`: 11-class label map and the training-set normalization statistics used in Experiment 4.
- Group-disjoint split: `實驗一/dataset/split_record.json` (group = appliance type × house; 855/219 records). The five extra splits are in `results/e7_splits_20261108/`.

## WHITED v1.1
Not redistributed. Obtain it from the dataset authors and place it in `data/whited_v11/`. The included classes, groups and folds are in `results/e10_whited_20261108/` (`split.json`, `sample_manifest.json`, `class_coverage.csv`).
