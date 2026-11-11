# Release notes

## v1.0.0 public release

Before the repository was made public, the following documentation and result-description files were edited to remove internal project-management notes: references to manuscript drafts and reviewer-response drafts, internal review-item identifiers, and draft response paragraphs (the `responses` field of the two `content_manifest.json` files). No code, data split, model setting, numerical result, figure or statistical output was changed.

Some SHA-256 manifests in `results/` were frozen before this cleanup and still record the hashes of the original files. The table below lists every edited file with its original and current SHA-256 (first 16 hex digits), so the manifests can be reconciled. Hash checks of these files against the frozen manifests will therefore report a mismatch; all other files are unchanged.

| File | Original SHA-256 | Current SHA-256 |
|---|---|---|
| `results/e14_noisefb_20261110/content_manifest.json` | `f2a00cb9abc55355…` | `125d8f554341d944…` |
| `results/e15_singlepass_20261110/content_manifest.json` | `8c3e06137b9f969a…` | `e6a31a555f05c563…` |
| `results/e10_whited_20261108/README.md` | `c1afbccd6d716f0c…` | `86ac7fde636921f1…` |
| `results/e11_wrg_20261108/README.md` | `3204233528d0c01a…` | `0243d9919e993605…` |
| `results/e12_measurement_20261108/README.md` | `2758abd97cc17643…` | `552688ac8034ec2b…` |
| `results/e14_noisefb_20261110/README.md` | `018e079b11d5516b…` | `2e0bd151dd2ebee9…` |
| `results/e15_singlepass_20261110/README.md` | `ed170217afb0f801…` | `05470b9532881d4f…` |
| `results/e1_e4_mac_20261106/followup_report.md` | `6b3a883c10fa052f…` | `b2e91f801a2d70af…` |
| `results/e1_e4_mac_20261106/tie_rule_vote_only_20261107/README.md` | `e4e108a756db7f8f…` | `436c4ef5b16cc235…` |
| `results/e6_mechanism_20261108/e6_report.md` | `8725b0113bbea630…` | `ca298b52d985ed3d…` |
| `results/e8_cpu_20261108/README.md` | `53994cb359ae126e…` | `145ce69d8280ecc0…` |
| `results/e9_preprocessing_20261108/README.md` | `7b7006a494d162ee…` | `804b6beaf9ab87b1…` |
| `results/e5_full_20261106/e5_supporting_evidence_20261106.md` | `a581024b0d72f377…` | `b63b9401701e5f33…` |
| `results/e5_full_20261106/completion_status.json` | `87647c3d9e56ac3d…` | `f0e5fabb158f9e83…` |
| `results/e7_s3_splits_20261108/README.md` | `8481ad8916153a96…` | `3abe530f4e467894…` |
| `results/e7_s3_splits_20261108/protocol.json` | `93effa7f82f7783c…` | `f1dce3c20e32326a…` |
| `results/e7_splits_20261108/README.md` | `4528eb2fcd0b38e6…` | `69653aa350828313…` |

Full original and current hashes: `docs/release_cleanup_hashes.json`.

Other corrections in this release:
- `README.md`, `requirements.txt`, `docs/REPRODUCE.md`: Experiment 1 software environment corrected to Python 3.9.25, PyTorch 2.6.0+cu124 and torchvision 0.21.0+cu124.
- `CITATION.cff`: article title updated to the revised title.
- `data/plaid_2014_published/README.md`: redistribution statement for the derived PLAID windows (CC BY 4.0).
