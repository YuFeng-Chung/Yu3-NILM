> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# Final strategy-selection rule (v4), confirmed by the authors

Clarification of the rule stated in the original manuscript: candidates are first compared with the existing criterion, and are tied only when that criterion cannot separate them. This record replaces the v3 rule ("if noise votes tie, compare mean F1").

- **Noise defenses**: in each fold, the candidate with the highest mean Macro-F1 over AWGN 0 dB and 20% impulses gets one vote; exact ties within a fold give each tied candidate a vote. The final choice is by total votes over five folds; candidates with the same highest vote count are all kept. Cross-fold mean F1 is descriptive only and is not used to break ties.
- **Missing-data defenses**: a candidate wins a fold only if it is strictly better than No-Defense in that fold. Candidates are ranked first by the smaller of the two win counts (random and consecutive missing), then by total wins; candidates tied on both are all kept. No-Defense is not itself ranked as a defense candidate.
- Winners are not chosen by listing order, mean score or outer-test results. All tied strategies are trained, evaluated and reported, and all distinct strategy pairs enter the same Holm family.
- This amendment was made by the authors after some inner-validation results had been seen but before any outer-test evaluation; the procedure is therefore not described as fully pre-registered. Earlier versions are kept.

Checks: 1,720 saved training/prediction artifacts were verified unchanged by SHA-256, 584,740 saved inner-validation predictions were recomputed, and a rule test confirmed that "equal votes, different mean F1" remains a tie. Entry points: `experiments/run_revision_followup_v4.py`, `experiments/analyze_revision_followup_v4.py`, protocol `protocol_v4.json`. Training data, folds, seeds, epochs and perturbation conditions are unchanged by the amendment.

Reporting convention: under the selected hyperparameters, ResNet18 and MobileNetV3-Small have identical effective settings for S3 and S4, and for dynamic masking and Combined. These pairs are therefore reported together, with the original strategy labels kept.
