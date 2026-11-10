> English translation of the original working note `prelaunch_report.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E14 pre-launch report (reported to the authors before final training, 2026-11-09)

- Original runner `experiments/run_revision_followup_v4.py` with engine `revision_followup_core.py`; all source hashes in `protocol_v4.json` verified. E14 uses an isolated entry `source/run_e14.py` and engine `source/e14_core.py` (diff: `source/core_changes.diff`); existing code is not modified.
- Epoch pilots (seed 42):
  - NoiseFB: best epoch 42, stopped at 72, validation Macro-F1 0.6955 → fixed_epochs 42.
  - PermFB: best epoch 49, stopped at 79, validation Macro-F1 0.7085 → fixed_epochs 49.
- Pilot rule: min 30 / patience 30 / max 150 / delta 0.001; highest clean validation F1 from epoch 30. The earlier v4 No-Defense pilots used patience 20; E14 follows the new specification (patience 30), so pilot patience is not identical.
- Final: 10 seeds (42–51) per model, 855 train / 219 test, 20 models; LR 0.0005, WD 0, dropout 0.5, Adam, batch 32, MPS.
- Evaluation: all E4 conditions (clean + 210 perturbed; 10 realizations), same sample IDs and perturbation formulas. Old model outputs are verified value-by-value against the saved E4 predictions.
- Noise uses population SD (correction = 0); random numbers come from a CPU `torch.Generator` with a fixed SHA-256 mapping (training key: seed, epoch, batch; evaluation key: sample ID), not Python's unstable hash.
- No E14 test result was read before epochs were chosen. Primary endpoint: pass 1 at impulses 5/10/20%; within each level, NoiseFB versus Yu3 and versus ZeroFB, two exact two-sided Wilcoxon tests with Holm correction. PermFB is analysed the same way as a secondary analysis; ShuffleFB and all other comparisons are exploratory and uncorrected.

## Runner provenance correction
The old ZeroFB/ShuffleFB No-Defense checkpoints were produced on 2026-11-06 by the initial `run_revision_followup.py` with the identical shared `revision_followup_core.py`, before v4 existed. The pilot/final functions are AST-identical between v1 and v4. ZeroFB used 32 epochs and ShuffleFB 36 (pilot patience 20). This corrects naming only; no design or training change. Evidence: `runner_provenance.json`.
