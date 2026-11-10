> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E8: computational cost on a controlled PC (Experiment 5; Table XII, Fig. 5; Supp. Tables S12–S13)

- **Data**: `latency_summary.csv`, `raw_accepted_timings.csv`, `memory_summary.csv`; complete raw records in `timing_*.json` and `memory_*.json`.
- **Integrity**: `independent_audit.json` and `measurement_audit.json` both report `all_checks_passed = true`: 60,000 accepted primary calls, 20 settings, 20 independent memory processes, no predefined drift flags.
- **Transparent amendment**: the first PyTorch 4-thread batch was disturbed by external load and yielded only 23 valid blocks in 40 attempts. It is kept in full (`initial_incomplete_timing_pytorch_4.json` and its summary). Following `rerun_amendment.json`, the complete setting was re-measured once after a 30-s low-load gate, with 30/30 valid blocks. This rerun is an amendment and was not part of the original protocol.
- **Provenance**: `protocol.json`, `execution_source_manifest.json`, `source/`, `environment_packages.json`. The ONNX exports and synthetic inputs are archived with the checkpoints.

Scope: Apple M2 CPU, FP32, batch size 1, synthetic 1 × 1 × 64 × 64 inputs; one S3 checkpoint (seed 42) per model; no retraining and no use of test data. Only the forward pass of each network is timed: preprocessing, imaging, end-to-end latency on real data, MACs/FLOPs, other hardware and deployment are not covered. Memory is the RSS after importing the shared runtime and the lifetime peak of the process, not activation-only memory.
