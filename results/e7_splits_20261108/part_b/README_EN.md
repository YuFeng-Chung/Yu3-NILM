> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E7-B: per-class analysis on the main split (Combined strategy)

Four models; perturbation realizations are first averaged within each training seed. Per-class metrics are mean / sample SD across seeds. Confusion matrices sum the counts over all seeds and realizations and are row-normalized. Perturbation realizations are not independent trainings.

A group is the proxy "appliance class × house", not a verified unique physical device, and the split is not whole-house isolation. See `protocol.json` for sources and statistical limits.
