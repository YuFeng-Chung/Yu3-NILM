> English translation of the original working note `README.md` (Chinese) in this folder. The original is kept unchanged because it, or the files it describes, is covered by the SHA-256 integrity records of this run. Operational details that only mattered while the run was in progress (process IDs, pause/resume times, report-drafting status) are omitted.

# E7-A: results across group splits (Combined strategy)

Four models; perturbation realizations are first averaged within each training seed. Per-class metrics are mean / sample SD across seeds. Confusion matrices sum the counts over all seeds and realizations and are row-normalized.

A group is the proxy "appliance class × house", not a verified unique physical device, and the split is not whole-house isolation.

The primary summary uses the five new splits. The six-split sensitivity analysis adds the main split (10 seeds) to the new splits (3 seeds each). Bootstrap intervals are conditional descriptions under dependent, overlapping splits; they are not used for claims of significance, equivalence or 95% population coverage.
