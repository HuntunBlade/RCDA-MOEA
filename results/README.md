# Curated result artifacts

This directory contains saved outputs retained for manuscript audit.

## Final-HV convention

Submission statistics define final HV as `hypervolume(final_archive_objectives)` using the shared dataset reference point. Historical `seed*_hv_log.json` files are retained because they contain useful trajectories, but they are not treated as an authoritative final-performance source when auditing old runs.

`audit_saved_results.py` rebuilds complete summaries from the saved final archives and writes the consolidated audit to `submission_audit/`.

## Contents

- `raw/ablation/`: 30-run synthetic four-variant feedback ablation.
- `final_20260910/tissp/`: Tissp RCDA/NSGA-II saved runs and diagnostics.
- `final_20260910/cocel_alt/`: Cocel-alt RCDA/NSGA-II saved runs and diagnostics.
- `raw_mouse/{rcda,nsga2}/`: Mouse saved final archives/runs.
- `raw_ustilago/{rcda,nsga2}/`: Ustilago saved final archives/runs.
- `final_20260910/controlled_ratio/`: controlled row/column-ratio experiment; primary HV columns refreshed from saved final fronts.
- `final_20260911/controlled_fixed_length/`: rerun with the common final-archive HV convention; final fronts are retained.
- `pc_sensitivity_20260911/`: Tissp NSGA-II crossover-probability sensitivity, refreshed against final-archive RCDA HV.
- `submission_audit/`: machine-readable final-HV audit.

Unverified TSTP, AMoDeBic, and OAEVOB adapters/results are not included in this submission package and are not presented as official reproductions.

Fresh real-data reruns require the verified matrices described in `../data/README.md`.
