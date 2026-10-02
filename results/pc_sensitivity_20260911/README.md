# NSGA-II Pc sensitivity (Tissp)

Exploratory check requested during the final pre-submission review.

- Dataset: `data/processed/tissp_gds596_genelevel_scaled.npy`
- Seeds: 1--10
- Population: 60
- Generations: 100
- min_rows/min_cols: 70/5
- init_density: 0.02
- mutation trigger rate: 0.3
- NSGA-II fixed Pc values: 0.7, 0.8, 0.95
- Comparator: the existing RCDA-MOEA runs for the same seeds 1--10
- Test: two-sided paired Wilcoxon, exploratory only (not added to the main Holm family)

See `tissp_summary.json` for the aggregate results and each `tissp_pc_*` directory for per-seed archives/HV logs.

Final HV values in the aggregate summary are recomputed from the saved final archive objectives with the shared Tissp reference point; trajectory logs are retained for inspection.
