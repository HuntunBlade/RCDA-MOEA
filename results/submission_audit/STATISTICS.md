# Submission statistics: common final-archive HV

Final HV is computed from each saved final Pareto archive after the last environmental-selection step.

| Dataset | RCDA mean ± sample SD | NSGA-II mean ± sample SD | raw Wilcoxon p | Holm p |
|---|---:|---:|---:|---:|
| Tissp | 0.0055802143 ± 0.00066143674 | 0.0057051955 ± 0.00074547126 | 0.556113 | 0.741481 |
| Ustilago | 0.00079399741 ± 6.2080568e-05 | 0.00083113506 ± 8.68178e-05 | 0.0919496 | 0.275849 |
| Mouse | 0.2089829 ± 0.018455678 | 0.21809291 ± 0.016207678 | 0.0576879 | 0.230752 |
| Cocel-GSE35896-alt | 0.00060224008 ± 4.8788773e-05 | 0.00061683304 ± 5.452645e-05 | 0.370741 | 0.741481 |

## Synthetic four-variant ablation (HV)

Friedman statistic = 3.88, p = 0.274717.

| Variant | mean ± sample SD |
|---|---:|
| full | 0.26311884 ± 0.017535711 |
| no_structural_diversity | 0.2649579 ± 0.021621198 |
| no_hv_feedback | 0.26365511 ± 0.020454435 |
| no_feedback | 0.27078063 ± 0.02205704 |

## Controlled fixed-length sensitivity

The corrected common-time-point rerun is stored under `results/final_20260911/controlled_fixed_length/`.
