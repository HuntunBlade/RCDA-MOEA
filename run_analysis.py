"""
Aggregates results/raw/<algo>/seed*_archive_objectives.npy + hv_log.json across
algorithms into the Table 2 (main comparison) / Table 3 (ablation) format from
the paper, and runs the Wilcoxon / Friedman+Holm tests from Section 5.5.

Usage:
    python run_analysis.py --data data/processed/tissp.npy \
        --results-root results/raw --algos rcda nsga2 tstp amodebic oaevob \
        --min-rows 100 --min-cols 5 --init-density 0.02

IGD reference front: computed ONCE per dataset by an independent, large-budget
search (see src/experiments/reference_front.py) -- NOT from the union of the
algorithms being compared here.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "src")
from experiments.metrics import igd, friedman_holm
from experiments.reference_front import get_or_compute_gold_reference_front
from rcda_moea.selection import hypervolume_2d
from rcda_moea.refpoint import get_or_compute_reference_point


def load_algo_results(root: str, algo: str, ref_point):
    algo_dir = os.path.join(root, algo)
    seeds = sorted(
        int(f.split("seed")[1].split("_")[0])
        for f in os.listdir(algo_dir) if f.endswith("_archive_objectives.npy")
    )
    fronts, hvs = [], []
    for s in seeds:
        fronts.append(np.load(os.path.join(algo_dir, f"seed{s}_archive_objectives.npy")))
        # Final performance is defined by the returned final archive, not by
        # an implementation-specific trajectory logging time point.
        hvs.append(hypervolume_2d(fronts[-1], np.asarray(ref_point, dtype=float)))
    return seeds, fronts, np.array(hvs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                    help="Same dataset .npy used for the runs being analyzed -- needed to "
                         "compute/cache the independent gold reference front for IGD.")
    p.add_argument("--results-root", default="results/raw")
    p.add_argument("--algos", nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-rows", type=int, default=2)
    p.add_argument("--min-cols", type=int, default=2)
    p.add_argument("--init-density", type=float, default=0.3)
    p.add_argument("--gold-front-seeds", type=int, default=10,
                    help="Number of large-budget RCDA-MOEA+NSGA2 run PAIRS used to build the "
                         "independent reference front (only spent once per dataset; cached).")
    a = p.parse_args()

    ref_point = get_or_compute_reference_point(a.data, a.min_rows, a.min_cols)
    per_algo = {}
    for algo in a.algos:
        seeds, fronts, hvs = load_algo_results(a.results_root, algo, ref_point)
        per_algo[algo] = {"seeds": seeds, "fronts": fronts, "hv": hvs}

    ref_front = get_or_compute_gold_reference_front(
        a.data, a.min_rows, a.min_cols, a.init_density, n_seeds=a.gold_front_seeds
    )
    print(f"(using independent gold reference front with {len(ref_front)} points)\n")

    print("=== Table 2: Main comparison (mean +/- std over seeds) ===")
    print(f"{'Method':<12}{'HV':>14}{'IGD':>14}")
    igd_per_algo = {}
    for algo, d in per_algo.items():
        igds = np.array([igd(ref_front, f) for f in d["fronts"]])
        igd_per_algo[algo] = igds
        print(f"{algo:<12}{d['hv'].mean():>8.5f}+-{d['hv'].std():<5.5f}"
              f"{igds.mean():>8.5f}+-{igds.std():<5.5f}")

    if len(per_algo) >= 3:
        min_runs = min(len(d["hv"]) for d in per_algo.values())
        hv_samples = {algo: d["hv"][:min_runs] for algo, d in per_algo.items()}
        print("\n=== Friedman + Holm on HV (Section 5.5) ===")
        result = friedman_holm(hv_samples, alpha=a.alpha)
        print(json.dumps(result, indent=2))

        igd_samples = {algo: igd_per_algo[algo][:min_runs] for algo in per_algo}
        print("\n=== Friedman + Holm on IGD (Section 5.5) ===")
        result_igd = friedman_holm(igd_samples, alpha=a.alpha)
        print(json.dumps(result_igd, indent=2))
    else:
        print("\nNeed >=3 algorithms with equal run counts for Friedman+Holm; "
              "use scipy.stats.wilcoxon directly for pairwise-only comparisons.")


if __name__ == "__main__":
    main()
