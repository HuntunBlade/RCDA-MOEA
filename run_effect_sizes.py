"""Paired HV effect sizes using final archive objectives.

The final performance metric is always recomputed from ``seed*_archive_objectives.npy``
with an explicitly supplied shared reference point. This avoids dependence on the
internal timing convention of an HV trajectory log.
"""
import argparse, json, os
import numpy as np
from scipy.stats import wilcoxon, rankdata
import sys
sys.path.insert(0, "src")
from rcda_moea.selection import hypervolume_2d


def load_hv(d, ref_point):
    vals = {}
    for f in os.listdir(d):
        if f.startswith("seed") and f.endswith("_archive_objectives.npy"):
            s = int(f[4:].split("_")[0])
            front = np.load(os.path.join(d, f))
            vals[s] = hypervolume_2d(front, np.asarray(ref_point, dtype=float))
    return vals


def rank_biserial_paired(d):
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz), method="average")
    wp = ranks[nz > 0].sum(); wm = ranks[nz < 0].sum()
    return float((wp - wm) / (wp + wm))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rcda", required=True); p.add_argument("--nsga2", required=True)
    p.add_argument("--ref-point", nargs=2, type=float, required=True, metavar=("MSR_REF", "F2_REF"))
    p.add_argument("--bootstrap", type=int, default=20000); p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--out", default=None); a = p.parse_args()
    x = load_hv(a.rcda, a.ref_point); y = load_hv(a.nsga2, a.ref_point)
    seeds = sorted(set(x) & set(y))
    if len(seeds) < 2:
        raise SystemExit("Need >=2 paired seeds in both directories")
    rc = np.array([x[s] for s in seeds], float); ns = np.array([y[s] for s in seeds], float)
    d = rc - ns; sd = d.std(ddof=1); dz = float(d.mean()/sd) if sd > 0 else float("inf")
    stat, pv = wilcoxon(d, alternative="two-sided")
    rng = np.random.default_rng(a.seed)
    means = np.empty(a.bootstrap)
    for i in range(a.bootstrap): means[i] = rng.choice(d, size=len(d), replace=True).mean()
    lo, hi = np.quantile(means, [.025, .975])
    out = {
        "metric_source": "final_archive_objectives",
        "reference_point": list(map(float, a.ref_point)),
        "n_pairs": len(d), "seeds": seeds,
        "rcda_mean": float(rc.mean()), "rcda_std_sample": float(rc.std(ddof=1)),
        "nsga2_mean": float(ns.mean()), "nsga2_std_sample": float(ns.std(ddof=1)),
        "mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
        "cohens_dz": dz, "rank_biserial": rank_biserial_paired(d),
        "wilcoxon_statistic": float(stat), "raw_p": float(pv),
        "bootstrap_mean_diff_ci95": [float(lo), float(hi)],
        "rcda_wins": int(np.sum(d > 0)), "nsga2_wins": int(np.sum(d < 0)), "ties": int(np.sum(d == 0)),
    }
    print(json.dumps(out, indent=2))
    if a.out:
        with open(a.out, "w") as f: json.dump(out, f, indent=2)

if __name__ == "__main__": main()
