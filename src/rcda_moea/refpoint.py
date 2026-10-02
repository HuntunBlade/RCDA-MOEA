"""
Adaptive hypervolume reference point.

The two objectives have very different natural scales:
  - f2 = 1 - |R||C|/(n*m) is exactly bounded in [0, 1).
  - MSR(B) has no fixed upper bound; its scale depends entirely on the
    variance of the input data matrix X and on how noisy/overlapping the
    biclusters are.

A HARD-CODED reference point (e.g. (1.5, 1.5)) silently breaks the moment a
dataset produces MSR values above that constant: hypervolume_2d filters out
any point that exceeds ref_point in any objective, so HV collapses toward
zero for reasons that have nothing to do with algorithm quality. This
function derives a data-dependent reference point instead, and caches it
next to the dataset so every algorithm compared on that dataset (RCDA-MOEA and NSGA-II) uses the identical value -- fairness
of the HV comparison depends on this being shared, not recomputed per run.
"""
from __future__ import annotations
import os
import json
import numpy as np
from .core import random_individual, evaluate


def compute_reference_point(X: np.ndarray, min_rows: int, min_cols: int,
                             n_samples: int = 300, seed: int = 12345,
                             margin: float = 1.15) -> tuple[float, float]:
    n, m = X.shape
    rng = np.random.default_rng(seed)
    msr_samples = []
    for _ in range(n_samples):
        ind = random_individual(n, m, min_rows, min_cols, rng, init_density=0.3)
        obj = evaluate(ind, X)
        msr_samples.append(obj[0])
    msr_samples = np.array(msr_samples)
    ref_msr = float(np.percentile(msr_samples, 99) * margin)
    # f2's true maximum for a feasible individual is bounded by minimum cardinality.
    ref_f2 = min(1.0, (1.0 - (min_rows * min_cols) / (n * m)) * margin)
    return ref_msr, ref_f2


def get_or_compute_reference_point(data_path: str, min_rows: int, min_cols: int) -> tuple[float, float]:
    """
    Cache the reference point as <data_path>.refpoint.json so that every
    script (run_rcda.py, run_baselines.py, run_ablation.py) invoked
    separately against the same dataset file uses the SAME reference point,
    without needing to pass it explicitly on the command line.
    """
    cache_path = data_path + ".refpoint.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("min_rows") == min_rows and cached.get("min_cols") == min_cols:
            return tuple(cached["ref_point"])
    X = np.load(data_path)
    ref = compute_reference_point(X, min_rows, min_cols)
    with open(cache_path, "w") as f:
        json.dump({"ref_point": list(ref), "min_rows": min_rows, "min_cols": min_cols}, f)
    return ref
