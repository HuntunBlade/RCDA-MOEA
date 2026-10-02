"""
Metrics (Section 5.4) and statistical testing (Section 5.5).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import rankdata


def igd(reference_front: np.ndarray, approx_front: np.ndarray) -> float:
    """
    IGD(P*, P) = (1/|P*|) * sum_{x in P*} min_{y in P} d(x, y)
    reference_front: (k, n_obj) array, ideally the best-known / combined Pareto front
    approx_front: (a, n_obj) array
    """
    if len(reference_front) == 0:
        return float("nan")
    if len(approx_front) == 0:
        return float("inf")
    dists = np.linalg.norm(reference_front[:, None, :] - approx_front[None, :, :], axis=2)
    return float(np.mean(dists.min(axis=1)))


def build_reference_front(*fronts: np.ndarray) -> np.ndarray:
    """Combine multiple algorithms' final archives and keep the non-dominated union
    as a practical reference front (no true optimum is known)."""
    all_pts = np.vstack([f for f in fronts if len(f) > 0])
    keep = []
    for i, p in enumerate(all_pts):
        dominated = np.any(np.all(all_pts <= p, axis=1) & np.any(all_pts < p, axis=1))
        if not dominated:
            keep.append(p)
    return np.array(keep)


def wilcoxon_pairwise(x: np.ndarray, y: np.ndarray):
    """Wilcoxon signed-rank test between two paired samples (e.g., per-seed HV of two methods)."""
    if len(x) != len(y):
        raise ValueError("Paired Wilcoxon requires equal-length, paired samples (same seeds).")
    stat, p = wilcoxon(x, y, alternative='two-sided')
    return {"statistic": float(stat), "p_value": float(p)}


def friedman_holm(samples: dict[str, np.ndarray], alpha: float = 0.05):
    """
    Friedman test across >=3 methods, each with the same number of paired runs,
    followed by Holm correction for pairwise post-hoc comparisons.
    samples: {method_name: array of per-seed metric values (same seeds/order)}
    """
    names = list(samples.keys())
    data = np.array([samples[n] for n in names])  # (k_methods, n_runs)
    stat, p = friedmanchisquare(*data)
    result = {"friedman_statistic": float(stat), "friedman_p": float(p), "alpha": alpha}

    # Holm post-hoc on average ranks (pairwise Wilcoxon p-values, Holm-corrected)
    k = len(names)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    raw_p = []
    for i, j in pairs:
        _, p_ij = wilcoxon(data[i], data[j], alternative='two-sided')
        raw_p.append(p_ij)
    order = np.argsort(raw_p)
    m = len(raw_p)
    holm_p = np.empty(m)
    for rank, idx in enumerate(order):
        holm_p[idx] = min(1.0, raw_p[idx] * (m - rank))
    # enforce monotonicity
    sorted_idx = order
    running_max = 0.0
    for idx in sorted_idx:
        running_max = max(running_max, holm_p[idx])
        holm_p[idx] = running_max

    result["pairwise"] = [
        {
            "pair": (names[i], names[j]),
            "raw_p": float(raw_p[k_]),
            "holm_p": float(holm_p[k_]),
            "significant": bool(holm_p[k_] < alpha),
        }
        for k_, (i, j) in enumerate(pairs)
    ]
    return result
