"""
Non-dominated sorting, crowding distance, and 2D hypervolume.
Standard NSGA-II machinery (Deb et al. 2002), used for environmental
selection (Section 4.6) and for the HV feedback signal (Section 4.5).
"""
from __future__ import annotations
import numpy as np
from .core import Individual


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b (minimization) if a <= b in all objectives and < in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(pop: list[Individual]) -> list[list[int]]:
    """
    Vectorized non-dominated sort: the O(N^2) pairwise domination check is done
    as ONE numpy broadcast comparison instead of N^2 individual Python-level
    dominates() calls, which is the dominant cost for larger populations.
    """
    N = len(pop)
    if N == 0:
        return []
    obj = np.stack([ind.objectives for ind in pop])          # (N, n_obj)
    diff = obj[:, None, :] - obj[None, :, :]                  # (N, N, n_obj)
    le_all = np.all(diff <= 0, axis=2)
    lt_any = np.any(diff < 0, axis=2)
    dom = le_all & lt_any                                     # dom[p, q]: p dominates q
    np.fill_diagonal(dom, False)

    S = [np.flatnonzero(dom[p]).tolist() for p in range(N)]
    n = dom.sum(axis=0)                                       # n[q] = # individuals dominating q

    fronts: list[list[int]] = [[]]
    for p in range(N):
        if n[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    pop[q].rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # remove trailing empty front
    return fronts


def crowding_distance(pop: list[Individual], front: list[int]) -> None:
    if not front:
        return
    n_obj = pop[front[0]].objectives.shape[0]
    for idx in front:
        pop[idx].crowding = 0.0
    for m in range(n_obj):
        front_sorted = sorted(front, key=lambda i: pop[i].objectives[m])
        f_min = pop[front_sorted[0]].objectives[m]
        f_max = pop[front_sorted[-1]].objectives[m]
        pop[front_sorted[0]].crowding = np.inf
        pop[front_sorted[-1]].crowding = np.inf
        denom = (f_max - f_min) if f_max > f_min else 1.0
        for k in range(1, len(front_sorted) - 1):
            prev_obj = pop[front_sorted[k - 1]].objectives[m]
            next_obj = pop[front_sorted[k + 1]].objectives[m]
            pop[front_sorted[k]].crowding += (next_obj - prev_obj) / denom


def environmental_selection(combined: list[Individual], N: int) -> list[Individual]:
    """Non-dominated sorting + crowding distance selection down to N individuals."""
    fronts = fast_non_dominated_sort(combined)
    selected: list[Individual] = []
    for front in fronts:
        crowding_distance(combined, front)
        if len(selected) + len(front) <= N:
            selected.extend(combined[i] for i in front)
        else:
            remaining = N - len(selected)
            front_sorted = sorted(front, key=lambda i: combined[i].crowding, reverse=True)
            selected.extend(combined[i] for i in front_sorted[:remaining])
            break
    return selected


def pareto_archive(pop: list[Individual]) -> list[Individual]:
    """Extract rank-0 (non-dominated) individuals as the archive."""
    fronts = fast_non_dominated_sort(pop)
    return [pop[i].copy() for i in fronts[0]] if fronts else []


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Exact 2D hypervolume (minimization). Points outside the reference box
    (any objective > ref_point for this minimization problem) are ignored;
    they are not clipped. The remaining points are swept in O(A log A).
    """
    if len(points) == 0:
        return 0.0
    pts = points[np.all(points <= ref_point, axis=1)]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    hv = 0.0
    prev_x = ref_point[0]
    best_y = ref_point[1]
    # sweep from largest f0 to smallest, tracking the running-min f1
    for x, y in pts[::-1]:
        if y < best_y:
            hv += (prev_x - x) * (best_y - y)
            best_y = y
            prev_x = x
    return float(hv)
