"""
Independent reference front for IGD (Section 5.4), replacing a self-referential
approach (build_reference_front over the SAME algorithms being compared).

WHY THIS MATTERS: if the reference front is just the non-dominated union of
the algorithms you're scoring, an algorithm that happens to explore an odd,
poorly-covered corner of objective space gets an artificially good-looking
IGD, while an algorithm that converges to a tight, well-covered region looks
worse even if that region is actually the best one.

This module instead runs a much larger, independent search (bigger
population, more generations, more seeds, combining BOTH RCDA-MOEA and
NSGA-II so the reference isn't biased toward either algorithm's own search
dynamics) ONCE per dataset, caches the result, and every subsequent
`run_analysis.py` call reuses that same fixed, independent front.
"""
from __future__ import annotations
import os
import json
import numpy as np
from rcda_moea.algorithm import RCDA_MOEA, RCDAConfig
from baselines.nsga2 import NSGA2, NSGA2Config
from rcda_moea.refpoint import get_or_compute_reference_point


def _non_dominated_union(points_list: list[np.ndarray]) -> np.ndarray:
    all_pts = np.vstack([p for p in points_list if len(p) > 0])
    if len(all_pts) == 0:
        return all_pts
    keep = np.ones(len(all_pts), dtype=bool)
    for i, p in enumerate(all_pts):
        if not keep[i]:
            continue
        dominated_by_others = np.any(
            np.all(all_pts <= p, axis=1) & np.any(all_pts < p, axis=1)
        )
        if dominated_by_others:
            keep[i] = False
    return np.unique(all_pts[keep], axis=0)


def compute_gold_reference_front(data_path: str, min_rows: int, min_cols: int,
                                  init_density: float, n_seeds: int = 10,
                                  population: int = 100, generations: int = 150,
                                  seed_start: int = 9001) -> np.ndarray:
    X = np.load(data_path)
    ref_point = get_or_compute_reference_point(data_path, min_rows, min_cols)
    all_fronts = []
    for i in range(n_seeds):
        seed = seed_start + i
        rcda_cfg = RCDAConfig(population_size=population, generations=generations,
                               min_rows=min_rows, min_cols=min_cols,
                               init_density=init_density, ref_point=ref_point)
        archive, _ = RCDA_MOEA(X, rcda_cfg, seed=seed).run()
        all_fronts.append(np.array([ind.objectives for ind in archive]))

        nsga_cfg = NSGA2Config(population_size=population, generations=generations,
                                min_rows=min_rows, min_cols=min_cols,
                                init_density=init_density, ref_point=ref_point)
        archive2, _ = NSGA2(X, nsga_cfg, seed=seed).run()
        all_fronts.append(np.array([ind.objectives for ind in archive2]))

        print(f"gold reference front: completed seed {seed} "
              f"({i + 1}/{n_seeds} pairs of large RCDA-MOEA+NSGA2 runs)")

    return _non_dominated_union(all_fronts)


def get_or_compute_gold_reference_front(data_path: str, min_rows: int, min_cols: int,
                                         init_density: float, **kwargs) -> np.ndarray:
    """Cache as <data_path>.gold_front.npy + a .json sidecar recording the
    settings used, so repeated `run_analysis.py` calls on the same dataset
    reuse the same independent front instead of recomputing."""
    cache_npy = data_path + ".gold_front.npy"
    cache_meta = data_path + ".gold_front.json"
    meta = {"min_rows": min_rows, "min_cols": min_cols, "init_density": init_density,
            "n_seeds": kwargs.get("n_seeds", 10),
            "population": kwargs.get("population", 100),
            "generations": kwargs.get("generations", 150),
            "seed_start": kwargs.get("seed_start", 9001)}
    if os.path.exists(cache_npy) and os.path.exists(cache_meta):
        with open(cache_meta) as f:
            cached_meta = json.load(f)
        if all(cached_meta.get(k) == v for k, v in meta.items()):
            return np.load(cache_npy)
        else:
            print("Cached gold reference front used different min_rows/min_cols/"
                  "init_density settings than the current run -- recomputing.")
    front = compute_gold_reference_front(data_path, min_rows, min_cols, init_density, **kwargs)
    np.save(cache_npy, front)
    with open(cache_meta, "w") as f:
        json.dump(meta, f)
    return front
