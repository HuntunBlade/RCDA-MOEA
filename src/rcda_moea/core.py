"""
Core data structures shared across RCDA-MOEA, the generic NSGA-II baseline,
and the adapter baselines (TSTP/AMoDeBic/OAEVOB).

An individual encodes a bicluster as a pair of boolean row/column masks,
following Section 3 (Problem Definition) of the paper:
    r = (r_1, ..., r_n),  c = (c_1, ..., c_m)
    x = [r | c]
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Individual:
    r: np.ndarray  # bool array, shape (n,)
    c: np.ndarray  # bool array, shape (m,)
    objectives: np.ndarray = field(default=None)  # [MSR, f2], filled by evaluate()
    rank: int = -1
    crowding: float = 0.0

    def copy(self) -> "Individual":
        ind = Individual(self.r.copy(), self.c.copy())
        if self.objectives is not None:
            ind.objectives = self.objectives.copy()
        return ind


def random_individual(n: int, m: int, min_rows: int, min_cols: int,
                       rng: np.random.Generator, init_density: float = 0.3) -> Individual:
    """Randomly initialize a bicluster respecting minimum cardinality."""
    r = rng.random(n) < init_density
    c = rng.random(m) < init_density
    r = _ensure_min_active(r, min_rows, rng)
    c = _ensure_min_active(c, min_cols, rng)
    return Individual(r, c)


def _ensure_min_active(mask: np.ndarray, min_active: int, rng: np.random.Generator) -> np.ndarray:
    """Repair operator: if fewer than min_active entries are True, activate more at random."""
    mask = mask.copy()
    n_active = int(mask.sum())
    if n_active < min_active:
        inactive_idx = np.flatnonzero(~mask)
        need = min_active - n_active
        need = min(need, len(inactive_idx))
        if need > 0:
            chosen = rng.choice(inactive_idx, size=need, replace=False)
            mask[chosen] = True
    return mask


def repair(ind: Individual, min_rows: int, min_cols: int, rng: np.random.Generator) -> Individual:
    """Enforce minimum-cardinality feasibility (Algorithm 1, line 10)."""
    ind.r = _ensure_min_active(ind.r, min_rows, rng)
    ind.c = _ensure_min_active(ind.c, min_cols, rng)
    return ind


def evaluate(ind: Individual, X: np.ndarray) -> np.ndarray:
    """
    Compute F(B) = [MSR(B), f2(B)] as defined in Section 3.

    MSR(B) = (1/|R||C|) * sum_{i in R, j in C} e_ij^2,
             e_ij = x_ij - xbar_i. - xbar_.j + xbar_..
    f2(B)  = 1 - |R||C| / (n*m)
    """
    n, m = X.shape
    R = np.flatnonzero(ind.r)
    C = np.flatnonzero(ind.c)
    sub = X[np.ix_(R, C)]
    row_means = sub.mean(axis=1, keepdims=True)
    col_means = sub.mean(axis=0, keepdims=True)
    grand_mean = sub.mean()
    residues = sub - row_means - col_means + grand_mean
    msr = float(np.mean(residues ** 2))
    f2 = 1.0 - (len(R) * len(C)) / (n * m)
    obj = np.array([msr, f2], dtype=float)
    ind.objectives = obj
    return obj


def evaluate_population(pop: list[Individual], X: np.ndarray) -> None:
    for ind in pop:
        evaluate(ind, X)
