"""
Variation operators: uniform crossover (row/column segments independently)
and dimension-specific row/column mutation (Add/Delete), matching the
"Crossover Operator" and "Row and Column Operators" subsections of Section 4.
"""
from __future__ import annotations
import numpy as np
from .core import Individual, repair


def tournament_select(pop: list[Individual], rng: np.random.Generator) -> Individual:
    """Binary tournament on non-domination rank, then crowding distance."""
    i, j = rng.integers(0, len(pop), size=2)
    a, b = pop[i], pop[j]
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding > b.crowding else b


def uniform_crossover_segment(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    mask = rng.random(a.shape[0]) < 0.5
    child1 = np.where(mask, a, b)
    child2 = np.where(mask, b, a)
    return child1, child2


def crossover(p1: Individual, p2: Individual, pc: float,
              rng: np.random.Generator) -> tuple[Individual, Individual]:
    """
    Uniform crossover applied independently to the row segment and the column
    segment. Uniform crossover is used (rather than single-point) because bits
    represent unordered set-membership of row/column indices with no positional
    dependency.
    """
    if rng.random() >= pc:
        return p1.copy(), p2.copy()
    r1, r2 = uniform_crossover_segment(p1.r, p2.r, rng)
    c1, c2 = uniform_crossover_segment(p1.c, p2.c, rng)
    return Individual(r1, c1), Individual(r2, c2)


def _flip_add(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    inactive = np.flatnonzero(~mask)
    if len(inactive) == 0:
        return mask
    mask = mask.copy()
    mask[rng.choice(inactive)] = True
    return mask


def _flip_delete(mask: np.ndarray, min_active: int, rng: np.random.Generator) -> np.ndarray:
    active = np.flatnonzero(mask)
    if len(active) <= min_active:
        return mask
    mask = mask.copy()
    mask[rng.choice(active)] = False
    return mask


def mutate(ind: Individual, W_R: float, W_C: float, mutation_rate: float,
           min_rows: int, min_cols: int, rng: np.random.Generator) -> Individual:
    """
    Dimension-specific mutation (Algorithm 1, line 9). For each mutation event,
    the row segment is chosen with probability W_R and the column segment with
    probability W_C; within the chosen segment, Add and Delete are applied with
    equal probability subject to the minimum-cardinality constraint.
    """
    if rng.random() >= mutation_rate:
        return ind
    if rng.random() < W_R:
        if rng.random() < 0.5:
            ind.r = _flip_add(ind.r, rng)
        else:
            ind.r = _flip_delete(ind.r, min_rows, rng)
    else:
        if rng.random() < 0.5:
            ind.c = _flip_add(ind.c, rng)
        else:
            ind.c = _flip_delete(ind.c, min_cols, rng)
    return ind


def make_offspring(pop: list[Individual], pc: float, W_R: float, W_C: float,
                    mutation_rate: float, min_rows: int, min_cols: int,
                    rng: np.random.Generator) -> list[Individual]:
    offspring = []
    N = len(pop)
    while len(offspring) < N:
        p1 = tournament_select(pop, rng)
        p2 = tournament_select(pop, rng)
        c1, c2 = crossover(p1, p2, pc, rng)
        c1 = mutate(c1, W_R, W_C, mutation_rate, min_rows, min_cols, rng)
        c2 = mutate(c2, W_R, W_C, mutation_rate, min_rows, min_cols, rng)
        c1 = repair(c1, min_rows, min_cols, rng)
        c2 = repair(c2, min_rows, min_cols, rng)
        offspring.append(c1)
        offspring.append(c2)
    return offspring[:N]
