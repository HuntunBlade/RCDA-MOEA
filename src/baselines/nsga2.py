"""
Generic NSGA-II baseline for the same biclustering encoding/objectives.
This is a faithful, runnable NSGA-II (Deb et al. 2002): fixed crossover
probability, uniform mutation across the row and column segments with no
diversity-based dimension allocation, and no HV-feedback crossover control.
It is NOT a reproduction of any specific paper's biclustering-NSGA-II variant;
it is the textbook algorithm applied to this problem's encoding, used as a
neutral baseline to isolate the effect of RCDA-MOEA's two feedback layers.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from rcda_moea.core import Individual, random_individual, evaluate_population, repair
from rcda_moea.selection import fast_non_dominated_sort, environmental_selection, pareto_archive, hypervolume_2d
from rcda_moea.operators import tournament_select, crossover, _flip_add, _flip_delete


@dataclass
class NSGA2Config:
    population_size: int = 60
    generations: int = 100
    min_rows: int = 2
    min_cols: int = 2
    mutation_rate: float = 0.3
    pc: float = 0.8               # fixed Pc used by the reported real-data performance baseline
    init_density: float = 0.3
    ref_point: tuple = (1.5, 1.5)


def _uniform_mutate(ind: Individual, mutation_rate: float, min_rows: int, min_cols: int,
                     rng: np.random.Generator) -> Individual:
    """Mutation with no dimension-specific weighting: row/col chosen with equal probability."""
    if rng.random() >= mutation_rate:
        return ind
    if rng.random() < 0.5:
        ind.r = _flip_add(ind.r, rng) if rng.random() < 0.5 else _flip_delete(ind.r, min_rows, rng)
    else:
        ind.c = _flip_add(ind.c, rng) if rng.random() < 0.5 else _flip_delete(ind.c, min_cols, rng)
    return ind


class NSGA2:
    def __init__(self, X: np.ndarray, config: NSGA2Config, seed: int = 1):
        self.X = np.asarray(X, dtype=float)
        self.n, self.m = self.X.shape
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.ref_point = np.array(config.ref_point, dtype=float)

    def _init_population(self):
        pop = [
            random_individual(self.n, self.m, self.cfg.min_rows, self.cfg.min_cols,
                               self.rng, self.cfg.init_density)
            for _ in range(self.cfg.population_size)
        ]
        evaluate_population(pop, self.X)
        return pop

    def run(self):
        cfg = self.cfg
        pop = self._init_population()
        fast_non_dominated_sort(pop)
        hv_log = []
        for t in range(1, cfg.generations + 1):
            offspring = []
            while len(offspring) < cfg.population_size:
                p1 = tournament_select(pop, self.rng)
                p2 = tournament_select(pop, self.rng)
                c1, c2 = crossover(p1, p2, cfg.pc, self.rng)
                c1 = _uniform_mutate(c1, cfg.mutation_rate, cfg.min_rows, cfg.min_cols, self.rng)
                c2 = _uniform_mutate(c2, cfg.mutation_rate, cfg.min_rows, cfg.min_cols, self.rng)
                c1 = repair(c1, cfg.min_rows, cfg.min_cols, self.rng)
                c2 = repair(c2, cfg.min_rows, cfg.min_cols, self.rng)
                offspring.extend([c1, c2])
            offspring = offspring[:cfg.population_size]
            evaluate_population(offspring, self.X)
            combined = pop + offspring
            pop = environmental_selection(combined, cfg.population_size)
            fast_non_dominated_sort(pop)
            archive_now = pareto_archive(pop)
            hv_log.append(hypervolume_2d(
                np.array([ind.objectives for ind in archive_now]), self.ref_point
            ))
        return pareto_archive(pop), hv_log
