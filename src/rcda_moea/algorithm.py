"""
RCDA-MOEA: Row-Column Diversity-Aware Multi-Objective Evolutionary Algorithm.
Implements Algorithm 1 of the paper end-to-end.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .core import Individual, random_individual, evaluate_population
from .diversity import compute_row_col_diversity, compute_mutation_weights
from .selection import (
    fast_non_dominated_sort, environmental_selection, pareto_archive, hypervolume_2d,
)
from .operators import make_offspring


@dataclass
class RCDAConfig:
    population_size: int = 60
    generations: int = 100
    min_rows: int = 2
    min_cols: int = 2
    mutation_rate: float = 0.3
    epsilon: float = 1e-2          # diversity-allocation smoothing term (Sec 4.3)
    allocation_sharpness: float = 1.0  # >1 amplifies a D_R/D_C imbalance into a
                                        # sharper W_R/W_C split; 1.0 = paper's original formula
    hv_window: int = 5             # L: HV-stagnation window length (Sec 4.5)
    hv_epsilon: float = 0.02       # epsilon_HV: RELATIVE stagnation threshold
                                    # (fraction of current HV, not an absolute value --
                                    # see algorithm.py note on why this changed from 1e-4 absolute)
    pc_base: float = 0.7           # P_c^base
    pc_max: float = 0.95           # P_c^max
    init_density: float = 0.3
    ref_point: tuple = (1.5, 1.5)  # reference point for HV; objectives are in [0,1]-ish range


class RCDA_MOEA:
    def __init__(self, X: np.ndarray, config: RCDAConfig, seed: int = 1):
        self.X = np.asarray(X, dtype=float)
        self.n, self.m = self.X.shape
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.ref_point = np.array(config.ref_point, dtype=float)
        self.diagnostics: list[dict] = []  # populated by run() when record_diagnostics=True

    def _init_population(self) -> list[Individual]:
        pop = [
            random_individual(self.n, self.m, self.cfg.min_rows, self.cfg.min_cols,
                               self.rng, self.cfg.init_density)
            for _ in range(self.cfg.population_size)
        ]
        evaluate_population(pop, self.X)
        return pop

    def run(self, record_diagnostics: bool = False):
        """Run RCDA-MOEA for exactly ``generations`` reproduction/selection steps.

        Returns
        -------
        archive
            Final non-dominated archive *after* the last environmental-selection step.
        hv_log
            Post-selection hypervolume trajectory. ``hv_log[t-1]`` is the HV after
            generation ``t`` has completed, so ``hv_log[-1]`` is exactly the HV of
            the returned final archive. This convention matches the NSGA-II baseline
            and prevents an off-by-one search-budget comparison.

        Notes
        -----
        RCDA's HV-stagnation controller must observe the population *before* creating
        the next offspring. It therefore keeps a separate internal feedback-HV
        history. When diagnostics are requested, ``hv``/``hv_feedback`` denote that
        pre-offspring controller signal and ``hv_post`` denotes the post-selection
        performance HV for the same generation. The controller itself is unchanged.
        """
        cfg = self.cfg
        pop = self._init_population()
        fast_non_dominated_sort(pop)
        self.diagnostics = []

        hv_log: list[float] = []
        feedback_hv_history: list[float] = []

        for t in range(1, cfg.generations + 1):
            # 1. Row/column structural diversity of the current parent population.
            D_R, D_C = compute_row_col_diversity(pop)

            # 2. Dimension-specific mutation allocation.
            W_R, W_C = compute_mutation_weights(
                D_R, D_C, cfg.epsilon, cfg.allocation_sharpness
            )

            # 3. Pre-offspring HV is the feedback signal used by the controller.
            archive_before = pareto_archive(pop)
            hv_feedback = hypervolume_2d(
                np.array([ind.objectives for ind in archive_before]), self.ref_point
            )
            feedback_hv_history.append(hv_feedback)
            if len(feedback_hv_history) > cfg.hv_window:
                prev_hv = feedback_hv_history[-1 - cfg.hv_window]
                threshold = cfg.hv_epsilon * max(abs(hv_feedback), 1e-6)
                stagnant = abs(hv_feedback - prev_hv) <= threshold
            else:
                stagnant = False
            pc = cfg.pc_max if stagnant else cfg.pc_base

            # 4-5. Offspring generation: crossover + dimension-specific mutation + repair.
            offspring = make_offspring(
                pop, pc, W_R, W_C, cfg.mutation_rate,
                cfg.min_rows, cfg.min_cols, self.rng,
            )
            evaluate_population(offspring, self.X)

            # 6. Environmental selection.
            pop = environmental_selection(pop + offspring, cfg.population_size)
            fast_non_dominated_sort(pop)

            # Performance HV is recorded AFTER the generation has completed.
            archive_after = pareto_archive(pop)
            hv_post = hypervolume_2d(
                np.array([ind.objectives for ind in archive_after]), self.ref_point
            )
            hv_log.append(hv_post)

            if record_diagnostics:
                self.diagnostics.append({
                    "gen": t,
                    "D_R": D_R,
                    "D_C": D_C,
                    "W_R": W_R,
                    "W_C": W_C,
                    "pc": pc,
                    "hv": hv_feedback,          # backward-compatible diagnostic key
                    "hv_feedback": hv_feedback,
                    "hv_post": hv_post,
                    "stagnant": stagnant,
                })

        archive = pareto_archive(pop)
        return archive, hv_log
