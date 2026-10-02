"""
Experiment runners. run_rcda_dataset matches the exact call signature used
by run_rcda.py (already in your package):

    run_rcda_dataset(a.data, a.out, a.runs, a.population, a.generations, a.seed_start)
"""
from __future__ import annotations
import os
import json
import numpy as np
from rcda_moea.algorithm import RCDA_MOEA, RCDAConfig
from rcda_moea.selection import hypervolume_2d
from rcda_moea.refpoint import get_or_compute_reference_point


def _save_run(out_dir: str, seed: int, archive, hv_log, diagnostics=None):
    os.makedirs(out_dir, exist_ok=True)
    objs = np.array([ind.objectives for ind in archive]) if archive else np.zeros((0, 2))
    np.save(os.path.join(out_dir, f"seed{seed}_archive_objectives.npy"), objs)
    with open(os.path.join(out_dir, f"seed{seed}_hv_log.json"), "w") as f:
        json.dump(hv_log, f)
    if diagnostics is not None:
        with open(os.path.join(out_dir, f"seed{seed}_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)


def run_rcda_dataset(data_path: str, out_dir: str, runs: int, population: int,
                      generations: int, seed_start: int = 1, config: RCDAConfig = None):
    """
    Runs RCDA-MOEA `runs` times (independent seeds) on the dataset at data_path
    (a .npy file containing the raw n x m data matrix) and saves each run's
    final Pareto-archive objectives and HV trajectory to out_dir.
    """
    X = np.load(data_path)
    results = []
    ref_min_rows = (config.min_rows if config else RCDAConfig().min_rows)
    ref_min_cols = (config.min_cols if config else RCDAConfig().min_cols)
    ref_point = get_or_compute_reference_point(data_path, ref_min_rows, ref_min_cols)
    cfg = config or RCDAConfig(population_size=population, generations=generations)
    cfg.population_size = population
    cfg.generations = generations
    cfg.ref_point = ref_point
    for i in range(runs):
        seed = seed_start + i
        cfg_i = cfg
        algo = RCDA_MOEA(X, cfg_i, seed=seed)
        archive, hv_log = algo.run(record_diagnostics=True)
        _save_run(out_dir, seed, archive, hv_log, diagnostics=algo.diagnostics)
        # Compute the reported final HV directly from the returned final archive.
        # This is deliberately independent of trajectory-logging conventions.
        final_hv = hypervolume_2d(
            np.array([ind.objectives for ind in archive]), np.asarray(cfg_i.ref_point, dtype=float)
        ) if archive else 0.0
        if hv_log and not np.isclose(final_hv, hv_log[-1], rtol=1e-12, atol=1e-15):
            raise RuntimeError("RCDA final archive HV does not match hv_log[-1]")
        results.append({"seed": seed, "final_hv": final_hv, "archive_size": len(archive)})
        print(f"[RCDA-MOEA] seed={seed} final_HV={final_hv:.5f} archive_size={len(archive)}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_baseline_dataset(algo_cls, config, data_path: str, out_dir: str, runs: int,
                          seed_start: int = 1, label: str = "baseline"):
    """Generic runner used by the validated NSGA-II comparison baseline."""
    X = np.load(data_path)
    results = []
    ref_point = get_or_compute_reference_point(data_path, config.min_rows, config.min_cols)
    config.ref_point = ref_point
    for i in range(runs):
        seed = seed_start + i
        algo = algo_cls(X, config, seed=seed)
        archive, hv_log = algo.run()
        _save_run(out_dir, seed, archive, hv_log)
        final_hv = hypervolume_2d(
            np.array([ind.objectives for ind in archive]), np.asarray(config.ref_point, dtype=float)
        ) if archive else 0.0
        if hv_log and not np.isclose(final_hv, hv_log[-1], rtol=1e-12, atol=1e-15):
            raise RuntimeError(f"{label} final archive HV does not match hv_log[-1]")
        results.append({"seed": seed, "final_hv": final_hv, "archive_size": len(archive)})
        print(f"[{label}] seed={seed} final_HV={final_hv:.5f} archive_size={len(archive)}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ablation_dataset(data_path: str, out_dir: str, runs: int, population: int,
                          generations: int, seed_start: int = 1,
                          min_rows: int = 2, min_cols: int = 2, init_density: float = 0.3,
                          epsilon: float = 1e-3, allocation_sharpness: float = 3.0,
                          hv_window: int = 5, hv_epsilon: float = 0.02,
                          pc_base: float = 0.7, pc_max: float = 0.95):
    """Run the four feedback variants reported in the manuscript.

    The default values are the documented synthetic-ablation settings.
    ``no_structural_diversity`` disables diversity-driven row/column allocation
    by setting ``allocation_sharpness=0`` (therefore W_R=W_C=0.5).
    ``no_hv_feedback`` fixes crossover at ``pc_base``. ``no_feedback`` disables
    both feedback paths. No duplicate fifth variant is generated.
    """
    ref_point = get_or_compute_reference_point(data_path, min_rows, min_cols)
    base = dict(
        population_size=population, generations=generations, ref_point=ref_point,
        min_rows=min_rows, min_cols=min_cols, init_density=init_density,
        epsilon=epsilon, allocation_sharpness=allocation_sharpness,
        hv_window=hv_window, hv_epsilon=hv_epsilon,
        pc_base=pc_base, pc_max=pc_max,
    )
    variants = {
        "full": RCDAConfig(**base),
        "no_structural_diversity": RCDAConfig(**{**base, "allocation_sharpness": 0.0}),
        "no_hv_feedback": RCDAConfig(**{**base, "pc_max": pc_base}),
        "no_feedback": RCDAConfig(**{**base, "allocation_sharpness": 0.0, "pc_max": pc_base}),
    }
    all_results = {}
    for name, cfg in variants.items():
        variant_out = os.path.join(out_dir, name)
        all_results[name] = run_rcda_dataset(
            data_path, variant_out, runs, population, generations, seed_start, config=cfg
        )
    return all_results
