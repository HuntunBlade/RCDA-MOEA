"""Run the four synthetic feedback-ablation variants reported in the manuscript."""
import argparse, sys, yaml
sys.path.insert(0, "src")
from experiments.runner import run_ablation_dataset

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/example_realistic.npy")
p.add_argument("--out", default="results/reproduced_ablation")
p.add_argument("--config", default="configs/synthetic_ablation.yaml")
p.add_argument("--runs", type=int, default=None)
p.add_argument("--population", type=int, default=None)
p.add_argument("--generations", type=int, default=None)
p.add_argument("--seed-start", type=int, default=None)
p.add_argument("--min-rows", type=int, default=2)
p.add_argument("--min-cols", type=int, default=2)
p.add_argument("--init-density", type=float, default=0.3)
a = p.parse_args()

with open(a.config, encoding="utf-8") as f:
    y = yaml.safe_load(f)
full = y["variants"]["full"]
run_ablation_dataset(
    a.data, a.out,
    runs=a.runs if a.runs is not None else int(y["runs"]),
    population=a.population if a.population is not None else int(y["population_size"]),
    generations=a.generations if a.generations is not None else int(y["generations"]),
    seed_start=a.seed_start if a.seed_start is not None else int(y["seed_start"]),
    min_rows=a.min_rows, min_cols=a.min_cols, init_density=a.init_density,
    epsilon=float(full["epsilon"]),
    allocation_sharpness=float(full["allocation_sharpness"]),
    hv_epsilon=float(full["hv_epsilon"]),
    pc_base=float(full["pc_base"]), pc_max=float(full["pc_max"]),
)
