"""Run the validated NSGA-II comparison baseline used by the manuscript."""
import argparse, sys
sys.path.insert(0, "src")
from baselines.nsga2 import NSGA2, NSGA2Config
from experiments.runner import run_baseline_dataset

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--algo", choices=["nsga2"], default="nsga2")
p.add_argument("--out", default="results/raw/nsga2")
p.add_argument("--runs", type=int, default=30)
p.add_argument("--population", type=int, default=60)
p.add_argument("--generations", type=int, default=100)
p.add_argument("--seed-start", type=int, default=1)
p.add_argument("--min-rows", type=int, default=2)
p.add_argument("--min-cols", type=int, default=2)
p.add_argument("--init-density", type=float, default=0.3)
p.add_argument("--pc", type=float, default=0.8,
               help="Reported real-data NSGA-II baseline uses Pc=0.8; controlled mechanism experiments use Pc=0.7 in their dedicated runners.")
a = p.parse_args()
config = NSGA2Config(population_size=a.population, generations=a.generations,
                     min_rows=a.min_rows, min_cols=a.min_cols,
                     init_density=a.init_density, pc=a.pc)
run_baseline_dataset(NSGA2, config, a.data, a.out, a.runs, a.seed_start, label="nsga2")
