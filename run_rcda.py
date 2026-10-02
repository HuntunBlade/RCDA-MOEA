import argparse,sys
sys.path.insert(0,"src")
from experiments.runner import run_rcda_dataset
from rcda_moea.algorithm import RCDAConfig
p=argparse.ArgumentParser()
p.add_argument("--data",required=True); p.add_argument("--out",default="results/raw/rcda")
p.add_argument("--runs",type=int,default=30); p.add_argument("--population",type=int,default=60)
p.add_argument("--generations",type=int,default=100); p.add_argument("--seed-start",type=int,default=1)
p.add_argument("--min-rows",type=int,default=2,
               help="Minimum genes per bicluster (repair floor). For real datasets, set this "
                    "well above 2 -- e.g. ~0.5%% of total genes -- to avoid degenerate tiny "
                    "biclusters. Default 2 is only sensible for small synthetic test data.")
p.add_argument("--min-cols",type=int,default=2,
               help="Minimum samples/conditions per bicluster (repair floor). Same caveat as --min-rows.")
p.add_argument("--init-density",type=float,default=0.3,
               help="Fraction of genes/samples initially active per individual. Default 0.3 is "
                    "fine for small synthetic matrices; on real gene-expression matrices "
                    "(thousands of genes) use something small, e.g. 0.01-0.05, or biclusters "
                    "will start absurdly large and evaluation will be slower than necessary.")
a=p.parse_args()
config = RCDAConfig(population_size=a.population, generations=a.generations,
                     min_rows=a.min_rows, min_cols=a.min_cols, init_density=a.init_density)
run_rcda_dataset(a.data,a.out,a.runs,a.population,a.generations,a.seed_start,config=config)
