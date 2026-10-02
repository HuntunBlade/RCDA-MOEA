"""Controlled row/column-ratio experiment for the RCDA-MOEA paper.

Keeps m fixed (default 100) and varies n/m over a prespecified grid. For each
ratio and synthetic-instance seed, RCDA-MOEA and NSGA-II are run with the SAME
algorithm seed and the SAME generated matrix. The primary outcome is paired
DeltaHV = HV_RCDA - HV_NSGAII. RCDA diagnostic trajectories D_R, D_C, W_R,
W_C, P_c, HV and stagnation are saved per run.

The default ratio grid matches the paper's proposed confirmatory experiment:
2, 5, 10, 20, 50, 100, 200, 500, 1000. High ratios can be computationally
expensive; use --ratios or --smoke for a development run.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from data.synthetic_realistic import generate_realistic_synthetic_matrix
from rcda_moea.algorithm import RCDA_MOEA, RCDAConfig
from baselines.nsga2 import NSGA2, NSGA2Config
from rcda_moea.refpoint import get_or_compute_reference_point
from rcda_moea.selection import hypervolume_2d


def _save_front(path, archive):
    arr=np.array([x.objectives for x in archive]) if archive else np.zeros((0,2))
    np.save(path, arr)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default='results/controlled_ratio')
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--ratios', nargs='+', type=float, default=[2,5,10,20,50,100,200,500,1000])
    ap.add_argument('--instances', type=int, default=20, help='number of independent synthetic matrices per ratio')
    ap.add_argument('--instance-start', type=int, default=1, help='first synthetic instance index')
    ap.add_argument('--runs-per-instance', type=int, default=1, help='paired algorithm seeds per matrix')
    ap.add_argument('--population', type=int, default=60)
    ap.add_argument('--generations', type=int, default=100)
    ap.add_argument('--epsilon', type=float, default=0.01)
    ap.add_argument('--sharpness', type=float, default=1.0)
    ap.add_argument('--hv-window', type=int, default=5)
    ap.add_argument('--hv-epsilon', type=float, default=0.02)
    ap.add_argument('--pc-base', type=float, default=0.7)
    ap.add_argument('--pc-max', type=float, default=0.95)
    ap.add_argument('--init-density', type=float, default=0.02)
    ap.add_argument('--smoke', action='store_true', help='ratios 2,20,100; 2 instances; pop 20; 20 generations')
    ap.add_argument('--discard-matrices', action='store_true', help='remove generated X.npy and refpoint cache after each instance; matrices are reproducible from seeds')
    a=ap.parse_args()
    if a.smoke:
        a.ratios=[2,20,100]; a.instances=2; a.runs_per_instance=1; a.population=20; a.generations=20
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for ratio in a.ratios:
        n=max(20,int(round(a.m*ratio)))
        # Keep planted structures proportional but bounded so the generator is valid.
        gene_lo=max(4,min(20,n//50)); gene_hi=max(gene_lo+2,min(60,n//15 if n>=30 else n-1))
        cond_lo=max(4,min(12,a.m//20)); cond_hi=max(cond_lo+2,min(25,a.m//5))
        for inst in range(a.instance_start,a.instance_start+a.instances):
            X=generate_realistic_synthetic_matrix(
                n_genes=n,n_conditions=a.m,n_modules=6,
                module_gene_range=(gene_lo,gene_hi),
                module_condition_range=(cond_lo,cond_hi),seed=100000+inst)
            ddir=out/f'ratio_{ratio:g}'/f'instance_{inst:02d}'
            ddir.mkdir(parents=True,exist_ok=True)
            data_path=ddir/'X.npy'; np.save(data_path,X)
            min_rows=max(2,int(round(0.005*n))); min_cols=max(2,int(round(0.03*a.m)))
            ref=get_or_compute_reference_point(str(data_path),min_rows,min_cols)
            for rep in range(a.runs_per_instance):
                seed=inst*1000+rep+1
                rcfg=RCDAConfig(population_size=a.population,generations=a.generations,
                    min_rows=min_rows,min_cols=min_cols,init_density=a.init_density,
                    epsilon=a.epsilon,allocation_sharpness=a.sharpness,
                    hv_window=a.hv_window,hv_epsilon=a.hv_epsilon,
                    pc_base=a.pc_base,pc_max=a.pc_max,ref_point=ref)
                ncfg=NSGA2Config(population_size=a.population,generations=a.generations,
                    min_rows=min_rows,min_cols=min_cols,init_density=a.init_density,
                    pc=a.pc_base,ref_point=ref)
                rcda=RCDA_MOEA(X,rcfg,seed=seed); ar,hvr=rcda.run(record_diagnostics=True)
                nsga=NSGA2(X,ncfg,seed=seed); an,hvn=nsga.run()
                _save_front(ddir/f'rcda_seed{seed}_front.npy',ar)
                _save_front(ddir/f'nsga2_seed{seed}_front.npy',an)
                (ddir/f'rcda_seed{seed}_diagnostics.json').write_text(json.dumps(rcda.diagnostics,indent=2))
                (ddir/f'rcda_seed{seed}_hv.json').write_text(json.dumps(hvr))
                (ddir/f'nsga2_seed{seed}_hv.json').write_text(json.dumps(hvn))
                hv_rcda = hypervolume_2d(np.array([x.objectives for x in ar]), np.asarray(ref))
                hv_nsga2 = hypervolume_2d(np.array([x.objectives for x in an]), np.asarray(ref))
                r={'ratio':ratio,'n':n,'m':a.m,'instance':inst,'seed':seed,
                   'hv_rcda':hv_rcda,'hv_nsga2':hv_nsga2,'delta_hv':hv_rcda-hv_nsga2,
                   'mean_D_R':float(np.mean([z['D_R'] for z in rcda.diagnostics])),
                   'mean_D_C':float(np.mean([z['D_C'] for z in rcda.diagnostics])),
                   'mean_W_R':float(np.mean([z['W_R'] for z in rcda.diagnostics])),
                   'mean_W_C':float(np.mean([z['W_C'] for z in rcda.diagnostics])),
                   'mean_pc':float(np.mean([z['pc'] for z in rcda.diagnostics])),
                   'stagnation_count':int(sum(z['stagnant'] for z in rcda.diagnostics))}
                rows.append(r); print(r)
            if a.discard_matrices:
                for q in (data_path, Path(str(data_path)+'.refpoint.json')):
                    try: q.unlink()
                    except FileNotFoundError: pass
    with open(out/'controlled_ratio_runs.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    meta=vars(a); (out/'experiment_config.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__': main()
