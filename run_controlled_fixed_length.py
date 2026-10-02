"""Supplementary controlled-ratio sensitivity check with fixed encoding length n+m.

This design removes the first-order confound that the main fixed-m experiment
has between n/m and binary chromosome length. It is deliberately supplementary:
with n+m fixed, matrix element count n*m still changes with the ratio, so it is
not a complete causal decomposition of all size effects.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'src')
from data.synthetic_realistic import generate_realistic_synthetic_matrix
from rcda_moea.algorithm import RCDA_MOEA, RCDAConfig
from baselines.nsga2 import NSGA2, NSGA2Config
from rcda_moea.refpoint import get_or_compute_reference_point
from rcda_moea.selection import hypervolume_2d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results/final_20260911/controlled_fixed_length')
    ap.add_argument('--total-length',type=int,default=500)
    ap.add_argument('--ratios',nargs='+',type=float,default=[2,10,50])
    ap.add_argument('--instances',type=int,default=8)
    ap.add_argument('--population',type=int,default=40)
    ap.add_argument('--generations',type=int,default=60)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for target in a.ratios:
        m=max(3,int(round(a.total_length/(1.0+target))))
        n=a.total_length-m
        actual=n/m
        gene_lo=max(4,min(20,n//50)); gene_hi=max(gene_lo+2,min(60,max(gene_lo+2,n//15)))
        cond_lo=max(2,min(6,max(2,m//6)))
        cond_hi=min(m, max(cond_lo+1, min(15,max(cond_lo+1,m//2))))
        if cond_hi<=cond_lo: cond_hi=cond_lo+1
        for inst in range(1,a.instances+1):
            X=generate_realistic_synthetic_matrix(n_genes=n,n_conditions=m,n_modules=6,
                module_gene_range=(gene_lo,gene_hi),module_condition_range=(cond_lo,cond_hi),seed=300000+inst)
            d=out/f'ratio_{target:g}'/f'instance_{inst:02d}'; d.mkdir(parents=True,exist_ok=True)
            path=d/'X.npy'; np.save(path,X)
            min_rows=max(2,int(round(.005*n))); min_cols=max(2,min(m-1,int(round(.03*m))))
            ref=get_or_compute_reference_point(str(path),min_rows,min_cols)
            seed=7000+inst
            rcfg=RCDAConfig(population_size=a.population,generations=a.generations,min_rows=min_rows,min_cols=min_cols,
                init_density=.02,epsilon=.01,allocation_sharpness=1,hv_window=5,hv_epsilon=.02,pc_base=.7,pc_max=.95,ref_point=ref)
            ncfg=NSGA2Config(population_size=a.population,generations=a.generations,min_rows=min_rows,min_cols=min_cols,
                init_density=.02,pc=.7,ref_point=ref)
            rc=RCDA_MOEA(X,rcfg,seed=seed); ar,hr=rc.run(record_diagnostics=True)
            ns=NSGA2(X,ncfg,seed=seed); an,hn=ns.run()
            fr=np.array([x.objectives for x in ar]); fn=np.array([x.objectives for x in an])
            np.save(d/f'rcda_seed{seed}_front.npy',fr); np.save(d/f'nsga2_seed{seed}_front.npy',fn)
            hv_rcda=hypervolume_2d(fr,np.asarray(ref)); hv_nsga2=hypervolume_2d(fn,np.asarray(ref))
            rows.append({'target_ratio':target,'actual_ratio':actual,'n':n,'m':m,'n_plus_m':n+m,'n_times_m':n*m,
                'instance':inst,'seed':seed,'hv_rcda':hv_rcda,'hv_nsga2':hv_nsga2,'delta_hv':hv_rcda-hv_nsga2,
                'mean_W_R':float(np.mean([z['W_R'] for z in rc.diagnostics])),'mean_W_C':float(np.mean([z['W_C'] for z in rc.diagnostics])),
                'mean_pc':float(np.mean([z['pc'] for z in rc.diagnostics])),'stagnation_count':int(sum(z['stagnant'] for z in rc.diagnostics))})
            # keep only compact diagnostics, not matrices, to control package size; generation is deterministic from seed
            (d/f'rcda_seed{seed}_diagnostics.json').write_text(json.dumps(rc.diagnostics))
            path.unlink(missing_ok=True); Path(str(path)+'.refpoint.json').unlink(missing_ok=True)
            print(rows[-1])
    with open(out/'controlled_fixed_length_runs.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out/'experiment_config.json').write_text(json.dumps(vars(a),indent=2))
if __name__=='__main__': main()
