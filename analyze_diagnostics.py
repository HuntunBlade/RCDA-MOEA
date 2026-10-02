"""Aggregate per-seed RCDA diagnostic trajectories into median/IQR CSVs and plots."""
from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

VARS=['D_R','D_C','W_R','W_C','pc','hv']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-dir',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); files=sorted(glob.glob(str(Path(a.results_dir)/'seed*_diagnostics.json')))
    if not files: raise SystemExit('No seed*_diagnostics.json found; rerun RCDA with the updated runner.')
    runs=[json.load(open(f)) for f in files]; T=min(map(len,runs)); rows=[]
    for t in range(T):
        row={'gen':t+1,'n_runs':len(runs)}
        for v in VARS:
            z=np.array([r[t][v] for r in runs],float); row[v+'_median']=np.median(z); row[v+'_q25']=np.quantile(z,.25); row[v+'_q75']=np.quantile(z,.75)
        st=np.array([r[t]['stagnant'] for r in runs],float); row['stagnant_rate']=st.mean(); rows.append(row)
    df=pd.DataFrame(rows); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); df.to_csv(out/'diagnostic_trajectory_summary.csv',index=False)
    for pair,name,ylabel in [(('D_R','D_C'),'diversity','Jaccard diversity'),(('W_R','W_C'),'weights','mutation allocation weight')]:
        fig,ax=plt.subplots()
        for v in pair: ax.plot(df.gen,df[v+'_median'],label=v); ax.fill_between(df.gen,df[v+'_q25'],df[v+'_q75'],alpha=.15)
        ax.set_xlabel('generation'); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{name}_trajectory.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(df.gen,df.pc_median,label='P_c'); ax2=ax.twinx(); ax2.plot(df.gen,df.stagnant_rate,label='stagnation rate'); ax.set_xlabel('generation'); ax.set_ylabel('P_c'); ax2.set_ylabel('fraction of runs stagnant'); fig.tight_layout(); fig.savefig(out/'pc_stagnation_trajectory.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(df.gen,df.hv_median); ax.fill_between(df.gen,df.hv_q25,df.hv_q75,alpha=.15); ax.set_xlabel('generation'); ax.set_ylabel('HV'); fig.tight_layout(); fig.savefig(out/'hv_trajectory.png',dpi=200); plt.close(fig)
    summary={'runs':len(runs),'generations':T,'mean_stagnation_rate':float(df.stagnant_rate.mean()),'mean_abs_weight_imbalance':float(np.mean(np.abs(df.W_R_median-df.W_C_median)))}
    json.dump(summary,open(out/'diagnostic_summary.json','w'),indent=2); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
