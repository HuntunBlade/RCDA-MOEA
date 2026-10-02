"""Summarize controlled n/m experiment and produce paper-ready CSV/plots."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
import matplotlib.pyplot as plt


def rbc(d):
    nz=d[d!=0]
    if len(nz)==0:return 0.0
    r=rankdata(np.abs(nz),method='average')
    return float((r[nz>0].sum()-r[nz<0].sum())/r.sum())

def boot_ci(d,B=10000,seed=20260910):
    rng=np.random.default_rng(seed); b=np.empty(B)
    for i in range(B): b[i]=rng.choice(d,len(d),replace=True).mean()
    return np.quantile(b,[.025,.975])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='results/controlled_ratio'); ap.add_argument('--out',default=None)
    a=ap.parse_args(); root=Path(a.root); out=Path(a.out or a.root); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(root/'controlled_ratio_runs.csv')
    rows=[]
    for ratio,g in df.groupby('ratio',sort=True):
        d=g.delta_hv.to_numpy(float); sd=d.std(ddof=1) if len(d)>1 else np.nan
        if len(d)>=2 and np.any(d!=0): stat,p=wilcoxon(d, alternative='two-sided')
        else: stat,p=np.nan,np.nan
        lo,hi=boot_ci(d) if len(d)>=2 else (np.nan,np.nan)
        rows.append({'ratio':ratio,'n':int(g.n.iloc[0]),'m':int(g.m.iloc[0]),'pairs':len(g),
          'mean_delta_hv':d.mean(),'median_delta_hv':np.median(d),'ci95_low':lo,'ci95_high':hi,
          'cohens_dz':d.mean()/sd if sd and np.isfinite(sd) and sd>0 else np.nan,
          'rank_biserial':rbc(d),'wilcoxon_p':p,'mean_abs_W_imbalance':np.mean(np.abs(g.mean_W_R-g.mean_W_C)),
          'mean_stagnation_count':g.stagnation_count.mean(),'mean_pc':g.mean_pc.mean()})
    s=pd.DataFrame(rows)
    # Holm correction across the nine prespecified ratio-wise Wilcoxon tests.
    ps=s['wilcoxon_p'].to_numpy(float); order=np.argsort(ps); adj=np.empty(len(ps)); running=0.0
    for j,i in enumerate(order):
        running=max(running,min(1.0,ps[i]*(len(ps)-j))); adj[i]=running
    s['holm_p']=adj
    s.to_csv(out/'controlled_ratio_summary.csv',index=False)
    # plot 1: delta HV vs ratio
    fig,ax=plt.subplots(); ax.errorbar(s.ratio,s.mean_delta_hv,yerr=[s.mean_delta_hv-s.ci95_low,s.ci95_high-s.mean_delta_hv],fmt='o-'); ax.axhline(0,linewidth=1); ax.set_xscale('log'); ax.set_xlabel('row/column ratio n/m'); ax.set_ylabel('paired mean ΔHV (RCDA - NSGA-II)'); fig.tight_layout(); fig.savefig(out/'controlled_ratio_delta_hv.png',dpi=200); plt.close(fig)
    # plot 2: allocation imbalance vs ratio
    fig,ax=plt.subplots(); ax.plot(s.ratio,s.mean_abs_W_imbalance,'o-'); ax.set_xscale('log'); ax.set_xlabel('row/column ratio n/m'); ax.set_ylabel('mean |W_R - W_C|'); fig.tight_layout(); fig.savefig(out/'controlled_ratio_weight_imbalance.png',dpi=200); plt.close(fig)
    print(s.to_string(index=False))
if __name__=='__main__': main()
