from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

def holm(ps):
    order=np.argsort(ps); out=np.empty(len(ps)); run=0
    for j,i in enumerate(order):
        run=max(run,min(1.0,ps[i]*(len(ps)-j))); out[i]=run
    return out

def rbc(d):
    z=d[d!=0]
    if not len(z): return 0.
    ranks=rankdata(np.abs(z)); return float((ranks[z>0].sum()-ranks[z<0].sum())/ranks.sum())

def ci(d,B=20000):
    rng=np.random.default_rng(20260910); b=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(B)])
    return np.quantile(b,[.025,.975])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='results/final_20260911/controlled_fixed_length'); a=ap.parse_args()
    root=Path(a.root); df=pd.read_csv(root/'controlled_fixed_length_runs.csv'); rows=[]
    for r,g in df.groupby('target_ratio',sort=True):
        d=g.delta_hv.to_numpy(float); lo,hi=ci(d); stat,p=wilcoxon(d,alternative='two-sided')
        sd=d.std(ddof=1)
        rows.append({'target_ratio':r,'actual_ratio':g.actual_ratio.iloc[0],'n':int(g.n.iloc[0]),'m':int(g.m.iloc[0]),'pairs':len(g),
          'mean_delta_hv':d.mean(),'ci95_low':lo,'ci95_high':hi,'cohens_dz':d.mean()/sd if sd>0 else np.nan,
          'rank_biserial':rbc(d),'raw_p':p,'mean_stagnation_count':g.stagnation_count.mean(),'mean_pc':g.mean_pc.mean()})
    s=pd.DataFrame(rows); s['holm_p']=holm(s.raw_p.to_numpy()); s.to_csv(root/'controlled_fixed_length_summary.csv',index=False)
    (root/'analysis_summary.json').write_text(s.to_json(orient='records',indent=2))
    print(s.to_string(index=False))
if __name__=='__main__': main()
