from pathlib import Path
import json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
root=Path('results/final_20260910/controlled_ratio'); out=root/'analysis'; out.mkdir(exist_ok=True)
rows=[]
for rd in sorted(root.glob('ratio_*'),key=lambda p:float(p.name.split('_')[1])):
 ratio=float(rd.name.split('_')[1]); runs=[]
 for f in rd.glob('instance_*/rcda_seed*_diagnostics.json'): runs.append(json.load(open(f)))
 T=min(map(len,runs))
 for t in range(T):
  z={'ratio':ratio,'gen':t+1,'n_runs':len(runs)}
  for v in ['D_R','D_C','W_R','W_C','pc','hv']:
   a=np.array([r[t][v] for r in runs],float); z[v+'_median']=np.median(a); z[v+'_q25']=np.quantile(a,.25); z[v+'_q75']=np.quantile(a,.75)
  z['stagnant_rate']=np.mean([r[t]['stagnant'] for r in runs]); rows.append(z)
df=pd.DataFrame(rows); df.to_csv(out/'controlled_ratio_trajectory_summary.csv',index=False)
sel=[2.,50.,1000.]
for varpair,name,ylabel in [(('D_R','D_C'),'selected_ratio_diversity_trajectories','Jaccard diversity'),(('W_R','W_C'),'selected_ratio_weight_trajectories','mutation allocation weight')]:
 fig,ax=plt.subplots(figsize=(7,4.2))
 for ratio in sel:
  g=df[df.ratio==ratio]
  for v in varpair: ax.plot(g.gen,g[v+'_median'],label=f'{v}, n/m={ratio:g}')
 ax.set_xlabel('generation'); ax.set_ylabel(ylabel); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=220); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.2))
for ratio in sel:
 g=df[df.ratio==ratio]; ax.plot(g.gen,g.pc_median,label=f'P_c, n/m={ratio:g}')
ax.set_xlabel('generation'); ax.set_ylabel('median crossover probability'); ax.legend(); fig.tight_layout(); fig.savefig(out/'selected_ratio_pc_trajectories.png',dpi=220); plt.close(fig)
print('saved',len(df),'trajectory rows')
