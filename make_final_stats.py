import json, numpy as np, pandas as pd
from pathlib import Path
x=json.load(open('results/final_20260910/pairwise_effect_sizes.json'))
# Holm correction helper
def holm(ps):
    names=list(ps); order=sorted(names,key=lambda k:ps[k]); m=len(order); out={}; run=0
    for i,k in enumerate(order):
        v=min(1,ps[k]*(m-i)); run=max(run,v); out[k]=run
    return out
hv={k:v['hv']['raw_p'] for k,v in x.items()}; ig={k:v['igd']['raw_p'] for k,v in x.items()}
hvh=holm(hv); ig_documented={k:ig[k] for k in ('Tissp','Cocel-GSE35896-alt')}; igh_documented=holm(ig_documented); igh_legacy_four=holm(ig)
rows=[]
for k,v in x.items():
    rows.append({'dataset':k,'n':v['shape'][0],'m':v['shape'][1],
      'hv_rcda_mean':v['hv']['rcda_mean'],'hv_rcda_sd':v['hv']['rcda_std_sample'],'hv_nsga2_mean':v['hv']['nsga2_mean'],'hv_nsga2_sd':v['hv']['nsga2_std_sample'],
      'hv_diff':v['hv']['mean_difference_rcda_minus_nsga2'],'hv_ci_low':v['hv']['bootstrap_mean_diff_ci95'][0],'hv_ci_high':v['hv']['bootstrap_mean_diff_ci95'][1],'hv_dz':v['hv']['cohens_dz'],'hv_rbc':v['hv']['rank_biserial'],'hv_p':v['hv']['raw_p'],'hv_holm':hvh[k],
      'igd_rcda_mean':v['igd']['rcda_mean'],'igd_rcda_sd':v['igd']['rcda_std_sample'],'igd_nsga2_mean':v['igd']['nsga2_mean'],'igd_nsga2_sd':v['igd']['nsga2_std_sample'],
      'igd_diff':v['igd']['mean_difference_rcda_minus_nsga2'],'igd_ci_low':v['igd']['bootstrap_mean_diff_ci95'][0],'igd_ci_high':v['igd']['bootstrap_mean_diff_ci95'][1],'igd_dz':v['igd']['cohens_dz'],'igd_rbc':v['igd']['rank_biserial'],'igd_p':v['igd']['raw_p'],'igd_holm_documented':igh_documented.get(k, np.nan),'igd_holm_legacy_four':igh_legacy_four[k]})
pd.DataFrame(rows).to_csv('results/final_20260910/real_dataset_statistics.csv',index=False)
json.dump({'hv_holm_four_dataset_family':hvh,'igd_holm_documented_family':igh_documented,'igd_holm_legacy_four_dataset_exploratory':igh_legacy_four},open('results/final_20260910/holm_adjusted.json','w'),indent=2)
print(json.dumps({'hv_holm_four_dataset_family':hvh,'igd_holm_documented_family':igh_documented,'igd_holm_legacy_four_dataset_exploratory':igh_legacy_four},indent=2))
