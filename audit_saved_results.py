"""Audit and normalize saved result artifacts for submission.

Key rule: final HV is recomputed from each saved final Pareto archive. Historical
HV trajectory files are retained as controller/per-generation records but are not
used as the definition of final performance.
"""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata
import sys
sys.path.insert(0, "src")
from rcda_moea.selection import hypervolume_2d
from experiments.metrics import friedman_holm

ROOT=Path("results")
AUD=ROOT/"submission_audit"; AUD.mkdir(parents=True,exist_ok=True)
BOOT_SEED=20260910; B=20000


def load_front_hv(d: Path, ref):
    out={}; sizes={}
    for p in d.glob("seed*_archive_objectives.npy"):
        seed=int(p.name[4:].split("_")[0]); f=np.load(p)
        out[seed]=hypervolume_2d(f,np.asarray(ref,float)); sizes[seed]=len(f)
    return out,sizes


def write_summary(d: Path, ref):
    vals,sizes=load_front_hv(d,ref)
    rows=[{"seed":s,"final_hv":vals[s],"archive_size":sizes[s],"metric_source":"final_archive_objectives"} for s in sorted(vals)]
    (d/"summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return vals


def rbc(d):
    z=d[d!=0]
    if not len(z): return 0.0
    r=rankdata(np.abs(z),method="average")
    return float((r[z>0].sum()-r[z<0].sum())/r.sum())


def paired_stats(x,y):
    seeds=sorted(set(x)&set(y)); a=np.array([x[s] for s in seeds]); b=np.array([y[s] for s in seeds]); d=a-b
    stat,p=wilcoxon(d,alternative="two-sided")
    rng=np.random.default_rng(BOOT_SEED); means=np.empty(B)
    for i in range(B): means[i]=rng.choice(d,len(d),replace=True).mean()
    lo,hi=np.quantile(means,[.025,.975]); sd=d.std(ddof=1)
    return {"rcda_mean":float(a.mean()),"rcda_std_sample":float(a.std(ddof=1)),
            "nsga2_mean":float(b.mean()),"nsga2_std_sample":float(b.std(ddof=1)),
            "mean_difference_rcda_minus_nsga2":float(d.mean()),"median_difference":float(np.median(d)),
            "cohens_dz":float(d.mean()/sd) if sd>0 else 0.0,"rank_biserial":rbc(d),
            "wilcoxon_statistic":float(stat),"raw_p":float(p),
            "bootstrap_mean_diff_ci95":[float(lo),float(hi)],
            "rcda_wins":int((d>0).sum()),"nsga2_wins":int((d<0).sum()),"ties":int((d==0).sum()),
            "metric_source":"final_archive_objectives"}


def holm(ps):
    order=sorted(ps,key=ps.get); m=len(order); out={}; running=0.0
    for i,k in enumerate(order):
        running=max(running,min(1.0,ps[k]*(m-i))); out[k]=running
    return out


def infer_ref_x_from_final_front(front, target_hv, ref_y=1.0):
    maxx=float(np.max(front[:,0])); r0=maxx+max(1e-12,abs(maxx)*1e-12)
    h0=hypervolume_2d(front,np.array([r0,ref_y])); h1=hypervolume_2d(front,np.array([r0+1.0,ref_y])); slope=h1-h0
    if slope<=0: raise ValueError("Cannot infer reference point: zero HV slope")
    rx=r0+(target_hv-h0)/slope
    check=hypervolume_2d(front,np.array([rx,ref_y]))
    if not np.isclose(check,target_hv,rtol=1e-9,atol=1e-12): raise ValueError((check,target_hv))
    return float(rx)


def update_controlled_ratio():
    root=ROOT/"final_20260910/controlled_ratio"; csvp=root/"controlled_ratio_runs.csv"
    if not csvp.exists(): return
    df=pd.read_csv(csvp); changed=[]
    for idx,row in df.iterrows():
        ratio=row["ratio"]; inst=int(row["instance"]); seed=int(row["seed"])
        d=root/f"ratio_{ratio:g}"/f"instance_{inst:02d}"
        fn=np.load(d/f"nsga2_seed{seed}_front.npy"); fr=np.load(d/f"rcda_seed{seed}_front.npy")
        hn=json.loads((d/f"nsga2_seed{seed}_hv.json").read_text())[-1]
        rx=infer_ref_x_from_final_front(fn,float(hn),1.0); ref=np.array([rx,1.0])
        hns=hypervolume_2d(fn,ref); hrc=hypervolume_2d(fr,ref)
        df.at[idx,"hv_nsga2"]=hns; df.at[idx,"hv_rcda"]=hrc; df.at[idx,"delta_hv"]=hrc-hns
        changed.append(abs((hrc-hns)-float(row["delta_hv"])))
    df.to_csv(csvp,index=False)
    return {"rows":len(df),"max_abs_delta_hv_change":float(max(changed))}


def main():
    refs=json.load(open(ROOT/"hv_reference_boundary_audit.json"))["datasets"]
    mapping={
      "Tissp":(ROOT/"final_20260910/tissp/rcda",ROOT/"final_20260910/tissp/nsga2",refs["Tissp"]["ref_point"]),
      "Ustilago":(ROOT/"raw_ustilago/rcda",ROOT/"raw_ustilago/nsga2",refs["Ustilago"]["ref_point"]),
      "Mouse":(ROOT/"raw_mouse/rcda",ROOT/"raw_mouse/nsga2",refs["Mouse"]["ref_point"]),
      "Cocel-GSE35896-alt":(ROOT/"final_20260910/cocel_alt/rcda",ROOT/"final_20260910/cocel_alt/nsga2",refs["Cocel-alt"]["ref_point"]),
    }
    pair_path=ROOT/"final_20260910/pairwise_effect_sizes.json"
    legacy=AUD/"legacy/pairwise_effect_sizes_legacy_hvlog.json"
    legacy.parent.mkdir(parents=True,exist_ok=True)
    if pair_path.exists() and not legacy.exists(): shutil.copy2(pair_path,legacy)
    pair=json.load(open(pair_path)) if pair_path.exists() else {}
    audit={"final_hv_definition":"hypervolume(final_archive_objectives)","datasets":{}}
    for name,(rd,nd,ref) in mapping.items():
        x=write_summary(rd,ref); y=write_summary(nd,ref); st=paired_stats(x,y)
        audit["datasets"][name]={"reference_point":ref,"n_pairs":len(set(x)&set(y)),**st}
        if name in pair: pair[name]["hv"]=st
    if pair: pair_path.write_text(json.dumps(pair,indent=2),encoding="utf-8")
    ps={k:v["raw_p"] for k,v in [(n,audit["datasets"][n]) for n in audit["datasets"]]}
    hv_holm=holm(ps); audit["hv_holm_four_dataset_family"]=hv_holm

    # Preserve IGD families from the existing pairwise file; only HV changed.
    if pair:
        ig={k:v["igd"]["raw_p"] for k,v in pair.items()}
        igdoc={k:ig[k] for k in ("Tissp","Cocel-GSE35896-alt")}
        holm_obj={"hv_holm_four_dataset_family":hv_holm,
                  "igd_holm_documented_family":holm(igdoc),
                  "igd_holm_legacy_four_dataset_exploratory":holm(ig)}
        (ROOT/"final_20260910/holm_adjusted.json").write_text(json.dumps(holm_obj,indent=2),encoding="utf-8")

    # Synthetic ablation: final-archive HV; IGD is unaffected by the HV logging convention.
    ref=json.load(open("data/processed/example_realistic.npy.refpoint.json"))["ref_point"]
    names=["full","no_structural_diversity","no_hv_feedback","no_feedback"]
    ab={}; arrays={}
    for n in names:
        v=write_summary(ROOT/"raw/ablation"/n,ref); arrays[n]=np.array([v[s] for s in sorted(v)])
        ab[n]={"mean":float(arrays[n].mean()),"std_sample":float(arrays[n].std(ddof=1))}
    abtest=friedman_holm(arrays)
    audit["synthetic_ablation"]={"reference_point":ref,"hv":ab,"tests":abtest}

    cr=update_controlled_ratio(); audit["controlled_ratio_archive_hv_refresh"]=cr
    (AUD/"archive_final_hv_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    pd.DataFrame([{"dataset":k,"reference_msr":mapping[k][2][0],"reference_f2":mapping[k][2][1],**v} for k,v in audit["datasets"].items()]).to_csv(AUD/"real_dataset_archive_final_hv.csv",index=False)
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
