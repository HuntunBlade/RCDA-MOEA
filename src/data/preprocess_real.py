"""Audited preprocessing for the two publicly identified GEO datasets.

This module implements only the preprocessing paths documented by the current
manuscript/package:

Tissp (GDS596 / GPL96)
    GDS SOFT -> retain non-empty IDENTIFIER and complete numeric GSM values ->
    aggregate duplicate probes mapping to the same IDENTIFIER by sample-wise
    mean -> whole-matrix z-score (population standard deviation, ddof=0).
    Expected audited shape: 14,063 x 158.

Cocel-alt sensitivity dataset (GSE35896 / GPL570)
    RMA series matrix -> map probe IDs through a GPL570 annotation table -> keep
    only unambiguous single Gene Symbol mappings with complete numeric values ->
    aggregate duplicate probes per Gene Symbol by sample-wise mean -> whole-
    matrix z-score (ddof=0).
    Expected audited shape: 20,848 x 62.

No zero-removal, TPM normalization, or generic fallback pipeline is applied.
Those belonged to legacy working scripts and are intentionally excluded from
this submission version.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, re
from pathlib import Path
import numpy as np
import pandas as pd


def _read_text(path: str) -> str:
    p = Path(path)
    raw = p.read_bytes()
    if p.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _between_markers(text: str, begin: str, end: str) -> str:
    lines = text.splitlines()
    i = next(i for i, x in enumerate(lines) if begin in x) + 1
    j = next(i for i, x in enumerate(lines) if end in x)
    return "\n".join(lines[i:j])


def global_zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = float(X.mean()); sd = float(X.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot z-score a constant/non-finite matrix")
    return (X - mu) / sd


def _gsm_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"GSM\d+", str(c).strip())]
    if not cols:
        raise ValueError("No GSM<digits> sample columns found")
    return cols


def preprocess_tissp(gds_soft_path: str) -> tuple[np.ndarray, dict]:
    text = _read_text(gds_soft_path)
    table = pd.read_csv(io.StringIO(_between_markers(text, "dataset_table_begin", "dataset_table_end")), sep="\t")
    if "IDENTIFIER" not in table.columns:
        raise ValueError("GDS SOFT table does not contain IDENTIFIER")
    samples = _gsm_columns(table)
    ids = table["IDENTIFIER"].astype(str).str.strip()
    numeric = table[samples].apply(pd.to_numeric, errors="coerce")
    keep = ids.notna() & (ids != "") & (ids.str.lower() != "nan") & numeric.notna().all(axis=1)
    df = numeric.loc[keep].copy(); df.insert(0, "gene", ids.loc[keep].values)
    grouped = df.groupby("gene", sort=True, as_index=True)[samples].mean()
    X = global_zscore(grouped.to_numpy(float))
    meta = {"dataset":"Tissp","source":"GDS596/GPL96","shape":list(X.shape),
            "mapping":"IDENTIFIER","duplicate_rule":"sample-wise mean","zscore_ddof":0}
    return X, meta


def _parse_platform_annotation(path: str) -> pd.DataFrame:
    text = _read_text(path)
    if "platform_table_begin" in text:
        body = _between_markers(text, "platform_table_begin", "platform_table_end")
        return pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
    # GEO *.annot files contain metadata lines beginning with '#'.
    return pd.read_csv(io.StringIO(text), sep="\t", comment="#", dtype=str)


def _pick_col(cols, candidates):
    norm = {str(c).strip().lower(): c for c in cols}
    for x in candidates:
        if x.lower() in norm:
            return norm[x.lower()]
    raise ValueError(f"Could not find any of {candidates}; columns={list(cols)[:30]}")


def _single_gene_symbol(x: str) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s in {"---", "NA", "N/A", "nan"}:
        return None
    # GPL570 uses ' /// ' for multi-mapping. Semicolon/comma lists are also
    # conservatively treated as ambiguous for the audited single-symbol rule.
    if "///" in s or ";" in s or "," in s:
        return None
    if re.search(r"\s//\s", s):
        return None
    return s


def preprocess_cocel_alt(series_matrix_path: str, gpl570_annotation_path: str) -> tuple[np.ndarray, dict]:
    text = _read_text(series_matrix_path)
    body = _between_markers(text, "series_matrix_table_begin", "series_matrix_table_end")
    expr = pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
    probe_col = expr.columns[0]
    samples = _gsm_columns(expr)
    numeric = expr[samples].apply(pd.to_numeric, errors="coerce")
    complete = numeric.notna().all(axis=1)
    expr2 = pd.DataFrame({"probe": expr.loc[complete, probe_col].astype(str).str.strip()})
    for c in samples: expr2[c] = numeric.loc[complete, c].to_numpy()

    ann = _parse_platform_annotation(gpl570_annotation_path)
    id_col = _pick_col(ann.columns, ["ID", "ID_REF", "Probe Set ID"])
    sym_col = _pick_col(ann.columns, ["Gene Symbol", "GENE_SYMBOL", "Gene symbol"])
    amap = ann[[id_col, sym_col]].copy()
    amap["probe"] = amap[id_col].astype(str).str.strip()
    amap["gene"] = amap[sym_col].map(_single_gene_symbol)
    amap = amap.dropna(subset=["gene"])[["probe", "gene"]].drop_duplicates("probe", keep=False)

    merged = expr2.merge(amap, on="probe", how="inner")
    grouped = merged.groupby("gene", sort=True, as_index=True)[samples].mean()
    X = global_zscore(grouped.to_numpy(float))
    meta = {"dataset":"Cocel-alt","source":"GSE35896/GPL570","shape":list(X.shape),
            "mapping":"unambiguous single Gene Symbol","duplicate_rule":"sample-wise mean","zscore_ddof":0}
    return X, meta


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()


def save_with_metadata(X: np.ndarray, meta: dict, out: str):
    p=Path(out); p.parent.mkdir(parents=True, exist_ok=True); np.save(p, X)
    meta=dict(meta); meta["sha256_npy"]=_sha256(p)
    p.with_suffix(p.suffix+".metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(meta,indent=2,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest="dataset",required=True)
    a=sub.add_parser("tissp"); a.add_argument("--gds-soft",required=True); a.add_argument("--out",required=True)
    b=sub.add_parser("cocel-alt"); b.add_argument("--series-matrix",required=True); b.add_argument("--platform-annotation",required=True); b.add_argument("--out",required=True)
    x=ap.parse_args()
    if x.dataset=="tissp": X,m=preprocess_tissp(x.gds_soft)
    else: X,m=preprocess_cocel_alt(x.series_matrix,x.platform_annotation)
    save_with_metadata(X,m,x.out)

if __name__=="__main__": main()
