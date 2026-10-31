import ast
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import scanpy as sc

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from classify_tumor_normal import classify_spot, strip_suffix

BASE = Path("/Users/arun/Desktop/Research")
RESULTS = BASE / "Results"
CONF_THRESH = 0.1
warnings.filterwarnings("ignore")

def gbm_markers():
    tree = ast.parse((CODE / "export_tumor_counts.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "GBM_MARKERS")
    return ast.literal_eval(node.value)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("section")
    args = ap.parse_args()
    out = RESULTS / args.dataset / args.section
    sr = pd.read_csv(out / "celltype_metadata_singleR.csv")[["barcode", "SingleR_label_main", "SingleR_score_main"]]
    nf = pd.read_csv(out / "neftel_curated.csv")[["barcode", "Neftel_state", "Neftel_confidence"]]
    m = sr.merge(nf, on="barcode", how="outer")
    m["Neftel_confidence"] = m["Neftel_confidence"].fillna(0.0)
    m["SingleR_label_main"] = m["SingleR_label_main"].fillna("unknown")
    res = m.apply(lambda r: classify_spot(r["SingleR_label_main"], r["Neftel_confidence"], CONF_THRESH), axis=1, result_type="expand")
    m["tumor_label"], m["reason"] = res[0], res[1]
    m["barcode_key"] = m["barcode"].map(strip_suffix)
    m.to_csv(out / f"{args.section}_tumor_normal_labels.csv", index=False)
    d = sc.read_h5ad(out / f"{args.section}_dimred.h5ad")
    tumor = set(m.loc[m["tumor_label"] == "Tumor", "barcode"])
    dt = d[d.obs_names.isin(tumor)].copy()
    genes = [g for g in gbm_markers() if g in dt.var_names]
    M = dt[:, genes].layers["norm_counts"]
    M = (M.toarray() if hasattr(M, "toarray") else np.asarray(M)).T.astype(np.float64)
    scipy.io.savemat(out / f"{args.section}_norm_counts_tumor.mat", {"M": M})
    pd.DataFrame({"barcode": dt.obs_names}).to_csv(out / f"{args.section}_tumor_barcodes.csv", index=False)
    pd.Series(genes).to_csv(out / f"{args.section}_unmix_genes.csv", index=False, header=["gene"])
    print(f"{args.section}: {int((m['tumor_label'] == 'Tumor').sum())} tumor / {len(m)} spots "
          f"({dt.n_obs} tumor spots after QC); UNMIX input {M.shape[0]} marker genes x {M.shape[1]} spots")
    print(m["tumor_label"].value_counts().to_string())

if __name__ == "__main__":
    main()