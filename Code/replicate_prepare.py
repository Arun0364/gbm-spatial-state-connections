import os
import sys
import glob
import gzip
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import scanpy as sc

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from Spatial_dimred import runDimredPipeline

BASE = Path("/Users/arun/Desktop/Research")
DATA = BASE / "Data"
RESULTS = BASE / "Results"
warnings.filterwarnings("ignore")

def find(folder, pattern):
    hits = sorted(glob.glob(str(folder / pattern)))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found in {folder}")
    return hits[0]

def load_counts(dataset, section):
    folder = DATA / dataset
    h5 = sorted(glob.glob(str(folder / f"*{section}_filtered_feature_bc_matrix.h5")))
    if h5:
        a = sc.read_10x_h5(h5[0])
        a.var["gene_id"] = a.var["gene_ids"].astype(str).values
    else:
        X = scipy.io.mmread(find(folder, f"*{section}_matrix.mtx.gz")).tocsr().T.tocsr()
        feats = pd.read_csv(find(folder, f"*{section}_features.tsv.gz"), sep="\t", header=None)
        bcs = pd.read_csv(find(folder, f"*{section}_barcodes.tsv.gz"), sep="\t", header=None)[0].astype(str).values
        a = sc.AnnData(X)
        a.obs_names = bcs
        a.var_names = feats[1].astype(str).values
        a.var["gene_id"] = feats[0].astype(str).values
    a.var_names_make_unique()
    pos_file = find(folder, f"*{section}_tissue_positions*.csv.gz")
    pos = pd.read_csv(pos_file, header=None)
    if not str(pos.iloc[0, 1]).strip().lstrip("-").isdigit():
        pos = pd.read_csv(pos_file)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = pos.set_index("barcode").loc[a.obs_names]
    for c in ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        a.obs[c] = pos[c].astype(int).values
    a = a[a.obs["in_tissue"] == 1].copy()
    a.obsm["spatial"] = np.vstack([a.obs["pxl_col_in_fullres"].to_numpy(), a.obs["pxl_row_in_fullres"].to_numpy()]).T
    a.obs["sample_ID"] = section
    a.obs["sample_name"] = section
    a.uns["spatial"] = {section: {"scalefactors": json.load(gzip.open(find(folder, f"*{section}_scalefactors_json.json.gz"))),
                                  "images": {}, "metadata": {"chemistry_description": "Visium", "source": "GEO"}}}
    a.var["mt"] = a.var_names.str.startswith(("MT-", "mt-"))
    a.X = sp.csr_matrix(a.X)
    a.layers["counts"] = a.X.copy()
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("section")
    args = ap.parse_args()
    out = RESULTS / args.dataset / args.section
    out.mkdir(parents=True, exist_ok=True)
    a = load_counts(args.dataset, args.section)
    a.write_h5ad(out / f"{args.section}_preprocessed_spatial_data.h5ad")
    mtx_dir = out / "counts_mtx"
    mtx_dir.mkdir(exist_ok=True)
    scipy.io.mmwrite(str(mtx_dir / "matrix.mtx"), a.layers["counts"].T.tocsc())
    pd.DataFrame({"id": a.var["gene_id"].values, "name": a.var_names, "type": "Gene Expression"}).to_csv(mtx_dir / "features.tsv", sep="\t", header=False, index=False)
    pd.Series(a.obs_names).to_csv(mtx_dir / "barcodes.tsv", sep="\t", header=False, index=False)
    sc.settings.figdir = str(out / "figures")
    d = runDimredPipeline(a, args.section, str(out / "dimred"))
    d.write_h5ad(out / f"{args.section}_dimred.h5ad")
    print(f"{args.section}: {a.n_obs} in-tissue spots, {a.n_vars} genes -> after QC {d.n_obs} spots, {d.n_vars} genes")

if __name__ == "__main__":
    main()