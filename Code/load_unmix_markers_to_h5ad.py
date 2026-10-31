import scanpy as sc
import scipy.io
import numpy as np
import pandas as pd
import os
import re

BASE      = "/Users/arun/Desktop/Research/Results"
unmix_dir = f"{BASE}/Unmix_tumor_k4_markers"
out_dir   = f"{BASE}/PostAA_tumor_unmix_k4_markers"

SAMPLES = {
    "G740_ST1": f"{BASE}/DimRed/G740_ST1/G740_ST1_pca_tsne_umap.h5ad",
    "G4_ST2":   f"{BASE}/DimRed/G4_ST2/G4_ST2_pca_tsne_umap.h5ad",
}

def strip_suffix(bc):
    return re.sub(r"-\d+$", "", str(bc))

for sample, h5ad_path in SAMPLES.items():
    print(f"\n{sample}")
    mat = scipy.io.loadmat(f"{unmix_dir}/{sample}_tumor_unmix_results.mat")
    F = mat["F"].astype(np.float32)
    C = mat["C"].astype(np.float32)
    a = float(mat["a"])
    print(f" F: {F.shape}, a={a:.2f}")

    ad = sc.read_h5ad(h5ad_path)
    ad.obs["barcode_key"] = [strip_suffix(b) for b in ad.obs_names]

    bc_df = pd.read_csv(f"{unmix_dir}/{sample}_tumor_barcodes.csv")
    tumor_bcs = [strip_suffix(b) for b in bc_df["barcode"].astype(str)]

    bc_to_row = {bc: i for i, bc in enumerate(tumor_bcs)}

    W_full = np.zeros((ad.n_obs, F.shape[1]), dtype=np.float32)
    argmax = np.full(ad.n_obs, -1, dtype=int)
    maxW = np.zeros(ad.n_obs, dtype=np.float32)

    matched = 0
    for idx, bk in enumerate(ad.obs["barcode_key"]):
        row = bc_to_row.get(bk, None)
        if row is not None:
            W_full[idx] = F[row]
            argmax[idx] = int(np.argmax(F[row]))
            maxW[idx] = float(np.max(F[row]))
            matched += 1
    
    print(f" Matched {matched}/{ad.n_obs} spots")

    ad.obsm["X_W"] = W_full
    ad.uns["X_Z"] = C.T
    ad.uns["unmix_simplex_vol"] = a
    ad.obs["argmax"] = argmax
    ad.obs["maxW"] = maxW
    for k in range(F.shape[1]):
        ad.obs[f"W{k}"] = W_full[:, k]

    sample_out = os.path.join(out_dir, sample)
    os.makedirs(sample_out, exist_ok=True)
    out_path = os.path.join(sample_out, f"{sample}_annotated.h5ad")
    ad.write_h5ad(out_path)
    print(f" Saved -> {out_path}")

print("\nDone.")