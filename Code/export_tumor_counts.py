import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io

BASE = "/Users/arun/Desktop/Research"

SAMPLES = {
    "G740_ST1": os.path.join(BASE, "Results/DimRed/G740_ST1/G740_ST1_pca_tsne_umap.h5ad"),
    "G4_ST2":   os.path.join(BASE, "Results/DimRed/G4_ST2/G4_ST2_pca_tsne_umap.h5ad"),
}

TUMOR_LABELS_DIR = os.path.join(BASE, "Results/TumorNormalClassification")
OUT_DIR          = os.path.join(BASE, "Results/Unmix_tumor_k4_markers")
os.makedirs(OUT_DIR, exist_ok=True)

GBM_MARKERS = [
    # AC
    "GFAP", "SOX9", "TNC", "HOPX", "PTPRZ1", "ANXA2",
    # MES
    "CHI3L1", "CD44", "VIM", "SERPINE1", "ANXA1", "C1S", "C1R",
    "LGALS1", "PLAUR", "MET", "ITGA5", "FOSL2", "FN1",
    "TGFBI", "COL1A2", "COL3A1", "STAT3", "IL6",
    # OPC
    "OLIG1", "OLIG2", "PDGFRA", "SOX10", "CSPG4", "PLP1",
    "NKX2-2", "DLL3", "SOX4", "MOG", "GPR17", "BCAN", "SNCG", "ALCAM",
    # NPC
    "DCX", "STMN2", "HES6", "CD24", "NEUROD1", "ASCL1",
    "SOX2", "NES", "NEUROD6", "SNAP25",
]

def strip_suffix(bc):
    return re.sub(r"-\d+$", "", str(bc))

for sample, h5ad_path in SAMPLES.items():
    print(f"\n{sample}")

    ad = sc.read_h5ad(h5ad_path)
    ad.obs["barcode_key"] = [strip_suffix(b) for b in ad.obs_names]

    tumor_labels   = pd.read_csv(os.path.join(TUMOR_LABELS_DIR, f"{sample}_tumor_normal_labels.csv"))
    tumor_barcodes = set(tumor_labels.loc[tumor_labels["tumor_label"] == "Tumor", "barcode"].astype(str))

    tumor_mask = np.array([bk in tumor_barcodes for bk in ad.obs["barcode_key"]])
    ad_t = ad[tumor_mask].copy()
    print(f"  Tumor spots: {ad_t.n_obs} / {ad.n_obs}")

    present_markers = [g for g in GBM_MARKERS if g in ad_t.var_names]
    missing_markers = [g for g in GBM_MARKERS if g not in ad_t.var_names]
    print(f"  Marker genes present: {len(present_markers)} / {len(GBM_MARKERS)}")
    if missing_markers:
        print(f"  Missing: {missing_markers}")
    
    ad_m = ad_t[:, present_markers].copy()

    M = ad_m.layers["norm_counts"]
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    M = M.T.astype(np.float64)
    print(f"  M shape: {M.shape}  (genes x spots)")

    barcodes = ad_t.obs_names.to_numpy()
    pd.DataFrame({"barcode": barcodes}).to_csv(
        os.path.join(OUT_DIR, f"{sample}_tumor_barcodes.csv"), index=False)

    scipy.io.savemat(os.path.join(OUT_DIR, f"{sample}_norm_counts_tumor.mat"), {"M": M})
    print(f"  Saved M and barcodes -> {OUT_DIR}/")

print("\nDone. Run Unmix_tumor_spatial.m in MATLAB next.")