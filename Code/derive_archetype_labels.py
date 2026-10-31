import os
import re
import numpy as np
import pandas as pd
import scipy.io
import scanpy as sc

BASE = "/Users/arun/Desktop/Research"

SAMPLES = {
    "G740_ST1": {
        "h5ad_dimred": os.path.join(BASE, "Results/DimRed/G740_ST1/G740_ST1_pca_tsne_umap.h5ad"),
        "mat":         os.path.join(BASE, "Results/Unmix_tumor_k4_markers/G740_ST1_tumor_unmix_results.mat"),
        "out_dir":     os.path.join(BASE, "Results/PostAA_tumor_unmix_k4_markers/G740_ST1"),
    },
    "G4_ST2": {
        "h5ad_dimred": os.path.join(BASE, "Results/DimRed/G4_ST2/G4_ST2_pca_tsne_umap.h5ad"),
        "mat":         os.path.join(BASE, "Results/Unmix_tumor_k4_markers/G4_ST2_tumor_unmix_results.mat"),
        "out_dir":     os.path.join(BASE, "Results/PostAA_tumor_unmix_k4_markers/G4_ST2"),
    },
}

GBM_MARKERS = {
    "AC":  ["GFAP", "SOX9", "TNC", "HOPX", "PTPRZ1", "ANXA2"],
    "MES": ["CHI3L1", "CD44", "VIM", "SERPINE1", "ANXA1", "C1S", "C1R",
            "LGALS1", "PLAUR", "MET", "ITGA5", "FOSL2", "FN1",
            "TGFBI", "COL1A2", "COL3A1", "STAT3", "IL6"],
    "OPC": ["OLIG1", "OLIG2", "PDGFRA", "SOX10", "CSPG4", "PLP1",
            "NKX2-2", "DLL3", "SOX4", "MOG", "GPR17", "BCAN", "SNCG", "ALCAM"],
    "NPC": ["DCX", "STMN2", "HES6", "CD24", "NEUROD1", "ASCL1",
            "SOX2", "NES", "NEUROD6", "SNAP25"],
}

ALL_MARKERS = [g for genes in GBM_MARKERS.values() for g in genes]

for sample, cfg in SAMPLES.items():
    print(f"\n{'='*60}\n  {sample}\n{'='*60}")

    ad = sc.read_h5ad(cfg["h5ad_dimred"])
    gene_order = [g for g in ALL_MARKERS if g in ad.var_names]
    missing = [g for g in ALL_MARKERS if g not in ad.var_names]
    print(f"  Genes in DimRed h5ad: {len(gene_order)}/{len(ALL_MARKERS)}")
    if missing:
        print(f"  Missing from DimRed h5ad: {missing}")

    mat = scipy.io.loadmat(cfg["mat"])
    C = mat["C"].astype(float)
    print(f"  C matrix shape: {C.shape}  (expected {len(gene_order)} x 4)")

    if C.shape[0] != len(gene_order):
        print(f"  WARNING: C rows ({C.shape[0]}) != gene_order ({len(gene_order)}). Check gene list.")
        continue

    C_df = pd.DataFrame(C, index=gene_order, columns=[f"Arch_{k}" for k in range(C.shape[1])])
    C_df.to_csv(os.path.join(cfg["out_dir"], f"{sample}_C_matrix.csv"))
    print(f"  C matrix saved.")

    z = C_df.subtract(C_df.mean(axis=1), axis=0).divide(C_df.std(axis=1) + 1e-9, axis=0)

    print(f"\n  State z-scores per archetype (C matrix):")
    summary_rows = []
    for k in range(C.shape[1]):
        col = f"Arch_{k}"
        state_scores = {}
        for state, genes in GBM_MARKERS.items():
            g_present = [g for g in genes if g in z.index]
            state_scores[state] = z.loc[g_present, col].mean() if g_present else np.nan
        dominant = max(state_scores, key=lambda s: state_scores[s] if not np.isnan(state_scores[s]) else -np.inf)
        scores_str = "  ".join([f"{s}={v:.3f}" for s, v in state_scores.items()])
        print(f"    Arch {k}: {scores_str}  → {dominant}")
        summary_rows.append({"archetype": k, "dominant_state": dominant,
                              **{f"zscore_{s}": v for s, v in state_scores.items()}})

    summary_df = pd.DataFrame(summary_rows).sort_values("archetype")
    summary_path = os.path.join(cfg["out_dir"], f"{sample}_archetype_state_labels.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  State label summary saved -> {summary_path}")

print("\nDone.")
