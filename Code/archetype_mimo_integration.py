import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr
import scipy.io

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K_RUN = 4

UNMIX_DIR = os.path.join(BASE, f"Results/Unmix_tumor_k{K_RUN}_markers")
MIMO_DIR = os.path.join(BASE, "Results/PI_MIMO_VISUALS_tumor_unmix_k4_markers")
OUT_DIR = os.path.join(BASE, f"Results/archetype_mimo_integration_k{K_RUN}_markers")

SAMPLES = {
    "G740_ST1": {
        "mat": os.path.join(UNMIX_DIR, "G740_ST1_tumor_unmix_results.mat"),
        "barcodes": os.path.join(UNMIX_DIR, "G740_ST1_tumor_barcodes.csv"),
        "mimo": os.path.join(MIMO_DIR, "G740_ST1/G740_ST1_spot_level_mimo_fractions.csv")
    },
    "G4_ST2": {
        "mat":      os.path.join(UNMIX_DIR, "G4_ST2_tumor_unmix_results.mat"),
        "barcodes": os.path.join(UNMIX_DIR, "G4_ST2_tumor_barcodes.csv"),
        "mimo":     os.path.join(MIMO_DIR,  "G4_ST2/G4_ST2_spot_level_mimo_fractions.csv"),
    }
}

PURE_W_THRESH = 0.5
SCS_QUANTILE = 0.90

os.makedirs(OUT_DIR, exist_ok=True)

all_corr  = {}
all_fracs = {}
all_n     = {}

def strip_suffix(bc):
    return re.sub(r"-\d+$", "", str(bc))

for sample, cfg in SAMPLES.items():
    print(f"\n{'='*60}\n {sample}\n{'='*60}")

    mat = scipy.io.loadmat(cfg["mat"])
    F = mat["F"].astype(np.float64)
    K = F.shape[1]
    w_cols = [f"W{i}" for i in range(K)]
    arch_labels = [f"Arch {i}" for i in range(K)]

    barcodes = pd.read_csv(cfg["barcodes"])["barcode"].to_numpy()
    df = pd.DataFrame(F, columns=w_cols)
    df["barcode"] = barcodes
    df["barcode_key"] = df["barcode"].str.replace(r"-\d+$", "", regex=True)

    maxW = F.max(axis=1)
    df["maxW"] = maxW
    df["argmax"] = F.argmax(axis=1).astype(int)
    df["pure"] = maxW >= PURE_W_THRESH
    cutoff = float(np.quantile(maxW, SCS_QUANTILE))
    df["spot_type"] = np.where(maxW >= cutoff, "SCS", "Bulk")

    mimo = pd.read_csv(cfg["mimo"])
    clone_cols = [c for c in mimo.columns if c.startswith("mimo_clone_")]
    n_clones = len(clone_cols)
    clone_labels = [f"Clone {i}" for i in range(n_clones)]

    merged = df.merge(mimo, on="barcode_key", how="inner")
    print(f" Archetype spots: {len(df)} MIMO spots: {len(mimo)} Merged: {len(merged)}")

    corr_mat = pd.DataFrame(index=arch_labels, columns=clone_labels, dtype=float)
    for wc, al in zip(w_cols, arch_labels):
        for cc, cl in zip(clone_cols, clone_labels):
            r, _ = spearmanr(merged[wc], merged[cc], nan_policy="omit")
            corr_mat.at[al, cl] = r
    
    print(f"\n Spearman r:\n{corr_mat.round(3).to_string()}")
    corr_mat.to_csv(os.path.join(OUT_DIR, f"{sample}_arch_mimo_spearman.csv"))

    fig, ax = plt.subplots(figsize=(n_clones * 1.6 + 1.8, K * 1.3 + 2.2))
    vals = corr_mat.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(n_clones))
    ax.set_xticklabels(clone_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(K))
    ax.set_yticklabels(arch_labels, fontsize=12)
    best_j = np.argmax(vals, axis=1)
    for i in range(K):
        for j in range(n_clones):
            v = vals[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if abs(v) > 0.45 else "black",
                    fontweight="bold" if j == best_j[i] else "normal")
        rect = plt.Rectangle((best_j[i] - 0.5, i - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Spearman r", fontsize=11)
    ax.set_title(f"{sample}  —  Archetype vs Clone  (Spearman r,  n={len(merged)})",
                 fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{sample}_arch_mimo_spearman.png"), dpi=150)
    plt.close()

    best_clone = corr_mat.idxmax(axis=1)
    best_clone_r = corr_mat.max(axis=1)
    print("\n Best MIMO clone per archetype:")
    for al in arch_labels:
        print(f" {al} -> {best_clone[al]} (r={best_clone_r[al]:.3f})")

    scs_df = merged[merged["spot_type"] == "SCS"].copy()
    print(f"\n SCS spots in merged: {len(scs_df)}")
    if len(scs_df) >= 5:
        fig, axes = plt.subplots(1, n_clones, figsize=(3 * n_clones, 4))
        for ax, cc, cl in zip(axes, clone_cols, clone_labels):
            data_v, positions, tick_labels = [], [], []
            for i in range(K):
                vals_i = scs_df[scs_df["argmax"] == i][cc].dropna().values
                if len(vals_i) >= 3:
                    data_v.append(vals_i)
                    positions.append(i)
                    tick_labels.append(arch_labels[i])
            if data_v:
                parts = ax.violinplot(data_v, positions=positions, showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_alpha(0.7)
            ax.set_xticks(positions)
            ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
            ax.set_title(cl, fontsize=9)
            ax.set_ylim(-0.05, 1.05)
        axes[0].set_ylabel("MIMO clone fraction")
        fig.suptitle(f"{sample} [K={K}] - MIMO clone fractions per archetype (SCS, n={len(scs_df)})",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{sample}_arch_mimo_violin_scs.png"), dpi=150)
        plt.close()

    mean_fracs = pd.DataFrame(index=arch_labels, columns=clone_labels, dtype=float)
    for i, al in enumerate(arch_labels):
        arch_spots = merged[merged["argmax"] == i]
        for cc, cl in zip(clone_cols, clone_labels):
            mean_fracs.at[al, cl] = arch_spots[cc].mean()

    # Fill NaN rows (archetypes with 0 argmax spots) with overall mean clone distribution
    overall_mean = pd.Series({cl: merged[cc].mean() for cc, cl in zip(clone_cols, clone_labels)})
    nan_rows = mean_fracs.index[mean_fracs.isna().all(axis=1)]
    if len(nan_rows) > 0:
        print(f"\n  [INFO] Archetypes with no argmax spots — filling with overall mean: {list(nan_rows)}")
        for al in nan_rows:
            mean_fracs.loc[al] = overall_mean

    colors_bar = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(K * 1.4 + 1.8, 4.5))
    bottom = np.zeros(K)
    for j, (cc, cl) in enumerate(zip(clone_cols, clone_labels)):
        heights = mean_fracs[cl].values.astype(float)
        ax.bar(range(K), heights, bottom=bottom, label=cl,
               color=colors_bar[j % len(colors_bar)], edgecolor="white", linewidth=0.5)
        for bi, (h, b) in enumerate(zip(heights, bottom)):
            if h > 0.06:
                ax.text(bi, b + h / 2, f"{h:.2f}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += heights
    ax.set_xticks(range(K))
    ax.set_xticklabels(arch_labels, fontsize=11)
    ax.set_ylabel("Mean MIMO clone fraction", fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10, frameon=False)
    ax.set_title(f"{sample}  —  Mean clone composition per archetype",
                 fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{sample}_arch_mimo_stackedbar.png"), dpi=150)
    plt.close()

    print("\n Mean clone fraction per archetype:")
    print(mean_fracs.round(3).to_string())

    all_corr[sample]  = corr_mat.copy()
    all_fracs[sample] = mean_fracs.copy()
    all_n[sample]     = len(merged)

    merged.to_csv(os.path.join(OUT_DIR, f"{sample}_arch_mimo_merged.csv"), index=False)
    print(f"\n [DONE] {sample}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (sample, cm_) in zip(axes, all_corr.items()):
    vals = cm_.values.astype(float)
    K_  = vals.shape[0]
    nc_ = vals.shape[1]
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(nc_))
    ax.set_xticklabels([f"Clone {j}" for j in range(nc_)], rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(K_))
    ax.set_yticklabels([f"Arch {i}" for i in range(K_)], fontsize=11)
    best_j = np.argmax(vals, axis=1)
    for i in range(K_):
        for j in range(nc_):
            v = vals[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(v) > 0.45 else "black",
                    fontweight="bold" if j == best_j[i] else "normal")
        rect = plt.Rectangle((best_j[i] - 0.5, i - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
    ax.set_title(f"{sample}  (n={all_n[sample]})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Spearman r")

fig.suptitle("Archetype vs Clone  —  Spearman r", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "combined_arch_mimo_spearman.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nCombined figure -> {OUT_DIR}/combined_arch_mimo_spearman.png")

print(f"\nAll outputs in: {OUT_DIR}/")