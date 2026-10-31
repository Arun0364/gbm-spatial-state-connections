import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/arun/Desktop/Research")
sys.path.insert(0, str(BASE / "Code"))
from spatial_coalescent_sim import simulate, state_profiles, DEFAULTS, STATES, ROOT, STATE_COLORS
from sim_aa_run import aa_weights

OUT = BASE / "Results/pca_root"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
warnings.filterwarnings("ignore")

TREES = {"Star": [("NPC", "AC"), ("NPC", "OPC"), ("NPC", "MES")],
         "Fork-AC-MES": [("NPC", "AC"), ("NPC", "OPC"), ("AC", "MES")]}
SEED = 7


def pca_fit(X, n=2):
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return mu, Vt[:n], var[:n]


def sim_panel(tree, mix):
    G = nx.DiGraph(TREES[tree])
    a, _, _ = simulate(G, SEED, mix=mix)
    W_true = a.obs[[f"W{k}" for k in range(len(STATES))]].values
    W, Caa, genes = aa_weights(a, STATES, SEED)
    X = a[:, genes].layers["counts"].toarray()
    X = X / X.sum(1, keepdims=True) * 1e4
    rng = np.random.default_rng(SEED)
    C = state_profiles(G, STATES, ROOT, {**DEFAULTS, "mix": mix}, rng)
    gi = [int(g[1:]) for g in genes]
    pure = np.exp(C[:, gi])
    pure = pure / pure.sum(1, keepdims=True) * 1e4
    mu, V, var = pca_fit(X)
    r_root = np.corrcoef(W_true[:, 0], W[:, 0])[0, 1]
    return dict(P=(X - mu) @ V.T, arch=(Caa.T - mu) @ V.T, pure=(pure - mu) @ V.T, color=W_true[:, 0],
                var=var, title=f"{tree}, {'clean spots' if mix == 0 else 'real purity'}",
                note=f"UNMIX purity {W.max(1).mean():.2f}; NPC weight r = {r_root:.2f}")


def real_panel(sample):
    d = BASE / f"Results/PostAA_tumor_unmix_k4_markers/{sample}"
    a = ad.read_h5ad(d / f"{sample}_ns_annotated.h5ad")
    C = pd.read_csv(d / f"{sample}_C_matrix.csv", index_col=0)
    lab = pd.read_csv(d / f"{sample}_archetype_state_labels.csv").set_index("archetype")["dominant_state"]
    tn = pd.read_csv(BASE / f"Results/TumorNormalClassification/{sample}_tumor_normal_labels.csv")
    tumor = set(tn.loc[tn["tumor_label"] == "Tumor", "barcode"])
    X = a[:, list(C.index)].layers["norm_counts"]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    W = a.obs[[f"W{k}" for k in range(4)]].values
    keep = a.obs["barcode_key"].astype(str).isin(tumor).values & ~np.isnan(W).any(1)
    X, W = X[keep], W[keep]
    mu, V, var = pca_fit(X)
    order = [int(lab[lab == s].index[0]) for s in STATES]
    return dict(P=(X - mu) @ V.T, arch=(C.values.T[order] - mu) @ V.T, pure=None, color=W[:, order[0]],
                var=var, title=f"{sample} (real)", note=f"{len(X):,} tumor spots; {len(C)} marker genes")


def label_offsets(pts):
    span = np.ptp(pts, axis=0).max()
    offs = []
    for k, p in enumerate(pts):
        close = [j for j in range(len(pts)) if j != k and np.linalg.norm(pts[j] - p) < 0.18 * span]
        offs.append((6, -16) if close and any(pts[j][0] > p[0] for j in close) else (6, 6))
    return offs


def draw(ax, d):
    sc = ax.scatter(d["P"][:, 0], d["P"][:, 1], c=d["color"], cmap="Greens", s=6, vmin=0, vmax=1,
                    edgecolors="none", alpha=0.8, rasterized=True)
    offs = label_offsets(d["arch"])
    for k, s in enumerate(STATES):
        col = STATE_COLORS[s]
        ax.scatter(*d["arch"][k], marker="^", s=150, color=col, edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(s, d["arch"][k], xytext=offs[k], textcoords="offset points", fontsize=11, fontweight="bold", color=col)
        if d["pure"] is not None:
            ax.scatter(*d["pure"][k], marker="*", s=260, color=col, edgecolors="black", linewidths=1, zorder=6)
    ax.margins(0.12)
    ax.set_title(d["title"], fontsize=13, fontweight="bold", loc="left", pad=20)
    ax.text(0, 1.015, d["note"], transform=ax.transAxes, fontsize=10, color="#52514e", va="bottom")
    ax.set_xlabel(f"PC1 ({100 * d['var'][0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * d['var'][1]:.0f}%)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return sc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panels = [sim_panel("Star", 0.0), sim_panel("Star", 0.75), sim_panel("Fork-AC-MES", 0.0), sim_panel("Fork-AC-MES", 0.75)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for ax, d in zip(axes.ravel(), panels):
        sc = draw(ax, d)
    h = [plt.Line2D([], [], marker="^", ls="", color="grey", markeredgecolor="black", markersize=10, label="UNMIX archetype"),
         plt.Line2D([], [], marker="*", ls="", color="grey", markeredgecolor="black", markersize=14, label="True pure state")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.45, 0.98))
    fig.colorbar(sc, ax=axes, shrink=0.6, label="True NPC fraction of spot")
    fig.savefig(OUT / "pca_root_simulated.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "pca_root_simulated.pdf", bbox_inches="tight")
    plt.close(fig)
    real = [real_panel("G4_ST2"), real_panel("G740_ST1")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.3})
    for ax, d in zip(axes, real):
        sc = draw(ax, d)
    fig.legend(handles=h[:1], loc="upper center", frameon=False, bbox_to_anchor=(0.45, 1.06))
    fig.colorbar(sc, ax=axes, shrink=0.8, label="NPC archetype weight")
    fig.savefig(OUT / "pca_root_real.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "pca_root_real.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([{"panel": d["title"], "note": d["note"], "pc1_var": round(d["var"][0], 3), "pc2_var": round(d["var"][1], 3)}
                  for d in panels + real]).to_csv(OUT / "pca_root_summary.csv", index=False)


if __name__ == "__main__":
    main()