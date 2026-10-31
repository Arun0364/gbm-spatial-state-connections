import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "figures"
SAMPLES = ["G4_ST2", "G740_ST1"]
STATES  = ["AC", "MES", "NPC", "OPC"]
COLORS  = {"AC": "#C44E52", "MES": "#ED7D31", "NPC": "#70AD47", "OPC": "#4472C4"}
GREY    = "#D9D9D9"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10})


def strip_bc(x):
    return re.sub(r"-\d+$", "", str(x))


def load(sample):
    with h5py.File(BASE / f"Results/PostAA_tumor_unmix_k4_markers/{sample}/{sample}_ns_annotated.h5ad", "r") as h:
        xy = pd.DataFrame(h["obsm/spatial"][:], columns=["x", "y"])
        xy["barcode_key"] = [strip_bc(b.decode()) for b in h["obs/_index"][:]]
    F = sio.loadmat(BASE / f"Results/Unmix_tumor_k4_markers/{sample}_tumor_unmix_results.mat")["F"]
    bc = pd.read_csv(BASE / f"Results/Unmix_tumor_k4_markers/{sample}_tumor_barcodes.csv")["barcode"].map(strip_bc)
    labels = pd.read_csv(BASE / f"Results/PostAA_tumor_unmix_k4_markers/{sample}/{sample}_archetype_state_labels.csv")
    arch_state = dict(zip(labels["archetype"].astype(int), labels["dominant_state"]))
    W = pd.DataFrame(F, columns=[arch_state[k] for k in range(F.shape[1])])
    W["barcode_key"] = bc.values
    merged = pd.read_csv(BASE / f"Results/archetype_mimo_integration_k4_markers/{sample}_arch_mimo_merged.csv")
    clones = merged[["barcode_key"] + [f"mimo_clone_{c}" for c in range(4)]]
    rho = pd.read_csv(BASE / f"Results/archetype_mimo_integration_k4_markers/{sample}_arch_mimo_spearman.csv", index_col=0)
    rho.index = [arch_state[int(re.search(r"\d+", i).group())] for i in rho.index]
    return xy, xy.merge(W, on="barcode_key"), xy.merge(clones, on="barcode_key"), rho


def base_axes(ax, xy, title):
    ax.scatter(xy["x"], xy["y"], s=1.2, c=GREY, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def weight_maps(data):
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.9))
    for r, s in enumerate(SAMPLES):
        xy, w, _, _ = data[s]
        for c, st in enumerate(STATES):
            ax = axes[r, c]
            base_axes(ax, xy, st if r == 0 else "")
            sc = ax.scatter(w["x"], w["y"], s=1.2, c=w[st], cmap="viridis", vmin=0, vmax=1, linewidths=0, rasterized=True)
        axes[r, 0].text(-0.08, 0.5, s, transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Archetype weight")
    fig.savefig(OUT_DIR / "Fig2_1_archetype_weights.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_1_archetype_weights.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def dominant_maps(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, s in zip(axes, SAMPLES):
        xy, w, _, _ = data[s]
        dom = w[STATES].values.argmax(1)
        counts = np.bincount(dom, minlength=4)
        base_axes(ax, xy, s)
        for k, st in enumerate(STATES):
            m = dom == k
            ax.scatter(w.loc[m, "x"], w.loc[m, "y"], s=1.4, c=COLORS[st], linewidths=0, rasterized=True,
                       label=f"{st} ({counts[k]:,})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, markerscale=5,
                  handletextpad=0.2, columnspacing=1.2, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=GREY, markersize=4, label="Non-tumor or excluded")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=9)
    fig.savefig(OUT_DIR / "Fig2_2_dominant_state.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_2_dominant_state.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clone_maps(data, matching):
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.9))
    for r, s in enumerate(SAMPLES):
        xy, _, cl, _ = data[s]
        for c in range(4):
            ax = axes[r, c]
            base_axes(ax, xy, f"Clone {c} ({matching[s][c]})")
            sc = ax.scatter(cl["x"], cl["y"], s=1.2, c=cl[f"mimo_clone_{c}"], cmap="magma", vmin=0, vmax=1,
                            linewidths=0, rasterized=True)
        axes[r, 0].text(-0.08, 0.5, s, transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center", fontsize=10)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("Clone fraction")
    fig.savefig(OUT_DIR / "Fig2_3_clone_fractions.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_3_clone_fractions.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def correlation_heatmaps(data, matching):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, s in zip(axes, SAMPLES):
        rho = data[s][3].loc[STATES]
        im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"Clone {c}" for c in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels(STATES)
        for i in range(4):
            for j in range(4):
                v = rho.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.45 else "black",
                        fontweight="bold" if matching[s][j] == STATES[i] else "normal")
        for j in range(4):
            i = STATES.index(matching[s][j])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
        n = len(data[s][2])
        ax.set_title(f"{s} (n = {n:,} spots)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Spearman r")
    fig.savefig(OUT_DIR / "Fig2_4_state_clone_correlation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_4_state_clone_correlation.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in SAMPLES}
    matching = {}
    for s in SAMPLES:
        rho = data[s][3].loc[STATES]
        rows, cols = linear_sum_assignment(-rho.values)
        matching[s] = {int(c): STATES[r] for r, c in zip(rows, cols)}
        dom = data[s][1][STATES].values.argmax(1)
        print(s, "AA spots:", len(data[s][1]), "dominant:", dict(zip(STATES, np.bincount(dom, minlength=4))),
              "clone spots:", len(data[s][2]), "matching:", matching[s])
    weight_maps(data)
    dominant_maps(data)
    clone_maps(data, matching)
    correlation_heatmaps(data, matching)
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()