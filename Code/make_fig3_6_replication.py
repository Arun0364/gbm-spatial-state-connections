import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from make_chapter3_figures import node

BASE = Path("/Users/arun/Desktop/Research")
RESULTS = BASE / "Results"
OUT_DIR = BASE / "figures"
CONNECTIONS = ["NPC-OPC", "AC-NPC", "MES-NPC", "AC-MES", "AC-OPC", "MES-OPC"]
POS = {"OPC": (0, 1.4), "NPC": (1.4, 1.4), "AC": (0, 0), "MES": (1.4, 0)}
GROUPS = [("GBM1 region 1 (serial sections)", ["GBM1-R1-1", "GBM1-R1-2", "GBM1-R1-3", "GBM1-R1-4"]),
          ("GBM1 region 2 (serial sections)", ["GBM1-R2-1", "GBM1-R2-2", "GBM1-R2-3", "GBM1-R2-4"]),
          ("GBM2 (four regions)", ["GBM2-R1", "GBM2-R2", "GBM2-R3", "GBM2-R4"]),
          ("GBM3 (four regions)", ["GBM3-R1", "GBM3-R2", "GBM3-R3", "GBM3-R4"])]
TREE_SECTIONS = GROUPS[0][1] + GROUPS[2][1]
METHODS = [("NJ", "NJ"), ("T-hat most probable", "$\\hat{T}$ most\nprobable")]
WEIGHTS = "marker"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 9})


def load_trees(section):
    return pd.read_csv(RESULTS / "GSE253080" / section / f"{section}_trees.csv")


def flagged():
    f = pd.read_csv(RESULTS / "replicate_summary" / "section_flags.csv")
    return set(f.loc[f["flag_tumor_call"] == 1, "section"])


def small_tree(ax, row, title, flag):
    for c in CONNECTIONS:
        if row[c]:
            a, b = c.split("-")
            ax.plot([POS[a][0], POS[b][0]], [POS[a][1], POS[b][1]], color="#444444", lw=1.6, zorder=1)
    for v, (x, y) in POS.items():
        node(ax, x, y, v, size=400)
    ax.set_xlim(-0.42, 1.82)
    ax.set_ylim(-0.42, 1.82)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + (" *" if flag else ""), fontsize=8.5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flags = flagged()
    rows, labels, seps = [], [], []
    for gname, secs in GROUPS:
        for s in secs:
            t = load_trees(s)
            g = t[t["weights"] == WEIGHTS]
            rows.append(100 * g[CONNECTIONS].sum().to_numpy() / len(g))
            labels.append(s + (" *" if s in flags else ""))
        seps.append(len(rows))
    M = np.array(rows)
    fig = plt.figure(figsize=(7.4, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 60 else "black")
    for k in seps[:-1]:
        ax.axhline(k - 0.5, color="black", lw=1.0)
    for (gname, secs), end in zip(GROUPS, seps):
        ax.text(len(CONNECTIONS) - 0.4, end - len(secs) / 2 - 0.5, gname, ha="left", va="center", fontsize=8.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(CONNECTIONS)))
    ax.set_xticklabels([c.replace("-", "–") for c in CONNECTIONS])
    ax.xaxis.tick_top()
    ax.text(-0.2, 1.06, "A", transform=ax.transAxes, fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, location="bottom", shrink=0.5, pad=0.02, aspect=30)
    cb.set_label("Share of the 16 marker-guided trees with the connection (%)")
    sub = gs[1].subgridspec(len(METHODS), len(TREE_SECTIONS), wspace=0.08, hspace=0.05)
    tops = []
    for i, (m, mlabel) in enumerate(METHODS):
        for j, s in enumerate(TREE_SECTIONS):
            t = load_trees(s)
            row = t[(t["weights"] == WEIGHTS) & (t["root"] == "NPC") & (t["method"] == m)].iloc[0]
            a = fig.add_subplot(sub[i, j])
            small_tree(a, row, s.split("-", 1)[1] if i == 0 else "", s in flags and i == 0)
            if i == 0:
                tops.append(a.get_position())
            if j == 0:
                a.text(-0.32, 0.5, mlabel, transform=a.transAxes, rotation=90, ha="center", va="center", fontsize=9)
            if i == 0 and j == 0:
                a.text(-0.55, 1.45, "B", transform=a.transAxes, fontsize=12, fontweight="bold")
    y = tops[0].y1 + 0.03
    for lo, hi, name in [(0, 3, "GBM1 region 1 (serial sections)"), (4, 7, "GBM2 (four regions)")]:
        x0, x1 = tops[lo].x0 + 0.01, tops[hi].x1 - 0.01
        fig.add_artist(plt.Line2D([x0, x1], [y - 0.008, y - 0.008], color="black", lw=0.8, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, y, name, ha="center", va="bottom", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_6_replication.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'Fig3_6_replication.png'}")


if __name__ == "__main__":
    main()