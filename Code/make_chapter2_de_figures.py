import sys
import itertools
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE    = Path("/Users/arun/Desktop/Research")
DE_DIR  = BASE / "Results/phylogeny_clone_lineage"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "figures"
SAMPLES = ["G4_ST2", "G740_ST1"]
LFC, NLP = 1.0, 10.0
XLIM    = 6.0
PROG, DIFF, NS = "#70AD47", "#C44E52", "#C8C8C8"
SOURCE_COLORS = {"GO:BP": "#4472C4", "KEGG": "#ED7D31", "REAC": "#8064A2"}
SOURCE_NAMES  = {"GO:BP": "GO biological process", "KEGG": "KEGG", "REAC": "Reactome"}
LABELS = {
    "G4_ST2":   ["FAM107A", "AQP1", "APOD", "MBP", "PLP1", "ISG15", "MX1",
                 "LOX", "LOXL2", "LGALS3", "CA9", "CHI3L1", "VEGFA"],
    "G740_ST1": ["MBP", "PLP1", "MOBP", "TOP2A", "CCND2", "CHI3L1", "VEGFA"],
}
TOP_TERMS = 8

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10})


def crosses(a, b, c, d):
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def label_order(points, x_text, ys, xscale, yscale):
    best, best_key = None, None
    for perm in itertools.permutations(range(len(points))):
        segs = [((x_text / xscale, ys[i] / yscale), (points[p][0] / xscale, points[p][1] / yscale)) for i, p in enumerate(perm)]
        n = sum(crosses(*segs[i], *segs[j]) for i in range(len(segs)) for j in range(i + 1, len(segs)))
        length = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in segs)
        key = (n, length)
        if best_key is None or key < best_key:
            best, best_key = perm, key
    return best


def volcano(ax, sample, letter):
    d = pd.read_csv(DE_DIR / f"{sample}_DE_prog_vs_diff.csv")
    ymax = d.loc[np.isfinite(d["neg_log10p"]), "neg_log10p"].max()
    ycap = np.ceil(ymax / 5) * 5 + 5
    capped_y = ~np.isfinite(d["neg_log10p"])
    capped_x = d["log2FC"].abs() > XLIM
    d["y"] = np.where(capped_y, ycap, d["neg_log10p"])
    d["x"] = d["log2FC"].clip(-XLIM, XLIM)
    up = (d["log2FC"] > LFC) & (d["neg_log10p"] > NLP)
    dn = (d["log2FC"] < -LFC) & (d["neg_log10p"] > NLP)
    d["c"] = np.where(up, PROG, np.where(dn, DIFF, NS))
    for mask, marker in [(~capped_y & ~capped_x, "o"), (capped_y | capped_x, "^")]:
        sub = d[mask]
        ax.scatter(sub["x"], sub["y"], c=sub["c"], s=7 if marker == "o" else 12, marker=marker,
                   linewidths=0, alpha=0.8, rasterized=True)
    for v in (-LFC, LFC):
        ax.axvline(v, color="#7F7F7F", lw=0.6, ls="--")
    ax.axhline(NLP, color="#7F7F7F", lw=0.6, ls="--")
    lab = d[d["gene"].isin(LABELS[sample])]
    step = ycap * 0.085
    for side, x_text, ha in [(lab["x"] < 0, -XLIM + 0.1, "left"), (lab["x"] > 0, XLIM - 0.1, "right")]:
        rows = lab[side].reset_index(drop=True)
        ys = [ycap - n * step for n in range(len(rows))]
        order = label_order(rows[["x", "y"]].values, x_text, ys, 2 * XLIM, ycap)
        for n, p in enumerate(order):
            r = rows.iloc[p]
            ax.annotate(r["gene"], (r["x"], r["y"]), xytext=(x_text, ys[n]), textcoords="data",
                        ha=ha, va="center", fontsize=8, style="italic",
                        arrowprops=dict(arrowstyle="-", color="#595959", lw=0.4, shrinkA=1, shrinkB=2))
    ax.set_xlim(-XLIM - 0.4, XLIM + 0.4)
    ax.set_ylim(-1, ycap + 4)
    ax.set_xlabel("log$_2$ fold change (progenitor-like / differentiated)")
    ax.set_ylabel("$-$log$_{10}$ p")
    ax.set_title(f"{sample}  ({up.sum()} up, {dn.sum()} down)")
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def volcano_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, s, l in zip(axes, SAMPLES, "AB"):
        volcano(ax, s, l)
    handles = [Patch(color=PROG, label="Progenitor-enriched"), Patch(color=DIFF, label="Differentiated-enriched"),
               Patch(color=NS, label="Not significant"),
               Line2D([], [], marker="^", ls="", color="#7F7F7F", label="Value beyond axis limit")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_5_volcano.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_5_volcano.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pathway_panel(ax, sample, kind, letter):
    p = pd.read_csv(DE_DIR / f"{sample}_pathways_{kind}.csv").nsmallest(TOP_TERMS, "p_value").iloc[::-1]
    y = np.arange(len(p))
    ax.barh(y, -np.log10(p["p_value"]), color=[SOURCE_COLORS.get(s, "#BFBFBF") for s in p["source"]], height=0.7)
    names = [textwrap.shorten(n[0].upper() + n[1:], 48, placeholder="...") for n in p["name"]]
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.axvline(-np.log10(0.05), color="#7F7F7F", lw=0.6, ls="--")
    ax.set_xlabel("$-$log$_{10}$ FDR")
    title = "progenitor-enriched genes" if kind == "progenitor" else "differentiated-enriched genes"
    ax.set_title(f"{sample}, {title}", loc="left")
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pathway_figure():
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.6))
    panels = [(s, k) for s in SAMPLES for k in ("progenitor", "differentiated")]
    for ax, (s, k), l in zip(axes, panels, "ABCD"):
        pathway_panel(ax, s, k, l)
    used = set()
    for s in SAMPLES:
        for k in ("progenitor", "differentiated"):
            used |= set(pd.read_csv(DE_DIR / f"{s}_pathways_{k}.csv").nsmallest(TOP_TERMS, "p_value")["source"])
    handles = [Patch(color=c, label=SOURCE_NAMES[s]) for s, c in SOURCE_COLORS.items() if s in used]
    handles.append(Line2D([], [], color="#7F7F7F", lw=0.6, ls="--", label="FDR = 0.05"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, -0.03), ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(OUT_DIR / "Fig2_6_pathways.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_6_pathways.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    volcano_figure()
    pathway_figure()
    print("Saved to", OUT_DIR)