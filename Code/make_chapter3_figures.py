import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import (nj_bootstrap, nj_state_graph, patristic, optimal_arborescence, greedy_arborescence,
                          case_name)
from plot_real_trees import load_centroids, load_that, layout_rooted

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "figures"
SAMPLES = ["G4_ST2", "G740_ST1"]
N_BOOT  = 1000
COLORS  = {"AC": "#C44E52", "MES": "#ED7D31", "MES1": "#ED7D31", "MES2": "#B85A16",
           "NPC": "#70AD47", "NPC1": "#70AD47", "NPC2": "#3F7A22", "OPC": "#4472C4"}

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 10})


def node(ax, x, y, v, size=520):
    ax.scatter(x, y, s=size, color=COLORS.get(v, "#999999"), zorder=2, edgecolor="white", linewidth=1.2)
    ax.text(x, y, v, ha="center", va="center", fontsize=7.5 if len(v) > 3 else 8, color="white", fontweight="bold", zorder=3)


def finish(ax, x, y, title, xlabel, letter):
    span = max(x.values()) or 1.0
    ax.set_xlim(-0.14 * span, 1.18 * span)
    ax.set_ylim(min(y.values()) - 0.7, max(y.values()) + 0.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")


def draw_tree(ax, G, root, length, label, title, xlabel, letter, min_frac=0.14, size=520):
    raw, _ = layout_rooted(G, root, length)
    floor = min_frac * (max(raw.values()) or 1.0)
    x, y = layout_rooted(G, root, lambda a, b: max(length(a, b), floor))
    for a, b in G.edges:
        ax.plot([x[a], x[a], x[b]], [y[a], y[b], y[b]], color="#444444", lw=1.4, zorder=1)
        ax.text((x[a] + x[b]) / 2, y[b] + 0.14, label(a, b), ha="center", va="bottom", fontsize=7.5)
    for v in G.nodes:
        node(ax, x[v], y[v], v, size)
    finish(ax, x, y, title, xlabel, letter)


def draw_nj(ax, T, labels, root, split_support, letter):
    anchor = next(iter(T.neighbors(root)))
    R = nx.bfs_tree(T, anchor)
    length = lambda a, b: T[a][b]["length"]
    x, y = layout_rooted(R, anchor, length)
    leaves = set(labels)
    for a, b in R.edges:
        ax.plot([x[a], x[a], x[b]], [y[a], y[b], y[b]], color="#444444", lw=1.4, zorder=1)
        span = max(x.values())
        if b in leaves and x[b] - x[a] < 0.2 * span:
            ax.text(x[b] + 0.08 * span, y[b], f"{length(a, b):.2f}", ha="left", va="center", fontsize=7.5)
        else:
            ax.text((x[a] + x[b]) / 2, y[b] + 0.14, f"{length(a, b):.2f}", ha="center", va="bottom", fontsize=7.5)
    for v in R.nodes:
        if v in leaves:
            node(ax, x[v], y[v], v)
        else:
            below = frozenset(u for u in nx.descendants(R, v) | {v} if u in leaves)
            sup = [s for k, s in split_support.items() if k == below or k == frozenset(leaves) - below]
            ax.scatter(x[v], y[v], s=18, color="#666666", zorder=2)
            if sup:
                ax.text(x[v] - 0.03 * max(x.values()), y[v] - 0.15, f"{round(sup[0] * N_BOOT)}/{N_BOOT}",
                        ha="right", va="top", fontsize=7.5, fontweight="bold")
    finish(ax, x, y, "NJ tree", "Branch length", letter)


def real_tree_figure(sample, number):
    X, labels = load_centroids(sample)
    T0, split_support, edge_support = nj_bootstrap(X, labels, "NPC", B=N_BOOT)
    G_nj = nj_state_graph(T0, labels, "NPC")
    P = patristic(T0, labels)
    pi = {s: i for i, s in enumerate(labels)}
    That, tl = load_that(sample)
    ti = {s: i for i, s in enumerate(tl)}
    greedy = greedy_arborescence(That, tl, "NPC")
    best = optimal_arborescence(That, tl, "NPC")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    draw_nj(axes[0, 0], T0, labels, "NPC", split_support, "A")
    draw_tree(axes[0, 1], G_nj, "NPC", lambda a, b: P[pi[a], pi[b]],
              lambda a, b: f"{P[pi[a], pi[b]]:.2f} ({round(edge_support[frozenset((a, b))] * N_BOOT)})",
              f"NJ state tree: {case_name(G_nj, 'NPC')}", "Tree distance", "B")
    tlab = lambda a, b: f"{That[ti[a], ti[b]]:.3f}"
    tlen = lambda a, b: 1 - That[ti[a], ti[b]]
    draw_tree(axes[1, 0], greedy, "NPC", tlen, tlab, f"Greedy T̂ tree: {case_name(greedy, 'NPC')}", "1 − T̂", "C", min_frac=0.25)
    draw_tree(axes[1, 1], best, "NPC", tlen, tlab, f"Most probable T̂ tree: {case_name(best, 'NPC')}", "1 − T̂", "D", min_frac=0.25)
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_{number}_{sample}_trees.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(sample, "NJ", case_name(G_nj, "NPC"), "| greedy", case_name(greedy, "NPC"), "| most probable", case_name(best, "NPC"))


def six_state_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, sample, letter in zip(axes, SAMPLES, "AB"):
        df = pd.read_csv(BASE / f"Results/phylogeny_clone_lineage/{sample}_subtype_centroids.csv", index_col=0)
        labels = list(df.index)
        T0, _, edge_support = nj_bootstrap(df.values.astype(float), labels, "NPC1", B=N_BOOT)
        G = nj_state_graph(T0, labels, "NPC1")
        P = patristic(T0, labels)
        pi = {s: i for i, s in enumerate(labels)}
        draw_tree(ax, G, "NPC1", lambda a, b: P[pi[a], pi[b]],
                  lambda a, b: f"{round(edge_support[frozenset((a, b))] * N_BOOT)}",
                  sample, "Tree distance", letter, min_frac=0.12, size=480)
        print(sample, "six-state:", sorted(G.edges), {f"{a}-{b}": round(edge_support[frozenset((a, b))] * N_BOOT) for a, b in G.edges})
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_3_six_state_trees.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def recovery_figure():
    d100 = pd.read_csv(BASE / "Results/sim_aa/aa_results.csv")
    d1000 = pd.read_csv(BASE / "Results/sim_aa/aa_results_1000.csv")
    groups = [("Clean spots", d100[d100.condition == "AA-baseline"]),
              ("Clean,\ncoalescent", d100[d100.condition == "AA-coalescent"]),
              ("Real purity", d1000),
              ("Real purity,\ncoalescent", d100[d100.condition == "AA-coalescent-real-purity"])]
    methods = [("NJ_exact", "NJ", "#2a78d6"), ("That_arb_exact", "Most probable T\u0302 tree", "#eb6834"),
               ("That_greedy_exact", "Greedy T\u0302 tree", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(groups))
    w = 0.26
    for i, (col, label, color) in enumerate(methods):
        vals = [100 * g[col].mean() for _, g in groups]
        xs = x + (i - 1) * (w + 0.01)
        ax.bar(xs, vals, w, color=color, edgecolor="white", linewidth=0.8, label=label)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(100 / 16, color="#555555", lw=0.9, ls="--", label="Chance (6%)")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in groups])
    ax.set_ylabel("Exact tree recovered (%)")
    ax.set_ylim(0, 100)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_4_recovery.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("recovery:", {n: {lab: round(100 * g[c].mean(), 1) for c, lab, _ in methods} for n, g in groups})


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_tree_figure("G4_ST2", 1)
    real_tree_figure("G740_ST1", 2)
    six_state_figure()
    recovery_figure()
    print("Saved to", OUT_DIR)
