import re
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
                          rank_arborescences, that_mst, case_name, compare)

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/tree_eval/real"
SAMPLES = ["G4_ST2", "G740_ST1"]
ROOT    = "NPC"
N_BOOT  = 1000
COLORS  = {"AC": "#C44E52", "MES": "#ED7D31", "NPC": "#70AD47", "OPC": "#4472C4"}


def load_centroids(sample):
    df = pd.read_csv(BASE / f"Results/phylogeny_clone_lineage/{sample}_arch_centroids.csv", index_col=0)
    labels = [re.search(r"\((\w+)\)", i).group(1) for i in df.index]
    return df.values.astype(float), labels


def load_that(sample):
    df = pd.read_csv(BASE / f"Results/that_phylogeny/{sample}_that_matrix.csv", index_col=0)
    return df.values.astype(float), list(df.index)


def layout_rooted(G, root, length):
    x, y, order = {root: 0.0}, {}, []
    def walk(v):
        kids = sorted(G.successors(v), key=lambda k: (len(nx.descendants(G, k)), k))
        for k in kids:
            x[k] = x[v] + length(v, k)
            walk(k)
        if not kids:
            y[v] = float(len(order))
            order.append(v)
        else:
            y[v] = float(np.mean([y[k] for k in kids]))
    walk(root)
    return x, y


def draw_state_tree(ax, G, root, length, label, title, xlabel, min_frac=0.12):
    raw_x, _ = layout_rooted(G, root, length)
    floor = min_frac * (max(raw_x.values()) or 1.0)
    x, y = layout_rooted(G, root, lambda a, b: max(length(a, b), floor))
    for a, b in G.edges:
        ax.plot([x[a], x[a], x[b]], [y[a], y[b], y[b]], color="#444444", lw=2, zorder=1)
        ax.text((x[a] + x[b]) / 2, y[b] + 0.12, label(a, b), ha="center", va="bottom", fontsize=10, color="#222222")
    for v in G.nodes:
        ax.scatter(x[v], y[v], s=900, color=COLORS.get(v, "#999999"), zorder=2, edgecolor="white", linewidth=1.5)
        ax.text(x[v], y[v], v, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)
    finish(ax, x, y, title, xlabel)


def draw_nj_phylogram(ax, T, labels, root, split_support):
    anchor = next(iter(T.neighbors(root)))
    R = nx.bfs_tree(T, anchor)
    length = lambda a, b: T[a][b]["length"]
    x, y = layout_rooted(R, anchor, length)
    leaves = set(labels)
    for a, b in R.edges:
        ax.plot([x[a], x[a], x[b]], [y[a], y[b], y[b]], color="#444444", lw=2, zorder=1)
        ax.text((x[a] + x[b]) / 2, y[b] + 0.12, f"{length(a, b):.2f}", ha="center", va="bottom", fontsize=9, color="#555555")
    for v in R.nodes:
        if v in leaves:
            ax.scatter(x[v], y[v], s=900, color=COLORS.get(v, "#999999"), zorder=2, edgecolor="white", linewidth=1.5)
            ax.text(x[v], y[v], v, ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=3)
        else:
            below = frozenset(u for u in nx.descendants(R, v) | {v} if u in leaves)
            sup = [s for k, s in split_support.items() if k == below or k == frozenset(leaves) - below]
            ax.scatter(x[v], y[v], s=40, color="#888888", zorder=2)
            if sup:
                ax.text(x[v] + 0.02 * max(x.values()), y[v] - 0.18, f"{round(sup[0] * N_BOOT)}/{N_BOOT}",
                        ha="left", va="top", fontsize=10, color="#1f4e79", fontweight="bold")
    finish(ax, x, y, "NJ tree (expression distance, to scale)", "Branch length (Euclidean, z-scored marker genes)")


def finish(ax, x, y, title, xlabel):
    span = max(x.values()) or 1.0
    ax.set_xlim(-0.12 * span, 1.15 * span)
    ax.set_ylim(min(y.values()) - 0.7, max(y.values()) + 0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_yticks([])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, pairs, ranks = [], [], []
    for sample in SAMPLES:
        X, labels = load_centroids(sample)
        T0, split_support, edge_support = nj_bootstrap(X, labels, ROOT, B=N_BOOT)
        nj_graph = nj_state_graph(T0, labels, ROOT)
        P = patristic(T0, labels)
        pidx = {s: i for i, s in enumerate(labels)}

        That, tl = load_that(sample)
        tidx = {s: i for i, s in enumerate(tl)}
        arb = greedy_arborescence(That, tl, ROOT)
        arb_best = optimal_arborescence(That, tl, ROOT)
        mst, Dm = that_mst(That, tl, ROOT)
        ranked = rank_arborescences(That, tl, ROOT)

        trees = {"NJ state graph": nj_graph, "T-hat arborescence": arb, "T-hat MST": mst}
        for method, G in trees.items():
            summary.append({"sample": sample, "method": method, "case": case_name(G, ROOT),
                            "edges": "; ".join(f"{a}>{b}" for a, b in sorted(G.edges))})
        summary.append({"sample": sample, "method": "T-hat most probable tree (secondary)",
                        "case": case_name(arb_best, ROOT),
                        "edges": "; ".join(f"{a}>{b}" for a, b in sorted(arb_best.edges))})
        names = list(trees)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append({"sample": sample, "first": names[i], "second": names[j],
                              "first_case": case_name(trees[names[i]], ROOT),
                              "second_case": case_name(trees[names[j]], ROOT),
                              **compare(trees[names[i]], trees[names[j]], ROOT)})
        top = ranked[0][1]
        for rank, (name, prob, _) in enumerate(ranked[:5], 1):
            ranks.append({"sample": sample, "rank": rank, "case": name, "probability_product": f"{prob:.3e}",
                          "relative_to_best": round(prob / top, 3)})

        fig, axes = plt.subplots(1, 4, figsize=(26, 6.2))
        draw_nj_phylogram(axes[0], T0, labels, ROOT, split_support)
        draw_state_tree(axes[1], nj_graph, ROOT, lambda a, b: P[pidx[a], pidx[b]],
                        lambda a, b: f"{P[pidx[a], pidx[b]]:.2f} ({round(edge_support[frozenset((a, b))] * N_BOOT)}/{N_BOOT})",
                        f"NJ as a state tree: {case_name(nj_graph, ROOT)}", "Tree distance between states")
        draw_state_tree(axes[2], arb, ROOT, lambda a, b: 1 - That[tidx[a], tidx[b]],
                        lambda a, b: f"T-hat = {That[tidx[a], tidx[b]]:.3f}",
                        f"T-hat arborescence: {case_name(arb, ROOT)}", "1 - T-hat (short = strong flow)")
        draw_state_tree(axes[3], mst, ROOT, lambda a, b: Dm[tidx[a], tidx[b]],
                        lambda a, b: f"D = {Dm[tidx[a], tidx[b]]:.3f}",
                        f"T-hat MST: {case_name(mst, ROOT)}", "D = 1 - (T-hat(j,k) + T-hat(k,j)) / 2")
        fig.suptitle(f"{sample}: lineage trees of cell states, rooted at NPC, branch lengths to scale", fontsize=15, fontweight="bold")
        fig.text(0.5, -0.04,
                 "Panel 1: NJ on expression distances; numbers on branches are lengths, blue numbers are bootstrap support "
                 f"(genes resampled, B={N_BOOT}).  Panel 2: the same NJ tree read as connections between states "
                 "(minimum spanning tree of NJ tree distances), with bootstrap support per connection.  "
                 "Panel 3: greedy directed tree of T-hat flows from NPC (grown from NPC by repeatedly adding the strongest flow from a state in the tree to a new state).  Panel 4: undirected tree of strongest flows.  "
                 "In panels 2 to 4, edges shorter than 12% of the longest path are drawn at 12% so nodes do not overlap; labels give the true values.",
                 ha="center", fontsize=10, color="#444444", wrap=True)
        fig.savefig(OUT_DIR / f"{sample}_trees.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{sample}: NJ={case_name(nj_graph, ROOT)}  T-hat arborescence={case_name(arb, ROOT)} "
              f"(most probable {case_name(arb_best, ROOT)})  T-hat MST={case_name(mst, ROOT)}")

    pd.DataFrame(summary).to_csv(OUT_DIR / "real_tree_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT_DIR / "real_tree_comparison.csv", index=False)
    pd.DataFrame(ranks).to_csv(OUT_DIR / "real_arborescence_ranking.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60):
        print(pd.DataFrame(pairs).drop(columns=["first_case", "second_case"]).to_string(index=False))
        print()
        print(pd.DataFrame(ranks).to_string(index=False))


if __name__ == "__main__":
    main()