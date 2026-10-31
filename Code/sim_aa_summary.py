import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import all_rooted_trees, case_name

BASE    = Path("/Users/arun/Desktop/Research")
AA_CSV  = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "Results/sim_aa/aa_results.csv"
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "Results/sim_aa"
STATES  = ["NPC", "AC", "OPC", "MES"]
METHODS = ["NJ", "That_arb", "That_greedy", "That_mst"]
NAMES   = {"NJ": "NJ", "That_arb": "T-hat most probable", "That_greedy": "T-hat greedy", "That_mst": "T-hat MST"}
SHAPE_ORDER = ["Star", "Relay", "Fork", "Chain"]
TREES   = {case_name(G, "NPC"): G for G in all_rooted_trees(STATES, "NPC")}
EDGES   = {n: {frozenset(e) for e in G.to_undirected().edges} for n, G in TREES.items()}

plt.rcParams.update({"font.family": "Arial", "font.size": 9})


def tree_order():
    return sorted(TREES, key=lambda n: (SHAPE_ORDER.index(n.split("-")[0]), n))


def roles(case):
    G = TREES[case]
    return {s: ("root" if s == "NPC" else "leaf" if G.out_degree(s) == 0 else "internal") for s in STATES}


def edge_precision(d, col):
    hit = tot = 0
    for true, est in zip(d.case, d[col]):
        tot += 3
        hit += len(EDGES[est] & EDGES[true])
    return 100 * hit / tot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(AA_CSV)
    rows = []
    for cond, g in d.groupby("condition"):
        row = {"condition": cond, "runs": len(g), "true_purity": g.true_purity.mean(), "aa_purity": g.aa_purity.mean(),
               "labels_correct_%": 100 * g.labels_correct.mean()}
        for m in METHODS:
            p = 100 * g[f"{m}_exact"].mean()
            row[f"{NAMES[m]} exact %"] = p
            row[f"{NAMES[m]} 95% margin"] = 196 * np.sqrt(p / 100 * (1 - p / 100) / len(g))
            row[f"{NAMES[m]} connections true %"] = edge_precision(g, f"{m}_case")
        agree = g[g.NJ_case == g.That_arb_case]
        row["NJ and most probable agree: runs"] = len(agree)
        row["when they agree: connections true %"] = edge_precision(agree, "NJ_case") if len(agree) else np.nan
        row["when they agree: whole tree right %"] = 100 * (agree.case == agree.NJ_case).mean() if len(agree) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("condition").round(2)
    summary.to_csv(OUT_DIR / "summary_by_condition.csv")
    print(summary.T.to_string())

    order = tree_order()
    for m in ["NJ", "That_arb", "That_greedy"]:
        t = d.pivot_table(index="case", columns="condition", values=f"{m}_exact", aggfunc="mean").reindex(order) * 100
        t.round(1).to_csv(OUT_DIR / f"per_topology_{m}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, m in zip(axes, ["NJ", "That_arb"]):
        t = pd.read_csv(OUT_DIR / f"per_topology_{m}.csv", index_col=0)
        ax.imshow(t.values, cmap="Greys", vmin=0, vmax=100, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, "" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else "black")
        ax.set_xticks(range(t.shape[1]))
        ax.set_xticklabels(t.columns, rotation=35, ha="right")
        ax.set_yticks(range(t.shape[0]))
        ax.set_yticklabels(t.index)
        ax.set_title(f"{NAMES[m]}: % exact")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_per_topology.png", dpi=200)
    plt.close(fig)

    long = []
    for _, r in d.iterrows():
        rl = roles(r.case)
        for s in STATES:
            long.append({"condition": r.condition, "role": rl[s], "w_r": r[f"w_r_{s}"], "profile_r": r[f"profile_r_{s}"]})
    long = pd.DataFrame(long)
    states = long.groupby(["condition", "role"])[["w_r", "profile_r"]].median().round(3)
    states.to_csv(OUT_DIR / "state_recovery_by_role.csv")
    print(states.to_string())


if __name__ == "__main__":
    main()
