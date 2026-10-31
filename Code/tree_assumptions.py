import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import ast
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import nj_bootstrap, nj_state_graph, patristic, euclid, zscore_cols
from three_point_test import describe
from ancestral_nodes import hidden_score
from make_chapter3_figures import draw_tree, N_BOOT
from real_marker_trees import SAMPLES, STATES, neftel_markers, norm_counts, centroids
from state_weights import BASE, load_tumor, marker_setup

RES = BASE / "Results/marker_guided"
OUT_DIR = RES / "chapter3"
FIG_DIR = RES / "figures"
ESTIMATES = ["archetype", "marker", "markerfree"]
SIX = ["NPC1", "NPC2", "AC", "OPC", "MES1", "MES2"]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 10})
warnings.filterwarnings("ignore")


def substate_markers():
    tree = ast.parse((CODE / "export_subtype_centroids.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "SUBSTATE_MARKERS")
    return ast.literal_eval(node.value)


def six_state_weights(ad, W, sub):
    genes = [g for g in dict.fromkeys(g for v in sub.values() for g in v) if g in ad.var_names]
    X = norm_counts(ad, genes)
    gi = {g: i for i, g in enumerate(genes)}
    score = {s: X[:, [gi[g] for g in sub[s] if g in gi]].mean(1) for s in ["MES1", "MES2", "NPC1", "NPC2"]}
    mes = np.where(score["MES1"] + score["MES2"] > 0, score["MES1"] + score["MES2"], 1.0)
    npc = np.where(score["NPC1"] + score["NPC2"] > 0, score["NPC1"] + score["NPC2"], 1.0)
    w = dict(zip(STATES, W.T))
    eff = {"NPC1": w["NPC"] * score["NPC1"] / npc, "NPC2": w["NPC"] * score["NPC2"] / npc, "AC": w["AC"], "OPC": w["OPC"],
           "MES1": w["MES"] * score["MES1"] / mes, "MES2": w["MES"] * score["MES2"] / mes}
    return np.column_stack([eff[s] for s in SIX]), X


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    markers, sub = neftel_markers(), substate_markers()
    rng = np.random.default_rng(0)
    clock, clock_boot, hidden, hidden_boot, six = [], [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, sample, letter in zip(axes, SAMPLES, "AB"):
        ad, _ = load_tumor(sample)
        wdf = pd.read_csv(RES / f"{sample}_weights.csv")
        genes, _ = marker_setup(ad, markers)
        X = norm_counts(ad, genes)
        for wn in ESTIMATES:
            C = centroids(X, wdf[[f"{wn}_{s}" for s in STATES]].values)
            clock.append({"sample": sample, "weights": wn, **describe(euclid(zscore_cols(C)), STATES)})
            score, near, nodes, total = hidden_score(euclid(zscore_cols(C)), STATES)
            hidden.append({"sample": sample, "weights": wn, "hidden_score": score, "farthest_node_nearest_state": near,
                           "nodes": "; ".join(f"{n['nearest_state']} {n['distance']:.2f} ({n['fraction']:.3f})" for n in nodes)})
            for b in range(N_BOOT):
                Cb = C[:, rng.integers(0, C.shape[1], C.shape[1])]
                D = euclid(zscore_cols(Cb))
                r = describe(D, STATES)
                clock_boot.append({"sample": sample, "weights": wn, "three_point_mean": r["three_point_mean"], "four_point": r["four_point"], "upgma": r["upgma"]})
                hidden_boot.append({"sample": sample, "weights": wn, "hidden_score": hidden_score(D, STATES)[0]})
        W6, X6 = six_state_weights(ad, wdf[[f"marker_{s}" for s in STATES]].values, sub)
        C6 = centroids(X6, W6)
        T0, _, support = nj_bootstrap(C6, SIX, "NPC1", B=N_BOOT)
        G = nj_state_graph(T0, SIX, "NPC1")
        P = patristic(T0, SIX)
        pi = {s: i for i, s in enumerate(SIX)}
        for a, b in sorted(G.edges):
            six.append({"sample": sample, "edge": f"{a}>{b}", "tree_distance": round(P[pi[a], pi[b]], 2),
                        "support": round(support[frozenset((a, b))] * N_BOOT)})
        draw_tree(ax, G, "NPC1", lambda a, b: P[pi[a], pi[b]], lambda a, b: f"{round(support[frozenset((a, b))] * N_BOOT)}",
                  sample, "Tree distance", letter, min_frac=0.12, size=480)
        print(f"{sample}: six-state edges {sorted(G.edges)}", flush=True)
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Fig3_3_six_state_trees.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    clock, clock_boot = pd.DataFrame(clock), pd.DataFrame(clock_boot)
    hidden, hidden_boot = pd.DataFrame(hidden), pd.DataFrame(hidden_boot)
    q = clock_boot.groupby(["sample", "weights"])[["three_point_mean", "four_point"]].quantile([0.025, 0.975]).unstack()
    q.columns = [f"{a}_{'lo' if b == 0.025 else 'hi'}" for a, b in q.columns]
    clock = clock.merge(q.reset_index(), on=["sample", "weights"])
    clock = clock.merge(clock_boot.groupby(["sample", "weights"])["upgma"].agg(lambda s: s.value_counts(normalize=True).round(3).to_dict()).rename("upgma_bootstrap").reset_index(),
                        on=["sample", "weights"])
    sim = pd.read_csv(BASE / "Results/ancestral_nodes/sim_hidden_score.csv")
    hq = hidden_boot.groupby(["sample", "weights"])["hidden_score"].quantile([0.025, 0.975]).unstack()
    hq.columns = ["hidden_lo", "hidden_hi"]
    hidden = hidden.merge(hq.reset_index(), on=["sample", "weights"])
    for cond in ["Clean", "Real purity"]:
        neg = sim[(sim["condition"] == cond) & (sim["kind"] == "observed")]["hidden_score"].values
        hidden[f"pct_above_observed_{cond.lower().replace(' ', '_')}"] = [round(100 * (neg < v).mean(), 1) for v in hidden["hidden_score"]]
    clock.to_csv(OUT_DIR / "clock_conditions.csv", index=False)
    hidden.to_csv(OUT_DIR / "hidden_ancestor.csv", index=False)
    pd.DataFrame(six).to_csv(OUT_DIR / "six_state_tree.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(clock[["sample", "weights", "three_point_mean", "three_point_mean_lo", "three_point_mean_hi", "four_point",
                     "four_point_lo", "four_point_hi", "nj", "upgma"]].round(3).to_string(index=False))
        print(hidden[["sample", "weights", "hidden_score", "hidden_lo", "hidden_hi", "farthest_node_nearest_state",
                      "pct_above_observed_clean", "pct_above_observed_real_purity"]].round(3).to_string(index=False))
        print(pd.DataFrame(six).to_string(index=False))


if __name__ == "__main__":
    main()