import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import dpt_and_graph, nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import that_hard_soft
from sim_marker_robust import corrupt
from marker_weights import weights_marker

BASE      = Path("/Users/arun/Desktop/Research")
OUT_CSV   = BASE / "Results/sim_weights_fix/marker_robust_scaled.csv"
WRONG     = [0.0, 0.25, 0.5, 0.75]
N_MARKERS = [30, 10]
SCALES    = ["none", "gene_sd"]


def one_run(case, edges, seed, tree_offset):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = tree_offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, mix=0.75)
        if (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= MIN_STATE_FRACTION:
            break
    true_m = json.loads(a.uns["state_markers"])
    rng = np.random.default_rng(seed_used + 11)
    W_true = a.obs[[f"W{k}" for k in range(len(STATES))]].values
    rows = []
    for n in N_MARKERS:
        for frac in WRONG:
            m = corrupt(true_m, list(a.var_names), frac, n, rng)
            genes = sorted(set(sum(m.values(), [])))
            X = a[:, genes].layers["counts"].toarray()
            X = X / X.sum(1, keepdims=True) * 1e4
            for sc_ in SCALES:
                W = weights_marker(X, genes, m, STATES, sc_)
                r = np.corrcoef(W_true.T, W.T)[:4, 4:]
                dpt, conn = dpt_and_graph(a, W, genes, STATES, ROOT)
                Th, _ = that_hard_soft(dpt, conn, W)
                row = {"case": case, "seed": seed, "n_markers": n, "wrong": frac, "scale": sc_,
                       "r_NPC": r[0, 0], "r_mean": np.diag(r).mean(),
                       "NJ_exact": int(rf_rooted(G, nj_graph(a, W, genes, STATES, ROOT), ROOT) == 0),
                       "That_arb_exact": int(rf_rooted(G, optimal_arborescence(Th, STATES, ROOT), ROOT) == 0)}
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(trees)
    cases = args.cases or names
    jobs = [(n, trees[n], s, 1000000 * (names.index(n) + 1)) for n in cases for s in range(1, args.seeds + 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(one_run, *j) for j in jobs]), 1):
            rows += f.result()
            if i % 80 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    d = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    s = d.groupby(["n_markers", "wrong", "scale"])[["r_NPC", "NJ_exact", "That_arb_exact"]].mean().round(3).unstack("scale")
    with pd.option_context("display.width", 200):
        print(s)


if __name__ == "__main__":
    main()