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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import dpt_and_graph, nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import weights_nnls, that_hard_soft

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_improve/marker_robust.csv"
WRONG     = [0.0, 0.25, 0.5, 0.75]
N_MARKERS = [30, 10]


def corrupt(markers, all_genes, frac, n, rng):
    out = {}
    for s, g in markers.items():
        keep = list(g[:n])
        k = int(round(frac * n))
        others = [x for x in all_genes if x not in keep]
        if k:
            idx = rng.choice(n, k, replace=False)
            for i, new in zip(idx, rng.choice(others, k, replace=False)):
                keep[i] = new
        out[s] = keep
    return out


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
            W = weights_nnls(X, genes, m)
            r = np.corrcoef(W_true.T, W.T)[:4, 4:]
            dpt, conn = dpt_and_graph(a, W, genes, STATES, ROOT)
            Th, Ts = that_hard_soft(dpt, conn, W)
            rec = {"NJ": nj_graph(a, W, genes, STATES, ROOT), "That_arb": optimal_arborescence(Th, STATES, ROOT),
                   "Soft_arb": optimal_arborescence(Ts, STATES, ROOT)}
            row = {"case": case, "seed": seed, "n_markers": n, "wrong": frac, "r_NPC": r[0, 0], "r_mean": np.diag(r).mean()}
            for k, H in rec.items():
                row[f"{k}_exact"] = int(rf_rooted(G, H, ROOT) == 0)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(trees)
    jobs = [(n, trees[n], s, 1000000 * (names.index(n) + 1)) for n in names for s in range(1, args.seeds + 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(one_run, *j) for j in jobs]), 1):
            rows += f.result()
            if i % 80 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    d = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False)
    print(d.groupby(["n_markers", "wrong"])[["r_NPC", "r_mean", "NJ_exact", "That_arb_exact", "Soft_arb_exact"]].mean().round(3))


if __name__ == "__main__":
    main()
