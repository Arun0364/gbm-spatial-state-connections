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
from scipy.optimize import linear_sum_assignment, nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence, greedy_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import dpt_and_graph, nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from unmix import unmix

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_improve/improve_results.csv"

CONDITIONS = {
    "Real-purity":            {"mix": 0.75},
    "Real-purity-coalescent": {"lineage": "coalescent", "mix": 0.75},
}
TOP_FRACTION = 0.05


def marker_matrix(a):
    genes = list(a.uns["marker_genes"])
    X = a[:, genes].layers["counts"].toarray()
    X = X / X.sum(1, keepdims=True) * 1e4
    return X, genes


def label_archetypes(C, genes, markers):
    Z = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-12)
    gidx = {g: i for i, g in enumerate(genes)}
    score = np.array([[Z[[gidx[g] for g in markers[s]], k].mean() for s in STATES] for k in range(len(STATES))])
    arch, st = linear_sum_assignment(-score)
    order = np.empty(len(STATES), dtype=int)
    order[st] = arch
    return order


def weights_aa(X, genes, markers, seed, restarts):
    C, _, F = unmix(X.T, len(STATES), restarts=restarts, seed=seed)
    return F[:, label_archetypes(C, genes, markers)]


def weights_nnls(X, genes, markers):
    L = np.log1p(X)
    Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
    gidx = {g: i for i, g in enumerate(genes)}
    score = np.column_stack([Z[:, [gidx[g] for g in markers[s] if g in gidx]].mean(1) for s in STATES])
    n_top = max(5, int(TOP_FRACTION * len(X)))
    ref = np.column_stack([X[np.argsort(score[:, k])[::-1][:n_top]].mean(0) for k in range(len(STATES))])
    W = np.array([nnls(ref, x)[0] for x in X])
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1
    return W / s


def that_hard_soft(dpt, conn, W):
    K = W.shape[1]
    k = W.argmax(1)
    coo = conn.tocoo()
    up = np.isfinite(dpt[coo.row]) & np.isfinite(dpt[coo.col]) & (dpt[coo.col] > dpt[coo.row])
    r, c = coo.row[up], coo.col[up]
    Th = np.zeros((K, K))
    np.add.at(Th, (k[r], k[c]), 1.0)
    Ts = W[r].T @ W[c]
    norm = lambda T: T / np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)
    return norm(Th), norm(Ts)


def one_run(cond, case, edges, seed, tree_offset):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = tree_offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, **CONDITIONS[cond])
        if (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= MIN_STATE_FRACTION:
            break
    else:
        raise RuntimeError("a state stayed too rare")
    W_true = a.obs[[f"W{k}" for k in range(len(STATES))]].values
    X, genes = marker_matrix(a)
    markers = json.loads(a.uns["state_markers"])
    Ws = {"aa5": weights_aa(X, genes, markers, seed_used, 5),
          "aa50": weights_aa(X, genes, markers, seed_used, 50),
          "nnls": weights_nnls(X, genes, markers)}
    row = {"condition": cond, "case": case, "seed": seed, "seed_used": seed_used, "n_spots": a.n_obs}
    for wn, W in Ws.items():
        r = np.corrcoef(W_true.T, W.T)[:len(STATES), len(STATES):]
        row[f"{wn}_purity"] = round(float(W.max(1).mean()), 4)
        for kk, s in enumerate(STATES):
            row[f"{wn}_r_{s}"] = round(float(r[kk, kk]), 4)
        dpt, conn = dpt_and_graph(a, W, genes, STATES, ROOT)
        Th, Ts = that_hard_soft(dpt, conn, W)
        rec = {"NJ": nj_graph(a, W, genes, STATES, ROOT),
               "That_greedy": greedy_arborescence(Th, STATES, ROOT),
               "That_arb": optimal_arborescence(Th, STATES, ROOT),
               "Soft_greedy": greedy_arborescence(Ts, STATES, ROOT),
               "Soft_arb": optimal_arborescence(Ts, STATES, ROOT)}
        for m, H in rec.items():
            row[f"{wn}_{m}_case"] = case_name(H, ROOT)
            row[f"{wn}_{m}_exact"] = int(rf_rooted(G, H, ROOT) == 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.condition, prev.case, prev.seed))
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(trees)
    jobs = [(c, n, trees[n], s, 1000000 * (names.index(n) + 1)) for c in args.conditions for n in names
            for s in range(1, args.seeds + 1) if (c, n, s) not in done]
    print(f"{len(jobs)} runs to do", flush=True)
    header = not out.exists()
    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as err:
                print(f"FAILED {futs[f][:2]} seed {futs[f][3]}: {err}", flush=True)
                continue
            pd.DataFrame([row]).to_csv(out, mode="a", header=header, index=False)
            header = False
            finished += 1
            if finished % 200 == 0 or finished == len(jobs):
                print(f"  {finished}/{len(jobs)} done", flush=True)


if __name__ == "__main__":
    main()
