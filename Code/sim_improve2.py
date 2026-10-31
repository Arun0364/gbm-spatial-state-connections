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
from scipy.optimize import nnls
from scipy.spatial import cKDTree

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence, greedy_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import marker_matrix, label_archetypes, weights_nnls
from unmix import unmix

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_improve/improve2_results.csv"
CONDITIONS = {"Clean": {}, "Clean-coalescent": {"lineage": "coalescent"},
              "Real-purity": {"mix": 0.75}, "Real-purity-coalescent": {"lineage": "coalescent", "mix": 0.75}}
NPC = STATES.index(ROOT)
TOP_FRACTION = 0.05
CONFIDENT = 0.5


def aa_labeled(X, genes, markers, seed, restarts):
    C, _, F = unmix(X.T, len(STATES), restarts=restarts, seed=seed)
    return F[:, label_archetypes(C, genes, markers)]


def nnls_iterative(X, genes, markers, rounds=3):
    W = weights_nnls(X, genes, markers)
    n_top = max(5, int(TOP_FRACTION * len(X)))
    for _ in range(rounds):
        ref = np.column_stack([X[np.argsort(W[:, k])[::-1][:n_top]].mean(0) for k in range(len(STATES))])
        W = np.array([nnls(ref, x)[0] for x in X])
        s = W.sum(1, keepdims=True)
        s[s == 0] = 1
        W = W / s
    return W


def smooth(W, xy, k=6, self_weight=0.5):
    nb = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    S = self_weight * W + (1 - self_weight) * W[nb].mean(1)
    return S / S.sum(1, keepdims=True)


def dpt_graph(a, genes, iroot, cache):
    if iroot in cache:
        return cache[iroot]
    import scanpy as sc
    am = a[:, genes].copy()
    am.X = am.layers["norm_counts"].toarray()
    sc.pp.scale(am, max_value=10)
    sc.tl.pca(am, n_comps=20, svd_solver="arpack")
    sc.pp.neighbors(am, n_neighbors=15, n_pcs=15)
    sc.tl.diffmap(am, n_comps=15)
    am.uns["iroot"] = int(iroot)
    sc.tl.dpt(am, n_dcs=10)
    cache[iroot] = (am.obs["dpt_pseudotime"].values, am.obsp["connectivities"].tocsr())
    return cache[iroot]


def that(dpt, conn, W, confident_only):
    K = W.shape[1]
    k = W.argmax(1)
    coo = conn.tocoo()
    up = np.isfinite(dpt[coo.row]) & np.isfinite(dpt[coo.col]) & (dpt[coo.col] > dpt[coo.row])
    if confident_only:
        mx = W.max(1)
        ok = mx >= np.quantile(mx, CONFIDENT)
        up &= ok[coo.row] & ok[coo.col]
    T = np.zeros((K, K))
    np.add.at(T, (k[coo.row[up]], k[coo.col[up]]), 1.0)
    return T / np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)


def one_run(cond, case, edges, seed, offset):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, **CONDITIONS[cond])
        if (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= MIN_STATE_FRACTION:
            break
    W_true = a.obs[[f"W{k}" for k in range(len(STATES))]].values
    X, genes = marker_matrix(a)
    markers = json.loads(a.uns["state_markers"])
    xy = a.obsm["spatial"]
    runs = [aa_labeled(X, genes, markers, seed_used + 101 * r, 5) for r in range(5)]
    Ws = {"aa5": runs[0], "aa50": aa_labeled(X, genes, markers, seed_used, 50), "aa_ensemble5": np.mean(runs, axis=0),
          "nnls": weights_nnls(X, genes, markers)}
    Ws["nnls_iter"] = nnls_iterative(X, genes, markers)
    Ws["aa5_smooth"] = smooth(Ws["aa5"], xy)
    Ws["nnls_iter_smooth"] = smooth(Ws["nnls_iter"], xy)
    row = {"condition": cond, "case": case, "seed": seed}
    cache = {}
    for wn, W in Ws.items():
        r = np.corrcoef(W_true.T, W.T)[:4, 4:]
        row[f"{wn}_r_root"] = round(float(r[NPC, NPC]), 4)
        row[f"{wn}_r_mean"] = round(float(np.diag(r).mean()), 4)
        dpt, conn = dpt_graph(a, genes, int(np.argmax(W[:, NPC])), cache)
        nj = nj_graph(a, W, genes, STATES, ROOT)
        row[f"{wn}_NJ_case"] = case_name(nj, ROOT)
        row[f"{wn}_NJ"] = int(rf_rooted(G, nj, ROOT) == 0)
        for conf in (False, True):
            T = that(dpt, conn, W, conf)
            tag = "_conf" if conf else ""
            arb = optimal_arborescence(T, STATES, ROOT)
            row[f"{wn}_arb{tag}_case"] = case_name(arb, ROOT)
            row[f"{wn}_arb{tag}"] = int(rf_rooted(G, arb, ROOT) == 0)
            row[f"{wn}_greedy{tag}"] = int(rf_rooted(G, greedy_arborescence(T, STATES, ROOT), ROOT) == 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.condition, prev.case, prev.seed))
    tr = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(tr)
    jobs = [(c, n, tr[n], s, 3000000 + 1000000 * (names.index(n) + 1)) for c in args.conditions for n in names
            for s in range(1, args.seeds + 1) if (c, n, s) not in done]
    print(f"{len(jobs)} runs to do", flush=True)
    header = not out.exists()
    k = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as err:
                print("FAILED", futs[f][:2], err, flush=True)
                continue
            pd.DataFrame([row]).to_csv(out, mode="a", header=header, index=False)
            header = False
            k += 1
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done", flush=True)


if __name__ == "__main__":
    main()
