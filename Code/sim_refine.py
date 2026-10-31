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

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence, greedy_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import marker_matrix, label_archetypes, weights_nnls
from sim_improve2 import CONDITIONS, dpt_graph, that
from unmix import unmix

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_refine/refine_results.csv"
NPC = STATES.index(ROOT)
TOP_FRACTION = 0.05
MAX_ITER = 50


def nnls_weights(ref, X):
    W = np.array([nnls(ref, x)[0] for x in X])
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1
    return W / s


def aa_labeled(X, genes, markers, seed):
    C, _, F = unmix(X.T, len(STATES), restarts=5, seed=seed)
    order = label_archetypes(C, genes, markers)
    return F[:, order], C[:, order]


def refine_hard(X, C0, genes, markers):
    L = np.log1p(X)
    centers = np.log1p(np.clip(C0.T, 0, None))
    assign = None
    for it in range(1, MAX_ITER + 1):
        d = ((L[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        centers = np.vstack([L[assign == k].mean(0) if (assign == k).any() else centers[k] for k in range(len(STATES))])
    ref = np.column_stack([X[assign == k].mean(0) if (assign == k).any() else np.clip(C0[:, k], 0, None)
                           for k in range(len(STATES))])
    order = label_archetypes(ref, genes, markers)
    ref = ref[:, order]
    return nnls_weights(ref, X), it, int((np.bincount(assign, minlength=len(STATES)) == 0).sum())


def refine_pure(X, W0, genes, markers, rounds=3):
    W = W0
    n_top = max(5, int(TOP_FRACTION * len(X)))
    for _ in range(rounds):
        ref = np.column_stack([X[np.argsort(W[:, k])[::-1][:n_top]].mean(0) for k in range(len(STATES))])
        ref = ref[:, label_archetypes(ref, genes, markers)]
        W = nnls_weights(ref, X)
    return W


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
    W_aa, C_aa = aa_labeled(X, genes, markers, seed_used)
    W_hard, n_iter, n_empty = refine_hard(X, C_aa, genes, markers)
    Ws = {"aa5": W_aa, "aa_hard": W_hard, "aa_pure": refine_pure(X, W_aa, genes, markers),
          "nnls": weights_nnls(X, genes, markers)}
    row = {"condition": cond, "case": case, "seed": seed, "hard_iter": n_iter, "hard_empty": n_empty}
    cache = {}
    for wn, W in Ws.items():
        r = np.corrcoef(W_true.T, W.T)[:4, 4:]
        row[f"{wn}_r_root"] = round(float(r[NPC, NPC]), 4)
        row[f"{wn}_r_mean"] = round(float(np.diag(r).mean()), 4)
        dpt, conn = dpt_graph(a, genes, int(np.argmax(W[:, NPC])), cache)
        nj = nj_graph(a, W, genes, STATES, ROOT)
        row[f"{wn}_NJ_case"] = case_name(nj, ROOT)
        row[f"{wn}_NJ"] = int(rf_rooted(G, nj, ROOT) == 0)
        T = that(dpt, conn, W, False)
        arb = optimal_arborescence(T, STATES, ROOT)
        row[f"{wn}_arb_case"] = case_name(arb, ROOT)
        row[f"{wn}_arb"] = int(rf_rooted(G, arb, ROOT) == 0)
        row[f"{wn}_greedy"] = int(rf_rooted(G, greedy_arborescence(T, STATES, ROOT), ROOT) == 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--cases", nargs="*", default=None)
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
    cases = args.cases or names
    jobs = [(c, n, tr[n], s, 3000000 + 1000000 * (names.index(n) + 1)) for c in args.conditions for n in cases
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