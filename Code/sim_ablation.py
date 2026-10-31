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
from scipy.optimize import linear_sum_assignment

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import all_rooted_trees, case_name, rf_rooted, optimal_arborescence, greedy_arborescence
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import that_matrices, nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import marker_matrix, label_archetypes
from unmix import unmix

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_ablation/ablation_results.csv"
CONDITIONS = {"Real-purity": {"mix": 0.75}, "Real-purity-coalescent": {"lineage": "coalescent", "mix": 0.75}}
VARIANTS = ["baseline", "true_labels", "true_root_weights", "true_dpt_start", "true_labels_and_root", "true_weights"]
METHODS  = ["NJ", "That_arb", "That_greedy"]
NPC = STATES.index(ROOT)


def dpt_graph(a, genes, iroot):
    import scanpy as sc
    am = a[:, genes].copy()
    am.X = am.layers["norm_counts"].toarray()
    sc.pp.scale(am, max_value=10)
    sc.tl.pca(am, n_comps=20, svd_solver="arpack")
    sc.pp.neighbors(am, n_neighbors=15, n_pcs=15)
    sc.tl.diffmap(am, n_comps=15)
    am.uns["iroot"] = int(iroot)
    sc.tl.dpt(am, n_dcs=10)
    return am.obs["dpt_pseudotime"].values, am.obsp["connectivities"].tocsr()


def replace_root(W, W_true):
    out = W.copy()
    others = [k for k in range(W.shape[1]) if k != NPC]
    rest = out[:, others].sum(1, keepdims=True)
    rest[rest == 0] = 1
    out[:, others] = out[:, others] / rest * (1 - W_true[:, [NPC]])
    out[:, NPC] = W_true[:, NPC]
    return out


def trees(a, W, genes, iroot, G):
    dpt, conn = dpt_graph(a, genes, iroot)
    T, _ = that_matrices(dpt, conn, W)
    rec = {"NJ": nj_graph(a, W, genes, STATES, ROOT), "That_arb": optimal_arborescence(T, STATES, ROOT),
           "That_greedy": greedy_arborescence(T, STATES, ROOT)}
    return {m: int(rf_rooted(G, H, ROOT) == 0) for m, H in rec.items()}


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
    C, _, F = unmix(X.T, len(STATES), restarts=5, seed=seed_used)
    W_marker = F[:, label_archetypes(C, genes, markers)]
    r = np.nan_to_num(np.corrcoef(W_true.T, F.T)[:4, 4:])
    oracle = np.empty(4, dtype=int)
    oracle[linear_sum_assignment(-r)[0]] = linear_sum_assignment(-r)[1]
    W_oracle = F[:, oracle]
    weights = {"baseline": W_marker, "true_labels": W_oracle, "true_root_weights": replace_root(W_marker, W_true),
               "true_dpt_start": W_marker, "true_labels_and_root": replace_root(W_oracle, W_true), "true_weights": W_true}
    true_start = int(np.argmax(W_true[:, NPC]))
    row = {"condition": cond, "case": case, "seed": seed,
           "labels_correct": int(np.array_equal(label_archetypes(C, genes, markers), oracle))}
    for v in VARIANTS:
        W = weights[v]
        start = true_start if v in ("true_dpt_start", "true_root_weights", "true_labels_and_root", "true_weights") else int(np.argmax(W[:, NPC]))
        for m, ok in trees(a, W, genes, start, G).items():
            row[f"{v}_{m}"] = ok
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=8)
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
            if k % 100 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done", flush=True)
    d = pd.read_csv(out)
    for c, g in d.groupby("condition"):
        print(f"\n{c} ({len(g)} runs): exact tree recovery (%)")
        tab = pd.DataFrame({m: [100 * g[f"{v}_{m}"].mean() for v in VARIANTS] for m in METHODS}, index=VARIANTS)
        print(tab.round(1).to_string())


if __name__ == "__main__":
    main()
