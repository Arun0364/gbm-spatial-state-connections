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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import (all_rooted_trees, case_name, rf_rooted, optimal_arborescence,
                          greedy_arborescence, that_mst)
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import dpt_and_graph, that_matrices, nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from unmix import unmix

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_aa/aa_results.csv"

CONDITIONS = {
    "AA-baseline":               {},
    "AA-real-purity":            {"mix": 0.75},
    "AA-coalescent":             {"lineage": "coalescent"},
    "AA-coalescent-real-purity": {"lineage": "coalescent", "mix": 0.75},
}

METHODS = ["NJ", "That_greedy", "That_arb", "That_mst"]
COLUMNS = (["condition", "case", "seed", "seed_used", "n_spots", "true_purity", "aa_purity",
            "labels_correct", "w_r_mean", "w_mae"] +
           [f"w_r_{s}" for s in STATES] + [f"profile_r_{s}" for s in STATES] +
           [f"{m}_{x}" for m in METHODS for x in ("case", "rf_rooted", "exact")])


def aa_weights(a, states, seed, restarts=5):
    genes = list(a.uns["marker_genes"])
    X = a[:, genes].layers["counts"].toarray()
    X = X / X.sum(1, keepdims=True) * 1e4
    C, _, F = unmix(X.T, len(states), restarts=restarts, seed=seed)
    Z = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-12)
    markers = json.loads(a.uns["state_markers"])
    gidx = {g: i for i, g in enumerate(genes)}
    score = np.array([[Z[[gidx[g] for g in markers[s]], k].mean() for s in states] for k in range(len(states))])
    arch, st = linear_sum_assignment(-score)
    order = np.empty(len(states), dtype=int)
    order[st] = arch
    return F[:, order], C[:, order], genes


def one_run(cond, case, edges, seed, tree_offset=0):
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
    W, Caa, genes = aa_weights(a, STATES, seed_used)
    r = np.corrcoef(W_true.T, W.T)[:len(STATES), len(STATES):]
    oracle_arch = linear_sum_assignment(-r)[1]
    X = a[:, genes].layers["counts"].toarray()
    X = X / X.sum(1, keepdims=True) * 1e4
    true_prof = ((W_true.T @ X) / W_true.sum(0)[:, None]).T
    rel_aa = Caa - Caa.mean(1, keepdims=True)
    rel_true = true_prof - true_prof.mean(1, keepdims=True)
    dpt, conn = dpt_and_graph(a, W, genes, STATES, ROOT)
    Tc, _ = that_matrices(dpt, conn, W)
    rec = {"NJ": nj_graph(a, W, genes, STATES, ROOT),
           "That_greedy": greedy_arborescence(Tc, STATES, ROOT),
           "That_arb": optimal_arborescence(Tc, STATES, ROOT),
           "That_mst": that_mst(Tc, STATES, ROOT)[0]}
    row = {"condition": cond, "case": case, "seed": seed, "seed_used": seed_used, "n_spots": a.n_obs,
           "true_purity": round(a.uns["purity"], 4), "aa_purity": round(float(W.max(1).mean()), 4),
           "labels_correct": int(np.array_equal(oracle_arch, np.arange(len(STATES)))),
           "w_r_mean": round(float(np.diag(r).mean()), 4),
           "w_mae": round(float(np.abs(W - W_true).mean()), 4)}
    for k, s in enumerate(STATES):
        row[f"w_r_{s}"] = round(float(r[k, k]), 4)
        row[f"profile_r_{s}"] = round(float(np.corrcoef(rel_aa[:, k], rel_true[:, k])[0, 1]), 4)
    for m, H in rec.items():
        row[f"{m}_case"] = case_name(H, ROOT)
        row[f"{m}_rf_rooted"] = round(rf_rooted(G, H, ROOT), 4)
        row[f"{m}_exact"] = int(row[f"{m}_rf_rooted"] == 0)
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
    print(f"{len(jobs)} runs to do ({len(done)} already in {out.name})", flush=True)
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
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out, mode="a", header=header, index=False)
            header = False
            finished += 1
            if finished % 50 == 0 or finished == len(jobs):
                print(f"  {finished}/{len(jobs)} done", flush=True)


if __name__ == "__main__":
    main()
