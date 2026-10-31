import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import (all_rooted_trees, case_name, rf_rooted, rf_unrooted, neighbor_joining, euclid,
                          zscore_cols, nj_state_graph, optimal_arborescence, greedy_arborescence, that_mst)
from spatial_coalescent_sim import simulate, STATES, ROOT

BASE    = Path("/Users/arun/Desktop/Research")
OUT_CSV = BASE / "Results/sim_grid/grid_results.csv"

CONDITIONS = {
    "Baseline":             {},
    "Mixed":                {"mix": 0.4},
    "Real-purity":          {"mix": 0.75},
    "Similar-states":       {"sigma_tree": 0.3},
    "Distinct-states":      {"sigma_tree": 1.0},
    "Slow-switching":       {"switch_rate": 0.02},
    "Fast-switching":       {"switch_rate": 0.08},
    "Data-genes":           {},
    "Noisy-W":              {},
    "Real-W-purity":        {"mix": 0.75},
    "Realistic":            {"mix": 0.75},
    "Coalescent":           {"lineage": "coalescent"},
    "Coalescent-realistic": {"lineage": "coalescent", "mix": 0.75},
    "Coalescent-constant":  {"lineage": "coalescent", "coal_growth": 0.0},
}

RECOVERY = {
    "Data-genes":           {"genes": "hvg"},
    "Noisy-W":              {"w_noise": 0.5},
    "Real-W-purity":        {"w_noise": 0.5},
    "Realistic":            {"genes": "hvg", "w_noise": 0.5},
    "Coalescent-realistic": {"genes": "hvg", "w_noise": 0.5},
}

MIN_STATE_FRACTION = 0.02
MAX_REDRAWS = 20

METHODS = ["NJ", "That_arb", "That_mst", "That_greedy", "That_w_arb", "That_w_mst"]
COLUMNS = (["condition", "case", "seed", "seed_used", "redraws", "states_dominant_somewhere", "n_spots",
            "purity", "w_purity", "true_marker_overlap", "moran_mean"] +
           [f"{m}_{x}" for m in METHODS for x in ("case", "rf_rooted", "rf_unrooted", "exact")])


def observed_inputs(a, states, opts, seed):
    W = a.obs[[f"W{k}" for k in range(len(states))]].values
    if opts.get("w_noise", 0) > 0:
        rng = np.random.default_rng(seed + 7)
        eps = opts["w_noise"]
        W = (1 - eps) * W + eps * rng.dirichlet(np.ones(len(states)), size=len(W))
    genes = list(a.uns["marker_genes"])
    if opts.get("genes") == "hvg":
        X = a.layers["norm_counts"].toarray()
        genes = list(a.var_names[np.argsort(X.var(0))[::-1][:len(genes)]])
    return W, genes


def dpt_and_graph(a, W, genes, states, root):
    import scanpy as sc
    am = a[:, genes].copy()
    am.X = am.layers["norm_counts"].toarray()
    sc.pp.scale(am, max_value=10)
    sc.tl.pca(am, n_comps=20, svd_solver="arpack")
    sc.pp.neighbors(am, n_neighbors=15, n_pcs=15)
    sc.tl.diffmap(am, n_comps=15)
    am.uns["iroot"] = int(np.argmax(W[:, states.index(root)]))
    sc.tl.dpt(am, n_dcs=10)
    return am.obs["dpt_pseudotime"].values, am.obsp["connectivities"].tocsr()


def that_matrices(dpt, conn, W):
    K = W.shape[1]
    k = W.argmax(1)
    rs = np.asarray(conn.sum(1)).ravel()
    rs[rs == 0] = 1
    coo = conn.tocoo()
    P = coo.data / rs[coo.row]
    up = np.isfinite(dpt[coo.row]) & np.isfinite(dpt[coo.col]) & (dpt[coo.col] > dpt[coo.row])
    Tc = np.zeros((K, K))
    Tw = np.zeros((K, K))
    np.add.at(Tc, (k[coo.row[up]], k[coo.col[up]]), 1.0)
    np.add.at(Tw, (k[coo.row[up]], k[coo.col[up]]), P[up])
    norm = lambda T: T / np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)
    return norm(Tc), norm(Tw)


def nj_graph(a, W, genes, states, root):
    X = a[:, genes].layers["norm_counts"].toarray()
    cent = (W.T @ X) / W.sum(0)[:, None]
    return nj_state_graph(neighbor_joining(euclid(zscore_cols(cent)), states), states, root)


def one_run(cond, case, edges, seed):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, **CONDITIONS[cond])
        if (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= MIN_STATE_FRACTION:
            break
    else:
        raise RuntimeError(f"a state stayed below {MIN_STATE_FRACTION:.0%} of cells in {MAX_REDRAWS} draws")
    W, genes = observed_inputs(a, STATES, RECOVERY.get(cond, {}), seed_used)
    dpt, conn = dpt_and_graph(a, W, genes, STATES, ROOT)
    Tc, Tw = that_matrices(dpt, conn, W)
    rec = {
        "NJ": nj_graph(a, W, genes, STATES, ROOT),
        "That_arb": optimal_arborescence(Tc, STATES, ROOT),
        "That_mst": that_mst(Tc, STATES, ROOT)[0],
        "That_greedy": greedy_arborescence(Tc, STATES, ROOT),
        "That_w_arb": optimal_arborescence(Tw, STATES, ROOT),
        "That_w_mst": that_mst(Tw, STATES, ROOT)[0],
    }
    row = {"condition": cond, "case": case, "seed": seed, "seed_used": seed_used, "redraws": attempt,
           "states_dominant_somewhere": int((np.bincount(W.argmax(1), minlength=len(STATES)) > 0).sum()),
           "n_spots": a.n_obs,
           "purity": round(a.uns["purity"], 4), "w_purity": round(float(W.max(1).mean()), 4),
           "true_marker_overlap": round(len(set(genes) & set(a.uns["marker_genes"])) / len(genes), 3),
           "moran_mean": round(float(np.mean(a.uns["morans_i"])), 4)}
    for m, H in rec.items():
        row[f"{m}_case"] = case_name(H, ROOT)
        row[f"{m}_rf_rooted"] = round(rf_rooted(G, H, ROOT), 4)
        row[f"{m}_rf_unrooted"] = round(rf_unrooted(G, H), 4)
        row[f"{m}_exact"] = int(row[f"{m}_rf_rooted"] == 0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        if list(prev.columns) != COLUMNS:
            prev.reindex(columns=COLUMNS).to_csv(out, index=False)
        done = set(zip(prev.condition, prev.case, prev.seed))
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    jobs = [(c, n, e, s) for c in args.conditions for n, e in sorted(trees.items())
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