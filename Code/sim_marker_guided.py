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
from tree_compare import all_rooted_trees, case_name, optimal_arborescence, greedy_arborescence, that_mst
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_grid_run import nj_graph, MIN_STATE_FRACTION, MAX_REDRAWS
from sim_improve_run import marker_matrix
from sim_improve2 import dpt_graph, that
from marker_weights import weights_gene_sd

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/marker_guided/marker_guided_sim"
METHODS = ["NJ", "That_greedy", "That_arb", "That_mst"]


def undirected(G):
    return sorted("-".join(sorted(e)) for e in G.to_undirected().edges)


def one_run(case, edges, seed, offset):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, mix=0.75)
        if (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= MIN_STATE_FRACTION:
            break
    X, genes = marker_matrix(a)
    W = weights_gene_sd(X, genes, json.loads(a.uns["state_markers"]), STATES)
    dpt, conn = dpt_graph(a, genes, int(np.argmax(W[:, STATES.index(ROOT)])), {})
    T = that(dpt, conn, W, False)
    trees = {"NJ": nj_graph(a, W, genes, STATES, ROOT), "That_greedy": greedy_arborescence(T, STATES, ROOT),
             "That_arb": optimal_arborescence(T, STATES, ROOT), "That_mst": that_mst(T, STATES, ROOT)[0]}
    row = {"case": case, "seed": seed, "true_edges": ";".join(undirected(G))}
    for m, H in trees.items():
        row[f"{m}_case"] = case_name(H, ROOT)
        row[f"{m}_edges"] = ";".join(undirected(H))
    return row


def precision(df, mask, label):
    out = []
    for name, sel in [("NJ and most probable both", lambda nj, ar: nj & ar), ("NJ only", lambda nj, ar: nj - ar),
                      ("most probable only", lambda nj, ar: ar - nj), ("any NJ edge", lambda nj, ar: nj),
                      ("any most probable edge", lambda nj, ar: ar)]:
        hits = total = 0
        for r in df[mask].itertuples():
            true = set(r.true_edges.split(";"))
            edges = sel(set(r.NJ_edges.split(";")), set(r.That_arb_edges.split(";")))
            total += len(edges)
            hits += len(edges & true)
        out.append({"subset": label, "tissues": int(mask.sum()), "edges_reported_by": name, "edges": total,
                    "precision": round(hits / total, 3) if total else np.nan})
    return out


def summarize(df, out_dir):
    df = df.copy()
    df["true_case"] = df["case"]
    for m in METHODS:
        df[f"{m}_exact"] = (df[f"{m}_case"] == df["true_case"]).astype(int)
    df["disagree_nj_arb"] = (df["NJ_edges"] != df["That_arb_edges"]).astype(int)
    df["disagree_nj_greedy"] = (df["NJ_edges"] != df["That_greedy_edges"]).astype(int)
    df["disagree_nj_both"] = df["disagree_nj_arb"] & df["disagree_nj_greedy"]
    overview = {"tissues": len(df), **{f"{m}_exact_pct": round(100 * df[f"{m}_exact"].mean(), 1) for m in METHODS},
                "NJ_vs_most_probable_disagree_pct": round(100 * df["disagree_nj_arb"].mean(), 1),
                "NJ_vs_greedy_disagree_pct": round(100 * df["disagree_nj_greedy"].mean(), 1),
                "NJ_vs_both_That_disagree_pct": round(100 * df["disagree_nj_both"].mean(), 1)}
    rows = precision(df, np.ones(len(df), bool), "all tissues")
    rows += precision(df, df["disagree_nj_arb"].astype(bool).values, "NJ and most probable disagree")
    rows += precision(df, df["disagree_nj_both"].astype(bool).values, "NJ disagrees with both T-hat trees (as in G4 and G740)")
    prec = pd.DataFrame(rows)
    shape = df.assign(shape=df["case"].str.split("-").str[0]).groupby("shape")[["disagree_nj_arb", "NJ_exact", "That_arb_exact"]].mean().mul(100).round(1)
    pd.DataFrame([overview]).to_csv(out_dir / "overview.csv", index=False)
    prec.to_csv(out_dir / "precision.csv", index=False)
    shape.to_csv(out_dir / "by_shape.csv")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60):
        print(pd.DataFrame([overview]).T.to_string(header=False))
        print(prec.to_string(index=False))
        print(shape.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / "marker_guided_trees.csv"
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = set(zip(prev.case, prev.seed))
    tr = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(tr)
    cases = args.cases or names
    jobs = [(n, tr[n], s, 1000000 * (names.index(n) + 1)) for n in cases for s in range(1, args.seeds + 1) if (n, s) not in done]
    print(f"{len(jobs)} runs to do", flush=True)
    header = not csv.exists()
    k = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, *j): j for j in jobs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as err:
                print("FAILED", futs[f][:2], err, flush=True)
                continue
            pd.DataFrame([row]).to_csv(csv, mode="a", header=header, index=False)
            header = False
            k += 1
            if k % 400 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done", flush=True)
    summarize(pd.read_csv(csv), out)


if __name__ == "__main__":
    main()