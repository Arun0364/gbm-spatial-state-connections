import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import re
import sys
import itertools
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import (neighbor_joining, nj_state_graph, distance_state_graph, euclid, zscore_cols,
                          all_rooted_trees, case_name, rf_rooted)

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/three_point"
SAMPLES = ["G4_ST2", "G740_ST1"]
ROOT    = "NPC"


def three_point(D):
    devs = []
    for i, j, k in itertools.combinations(range(len(D)), 3):
        d = np.sort([D[i, j], D[i, k], D[j, k]])
        devs.append((d[2] - d[1]) / d[2])
    return float(np.mean(devs)), float(np.max(devs))


def four_point(D):
    i, j, k, l = range(4)
    s = np.sort([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])
    return float((s[2] - s[1]) / s[2])


def upgma_state_graph(D, labels, root):
    Z = linkage(squareform(D, checks=False), method="average")
    coph = squareform(cophenet(Z))
    return distance_state_graph(coph, labels, root)


def describe(D, labels):
    nj = nj_state_graph(neighbor_joining(D, labels), labels, ROOT)
    up = upgma_state_graph(D, labels, ROOT)
    tp_mean, tp_max = three_point(D)
    return {"three_point_mean": tp_mean, "three_point_max": tp_max, "four_point": four_point(D),
            "nj": case_name(nj, ROOT), "upgma": case_name(up, ROOT), "nj_vs_upgma_rf": rf_rooted(nj, up, ROOT)}


def real(n_boot, seed=0):
    rows, boots = [], []
    rng = np.random.default_rng(seed)
    for s in SAMPLES:
        df = pd.read_csv(BASE / f"Results/phylogeny_clone_lineage/{s}_arch_centroids.csv", index_col=0)
        labels = [re.search(r"\((\w+)\)", i).group(1) for i in df.index]
        X = df.values.astype(float)
        rows.append({"sample": s, "n_genes": X.shape[1], **describe(euclid(zscore_cols(X)), labels)})
        for b in range(n_boot):
            Xb = X[:, rng.integers(0, X.shape[1], X.shape[1])]
            boots.append({"sample": s, "boot": b, **describe(euclid(zscore_cols(Xb)), labels)})
    return pd.DataFrame(rows), pd.DataFrame(boots)


def sim_one(case, edges, seed, offset, cond):
    warnings.filterwarnings("ignore")
    import networkx as nx
    from spatial_coalescent_sim import simulate, STATES
    from sim_aa_run import aa_weights
    G = nx.DiGraph(edges)
    for attempt in range(20):
        seed_used = offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, **({"mix": 0.75} if cond == "Real purity" else {}))
        if (np.bincount(state, minlength=4) / len(state)).min() >= 0.02:
            break
    out = []
    for wname in ("true", "unmix"):
        W = a.obs[[f"W{k}" for k in range(4)]].values if wname == "true" else aa_weights(a, STATES, seed_used)[0]
        genes = list(a.uns["marker_genes"])
        Xg = a[:, genes].layers["norm_counts"].toarray()
        cent = (W.T @ Xg) / W.sum(0)[:, None]
        r = describe(euclid(zscore_cols(cent)), STATES)
        out.append({"condition": cond, "weights": wname, "case": case, "seed": seed, **r,
                    "nj_exact": int(r["nj"] == case), "upgma_exact": int(r["upgma"] == case)})
    return out


def simulate_all(seeds, workers):
    from spatial_coalescent_sim import STATES
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(trees)
    jobs = [(n, trees[n], s, 1000000 * (names.index(n) + 1), c)
            for c in ("Clean", "Real purity") for n in names for s in range(1, seeds + 1)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(sim_one, *zip(*jobs)):
            rows += res
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    point, boots = real(args.boot)
    point.to_csv(out / "real_conditions.csv", index=False)
    boots.to_csv(out / "real_bootstrap.csv", index=False)
    print("REAL SECTIONS")
    print(point.round(3).to_string(index=False))
    q = boots.groupby("sample")[["three_point_mean", "four_point"]].quantile([0.025, 0.975]).round(3)
    print(q.to_string())
    print(boots.groupby("sample")["upgma"].value_counts(normalize=True).round(3).to_string())
    sim = simulate_all(args.seeds, args.workers)
    sim.to_csv(out / "sim_conditions.csv", index=False)
    print("\nSIMULATION (true tree known)")
    print(sim.groupby(["condition", "weights"])[["three_point_mean", "four_point", "nj_exact", "upgma_exact"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
