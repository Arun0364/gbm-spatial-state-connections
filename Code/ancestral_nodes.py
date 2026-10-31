import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import re
import sys
import json
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import networkx as nx

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import neighbor_joining, euclid, zscore_cols, all_rooted_trees, case_name

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/ancestral_nodes"
SAMPLES = ["G4_ST2", "G740_ST1"]
OBS     = ["NPC", "AC", "OPC", "MES"]
HIDDEN  = "H"


def node_distances(D, labels):
    T = neighbor_joining(D, labels)
    total = sum(d["length"] for _, _, d in T.edges(data=True))
    out = []
    for v in T.nodes:
        if v in labels:
            continue
        dist = nx.single_source_dijkstra_path_length(T, v, weight="length")
        near = min(labels, key=lambda s: dist[s])
        out.append({"node": v, "nearest_state": near, "distance": dist[near], "fraction": dist[near] / total if total > 0 else 0.0})
    return out, total


def hidden_score(D, labels):
    nodes, total = node_distances(D, labels)
    far = max(nodes, key=lambda r: r["fraction"])
    return far["fraction"], far["nearest_state"], nodes, total


def real(n_boot, seed=0):
    rng = np.random.default_rng(seed)
    rows, boots, nodes_out = [], [], []
    for s in SAMPLES:
        df = pd.read_csv(BASE / f"Results/phylogeny_clone_lineage/{s}_arch_centroids.csv", index_col=0)
        labels = [re.search(r"\((\w+)\)", i).group(1) for i in df.index]
        X = df.values.astype(float)
        score, near, nodes, total = hidden_score(euclid(zscore_cols(X)), labels)
        rows.append({"sample": s, "hidden_score": score, "farthest_node_nearest_state": near, "tree_length": total})
        for n in nodes:
            nodes_out.append({"sample": s, **n})
        for b in range(n_boot):
            Xb = X[:, rng.integers(0, X.shape[1], X.shape[1])]
            boots.append({"sample": s, "boot": b, "hidden_score": hidden_score(euclid(zscore_cols(Xb)), labels)[0]})
    return pd.DataFrame(rows), pd.DataFrame(nodes_out), pd.DataFrame(boots)


def sim_one(kind, case, edges, seed, cond):
    warnings.filterwarnings("ignore")
    from spatial_coalescent_sim import simulate
    G = nx.DiGraph(edges)
    states = OBS if kind == "observed" else [HIDDEN] + OBS
    root = "NPC" if kind == "observed" else HIDDEN
    extra = {"mix": 0.75} if cond == "Real purity" else {}
    for attempt in range(20):
        seed_used = 7000000 + seed + 10000 * attempt + 100 * (hash(case) % 997)
        a, _, state = simulate(G, seed_used, states=states, root=root, **extra)
        if (np.bincount(state, minlength=len(states)) / len(state)).min() >= 0.02:
            break
    markers = json.loads(a.uns["state_markers"])
    genes = sorted(set(sum((markers[s] for s in OBS), [])))
    Xg = a[:, genes].layers["norm_counts"].toarray()
    W = a.obs[[f"W{states.index(s)}" for s in OBS]].values
    cent = (W.T @ Xg) / W.sum(0)[:, None]
    score, near, _, _ = hidden_score(euclid(zscore_cols(cent)), OBS)
    return {"kind": kind, "condition": cond, "case": case, "seed": seed, "hidden_score": score, "nearest_state": near,
            "hidden_children": G.out_degree(HIDDEN) if kind == "hidden" else 0}


def simulate_all(seeds, n_hidden_trees, workers, seed=1):
    rng = np.random.default_rng(seed)
    obs_trees = {case_name(G, "NPC"): list(G.edges) for G in all_rooted_trees(OBS, "NPC")}
    hid = [G for G in all_rooted_trees([HIDDEN] + OBS, HIDDEN) if G.out_degree(HIDDEN) >= 2]
    pick = rng.choice(len(hid), min(n_hidden_trees, len(hid)), replace=False)
    hid_trees = {"H>" + ";".join(sorted(f"{u}-{v}" for u, v in hid[i].edges)): list(hid[i].edges) for i in pick}
    jobs = [("observed", n, e, s, c) for c in ("Clean", "Real purity") for n, e in obs_trees.items() for s in range(1, seeds + 1)]
    jobs += [("hidden", n, e, s, c) for c in ("Clean", "Real purity") for n, e in hid_trees.items() for s in range(1, seeds + 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(sim_one, *zip(*jobs), chunksize=4))), len(hid)


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--hidden-trees", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, nodes, boots = real(args.boot)
    rows.to_csv(out / "real_hidden_score.csv", index=False)
    nodes.to_csv(out / "real_internal_nodes.csv", index=False)
    boots.to_csv(out / "real_bootstrap.csv", index=False)
    print("REAL: internal nodes of the NJ tree (distance to nearest observed state, as a fraction of tree length)")
    print(nodes.round(3).to_string(index=False))
    print(rows.round(3).to_string(index=False))
    print(boots.groupby("sample")["hidden_score"].quantile([0.025, 0.5, 0.975]).round(3).to_string())
    sim, n_possible = simulate_all(args.seeds, args.hidden_trees, args.workers)
    sim.to_csv(out / "sim_hidden_score.csv", index=False)
    print(f"\nSIMULATION ({n_possible} possible trees with a hidden ancestor of 2 or more states; sampled {args.hidden_trees})")
    for c, g in sim.groupby("condition"):
        neg, pos = g[g.kind == "observed"].hidden_score, g[g.kind == "hidden"].hidden_score
        print(f"{c}: all ancestors observed median {neg.median():.3f} (95th pct {neg.quantile(0.95):.3f}); "
              f"hidden ancestor median {pos.median():.3f} (5th pct {pos.quantile(0.05):.3f}); AUC {auc(neg, pos):.3f}")
        for r in rows.itertuples():
            print(f"   {r.sample}: score {r.hidden_score:.3f} is above {100 * (neg < r.hidden_score).mean():.0f}% of observed-ancestor runs "
                  f"and above {100 * (pos < r.hidden_score).mean():.0f}% of hidden-ancestor runs")


if __name__ == "__main__":
    main()
