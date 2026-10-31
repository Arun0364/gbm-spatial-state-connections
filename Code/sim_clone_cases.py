import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import argparse
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import all_rooted_trees, case_name
from spatial_coalescent_sim import simulate, STATES, ROOT
from sim_aa_run import aa_weights

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/sim_clones/cases"

CONDITIONS = {
    "Clones-clean":       {"n_clones": 4, "clone_window": (3, 4)},
    "Clones-real-purity": {"n_clones": 4, "clone_window": (3, 4), "mix": 0.75},
}
N_PSEUDOBULK       = 300
K_NEIGHBORS        = 15
BULKS_PER_CASE     = 10
GENES_PER_INTERVAL = 10
MIN_FRACTION       = 0.05
MAX_REDRAWS        = 20


def pseudobulks(xy, rng):
    anchors = rng.choice(len(xy), N_PSEUDOBULK, replace=False)
    return cKDTree(xy).query(xy[anchors], k=K_NEIGHBORS)[1]


def to_intervals(X):
    n = X.shape[1] // GENES_PER_INTERVAL
    return X[:, :n * GENES_PER_INTERVAL].reshape(len(X), n, GENES_PER_INTERVAL).mean(2)


def build(cond, case, edges, seed, tree_offset, out_dir):
    warnings.filterwarnings("ignore")
    import networkx as nx
    G = nx.DiGraph(edges)
    for attempt in range(MAX_REDRAWS):
        seed_used = tree_offset + seed + 10000 * attempt
        a, _, state = simulate(G, seed_used, **CONDITIONS[cond])
        ok_states = (np.bincount(state, minlength=len(STATES)) / len(state)).min() >= 0.02
        f = a.uns.get("clone_fraction_cells", np.zeros(1))
        if ok_states and len(f) == 4 and f.min() >= MIN_FRACTION:
            break
    else:
        raise RuntimeError("could not draw a tissue with four clones and four states")
    rng = np.random.default_rng(seed_used + 5)
    W, _, _ = aa_weights(a, STATES, seed_used)
    counts = a.layers["counts"].toarray().astype(float)
    mem = pseudobulks(a.obsm["spatial"], rng)
    tumor = to_intervals(np.stack([counts[m].mean(0) for m in mem]))
    fref = np.stack([W[m].mean(0) for m in mem])
    fref = fref / fref.sum(1, keepdims=True)
    cref = to_intervals((W.T @ counts) / W.sum(0)[:, None])
    order = rng.permutation(N_PSEUDOBULK)
    out = Path(out_dir) / cond / f"{case}__s{seed:03d}"
    out.mkdir(parents=True, exist_ok=True)
    for c in range(N_PSEUDOBULK // BULKS_PER_CASE):
        idx = order[c * BULKS_PER_CASE:(c + 1) * BULKS_PER_CASE]
        np.savez(out / f"case_{c:02d}.npz", TumorSample=tumor[idx].T, CRefer=cref.T, FRefer=fref[idx].T, bulks=idx)
    Q = a.obs[[f"Q{c}" for c in range(4)]].values
    W_true = a.obs[[f"W{k}" for k in range(len(STATES))]].values
    np.savez(out / "truth.npz", members=mem, Q=Q, W_true=W_true, W_aa=W, clone_cn=a.uns["clone_cn"],
             clone_parent=np.array(json.loads(a.uns["clone_parent"])))
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--trees", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    trees = {case_name(G, ROOT): list(G.edges) for G in all_rooted_trees(STATES, ROOT)}
    names = sorted(trees)
    use = args.trees or names
    jobs = [(c, n, trees[n], s, 1000000 * (names.index(n) + 1), str(out_dir)) for c in args.conditions for n in use
            for s in range(1, args.seeds + 1) if not (out_dir / c / f"{n}__s{s:03d}" / "truth.npz").exists()]
    print(f"{len(jobs)} tissues to build", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as err:
                print(f"FAILED {futs[f][:2]} seed {futs[f][3]}: {err}", flush=True)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} built", flush=True)


if __name__ == "__main__":
    main()
