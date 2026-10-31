import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, "/Users/arun/Desktop/Research/Code")
warnings.filterwarnings("ignore")
from state_trees import WEIGHTS_DIR, build
from real_marker_trees import AA_DIR, SAMPLES, STATES, ROOT, norm_counts, neftel_markers
from real_root_check import RootedPseudotime
from state_weights import load_tumor, marker_setup
from tree_compare import greedy_arborescence, optimal_arborescence, that_mst, rank_arborescences, case_name, rf_rooted

def edges(G):
    return sorted("-".join(sorted(e)) for e in G.to_undirected().edges)

def spatial_that(dpt, W, xy):
    tree = cKDTree(xy)
    d = np.median(tree.query(xy, k=2)[0][:, 1])
    pairs = np.array(sorted(tree.query_pairs(1.2 * d)))
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    k = W.argmax(1)
    up = dpt[col] > dpt[row]
    T = np.zeros((len(STATES), len(STATES)))
    np.add.at(T, (k[row[up]], k[col[up]]), 1.0)
    deg = np.bincount(row, minlength=len(xy))
    return T / T.sum(1, keepdims=True), deg

markers = neftel_markers()
out = []
for sample in SAMPLES:
    ad, _ = load_tumor(sample)
    wdf = pd.read_csv(WEIGHTS_DIR / f"{sample}_weights.csv")
    assert (wdf["barcode_key"].values == ad.obs["barcode_key"].values).all()
    xy = wdf[["x", "y"]].to_numpy(float)
    genes, base = marker_setup(ad, markers)
    X = norm_counts(ad, genes)
    dpt_genes = [g for g in pd.read_csv(AA_DIR / sample / f"{sample}_archetype_markers.csv")["gene"].unique() if g in ad.var_names]
    pt = RootedPseudotime(ad, dpt_genes)
    for wn in ["marker", "markerfree", "archetype"]:
        W = wdf[[f"{wn}_{s}" for s in STATES]].values
        trees, _, Te = build(pt, W, X, boot=False)
        dpt = pt.cache[int(np.argmax(W[:, STATES.index(ROOT)]))]
        Ts, deg = spatial_that(dpt, W, xy)
        ts = {"greedy": greedy_arborescence(Ts, STATES, ROOT), "most probable": optimal_arborescence(Ts, STATES, ROOT), "MST": that_mst(Ts, STATES, ROOT)[0]}
        te = {"greedy": trees["T-hat greedy"], "most probable": trees["T-hat most probable"], "MST": trees["T-hat MST"]}
        rk = rank_arborescences(Ts, STATES, ROOT)
        for m in ts:
            out.append({"sample": sample, "weights": wn, "tree": m, "expression": case_name(te[m], ROOT), "spatial": case_name(ts[m], ROOT),
                        "RF": round(rf_rooted(te[m], ts[m], ROOT), 2), "spatial_edges": "; ".join(edges(ts[m])), "expr_edges": "; ".join(edges(te[m]))})
        print(f"\n== {sample} {wn}: NJ {case_name(trees['NJ'], ROOT)} | spatial neighbors per spot median {int(np.median(deg))}")
        print("expression T-hat\n", pd.DataFrame(Te, index=STATES, columns=STATES).round(3).to_string())
        print("spatial T-hat\n", pd.DataFrame(Ts, index=STATES, columns=STATES).round(3).to_string())
        print("spatial ranking:", [(n, round(p / rk[0][1], 2)) for n, p, _ in rk[:5]])
df = pd.DataFrame(out)
print("\n", df.to_string(index=False))
df.to_csv(sys.argv[1] + "/spatial_that_summary.csv", index=False)
