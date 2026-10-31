import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import (neighbor_joining, euclid, zscore_cols, nj_state_graph, greedy_arborescence,
                          optimal_arborescence, that_mst)
from marker_weights import weights_gene_sd as weights_nnls
from real_marker_trees import BASE, SAMPLES, STATES, neftel_markers, load, norm_counts, centroids
from real_marker_sensitivity import Pseudotime, edges_of, KEY
from real_markerfree_trees import markerfree_weights

OUT_DIR = BASE / "Results/real_marker_trees_v2"
warnings.filterwarnings("ignore")


class RootedPseudotime(Pseudotime):
    def that_from(self, W, root):
        iroot = int(np.argmax(W[:, STATES.index(root)]))
        if iroot not in self.cache:
            self.am.uns["iroot"] = iroot
            sc.tl.dpt(self.am, n_dcs=10)
            self.cache[iroot] = self.am.obs["dpt_pseudotime"].values.copy()
        dpt = self.cache[iroot]
        k = W.argmax(1)
        up = dpt[self.col] > dpt[self.row]
        T = np.zeros((len(STATES), len(STATES)))
        np.add.at(T, (k[self.row[up]], k[self.col[up]]), 1.0)
        return T / np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    rows = []
    for sample in SAMPLES:
        ad, W_aa, dpt_genes = load(sample)
        pt = RootedPseudotime(ad, dpt_genes)
        nj_genes = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
        X = norm_counts(ad, nj_genes)
        present = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
        Wdf, _, _ = markerfree_weights(sample, markers)
        Ws = {"archetype": W_aa, "marker NNLS": weights_nnls(X, nj_genes, present),
              "marker-free": Wdf.loc[ad.obs["barcode_key"].values].values}
        for wn, W in Ws.items():
            T0 = neighbor_joining(euclid(zscore_cols(centroids(X, W))), STATES)
            for root in STATES:
                That = pt.that_from(W, root)
                trees = {"NJ": nj_state_graph(T0, STATES, root),
                         "T-hat greedy": greedy_arborescence(That, STATES, root),
                         "T-hat most probable": optimal_arborescence(That, STATES, root),
                         "T-hat MST": that_mst(That, STATES, root)[0]}
                for m, G in trees.items():
                    rows.append({"sample": sample, "root": root, "weights": wn, "method": m,
                                 "edges": "; ".join(sorted(f"{a}>{b}" for a, b in G.edges)),
                                 **{e: int(e in edges_of(G)) for e in KEY[sample]}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "real_root_check.csv", index=False)
    out = []
    for sample in SAMPLES:
        d = df[df["sample"] == sample]
        for root in STATES:
            r = d[d["root"] == root]
            out.append({"sample": sample, "root": root, "trees": len(r),
                        **{f"{e} (of {len(r)})": int(r[e].sum()) for e in KEY[sample]},
                        **{f"{e}, T-hat only (of {int((r['method'] != 'NJ').sum())})": int(r.loc[r["method"] != "NJ", e].sum()) for e in KEY[sample]}})
    summ = pd.DataFrame(out)
    summ.to_csv(OUT_DIR / "real_root_check_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()