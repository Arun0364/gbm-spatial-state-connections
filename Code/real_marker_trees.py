import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import ast
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import (nj_bootstrap, nj_state_graph, patristic, greedy_arborescence, optimal_arborescence,
                          rank_arborescences, that_mst, case_name, compare)
from marker_weights import weights_gene_sd as weights_nnls

BASE    = Path("/Users/arun/Desktop/Research")
AA_DIR  = BASE / "Results/PostAA_tumor_unmix_k4_markers"
INT_DIR = BASE / "Results/archetype_mimo_integration_k4_markers"
OUT_DIR = BASE / "Results/real_marker_trees_v2"
SAMPLES = ["G4_ST2", "G740_ST1"]
STATES  = ["NPC", "AC", "OPC", "MES"]
ROOT    = "NPC"
N_BOOT  = 1000
warnings.filterwarnings("ignore")


def neftel_markers():
    tree = ast.parse((CODE / "export_clone_centroids_with_states.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "STATE_MARKERS")
    return ast.literal_eval(node.value)


def load(sample):
    sdir = AA_DIR / sample
    ad = sc.read_h5ad(sdir / f"{sample}_ns_annotated.h5ad")
    ad.obs["barcode_key"] = [re.sub(r"-\d+$", "", str(b)) for b in ad.obs_names]
    merged = pd.read_csv(INT_DIR / f"{sample}_arch_mimo_merged.csv")
    lab = pd.read_csv(sdir / f"{sample}_archetype_state_labels.csv")
    arch_to_state = dict(zip(lab["archetype"].astype(int), lab["dominant_state"]))
    obs = ad.obs[["barcode_key"]].merge(merged[["barcode_key", "W0", "W1", "W2", "W3"]], on="barcode_key", how="left")
    keep = ~obs["W0"].isna().values
    ad = ad[keep].copy()
    W_arch = obs.loc[keep, ["W0", "W1", "W2", "W3"]].values
    order = [next(k for k, s in arch_to_state.items() if s == st) for st in STATES]
    W_aa = W_arch[:, order]
    dpt_genes = [g for g in pd.read_csv(sdir / f"{sample}_archetype_markers.csv")["gene"].unique() if g in ad.var_names]
    return ad, W_aa, dpt_genes


def norm_counts(ad, genes):
    M = ad[:, genes].layers["norm_counts"]
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def centroids(X, W):
    return np.vstack([(X * W[:, k][:, None]).sum(0) / W[:, k].sum() for k in range(W.shape[1])])


def that_matrix(ad, genes, W):
    am = ad[:, genes].copy()
    am.X = norm_counts(ad, genes)
    sc.pp.scale(am, max_value=10)
    sc.tl.pca(am, n_comps=20, svd_solver="arpack")
    sc.pp.neighbors(am, n_neighbors=15, n_pcs=15)
    sc.tl.diffmap(am, n_comps=15)
    am.uns["iroot"] = int(np.argmax(W[:, STATES.index(ROOT)]))
    sc.tl.dpt(am, n_dcs=10)
    dpt = am.obs["dpt_pseudotime"].values
    k = W.argmax(1)
    coo = am.obsp["connectivities"].tocoo()
    up = dpt[coo.col] > dpt[coo.row]
    T = np.zeros((len(STATES), len(STATES)))
    np.add.at(T, (k[coo.row[up]], k[coo.col[up]]), 1.0)
    return T / np.where(T.sum(1, keepdims=True) > 0, T.sum(1, keepdims=True), 1)


def trees(ad, W, nj_genes, dpt_genes):
    X = norm_counts(ad, nj_genes)
    T0, split_support, edge_support = nj_bootstrap(centroids(X, W), STATES, ROOT, B=N_BOOT)
    nj = nj_state_graph(T0, STATES, ROOT)
    That = that_matrix(ad, dpt_genes, W)
    ranked = rank_arborescences(That, STATES, ROOT)
    out = {"NJ": nj, "T-hat greedy": greedy_arborescence(That, STATES, ROOT),
           "T-hat most probable": optimal_arborescence(That, STATES, ROOT), "T-hat MST": that_mst(That, STATES, ROOT)[0]}
    support = {f"{a}-{b}": round(edge_support[frozenset((a, b))] * N_BOOT) for a, b in nj.edges}
    return out, That, ranked, support


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    summary, weights, ranks, thats = [], [], [], []
    for sample in SAMPLES:
        ad, W_aa, dpt_genes = load(sample)
        nj_genes = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
        present = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
        W_mk = weights_nnls(norm_counts(ad, nj_genes), nj_genes, present)
        print(f"{sample}: {len(ad)} spots; NJ genes {len(nj_genes)}; markers per state "
              + ", ".join(f"{s} {len(present[s])}" for s in STATES))
        res = {}
        for wn, W in [("archetype", W_aa), ("marker NNLS", W_mk)]:
            tr, That, ranked, support = trees(ad, W, nj_genes, dpt_genes)
            res[wn] = tr
            for m, G in tr.items():
                summary.append({"sample": sample, "weights": wn, "method": m, "case": case_name(G, ROOT),
                                "edges": "; ".join(f"{a}>{b}" for a, b in sorted(G.edges)),
                                "nj_support": "; ".join(f"{e} {v}/{N_BOOT}" for e, v in support.items()) if m == "NJ" else ""})
            top = ranked[0][1]
            for rank, (name, prob, _) in enumerate(ranked[:5], 1):
                ranks.append({"sample": sample, "weights": wn, "rank": rank, "case": name, "relative_to_best": round(prob / top, 3)})
            thats.append(pd.DataFrame(That, index=STATES, columns=STATES).round(4).assign(sample=sample, weights=wn))
        dom_aa, dom_mk = W_aa.argmax(1), W_mk.argmax(1)
        for k, s in enumerate(STATES):
            weights.append({"sample": sample, "state": s, "r_archetype_vs_marker": round(float(np.corrcoef(W_aa[:, k], W_mk[:, k])[0, 1]), 3),
                            "dominant_archetype": int((dom_aa == k).sum()), "dominant_marker": int((dom_mk == k).sum()),
                            "mean_weight_archetype": round(float(W_aa[:, k].mean()), 3), "mean_weight_marker": round(float(W_mk[:, k].mean()), 3)})
        weights.append({"sample": sample, "state": "purity (mean max weight)", "mean_weight_archetype": round(float(W_aa.max(1).mean()), 3),
                        "mean_weight_marker": round(float(W_mk.max(1).mean()), 3)})
        for m in res["archetype"]:
            c = compare(res["archetype"][m], res["marker NNLS"][m], ROOT)
            print(f"  {m:20s} archetype {case_name(res['archetype'][m], ROOT):18s} marker {case_name(res['marker NNLS'][m], ROOT):18s} {c}")
    pd.DataFrame(summary).to_csv(OUT_DIR / "real_marker_tree_summary.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT_DIR / "real_marker_weights.csv", index=False)
    pd.DataFrame(ranks).to_csv(OUT_DIR / "real_marker_arborescence_ranking.csv", index=False)
    pd.concat(thats).to_csv(OUT_DIR / "real_marker_that_matrices.csv")


if __name__ == "__main__":
    main()