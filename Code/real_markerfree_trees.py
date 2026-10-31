import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import case_name, compare
from sim_improve_run import label_archetypes
from marker_weights import weights_gene_sd as weights_nnls
from sim_refine import refine_pure
from unmix import unmix
from real_marker_trees import (BASE, AA_DIR, SAMPLES, STATES, ROOT, N_BOOT, neftel_markers, load, norm_counts, trees)
import scanpy as sc

OUT_DIR = BASE / "Results/real_marker_trees_v2"
N_RUNS = 5
warnings.filterwarnings("ignore")


def tumor_spots(sample):
    import re
    tn = pd.read_csv(BASE / f"Results/TumorNormalClassification/{sample}_tumor_normal_labels.csv")
    tumor = set(tn.loc[tn["tumor_label"] == "Tumor", "barcode"])
    a = sc.read_h5ad(AA_DIR / sample / f"{sample}_ns_annotated.h5ad")
    a.obs["barcode_key"] = [re.sub(r"-\d+$", "", str(b)) for b in a.obs_names]
    return a[a.obs["barcode_key"].isin(tumor).values].copy()


def markerfree_weights(sample, markers):
    a = tumor_spots(sample)
    genes = list(pd.read_csv(AA_DIR / sample / f"{sample}_C_matrix.csv", index_col=0).index)
    X = norm_counts(a, genes)
    present = {s: [g for g in markers[s] if g in genes] for s in STATES}
    runs = []
    for r in range(N_RUNS):
        C, _, F = unmix(X.T, len(STATES), restarts=5, seed=101 * r)
        runs.append(F[:, label_archetypes(C, genes, present)])
    W_ens = np.mean(runs, axis=0)
    W = refine_pure(X, W_ens, genes, present)
    return pd.DataFrame(W, index=a.obs["barcode_key"].values, columns=STATES), runs, len(a)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    summary, compare_rows, weights = [], [], []
    for sample in SAMPLES:
        ad, W_aa, dpt_genes = load(sample)
        nj_genes = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
        present = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
        W_mk = weights_nnls(norm_counts(ad, nj_genes), nj_genes, present)
        Wdf, runs, n_tumor = markerfree_weights(sample, markers)
        W_mf = Wdf.loc[ad.obs["barcode_key"].values].values
        print(f"{sample}: UNMIX on {n_tumor} tumor spots, trees on {len(ad)} spots")
        res = {}
        for wn, W in [("archetype", W_aa), ("marker NNLS", W_mk), ("marker-free", W_mf)]:
            tr, That, ranked, support = trees(ad, W, nj_genes, dpt_genes)
            res[wn] = tr
            for m, G in tr.items():
                summary.append({"sample": sample, "weights": wn, "method": m, "case": case_name(G, ROOT),
                                "edges": "; ".join(f"{a}>{b}" for a, b in sorted(G.edges)),
                                "nj_support": "; ".join(f"{e} {v}/{N_BOOT}" for e, v in support.items()) if m == "NJ" else "",
                                "next_best_relative": round(ranked[1][1] / ranked[0][1], 3) if m == "T-hat most probable" else ""})
        for k, s in enumerate(STATES):
            weights.append({"sample": sample, "state": s,
                            "r_marker_free_vs_archetype": round(float(np.corrcoef(W_mf[:, k], W_aa[:, k])[0, 1]), 3),
                            "r_marker_free_vs_marker": round(float(np.corrcoef(W_mf[:, k], W_mk[:, k])[0, 1]), 3),
                            "dominant_archetype": int((W_aa.argmax(1) == k).sum()), "dominant_marker": int((W_mk.argmax(1) == k).sum()),
                            "dominant_marker_free": int((W_mf.argmax(1) == k).sum())})
        for m in res["archetype"]:
            mk_vs_mf = compare(res["marker NNLS"][m], res["marker-free"][m], ROOT)
            compare_rows.append({"sample": sample, "method": m,
                                 "archetype": case_name(res["archetype"][m], ROOT),
                                 "marker NNLS": case_name(res["marker NNLS"][m], ROOT),
                                 "marker-free": case_name(res["marker-free"][m], ROOT),
                                 "marker_vs_markerfree_rf": mk_vs_mf["rf_rooted"],
                                 "shared_marker_markerfree": mk_vs_mf["shared_edges"]})
            print(f"  {m:20s} archetype {compare_rows[-1]['archetype']:18s} marker {compare_rows[-1]['marker NNLS']:18s} "
                  f"marker-free {compare_rows[-1]['marker-free']:18s} shared(marker, free): {mk_vs_mf['shared_edges']}")
    pd.DataFrame(summary).to_csv(OUT_DIR / "real_markerfree_tree_summary.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(OUT_DIR / "real_three_way_comparison.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT_DIR / "real_markerfree_weights.csv", index=False)


if __name__ == "__main__":
    main()