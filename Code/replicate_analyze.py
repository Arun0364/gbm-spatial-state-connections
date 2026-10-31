import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import ast
import sys
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import scanpy as sc
from scipy.optimize import linear_sum_assignment

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import case_name
from real_marker_trees import STATES, ROOT, neftel_markers, norm_counts
from real_marker_sensitivity import edges_of
from real_root_check import RootedPseudotime
from sim_improve_run import label_archetypes
from sim_refine import refine_pure
from marker_weights import weights_gene_sd
from unmix import unmix
from state_trees import build

BASE = Path("/Users/arun/Desktop/Research")
RESULTS = BASE / "Results"
ESTIMATES = ["archetype", "marker", "markerfree"]
CONNECTIONS = ["NPC-OPC", "AC-NPC", "MES-NPC", "AC-MES", "AC-OPC", "MES-OPC"]
warnings.filterwarnings("ignore")


def gbm_markers_by_state():
    tree = ast.parse((CODE / "derive_archetype_labels.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "GBM_MARKERS")
    return ast.literal_eval(node.value)


def label_like_original(C, genes):
    df = pd.DataFrame(C, index=genes)
    z = df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1) + 1e-9, axis=0)
    S = np.array([[z.loc[[g for g in gl if g in z.index], k].mean() for s, gl in gbm_markers_by_state().items()] for k in range(C.shape[1])])
    names = list(gbm_markers_by_state())
    greedy = [names[i] for i in np.nanargmax(S, axis=1)]
    rows, cols = linear_sum_assignment(-np.nan_to_num(S, nan=-1e9))
    assigned = {int(r): names[c] for r, c in zip(rows, cols)}
    unique = len(set(greedy)) == len(greedy)
    labels = greedy if unique else [assigned[k] for k in range(C.shape[1])]
    return labels, greedy, unique, pd.DataFrame(S, columns=[f"zscore_{n}" for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("section")
    args = ap.parse_args()
    out = RESULTS / args.dataset / args.section
    d = sc.read_h5ad(out / f"{args.section}_dimred.h5ad")
    bc = pd.read_csv(out / f"{args.section}_tumor_barcodes.csv")["barcode"].astype(str).tolist()
    ad = d[bc].copy()
    ug = pd.read_csv(out / f"{args.section}_unmix_genes.csv")["gene"].tolist()
    res = scipy.io.loadmat(out / f"{args.section}_tumor_unmix_results.mat")
    C, F = res["C"].astype(float), res["F"].astype(float)
    labels, greedy, unique, zs = label_like_original(C, ug)
    zs.insert(0, "archetype", range(C.shape[1]))
    zs["label_original_rule"] = greedy
    zs["label_used"] = labels
    zs.to_csv(out / f"{args.section}_archetype_state_labels.csv", index=False)
    W_aa = F[:, [labels.index(s) for s in STATES]]
    markers = neftel_markers()
    genes = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
    present = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
    X = norm_counts(ad, genes)
    W_mk = weights_gene_sd(X, genes, present)
    Xu = norm_counts(ad, ug)
    pres_u = {s: [g for g in markers[s] if g in ug] for s in STATES}
    runs = []
    for r in range(5):
        Cr, _, Fr = unmix(Xu.T, len(STATES), restarts=5, seed=101 * r)
        runs.append(Fr[:, label_archetypes(Cr, ug, pres_u)])
    W_mf = refine_pure(Xu, np.mean(runs, axis=0), ug, pres_u)
    Ws = {"archetype": W_aa, "marker": W_mk, "markerfree": W_mf}
    wdf = pd.DataFrame({"barcode": ad.obs_names, "x": ad.obsm["spatial"][:, 0], "y": ad.obsm["spatial"][:, 1]})
    for wn, W in Ws.items():
        for k, s in enumerate(STATES):
            wdf[f"{wn}_{s}"] = W[:, k]
    wdf.to_csv(out / f"{args.section}_weights.csv", index=False)
    pt = RootedPseudotime(ad, ug)
    rows, comp = [], []
    for wn, W in Ws.items():
        dom = np.bincount(W.argmax(1), minlength=len(STATES))
        comp.append({"section": args.section, "weights": wn, "spots": len(W), "purity": round(float(W.max(1).mean()), 3),
                     **{f"dominant_{s}_pct": round(100 * dom[k] / len(W), 1) for k, s in enumerate(STATES)}})
        for root in STATES:
            trees, _, _ = build(pt, W, X, root=root)
            for m, G in trees.items():
                rows.append({"section": args.section, "weights": wn, "root": root, "method": m,
                             "tree": case_name(G, root) if root == ROOT else "; ".join(sorted(f"{a}>{b}" for a, b in G.edges)),
                             **{c: int(c in edges_of(G)) for c in CONNECTIONS}})
    trees = pd.DataFrame(rows)
    trees.to_csv(out / f"{args.section}_trees.csv", index=False)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / f"{args.section}_composition.csv", index=False)
    support = trees.groupby("weights")[CONNECTIONS].sum()
    support.loc["total (of 48)"] = support.sum()
    support.to_csv(out / f"{args.section}_connection_support.csv")
    main_trees = trees[trees["root"] == ROOT].pivot(index="weights", columns="method", values="tree")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(f"{args.section}: {len(ad)} tumor spots; UNMIX genes {len(ug)}; Neftel marker genes {len(genes)}; "
              f"archetype labels {greedy}{'' if unique else ' (duplicate, resolved by assignment)'}")
        print(comp.to_string(index=False))
        print(main_trees.to_string())
        print(support.to_string())


if __name__ == "__main__":
    main()