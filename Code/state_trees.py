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
from tree_compare import (neighbor_joining, euclid, zscore_cols, nj_bootstrap, nj_state_graph, greedy_arborescence,
                          optimal_arborescence, rank_arborescences, that_mst, case_name)
from real_marker_trees import AA_DIR, SAMPLES, STATES, ROOT, N_BOOT, neftel_markers, norm_counts, centroids
from real_marker_sensitivity import NORMAL, HYPOXIA, edges_of
from real_root_check import RootedPseudotime
from marker_weights import weights_gene_sd
from state_weights import BASE, load_tumor, marker_setup

WEIGHTS_DIR = BASE / "Results/marker_guided"
OUT_DIR = BASE / "Results/marker_guided"
N_BALANCED = 100
KEY = {"G4_ST2": ["NPC-OPC", "AC-NPC", "AC-MES", "AC-OPC"], "G740_ST1": ["MES-NPC", "AC-OPC", "AC-NPC", "NPC-OPC"]}
METHODS = ["NJ", "T-hat greedy", "T-hat most probable", "T-hat MST"]
warnings.filterwarnings("ignore")


def build(pt, W, nj_X, root=ROOT, boot=False):
    if boot:
        T0, _, support = nj_bootstrap(centroids(nj_X, W), STATES, root, B=N_BOOT)
    else:
        T0, support = neighbor_joining(euclid(zscore_cols(centroids(nj_X, W))), STATES), None
    That = pt.that_from(W, root)
    trees = {"NJ": nj_state_graph(T0, STATES, root), "T-hat greedy": greedy_arborescence(That, STATES, root),
             "T-hat most probable": optimal_arborescence(That, STATES, root), "T-hat MST": that_mst(That, STATES, root)[0]}
    return trees, support, That


def record(sample, check, rep, weights, root, trees, W, support=None):
    dom = np.bincount(W.argmax(1), minlength=len(STATES))
    row = {"sample": sample, "check": check, "rep": rep, "weights": weights, "root": root,
           **{m: case_name(G, root) if root == ROOT else "; ".join(sorted(f"{a}>{b}" for a, b in G.edges)) for m, G in trees.items()},
           "dominant": STATES[int(dom.argmax())], **{f"dominant_{s}": int(dom[k]) for k, s in enumerate(STATES)}}
    for e in KEY[sample]:
        row[e] = sum(e in edges_of(G) for G in trees.values())
    if support is not None:
        row["nj_support"] = "; ".join(f"{'-'.join(sorted(e))} {round(v * N_BOOT)}/{N_BOOT}" for e, v in sorted(support.items(), key=lambda x: sorted(x[0])))
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    rows, thats, ranks = [], [], []
    for sample in SAMPLES:
        ad, _ = load_tumor(sample)
        wdf = pd.read_csv(WEIGHTS_DIR / f"{sample}_weights.csv")
        assert (wdf["barcode_key"].values == ad.obs["barcode_key"].values).all()
        Ws = {w: wdf[[f"{w}_{s}" for s in STATES]].values for w in ["archetype", "marker", "markerfree"]}
        genes, base = marker_setup(ad, markers)
        X = norm_counts(ad, genes)
        dpt_genes = [g for g in pd.read_csv(AA_DIR / sample / f"{sample}_archetype_markers.csv")["gene"].unique() if g in ad.var_names]
        pt = RootedPseudotime(ad, dpt_genes)
        for wn, W in Ws.items():
            trees, sup, That = build(pt, W, X, boot=True)
            rows.append(record(sample, "main", 0, wn, ROOT, trees, W, sup))
            thats.append(pd.DataFrame(That, index=STATES, columns=STATES).round(4).assign(sample=sample, weights=wn))
            ranked = rank_arborescences(That, STATES, ROOT)
            for r, (name, prob, _) in enumerate(ranked[:5], 1):
                ranks.append({"sample": sample, "weights": wn, "rank": r, "case": name, "relative_to_best": round(prob / ranked[0][1], 3)})
            for root in STATES:
                if root != ROOT:
                    trees, _, _ = build(pt, W, X, root=root)
                    rows.append(record(sample, "root", 0, wn, root, trees, W))
        sets = {"no normal-cell markers": {s: [g for g in base[s] if g not in NORMAL] for s in STATES},
                "no hypoxia markers": {s: [g for g in base[s] if s != "MES" or g not in HYPOXIA] for s in STATES}}
        sets["no normal-cell or hypoxia markers"] = {s: [g for g in sets["no normal-cell markers"][s] if s != "MES" or g not in HYPOXIA] for s in STATES}
        for name, ms in sets.items():
            W = weights_gene_sd(X, genes, ms)
            trees, sup, _ = build(pt, W, X, boot=True)
            rows.append(record(sample, name, 0, "marker", ROOT, trees, W, sup))
        allm = set(g for s in markers.values() for g in s)
        other = [g for g in ad.var_names if g not in allm]
        trees, sup, _ = build(pt, Ws["marker"], norm_counts(ad, other), boot=True)
        rows.append(record(sample, f"NJ on {len(other)} non-marker genes", 0, "marker", ROOT, trees, Ws["marker"], sup))
        clone = set(pd.read_csv(BASE / f"Results/archetype_mimo_integration_k4_markers/{sample}_arch_mimo_merged.csv")["barcode_key"])
        sub = ad.obs["barcode_key"].isin(clone).values
        W_sub = weights_gene_sd(X[sub], genes, base)
        trees, sup, _ = build(RootedPseudotime(ad[sub].copy(), dpt_genes), W_sub, X[sub], boot=True)
        rows.append(record(sample, f"weights from clone subset ({int(sub.sum())} spots)", 0, "marker", ROOT, trees, W_sub, sup))
        k = min(len(v) for v in base.values())
        rng = np.random.default_rng(0)
        for rep in range(1, N_BALANCED + 1):
            W = weights_gene_sd(X, genes, {s: list(rng.choice(base[s], k, replace=False)) for s in STATES})
            trees, _, _ = build(pt, W, X)
            rows.append(record(sample, f"balanced, {k} per state", rep, "marker", ROOT, trees, W))
        for s in STATES:
            for g in base[s]:
                W = weights_gene_sd(X, genes, {t: [x for x in base[t] if t != s or x != g] for t in STATES})
                trees, _, _ = build(pt, W, X)
                rows.append(record(sample, "leave one marker out", g, "marker", ROOT, trees, W))
        print(f"{sample}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "trees_all_checks.csv", index=False)
    pd.concat(thats).to_csv(OUT_DIR / "that_matrices.csv")
    pd.DataFrame(ranks).to_csv(OUT_DIR / "arborescence_ranking.csv", index=False)
    out = []
    for (sample, check, weights), g in df.groupby(["sample", "check", "weights"], sort=False):
        r = {"sample": sample, "check": check, "weights": weights, "runs": len(g)}
        if check != "root":
            for m in METHODS:
                vc = g[m].value_counts()
                r[m] = vc.index[0] if len(g) == 1 else f"{vc.index[0]} ({100 * vc.iloc[0] / len(g):.0f}%)"
        n_trees = 4 * len(g)
        for e in KEY[sample]:
            r[e] = f"{int(g[e].sum())}/{n_trees}"
        vc = g["dominant"].value_counts()
        r["dominant"] = vc.index[0] if len(g) == 1 else f"{vc.index[0]} ({100 * vc.iloc[0] / len(g):.0f}%)"
        out.append(r)
    summ = pd.DataFrame(out)
    summ.to_csv(OUT_DIR / "trees_summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 26):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()