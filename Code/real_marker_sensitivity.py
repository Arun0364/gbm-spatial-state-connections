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
from tree_compare import (neighbor_joining, euclid, zscore_cols, nj_bootstrap, nj_state_graph, greedy_arborescence,
                          optimal_arborescence, that_mst, case_name)
from marker_weights import weights_gene_sd as weights_nnls
from real_marker_trees import BASE, SAMPLES, STATES, ROOT, N_BOOT, neftel_markers, load, norm_counts, centroids

OUT_DIR = BASE / "Results/real_marker_trees_v2"
N_BALANCED = 100
NORMAL = ["MBP", "PLP1", "CNP", "MYRF", "PLLP", "GFAP", "AQP4", "SPARCL1", "AGT", "MLC1", "STMN2", "MAP2", "NRXN1"]
HYPOXIA = ["HILPDA", "DDIT3", "NDRG1", "HERPUD1", "TRIB3", "EPAS1", "EGLN3", "BNIP3", "BNIP3L", "GDF15", "NAMPT",
           "ADM", "ENO2", "SLC2A1", "SLC2A3", "PFKP", "PGK1", "LDHA", "MT1X", "VEGFA"]
KEY = {"G4_ST2": ["NPC-OPC", "AC-MES", "AC-NPC", "AC-OPC"], "G740_ST1": ["MES-NPC", "AC-OPC", "AC-NPC"]}
warnings.filterwarnings("ignore")


class Pseudotime:
    def __init__(self, ad, genes):
        am = ad[:, genes].copy()
        am.X = norm_counts(ad, genes)
        sc.pp.scale(am, max_value=10)
        sc.tl.pca(am, n_comps=20, svd_solver="arpack")
        sc.pp.neighbors(am, n_neighbors=15, n_pcs=15)
        sc.tl.diffmap(am, n_comps=15)
        self.am, self.cache = am, {}
        coo = am.obsp["connectivities"].tocoo()
        self.row, self.col = coo.row, coo.col

    def that(self, W):
        iroot = int(np.argmax(W[:, STATES.index(ROOT)]))
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


def edges_of(G):
    return {"-".join(sorted(e)) for e in G.to_undirected().edges}


def run_variant(ad, pt, W, nj_X, boot):
    if boot:
        T0, _, edge_support = nj_bootstrap(centroids(nj_X, W), STATES, ROOT, B=N_BOOT)
    else:
        T0, edge_support = neighbor_joining(euclid(zscore_cols(centroids(nj_X, W))), STATES), None
    That = pt.that(W)
    trees = {"NJ": nj_state_graph(T0, STATES, ROOT), "T-hat greedy": greedy_arborescence(That, STATES, ROOT),
             "T-hat most probable": optimal_arborescence(That, STATES, ROOT), "T-hat MST": that_mst(That, STATES, ROOT)[0]}
    return trees, edge_support


def record(sample, variant, rep, trees, W, n_markers, support=None):
    dom = np.bincount(W.argmax(1), minlength=len(STATES))
    row = {"sample": sample, "variant": variant, "rep": rep, "n_markers": n_markers,
           **{f"{m}": case_name(G, ROOT) for m, G in trees.items()},
           **{f"dominant_{s}": int(dom[k]) for k, s in enumerate(STATES)}}
    for e in KEY[sample]:
        row[f"n_trees_{e}"] = sum(e in edges_of(G) for G in trees.values())
    if support is not None:
        row["nj_support"] = "; ".join(f"{'-'.join(sorted(e))} {round(v * N_BOOT)}/{N_BOOT}" for e, v in support.items())
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    rows = []
    for sample in SAMPLES:
        ad, W_aa, dpt_genes = load(sample)
        pt = Pseudotime(ad, dpt_genes)
        all_markers = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
        X_mk = norm_counts(ad, all_markers)
        other = [g for g in ad.var_names if g not in set(g for s in markers.values() for g in s)]
        X_other = norm_counts(ad, other)
        base = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
        sets = {"full": base,
                "no normal-cell markers": {s: [g for g in base[s] if g not in NORMAL] for s in STATES},
                "no hypoxia markers": {s: [g for g in base[s] if s != "MES" or g not in HYPOXIA] for s in STATES}}
        sets["no normal-cell or hypoxia markers"] = {s: [g for g in sets["no normal-cell markers"][s] if s != "MES" or g not in HYPOXIA] for s in STATES}
        for name, ms in sets.items():
            W = weights_nnls(X_mk, all_markers, ms)
            trees, sup = run_variant(ad, pt, W, X_mk, True)
            rows.append(record(sample, name, 0, trees, W, " ".join(f"{s}{len(ms[s])}" for s in STATES), sup))
        W = weights_nnls(X_mk, all_markers, base)
        trees, sup = run_variant(ad, pt, W, X_other, True)
        rows.append(record(sample, f"NJ on {len(other)} non-marker genes", 0, trees, W, "full", sup))
        k = min(len(v) for v in base.values())
        rng = np.random.default_rng(0)
        for rep in range(1, N_BALANCED + 1):
            ms = {s: list(rng.choice(base[s], k, replace=False)) for s in STATES}
            W = weights_nnls(X_mk, all_markers, ms)
            trees, _ = run_variant(ad, pt, W, X_mk, False)
            rows.append(record(sample, f"balanced, {k} per state", rep, trees, W, f"{k} each"))
        for s in STATES:
            for g in base[s]:
                ms = {t: [x for x in base[t] if t != s or x != g] for t in STATES}
                W = weights_nnls(X_mk, all_markers, ms)
                trees, _ = run_variant(ad, pt, W, X_mk, False)
                rows.append(record(sample, "leave one marker out", g, trees, W, f"without {g} ({s})"))
        print(f"{sample}: done ({len(pt.cache)} pseudotime roots)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "real_marker_sensitivity.csv", index=False)
    methods = ["NJ", "T-hat greedy", "T-hat most probable", "T-hat MST"]
    out = []
    for (sample, variant), g in df.groupby(["sample", "variant"], sort=False):
        r = {"sample": sample, "variant": variant, "runs": len(g)}
        for m in methods:
            vc = g[m].value_counts()
            r[m] = f"{vc.index[0]} ({100 * vc.iloc[0] / len(g):.0f}%)" if len(g) > 1 else vc.index[0]
        for e in KEY[sample]:
            r[f"{e} (mean trees of 4)"] = round(g[f"n_trees_{e}"].mean(), 2)
        dom = g[[f"dominant_{s}" for s in STATES]].values.argmax(1)
        r["most common dominant state"] = f"{STATES[np.bincount(dom, minlength=4).argmax()]} ({100 * np.bincount(dom, minlength=4).max() / len(g):.0f}%)"
        out.append(r)
    summ = pd.DataFrame(out)
    summ.to_csv(OUT_DIR / "real_marker_sensitivity_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()