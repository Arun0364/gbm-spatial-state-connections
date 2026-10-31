import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import json
import urllib.request
from scipy.spatial import cKDTree
from scipy.stats import norm, mannwhitneyu, spearmanr
from scipy.optimize import linear_sum_assignment

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from real_marker_trees import SAMPLES, STATES, norm_counts
from state_weights import BASE, load_tumor

WEIGHTS_DIR = BASE / "Results/marker_guided"
OUT_DIR = BASE / "Results/marker_guided/chapter2"
ESTIMATES = ["archetype", "marker", "markerfree"]
PROGENITOR, DIFFERENTIATED = {"NPC", "OPC"}, {"AC", "MES"}
N_NEIGH, N_PERMS = 6, 999
LFC, NEGLOGP = 1.0, 10.0
warnings.filterwarnings("ignore")


def morans_i(z, A, S0):
    z = z - z.mean()
    return len(z) / S0 * (z @ (A @ z)) / (z @ z)


def morans_table(sample, xy, Ws):
    N = len(xy)
    _, nn = cKDTree(xy).query(xy, N_NEIGH + 1)
    A = sp.csr_matrix((np.ones(N * N_NEIGH), (np.repeat(np.arange(N), N_NEIGH), nn[:, 1:].ravel())), shape=(N, N))
    S0 = A.sum()
    S1 = 0.5 * (A + A.T).power(2).sum()
    S2 = ((np.asarray(A.sum(1)).ravel() + np.asarray(A.sum(0)).ravel()) ** 2).sum()
    EI = -1 / (N - 1)
    VI = (N * N * S1 - N * S2 + 3 * S0 ** 2) / ((N * N - 1) * S0 ** 2) - EI ** 2
    rows = []
    for wn, W in Ws.items():
        rng = np.random.default_rng(0)
        for k, s in enumerate(STATES):
            I = morans_i(W[:, k], A, S0)
            perm = np.array([morans_i(rng.permutation(W[:, k]), A, S0) for _ in range(N_PERMS)])
            rows.append({"sample": sample, "weights": wn, "state": s, "I": round(I, 4),
                         "pval_norm": norm.sf((I - EI) / np.sqrt(VI)), "pval_perm": (1 + (perm >= I).sum()) / (N_PERMS + 1)})
    return rows


def clone_table(sample, keys, Ws):
    merged = pd.read_csv(BASE / f"Results/archetype_mimo_integration_k4_markers/{sample}_arch_mimo_merged.csv")
    clones = [f"mimo_clone_{c}" for c in range(4)]
    pos = pd.Series(np.arange(len(keys)), index=keys)
    idx = pos.loc[merged["barcode_key"]].values
    rows, mats = [], []
    for wn, W in Ws.items():
        R = np.array([[spearmanr(W[idx, k], merged[c])[0] for c in clones] for k in range(len(STATES))])
        si, ci = linear_sum_assignment(-R)
        mats.append(pd.DataFrame(R, index=STATES, columns=clones).round(3).assign(sample=sample, weights=wn))
        for k, c in zip(si, ci):
            rows.append({"sample": sample, "weights": wn, "state": STATES[k], "matched_clone": clones[c], "spearman_r": round(R[k, c], 3)})
    return rows, mats, len(idx)


def run_de(expr, a, b, genes):
    out = []
    for i, g in enumerate(genes):
        x, y = expr[a, i], expr[b, i]
        lfc = np.log2((x.mean() + 1e-9) / (y.mean() + 1e-9))
        try:
            p = mannwhitneyu(x, y, alternative="two-sided")[1]
        except ValueError:
            p = 1.0
        out.append((g, lfc, -np.log10(p + 1e-300)))
    return pd.DataFrame(out, columns=["gene", "log2FC", "neg_log10p"]).sort_values("neg_log10p", ascending=False)


def gprofiler(genes):
    cols = ["source", "name", "p_value", "intersection_size", "term_size"]
    if not genes:
        return pd.DataFrame(columns=cols)
    body = json.dumps({"organism": "hsapiens", "query": genes, "sources": ["GO:BP", "KEGG", "REAC"],
                       "user_threshold": 0.05, "significance_threshold_method": "fdr", "no_evidences": True}).encode()
    req = urllib.request.Request("https://biit.cs.ut.ee/gprofiler/api/gost/profile/", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read()).get("result") or []
    return pd.DataFrame(res)[cols] if res else pd.DataFrame(columns=cols)


def de_and_pathways(sample, ad, W, label, rng):
    dom = np.array(STATES)[W.argmax(1)]
    prog = np.where(np.isin(dom, list(PROGENITOR)))[0]
    diff = np.where(np.isin(dom, list(DIFFERENTIATED)))[0]
    n_prog, n_diff = len(prog), len(diff)
    if n_diff > n_prog:
        diff = rng.choice(diff, n_prog, replace=False)
    elif n_prog > n_diff:
        prog = rng.choice(prog, n_diff, replace=False)
    a = np.zeros(ad.n_obs, bool)
    a[prog] = True
    b = np.zeros(ad.n_obs, bool)
    b[diff] = True
    genes = list(ad.var_names)
    de = run_de(norm_counts(ad, genes), a, b, genes)
    de.to_csv(OUT_DIR / f"{sample}_{label}_DE_prog_vs_diff.csv", index=False)
    sig = de[de["neg_log10p"] > NEGLOGP]
    up = sig[sig["log2FC"] > LFC].nlargest(200, "neg_log10p")["gene"].tolist()
    down = sig[sig["log2FC"] < -LFC].nlargest(200, "neg_log10p")["gene"].tolist()
    pw = {}
    for kind, gl in [("progenitor", up), ("differentiated", down)]:
        pw[kind] = gprofiler(gl)
        pw[kind].to_csv(OUT_DIR / f"{sample}_{label}_pathways_{kind}.csv", index=False)
    top = {k: (f"{v.nsmallest(1, 'p_value')['name'].iloc[0]} (FDR {v['p_value'].min():.1e})" if len(v) else "none") for k, v in pw.items()}
    return {"sample": sample, "weights": label, "progenitor_spots": n_prog, "differentiated_spots": n_diff,
            "spots_per_group_used": int(a.sum()), "sig_genes": len(sig),
            "progenitor_enriched": int((sig["log2FC"] > LFC).sum()), "differentiated_enriched": int((sig["log2FC"] < -LFC).sum()),
            "top_progenitor_pathway": top["progenitor"], "top_differentiated_pathway": top["differentiated"]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp, morans, clone_rows, clone_mats, de_rows = [], [], [], [], []
    for sample in SAMPLES:
        ad, _ = load_tumor(sample)
        wdf = pd.read_csv(WEIGHTS_DIR / f"{sample}_weights.csv")
        assert (wdf["barcode_key"].values == ad.obs["barcode_key"].values).all()
        Ws = {w: wdf[[f"{w}_{s}" for s in STATES]].values for w in ESTIMATES}
        xy = wdf[["x", "y"]].values
        for wn, W in Ws.items():
            dom = np.bincount(W.argmax(1), minlength=len(STATES))
            comp.append({"sample": sample, "weights": wn, "spots": len(W), "purity": round(float(W.max(1).mean()), 3),
                         **{f"dominant_{s}": int(dom[k]) for k, s in enumerate(STATES)},
                         **{f"dominant_{s}_pct": round(100 * dom[k] / len(W), 1) for k, s in enumerate(STATES)},
                         **{f"mean_weight_{s}": round(float(W[:, k].mean()), 3) for k, s in enumerate(STATES)}})
        morans += morans_table(sample, xy, Ws)
        r, m, n_clone = clone_table(sample, wdf["barcode_key"].values, Ws)
        clone_rows += r
        clone_mats += m
        for wn in ["marker", "markerfree"]:
            de_rows.append(de_and_pathways(sample, ad, Ws[wn], wn, np.random.default_rng(42)))
        print(f"{sample}: done ({len(ad)} spots; {n_clone} with clone fractions)", flush=True)
    for name, rows in [("composition", comp), ("morans_i", morans), ("clone_matching", clone_rows), ("de_summary", de_rows)]:
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{name}.csv", index=False)
    pd.concat(clone_mats).to_csv(OUT_DIR / "clone_spearman_matrices.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70):
        print(pd.DataFrame(comp)[["sample", "weights", "purity"] + [f"dominant_{s}_pct" for s in STATES]].to_string(index=False))
        print(pd.DataFrame(morans).pivot_table(index=["sample", "weights"], columns="state", values="I").round(3)[STATES].to_string())
        print(pd.DataFrame(clone_rows).pivot_table(index=["sample", "weights"], columns="state", values="spearman_r")[STATES].to_string())
        print(pd.DataFrame(de_rows).to_string(index=False))


if __name__ == "__main__":
    main()