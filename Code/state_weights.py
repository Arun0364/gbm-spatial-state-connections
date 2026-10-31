import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import re
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import scanpy as sc

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from real_marker_trees import BASE, AA_DIR, SAMPLES, STATES, neftel_markers, norm_counts
from real_markerfree_trees import markerfree_weights
from marker_weights import weights_gene_sd

OUT_DIR = BASE / "Results/marker_guided"
UNMIX_DIR = BASE / "Results/Unmix_tumor_k4_markers"
warnings.filterwarnings("ignore")


def strip_bc(x):
    return re.sub(r"-\d+$", "", str(x))


def load_tumor(sample):
    bc = pd.read_csv(UNMIX_DIR / f"{sample}_tumor_barcodes.csv")["barcode"].map(strip_bc).tolist()
    ad = sc.read_h5ad(AA_DIR / sample / f"{sample}_ns_annotated.h5ad")
    ad.obs["barcode_key"] = [strip_bc(b) for b in ad.obs_names]
    pos = pd.Series(np.arange(ad.n_obs), index=ad.obs["barcode_key"].values)
    ad = ad[pos.loc[bc].values].copy()
    F = sio.loadmat(UNMIX_DIR / f"{sample}_tumor_unmix_results.mat")["F"]
    lab = pd.read_csv(AA_DIR / sample / f"{sample}_archetype_state_labels.csv")
    arch_to_state = dict(zip(lab["archetype"].astype(int), lab["dominant_state"]))
    W_aa = F[:, [next(k for k, s in arch_to_state.items() if s == st) for st in STATES]]
    return ad, W_aa


def marker_setup(ad, markers):
    genes = [g for g in dict.fromkeys(g for s in markers.values() for g in s) if g in ad.var_names]
    present = {s: [g for g in markers[s] if g in ad.var_names] for s in STATES}
    return genes, present


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    markers = neftel_markers()
    summary = []
    for sample in SAMPLES:
        ad, W_aa = load_tumor(sample)
        genes, present = marker_setup(ad, markers)
        W_mk = weights_gene_sd(norm_counts(ad, genes), genes, present)
        Wdf, _, _ = markerfree_weights(sample, markers)
        W_mf = Wdf.loc[ad.obs["barcode_key"].values].values
        out = pd.DataFrame({"barcode_key": ad.obs["barcode_key"].values,
                            "x": ad.obsm["spatial"][:, 0], "y": ad.obsm["spatial"][:, 1]})
        for name, W in [("archetype", W_aa), ("marker", W_mk), ("markerfree", W_mf)]:
            for k, s in enumerate(STATES):
                out[f"{name}_{s}"] = W[:, k]
            out[f"{name}_dominant"] = np.array(STATES)[W.argmax(1)]
            dom = np.bincount(W.argmax(1), minlength=len(STATES))
            summary.append({"sample": sample, "weights": name, "spots": len(W), "purity": round(float(W.max(1).mean()), 3),
                            **{f"dominant_{s}": int(dom[k]) for k, s in enumerate(STATES)},
                            **{f"r_with_marker_{s}": round(float(np.corrcoef(W[:, k], W_mk[:, k])[0, 1]), 3) for k, s in enumerate(STATES)}})
        out.to_csv(OUT_DIR / f"{sample}_weights.csv", index=False)
        print(f"{sample}: {len(out)} tumor spots; marker genes {len(genes)} "
              + ", ".join(f"{s} {len(present[s])}" for s in STATES))
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_DIR / "weights_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()