import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/arun/Desktop/Research/Code")
warnings.filterwarnings("ignore")
from state_trees import WEIGHTS_DIR
from real_marker_trees import SAMPLES, STATES, ROOT, norm_counts, neftel_markers, centroids
from state_weights import load_tumor, marker_setup
from tree_compare import neighbor_joining, euclid, zscore_cols, nj_bootstrap, nj_leaf_splits

out = sys.argv[1]
markers = neftel_markers()
for sample in SAMPLES:
    ad, _ = load_tumor(sample)
    wdf = pd.read_csv(WEIGHTS_DIR / f"{sample}_weights.csv")
    genes, _ = marker_setup(ad, markers)
    X = norm_counts(ad, genes)
    W = wdf[[f"marker_{s}" for s in STATES]].values
    C = centroids(X, W)
    pd.DataFrame(C, index=STATES, columns=genes).to_csv(f"{out}/{sample}_centroids.csv")
    T0, split_sup, edge_sup = nj_bootstrap(C, STATES, ROOT, B=1000)
    splits = nj_leaf_splits(T0, STATES)
    print(sample, "genes", len(genes), "python splits", [sorted(map(sorted, s)) if not isinstance(next(iter(s)), str) else sorted(s) for s in splits], "support", {str(sorted(s) if isinstance(next(iter(s)), str) else s): v for s, v in split_sup.items()})
