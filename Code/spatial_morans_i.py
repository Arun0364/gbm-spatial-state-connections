import os
import re
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import h5py
from scipy.spatial import cKDTree
from scipy.stats import norm

BASE      = "/Users/arun/Desktop/Research"
AA_DIR    = f"{BASE}/Results/PostAA_tumor_unmix_k4_markers"
UNMIX_DIR = f"{BASE}/Results/Unmix_tumor_k4_markers"
OUT_DIR   = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/Results/morans_i_markers"
N_NEIGH   = 6
N_PERMS   = 999
os.makedirs(OUT_DIR, exist_ok=True)


def strip_bc(x):
    return re.sub(r"-\d+$", "", str(x))


def morans_i(z, A, S0):
    z = z - z.mean()
    return len(z) / S0 * (z @ (A @ z)) / (z @ z)


for sample in ["G740_ST1", "G4_ST2"]:
    print(f"\n{sample}")

    labels_df   = pd.read_csv(f"{AA_DIR}/{sample}/{sample}_archetype_state_labels.csv")
    STATE_NAMES = dict(zip(labels_df["archetype"].astype(int), labels_df["dominant_state"]))

    F  = sio.loadmat(f"{UNMIX_DIR}/{sample}_tumor_unmix_results.mat")["F"]
    bc = pd.read_csv(f"{UNMIX_DIR}/{sample}_tumor_barcodes.csv")["barcode"].map(strip_bc).to_numpy()

    with h5py.File(f"{AA_DIR}/{sample}/{sample}_ns_annotated.h5ad", "r") as h:
        coords = h["obsm/spatial"][:].astype(float)
        keys   = [strip_bc(b.decode()) for b in h["obs/_index"][:]]
    pos = pd.Series(np.arange(len(keys)), index=keys)
    P = coords[pos.loc[bc].to_numpy()]
    N = len(P)
    print(f"  Tumor spots: {N}")

    _, nn = cKDTree(P).query(P, N_NEIGH + 1)
    A = sp.csr_matrix((np.ones(N * N_NEIGH), (np.repeat(np.arange(N), N_NEIGH), nn[:, 1:].ravel())), shape=(N, N))
    S0 = A.sum()
    S1 = 0.5 * (A + A.T).power(2).sum()
    S2 = ((np.asarray(A.sum(1)).ravel() + np.asarray(A.sum(0)).ravel()) ** 2).sum()
    EI = -1 / (N - 1)
    VI = (N * N * S1 - N * S2 + 3 * S0 ** 2) / ((N * N - 1) * S0 ** 2) - EI ** 2

    rng = np.random.default_rng(0)
    rows = []
    for k in range(F.shape[1]):
        w = F[:, k]
        I = morans_i(w, A, S0)
        perm = np.array([morans_i(rng.permutation(w), A, S0) for _ in range(N_PERMS)])
        rows.append({"state": STATE_NAMES[k], "archetype": f"W{k}", "I": round(I, 4),
                     "pval_norm": norm.sf((I - EI) / np.sqrt(VI)),
                     "pval_perm": (1 + (perm >= I).sum()) / (N_PERMS + 1)})
    result = pd.DataFrame(rows).set_index("state")
    print(result.to_string())

    out_path = f"{OUT_DIR}/{sample}_moransI.csv"
    result.to_csv(out_path)
    print(f"  Saved -> {out_path}")

print("\nDone.")