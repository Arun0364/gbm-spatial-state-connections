import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

BASE = "/Users/arun/Desktop/Research"

SAMPLES = {
    "G740_ST1": {
        "h5ad":    f"{BASE}/Results/PostAA_tumor_unmix_k4_markers/G740_ST1/G740_ST1_ns_annotated.h5ad",
        "pb_expr": f"{BASE}/Results/PostAA_tumor_unmix_k4_markers/G740_ST1/G740_ST1_pseudo_bulk_expression.csv",
    },
    "G4_ST2": {
        "h5ad":    f"{BASE}/Results/PostAA_tumor_unmix_k4_markers/G4_ST2/G4_ST2_ns_annotated.h5ad",
        "pb_expr": f"{BASE}/Results/PostAA_tumor_unmix_k4_markers/G4_ST2/G4_ST2_pseudo_bulk_expression.csv",
    },
}

OUT_DIR     = f"{BASE}/Results/pseudobulk_memberships_from_csv"
K_NEIGHBORS = 15

os.makedirs(OUT_DIR, exist_ok=True)

def strip_bc(x):
    return re.sub(r"-\d+$", "", str(x))

for sample, cfg in SAMPLES.items():
    print(f"\n{'='*60}\n  {sample}\n{'='*60}")

    pb = pd.read_csv(cfg["pb_expr"], index_col=0)
    anchor_pattern = re.compile(r"pb_\d+_anchor(\d+)")
    anchor_indices = []
    for row_name in pb.index:
        m = anchor_pattern.match(str(row_name))
        if not m:
            raise ValueError(f"Unexpected row name format: {row_name}")
        anchor_indices.append(int(m.group(1)))
    anchor_indices = np.array(anchor_indices)
    n_bulks = len(anchor_indices)
    print(f"  Pseudo-bulk anchors extracted: {n_bulks}")

    ad = sc.read_h5ad(cfg["h5ad"])
    ad.obs["barcode_key"] = [strip_bc(b) for b in ad.obs_names]
    coords = np.asarray(ad.obsm["spatial"], dtype=float)
    barcodes_all     = ad.obs_names.to_numpy()
    barcodes_key_all = ad.obs["barcode_key"].to_numpy()
    n_spots = ad.n_obs
    print(f"  h5ad spots: {n_spots}")

    k = min(K_NEIGHBORS, n_spots)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(coords)

    rows = []
    for bulk_id, anchor_idx in enumerate(anchor_indices):
        if anchor_idx >= n_spots:
            print(f"  WARNING: anchor_idx {anchor_idx} out of range (n_spots={n_spots}), skipping bulk_id {bulk_id}")
            continue
        anchor_coord = coords[anchor_idx].reshape(1, -1)
        dists, neigh_locs = nn.kneighbors(anchor_coord, return_distance=True)
        neigh_locs = neigh_locs.ravel()
        dists      = dists.ravel()
        w = np.ones(len(neigh_locs), dtype=float) / len(neigh_locs)

        for gi, wi, di in zip(neigh_locs, w, dists):
            rows.append({
                "dataset":            sample,
                "bulk_id":            bulk_id,
                "center_barcode":     barcodes_all[anchor_idx],
                "center_barcode_key": barcodes_key_all[anchor_idx],
                "barcode":            barcodes_all[gi],
                "barcode_key":        barcodes_key_all[gi],
                "weight":             float(wi),
                "distance":           float(di),
            })

    mem_df = pd.DataFrame(rows)
    out_sample_dir = os.path.join(OUT_DIR, sample)
    os.makedirs(out_sample_dir, exist_ok=True)
    out_path = os.path.join(out_sample_dir, f"{sample}_bulk_membership.csv")
    mem_df.to_csv(out_path, index=False)

    print(f"  Membership rows: {len(mem_df)}")
    print(f"  Saved -> {out_path}")

print("\nDone.")
