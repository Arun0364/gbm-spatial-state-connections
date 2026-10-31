import os
import re
import numpy as np
import pandas as pd

BASE = "/Users/arun/Desktop/Research"

SAMPLES = {
    "G740_ST1": {
        "membership": f"{BASE}/Results/pseudobulk_memberships_from_csv/G740_ST1/G740_ST1_bulk_membership.csv",
        "results_dir": f"{BASE}/FISH_deconvolution/MIMO/results/7_3_26_ST1_1.0",
        "n_bulks": 300,
        "n_cases": 30,
        "bulks_per_case": 10,
        "seed": 42,
    },
    "G4_ST2": {
        "membership": f"{BASE}/Results/pseudobulk_memberships_from_csv/G4_ST2/G4_ST2_bulk_membership.csv",
        "results_dir": f"{BASE}/FISH_deconvolution/MIMO/results/7_3_26_ST2_1.0",
        "n_bulks": 300,
        "n_cases": 30,
        "bulks_per_case": 10,
        "seed": 42,
    },
}

OUT_DIR = f"{BASE}/Results/PI_MIMO_VISUALS_7_3_26_wref_k4"

def strip_bc(x):
    return re.sub(r"-\d+$", "", str(x))

def parse_sample_output(path):
    lines = open(path).read().splitlines()
    blocks, cur = [], []
    for ln in lines:
        if ln.strip() == "---":
            if cur:
                blocks.append(cur)
            cur = []
            continue
        if ln.strip():
            cur.append(ln)
    if cur:
        blocks.append(cur)

    def parse_block(block):
        rows = []
        for ln in block:
            ln = ln.strip()
            if not ln.startswith("- ["):
                continue
            inside = ln.split("[", 1)[1].rsplit("]", 1)[0]
            rows.append([float(x.strip()) for x in inside.split(",") if x.strip()])
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, 0))

    F_raw = parse_block(blocks[1]) if len(blocks) > 1 else np.zeros((0, 0))
    return F_raw

for sample, cfg in SAMPLES.items():
    print(f"\n{'='*60}\n  {sample}\n{'='*60}")

    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(cfg["n_bulks"])[:cfg["n_cases"] * cfg["bulks_per_case"]]
    chunks = np.array_split(perm, cfg["n_cases"])

    bulk_to_case_col = {}
    for case_id, chunk in enumerate(chunks):
        for col_idx, bulk_idx in enumerate(chunk):
            bulk_to_case_col[int(bulk_idx)] = (case_id, col_idx)

    txt_pattern = "sample_output_{:03d}_0.2_0.2_0.0.txt"
    case_F = {}
    for case_id in range(cfg["n_cases"]):
        path = os.path.join(cfg["results_dir"], txt_pattern.format(case_id))
        F_raw = parse_sample_output(path)
        case_F[case_id] = F_raw
        if case_id == 0:
            n_clones = F_raw.shape[0]
            print(f"  F shape (case 0): {F_raw.shape}  → {n_clones} clones")

    mem = pd.read_csv(cfg["membership"])
    mem["barcode_key"] = mem["barcode_key"].astype(str).map(strip_bc)
    mem["bulk_id"] = mem["bulk_id"].astype(int)

    rows = []
    n_missing = 0
    for bc, g in mem.groupby("barcode_key", sort=False):
        clone_vec = np.zeros(n_clones, dtype=float)
        total_w = 0.0

        for _, row in g.iterrows():
            bid = int(row["bulk_id"])
            w   = float(row["weight"])
            if bid not in bulk_to_case_col:
                n_missing += 1
                continue
            case_id, col_idx = bulk_to_case_col[bid]
            F = case_F[case_id]
            if col_idx >= F.shape[1]:
                continue
            clone_vec += w * F[:, col_idx]
            total_w   += w

        if total_w > 0:
            clone_vec /= total_w

        vals = {"barcode_key": bc}
        for k in range(n_clones):
            vals[f"mimo_clone_{k}"] = float(clone_vec[k])
        vals["mimo_argmax"]  = int(np.argmax(clone_vec))
        vals["mimo_maxfrac"] = float(np.max(clone_vec))
        rows.append(vals)

    if n_missing:
        print(f"  WARNING: {n_missing} membership rows had bulk_id not in permutation")

    spot_frac = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, sample, f"{sample}_spot_level_mimo_fractions.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    spot_frac.to_csv(out_path, index=False)
    print(f"  Spots assigned: {len(spot_frac)}")
    print(f"  Saved -> {out_path}")

print("\nDone.")
