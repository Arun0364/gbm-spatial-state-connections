import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/arun/Desktop/Research"
SAMPLES = ["G740_ST1", "G4_ST2"]

PATHS = {
    s: {
        "singleR":   os.path.join(BASE, f"Results/CellTypeAssignment/{s}_SingleR/celltype_metadata_singleR.csv"),
        "neftel":    os.path.join(BASE, "Results/neftel_outputs_curated/visium_all_spots_neftel_curated.csv"),
        "positions": os.path.join(BASE, f"Data/{s}/spatial/tissue_positions.csv"),
        "spotlight": os.path.join(BASE, f"Results/SPOTlight_unmix_k4/{s}_spotlight_weights.csv"),
    }
    for s in SAMPLES
}

OUT_DIR = os.path.join(BASE, "Results/TumorNormalClassification")
os.makedirs(OUT_DIR, exist_ok=True)

DEFINITE_NORMAL = {
    "Endothelial_cells", "Neurons", "Macrophage", "Neutrophils", "CMP",
    "Smooth_muscle_cells", "Keratinocytes", "Chondrocytes",
    "NK_cell", "T_cells", "B_cell", "Erythrocyte", "Monocyte", "DC", "Platelets",
}

TUMOR_AMBIGUOUS = {
    "Astrocyte",
    "Neuroepithelial_cell",
    "Tissue_stem_cells",
}

def strip_suffix(bc):
    import re
    return re.sub(r"-\d+$", "", str(bc))

def classify_spot(singleR_label, neftel_confidence, conf_thresh):
    if singleR_label in DEFINITE_NORMAL:
        return "Normal", f"SingleR={singleR_label} (definite normal)"
    if singleR_label in TUMOR_AMBIGUOUS:
        if neftel_confidence >= conf_thresh:
            return "Tumor", f"SingleR={singleR_label} (ambiguous) + Neftel_conf={neftel_confidence:.3f}>={conf_thresh}"
        else:
            return "Normal", f"SingleR={singleR_label} (ambiguous) but Neftel_conf={neftel_confidence:.3f}<{conf_thresh}"
    if neftel_confidence >= conf_thresh:
        return "Tumor", f"SingleR={singleR_label} (unknown) + Neftel_conf={neftel_confidence:.3f}>={conf_thresh}"
    return "Normal", f"SingleR={singleR_label} (unknown) + Neftel_conf={neftel_confidence:.3f}<{conf_thresh}"

def run_classification(sample, conf_thresh, spotlight_thresh=0.4):
    cfg = PATHS[sample]
    print(f"\n{'='*60}\n{sample}\n{'='*60}")

    singleR = pd.read_csv(cfg["singleR"])
    singleR["barcode_key"] = singleR["barcode"].apply(strip_suffix)
    singleR = singleR[["barcode_key", "SingleR_label_main", "SingleR_score_main"]].copy()
    print(f"  SingleR spots: {len(singleR)}")
    for lbl, cnt in singleR["SingleR_label_main"].value_counts().items():
        tag = "[NORMAL]" if lbl in DEFINITE_NORMAL else ("[AMBIGUOUS]" if lbl in TUMOR_AMBIGUOUS else "[UNKNOWN]")
        print(f"    {lbl:<30} {cnt:>5}  {tag}")

    neftel_all = pd.read_csv(cfg["neftel"])
    neftel = neftel_all[neftel_all["dataset"] == sample].copy()
    neftel["barcode_key"] = neftel["barcode"].apply(strip_suffix)
    neftel = neftel[["barcode_key", "Neftel_state", "Neftel_confidence"]].copy()
    print(f"  Neftel spots:  {len(neftel)}")

    merged = singleR.merge(neftel, on="barcode_key", how="outer", indicator=True)
    only_singleR = (merged["_merge"] == "left_only").sum()
    only_neftel  = (merged["_merge"] == "right_only").sum()
    if only_singleR > 0:
        warnings.warn(f"  {only_singleR} barcodes in SingleR only (no Neftel match)")
    if only_neftel > 0:
        warnings.warn(f"  {only_neftel} barcodes in Neftel only (no SingleR match)")
    merged = merged[merged["_merge"] != "right_only"].drop(columns="_merge")
    merged["Neftel_confidence"]  = merged["Neftel_confidence"].fillna(0.0)
    merged["Neftel_state"]       = merged["Neftel_state"].fillna("unknown")
    merged["SingleR_label_main"] = merged["SingleR_label_main"].fillna("unknown")

    results = merged.apply(
        lambda row: classify_spot(row["SingleR_label_main"], row["Neftel_confidence"], conf_thresh),
        axis=1, result_type="expand"
    )
    results.columns = ["tumor_label", "reason"]
    merged = pd.concat([merged, results], axis=1)

    n_tumor  = (merged["tumor_label"] == "Tumor").sum()
    n_normal = (merged["tumor_label"] == "Normal").sum()
    print(f"\n  Classification (conf_thresh={conf_thresh}):")
    print(f"    Tumor:  {n_tumor:>5}  ({100*n_tumor/len(merged):.1f}%)")
    print(f"    Normal: {n_normal:>5}  ({100*n_normal/len(merged):.1f}%)")

    if os.path.exists(cfg["spotlight"]):
        sp = pd.read_csv(cfg["spotlight"])
        sp["barcode_key"] = sp["barcode"].apply(strip_suffix)
        tumor_cols = [c for c in ["AC_like", "MES_like", "OPC_like", "NPC_like"] if c in sp.columns]
        sp["sp_tumor_sum"] = sp[tumor_cols].sum(axis=1)
        sp["sp_tumor"] = sp["sp_tumor_sum"] >= spotlight_thresh
        comp = merged.merge(sp[["barcode_key", "sp_tumor"]], on="barcode_key", how="left")
        new_t = comp["tumor_label"] == "Tumor"
        old_t = comp["sp_tumor"].fillna(False)
        print(f"\n  vs SPOTlight (sum>={spotlight_thresh}):")
        print(f"    SPOTlight Tumor:         {old_t.sum()}")
        print(f"    SPOTlight Normal:        {(~old_t).sum()}")
        print(f"    Agreement (both Tumor):  {(new_t & old_t).sum()}")
        print(f"    Agreement (both Normal): {(~new_t & ~old_t).sum()}")
        print(f"    New=Tumor, Old=Normal:   {(new_t & ~old_t).sum()}")
        print(f"    New=Normal, Old=Tumor:   {(~new_t & old_t).sum()}")

    merged.rename(columns={"barcode_key": "barcode"}, inplace=True)
    out_csv = os.path.join(OUT_DIR, f"{sample}_tumor_normal_labels.csv")
    merged.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv}")

    pos = pd.read_csv(cfg["positions"])
    pos["barcode_key"] = pos["barcode"].apply(strip_suffix)
    pos = pos[pos["in_tissue"] == 1]
    plot_df = merged.merge(
        pos[["barcode_key", "array_row", "array_col"]],
        left_on="barcode", right_on="barcode_key", how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{sample} — Tumor vs Normal Classification", fontsize=13, fontweight="bold")

    color_map = {"Tumor": "#d62728", "Normal": "#1f77b4"}
    for label, grp in plot_df.groupby("tumor_label"):
        axes[0].scatter(grp["array_col"], -grp["array_row"],
                        c=color_map.get(label, "grey"), s=6, alpha=0.7, linewidths=0, label=label)
    axes[0].set_title("Tumor / Normal")
    axes[0].set_aspect("equal"); axes[0].axis("off")
    axes[0].legend(loc="lower right", markerscale=3, framealpha=0.8)

    cmap = plt.get_cmap("tab20")
    all_labels = sorted(plot_df["SingleR_label_main"].unique())
    label_colors = {lbl: cmap(i % 20) for i, lbl in enumerate(all_labels)}
    for lbl, grp in plot_df.groupby("SingleR_label_main"):
        axes[1].scatter(grp["array_col"], -grp["array_row"],
                        c=[label_colors[lbl]], s=6, alpha=0.7, linewidths=0, label=lbl)
    axes[1].set_title("SingleR Labels")
    axes[1].set_aspect("equal"); axes[1].axis("off")
    axes[1].legend(loc="lower right", markerscale=3, framealpha=0.8, fontsize=7)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{sample}_spatial_tumor_normal.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {out_png}")

    return {"sample": sample, "n_total": len(merged), "n_tumor": int(n_tumor), "n_normal": int(n_normal)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--spotlight_thresh", type=float, default=0.4)
    args = parser.parse_args()
    for sample in SAMPLES:
        run_classification(sample, args.conf_thresh, args.spotlight_thresh)

if __name__ == "__main__":
    main()
