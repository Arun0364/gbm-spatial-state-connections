import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io

BASE = Path("/Users/arun/Desktop/Research")
RESULTS = BASE / "Results"
OUT = RESULTS / "replicate_summary"
CONNECTIONS = ["NPC-OPC", "AC-NPC", "MES-NPC", "AC-MES", "AC-OPC", "MES-OPC"]


def tumor_of(dataset, section):
    if dataset == "GSE253080":
        return section.split("-")[0]
    return {"SPA1_D": "B6", "SPA2_A": "B7", "SPA3_D": "B8", "SPA5_D": "B9", "SPA7_D": "B10", "SPA8_A": "B11"}.get(section, section)


def section_flags(folder, section):
    m = pd.read_csv(folder / f"{section}_tumor_normal_labels.csv")
    astro = 100 * m["SingleR_label_main"].isin(["Astrocyte", "Tissue_stem_cells", "Neuroepithelial_cell"]).mean()
    tumor = 100 * (m["tumor_label"] == "Tumor").mean()
    F = scipy.io.loadmat(folder / f"{section}_tumor_unmix_results.mat")["F"]
    top = 100 * np.bincount(F.argmax(1), minlength=F.shape[1]).max() / len(F)
    lab = pd.read_csv(folder / f"{section}_archetype_state_labels.csv")["label_original_rule"]
    return {"tumor_pct": round(tumor, 1), "ambiguous_label_pct": round(astro, 1), "top_archetype_pct": round(top, 1),
            "flag_tumor_call": int(tumor < 65 or astro < 60), "flag_dominant_archetype": int(top > 85),
            "flag_duplicate_labels": int(lab.nunique() < len(lab))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp, sup, agree, flags = [], [], [], []
    for f in sorted(glob.glob(str(RESULTS / "GSE*" / "*" / "*_trees.csv"))):
        p = Path(f)
        dataset, section = p.parts[-3], p.parts[-2]
        tumor = tumor_of(dataset, section)
        flags.append({"dataset": dataset, "tumor": tumor, "section": section, **section_flags(p.parent, section)})
        c = pd.read_csv(p.parent / f"{section}_composition.csv")
        c.insert(0, "tumor", tumor)
        c.insert(0, "dataset", dataset)
        comp.append(c)
        t = pd.read_csv(p)
        s = t.groupby("weights")[CONNECTIONS].sum().reset_index()
        s.insert(0, "section", section)
        s.insert(0, "tumor", tumor)
        s.insert(0, "dataset", dataset)
        sup.append(s)
        for (w, root), g in t.groupby(["weights", "root"]):
            tr = g.set_index("method")["tree"]
            agree.append({"dataset": dataset, "tumor": tumor, "section": section, "weights": w, "root": root,
                          "all_four_agree": int(tr.nunique() == 1),
                          "nj_disagrees_with_both_that": int(tr["NJ"] != tr["T-hat greedy"] and tr["NJ"] != tr["T-hat most probable"]),
                          **{m: tr[m] for m in tr.index}})
    comp, sup, agree, flags = pd.concat(comp), pd.concat(sup), pd.DataFrame(agree), pd.DataFrame(flags)
    flags.to_csv(OUT / "section_flags.csv", index=False)
    comp.to_csv(OUT / "composition_all.csv", index=False)
    sup.to_csv(OUT / "connection_support_all.csv", index=False)
    agree.to_csv(OUT / "method_agreement_all.csv", index=False)
    by_tumor = sup.groupby(["tumor", "weights"])[CONNECTIONS].sum()
    n = sup.groupby(["tumor", "weights"]).size().rename("sections")
    by_tumor = pd.concat([n, by_tumor], axis=1)
    by_tumor.to_csv(OUT / "connection_support_by_tumor.csv")
    ag = agree.groupby("weights")[["all_four_agree", "nj_disagrees_with_both_that"]].mean().mul(100).round(1)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_rows", 200):
        print(flags.to_string(index=False))
        print()
        print(comp.drop(columns=["spots"]).to_string(index=False))
        print("\nConnection support by tumor (16 trees per section per weight set)")
        print(by_tumor.to_string())
        print("\nPercent of section x root combinations")
        print(ag.to_string())


if __name__ == "__main__":
    main()