from pathlib import Path
import pandas as pd

BASE = Path("/Users/arun/Desktop/Research")
SUMMARY = BASE / "Results" / "replicate_summary"
OUT = SUMMARY / "leaveout"
CONNECTIONS = ["NPC-OPC", "AC-NPC", "MES-NPC", "AC-MES", "AC-OPC", "MES-OPC"]
STRONG = 12


def subsets(flags):
    tc = set(flags.loc[flags["flag_tumor_call"] == 1, "section"])
    da = set(flags.loc[flags["flag_dominant_archetype"] == 1, "section"])
    dup = set(flags.loc[flags["flag_duplicate_labels"] == 1, "section"])
    allsec = set(flags["section"])
    return {"all 22": allsec,
            "drop tumor-call flags": allsec - tc,
            "drop tumor-call + dominant-archetype flags": allsec - tc - da,
            "clean only (no flags)": allsec - tc - da - dup}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flags = pd.read_csv(SUMMARY / "section_flags.csv")
    sup = pd.read_csv(SUMMARY / "connection_support_all.csv")
    agree = pd.read_csv(SUMMARY / "method_agreement_all.csv")
    rows_sec, rows_tum, rows_ag = [], [], []
    for name, keep in subsets(flags).items():
        s = sup[sup["section"].isin(keep)]
        a = agree[agree["section"].isin(keep)]
        for w, g in s.groupby("weights"):
            rows_sec.append({"subset": name, "weights": w, "sections": len(g),
                             **{c: int((g[c] >= STRONG).sum()) for c in CONNECTIONS}})
            t = g.groupby("tumor")[CONNECTIONS].mean()
            rows_tum.append({"subset": name, "weights": w, "tumors": len(t),
                             **{c: int((t[c] >= STRONG).sum()) for c in CONNECTIONS}})
        for w, g in a.groupby("weights"):
            rows_ag.append({"subset": name, "weights": w, "section_roots": len(g),
                            "all_four_agree_pct": round(100 * g["all_four_agree"].mean(), 1),
                            "nj_disagrees_with_both_that_pct": round(100 * g["nj_disagrees_with_both_that"].mean(), 1)})
    sec, tum, ag = pd.DataFrame(rows_sec), pd.DataFrame(rows_tum), pd.DataFrame(rows_ag)
    sec.to_csv(OUT / "strong_connections_by_section.csv", index=False)
    tum.to_csv(OUT / "strong_connections_by_tumor.csv", index=False)
    ag.to_csv(OUT / "method_agreement.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(f"Sections where a connection is in >= {STRONG} of 16 trees")
        print(sec.to_string(index=False))
        print(f"\nTumors where the mean support across its sections is >= {STRONG} of 16")
        print(tum.to_string(index=False))
        print("\nMethod agreement")
        print(ag.to_string(index=False))


if __name__ == "__main__":
    main()