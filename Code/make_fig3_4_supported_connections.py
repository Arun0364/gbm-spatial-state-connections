from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/arun/Desktop/Research")
OUT = BASE / "figures"
SAMPLES = ["G4_ST2", "G740_ST1"]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10})

d = pd.read_csv(BASE / "Results/marker_guided/trees_all_checks.csv")
keys = {"G4_ST2": ["NPC-OPC", "AC-MES", "AC-NPC", "AC-OPC"], "G740_ST1": ["MES-NPC", "AC-OPC", "NPC-OPC", "AC-NPC"]}
cols = [("Archetype", (d["check"].isin(["main", "root"])) & (d["weights"] == "archetype")),
        ("Marker-\nguided", (d["check"].isin(["main", "root"])) & (d["weights"] == "marker")),
        ("Marker-free", (d["check"].isin(["main", "root"])) & (d["weights"] == "markerfree")),
        ("No\nnormal-cell\nmarkers", d["check"] == "no normal-cell markers"),
        ("No\nhypoxia\nmarkers", d["check"] == "no hypoxia markers"),
        ("Clone-\nsubset\nweights", d["check"].str.startswith("weights from clone subset")),
        ("NJ on\nother\ngenes", d["check"].str.startswith("NJ on")),
        ("Balanced\nmarkers", d["check"].str.startswith("balanced")),
        ("Leave one\nmarker\nout", d["check"] == "leave one marker out")]
fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0))
for ax, s, l in zip(axes, SAMPLES, "AB"):
    M = 100 * np.array([[d.loc[m & (d["sample"] == s), e].sum() / (4 * (m & (d["sample"] == s)).sum()) for _, m in cols] for e in keys[s]])
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 60 else "black")
    ax.set_yticks(range(len(keys[s])))
    ax.set_yticklabels([e.replace("-", "–") for e in keys[s]])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for c, _ in cols] if s == SAMPLES[-1] else [""] * len(cols), fontsize=7.5)
    ax.axvline(2.5, color="black", lw=1.0)
    ax.set_title(s, loc="left")
    ax.text(-0.13, 1.05, l, transform=ax.transAxes, fontsize=12, fontweight="bold")
fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02).set_label("Share of trees with the connection (%)")
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Fig3_4_supported_connections.{ext}", dpi=300, bbox_inches="tight")
