from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/arun/Desktop/Research")
OUT = BASE / "figures"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10})

real = lambda df: df[df["condition"].str.startswith("Real")]
i2 = real(pd.read_csv(BASE / "Results/sim_improve/improve2_results.csv"))
r2 = real(pd.read_csv(BASE / "Results/sim_refine/refine2_results.csv"))
wf = real(pd.read_csv(BASE / "Results/sim_weights_fix/weights_fix_results.csv"))
rows = [("UNMIX\n(archetypes)", i2["aa5_NJ"], i2["aa5_arb"]),
        ("Average of\n5 UNMIX runs", i2["aa_ensemble5_NJ"], i2["aa_ensemble5_arb"]),
        ("Marker-free\n(average + pure spots)", r2["ens_pure_NJ"], r2["ens_pure_arb"]),
        ("Marker-guided\n(unscaled)", wf["none_NJ"], wf["none_arb"]),
        ("Marker-guided\n(scaled)", wf["gene_sd_NJ"], wf["gene_sd_arb"])]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
x = np.arange(len(rows))
for i, (label, color) in enumerate([("NJ", "#2a78d6"), ("Most probable T̂ tree", "#eb6834")]):
    vals = [100 * r[1 + i].mean() for r in rows]
    xs = x + (i - 0.5) * 0.38
    ax.bar(xs, vals, 0.36, color=color, edgecolor="white", linewidth=0.8, label=label)
    for xi, v in zip(xs, vals):
        ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax.axhline(100 / 16, color="#555555", lw=0.9, ls="--", label="Chance (6%)")
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=8)
ax.set_ylabel("Exact tree recovered (%)")
ax.set_ylim(0, 105)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"Fig3_1_weight_estimates.{ext}", dpi=300, bbox_inches="tight")
print({r[0].replace("\n", " "): (round(100 * r[1].mean(), 1), round(100 * r[2].mean(), 1), len(r[1])) for r in rows})
