import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import textwrap
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE = Path("/Users/arun/Desktop/Research/Code")
sys.path.insert(0, str(CODE))
from tree_compare import nj_bootstrap, nj_state_graph, patristic, optimal_arborescence, greedy_arborescence, case_name
from make_chapter3_figures import draw_nj, draw_tree, N_BOOT
from make_chapter2_de_figures import label_order, SOURCE_COLORS, SOURCE_NAMES, PROG, DIFF, NS
from real_marker_trees import AA_DIR, SAMPLES, ROOT, neftel_markers, norm_counts, centroids
from real_root_check import RootedPseudotime
from state_weights import BASE, load_tumor, marker_setup

RES = BASE / "Results/marker_guided"
CH2 = RES / "chapter2"
OUT_DIR = RES / "figures"
TREE_STATES = ["NPC", "AC", "OPC", "MES"]
MAP_STATES = ["AC", "MES", "NPC", "OPC"]
COLORS = {"AC": "#C44E52", "MES": "#ED7D31", "NPC": "#70AD47", "OPC": "#4472C4"}
GREY = "#D9D9D9"
LFC, NLP, XLIM, TOP_TERMS, N_LABELS = 1.0, 10.0, 6.0, 8, 5
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10})
warnings.filterwarnings("ignore")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def spatial(ax, w, title):
    ax.scatter(w["x"], w["y"], s=1.2, c=GREY, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def weight_maps(W):
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.9))
    for r, s in enumerate(SAMPLES):
        for c, st in enumerate(MAP_STATES):
            spatial(axes[r, c], W[s], st if r == 0 else "")
            sc = axes[r, c].scatter(W[s]["x"], W[s]["y"], s=1.2, c=W[s][f"marker_{st}"], cmap="viridis", vmin=0, vmax=1,
                                    linewidths=0, rasterized=True)
        axes[r, 0].text(-0.08, 0.5, s, transform=axes[r, 0].transAxes, rotation=90, ha="center", va="center")
    fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02).set_label("Marker-based weight")
    save(fig, "Fig2_1_marker_weights")


def dominant_maps(W):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, s in zip(axes, SAMPLES):
        spatial(ax, W[s], s)
        for st in MAP_STATES:
            m = W[s]["marker_dominant"] == st
            ax.scatter(W[s].loc[m, "x"], W[s].loc[m, "y"], s=1.4, c=COLORS[st], linewidths=0, rasterized=True,
                       label=f"{st} ({int(m.sum()):,})")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, markerscale=5,
                  handletextpad=0.2, columnspacing=1.2, fontsize=9)
    save(fig, "Fig2_2_dominant_state")


def clone_figures(W):
    mats = pd.read_csv(CH2 / "clone_spearman_matrices.csv", index_col=0)
    match = pd.read_csv(CH2 / "clone_matching.csv")
    match = match[match["weights"] == "marker"]
    clones = [f"mimo_clone_{c}" for c in range(4)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, s in zip(axes, SAMPLES):
        R = mats[(mats["sample"] == s) & (mats["weights"] == "marker")].loc[MAP_STATES, clones].values
        pairs = {(r.state, int(r.matched_clone[-1])) for r in match[match["sample"] == s].itertuples()}
        im = ax.imshow(R, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"Clone {c}" for c in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels(MAP_STATES)
        for i, st in enumerate(MAP_STATES):
            for j in range(4):
                hit = (st, j) in pairs
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(R[i, j]) > 0.45 else "black", fontweight="bold" if hit else "normal")
                if hit:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
        merged = pd.read_csv(BASE / f"Results/archetype_mimo_integration_k4_markers/{s}_arch_mimo_merged.csv")
        ax.set_title(f"{s} (n = {len(merged):,} spots)")
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02).set_label("Spearman r")
    save(fig, "Fig2_4_state_clone_correlation")


def volcano(ax, sample, letter):
    d = pd.read_csv(CH2 / f"{sample}_marker_DE_prog_vs_diff.csv")
    ymax = d.loc[np.isfinite(d["neg_log10p"]), "neg_log10p"].max()
    ycap = np.ceil(ymax / 5) * 5 + 5
    capped = ~np.isfinite(d["neg_log10p"]) | (d["log2FC"].abs() > XLIM)
    d["y"] = np.where(~np.isfinite(d["neg_log10p"]), ycap, d["neg_log10p"])
    d["x"] = d["log2FC"].clip(-XLIM, XLIM)
    up = (d["log2FC"] > LFC) & (d["neg_log10p"] > NLP)
    dn = (d["log2FC"] < -LFC) & (d["neg_log10p"] > NLP)
    d["c"] = np.where(up, PROG, np.where(dn, DIFF, NS))
    for mask, marker in [(~capped, "o"), (capped, "^")]:
        sub = d[mask]
        ax.scatter(sub["x"], sub["y"], c=sub["c"], s=7 if marker == "o" else 12, marker=marker, linewidths=0, alpha=0.8, rasterized=True)
    for v in (-LFC, LFC):
        ax.axvline(v, color="#7F7F7F", lw=0.6, ls="--")
    ax.axhline(NLP, color="#7F7F7F", lw=0.6, ls="--")
    step = ycap * 0.085
    for sel, x_text, ha in [(dn, -XLIM + 0.1, "left"), (up, XLIM - 0.1, "right")]:
        rows = d[sel].nlargest(N_LABELS, "neg_log10p").reset_index(drop=True)
        if rows.empty:
            continue
        ys = [ycap - n * step for n in range(len(rows))]
        for n, p in enumerate(label_order(rows[["x", "y"]].values, x_text, ys, 2 * XLIM, ycap)):
            r = rows.iloc[p]
            ax.annotate(r["gene"], (r["x"], r["y"]), xytext=(x_text, ys[n]), ha=ha, va="center", fontsize=8, style="italic",
                        arrowprops=dict(arrowstyle="-", color="#595959", lw=0.4, shrinkA=1, shrinkB=2))
    ax.set_xlim(-XLIM - 0.4, XLIM + 0.4)
    ax.set_ylim(-1, ycap + 4)
    ax.set_xlabel("log$_2$ fold change (progenitor-like / differentiated)")
    ax.set_ylabel("$-$log$_{10}$ p")
    ax.set_title(f"{sample}  ({int(up.sum())} up, {int(dn.sum())} down)")
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def volcano_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, s, l in zip(axes, SAMPLES, "AB"):
        volcano(ax, s, l)
    handles = [Patch(color=PROG, label="Progenitor-enriched"), Patch(color=DIFF, label="Differentiated-enriched"),
               Patch(color=NS, label="Not significant"), Line2D([], [], marker="^", ls="", color="#7F7F7F", label="Value beyond axis limit")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "Fig2_5_volcano")


def pathway_figure():
    panels = [(s, k) for s in SAMPLES for k in ("progenitor", "differentiated")]
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.6))
    used = set()
    for ax, (s, k), l in zip(axes, panels, "ABCD"):
        p = pd.read_csv(CH2 / f"{s}_marker_pathways_{k}.csv").nsmallest(TOP_TERMS, "p_value").iloc[::-1]
        used |= set(p["source"])
        y = np.arange(len(p))
        ax.barh(y, -np.log10(p["p_value"]), color=[SOURCE_COLORS.get(x, "#BFBFBF") for x in p["source"]], height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.shorten(n[0].upper() + n[1:], 48, placeholder="...") for n in p["name"]], fontsize=8.5)
        ax.axvline(-np.log10(0.05), color="#7F7F7F", lw=0.6, ls="--")
        ax.set_xlabel("$-$log$_{10}$ FDR")
        ax.set_title(f"{s}, {k}-enriched genes" + ("" if len(p) else " (no significant terms)"), loc="left")
        ax.text(-0.02, 1.08, l, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Patch(color=c, label=SOURCE_NAMES[x]) for x, c in SOURCE_COLORS.items() if x in used]
    handles.append(Line2D([], [], color="#7F7F7F", lw=0.6, ls="--", label="FDR = 0.05"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, -0.03), ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(h_pad=1.6)
    save(fig, "Fig2_6_pathways")


def tree_figures(markers):
    for number, sample in [(1, "G4_ST2"), (2, "G740_ST1")]:
        ad, _ = load_tumor(sample)
        wdf = pd.read_csv(RES / f"{sample}_weights.csv")
        W = wdf[[f"marker_{s}" for s in TREE_STATES]].values
        genes, _ = marker_setup(ad, markers)
        T0, split_support, edge_support = nj_bootstrap(centroids(norm_counts(ad, genes), W), TREE_STATES, ROOT, B=N_BOOT)
        G_nj = nj_state_graph(T0, TREE_STATES, ROOT)
        P = patristic(T0, TREE_STATES)
        pi = {s: i for i, s in enumerate(TREE_STATES)}
        dpt_genes = [g for g in pd.read_csv(AA_DIR / sample / f"{sample}_archetype_markers.csv")["gene"].unique() if g in ad.var_names]
        That = RootedPseudotime(ad, dpt_genes).that_from(W, ROOT)
        greedy = greedy_arborescence(That, TREE_STATES, ROOT)
        best = optimal_arborescence(That, TREE_STATES, ROOT)
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
        draw_nj(axes[0, 0], T0, TREE_STATES, ROOT, split_support, "A")
        draw_tree(axes[0, 1], G_nj, ROOT, lambda a, b: P[pi[a], pi[b]],
                  lambda a, b: f"{P[pi[a], pi[b]]:.2f} ({round(edge_support[frozenset((a, b))] * N_BOOT)})",
                  f"NJ state tree: {case_name(G_nj, ROOT)}", "Tree distance", "B")
        tlab = lambda a, b: f"{That[pi[a], pi[b]]:.3f}"
        tlen = lambda a, b: 1 - That[pi[a], pi[b]]
        draw_tree(axes[1, 0], greedy, ROOT, tlen, tlab, f"Greedy T̂ tree: {case_name(greedy, ROOT)}", "1 − T̂", "C", min_frac=0.25)
        draw_tree(axes[1, 1], best, ROOT, tlen, tlab, f"Most probable T̂ tree: {case_name(best, ROOT)}", "1 − T̂", "D", min_frac=0.25)
        fig.tight_layout(h_pad=2.2, w_pad=2.0)
        save(fig, f"Fig3_{number}_{sample}_trees")
        print(sample, "NJ", case_name(G_nj, ROOT), "| greedy", case_name(greedy, ROOT), "| most probable", case_name(best, ROOT))


def support_figure():
    d = pd.read_csv(RES / "trees_all_checks.csv")
    keys = {"G4_ST2": ["NPC-OPC", "AC-MES", "AC-NPC", "AC-OPC"], "G740_ST1": ["MES-NPC", "AC-OPC", "NPC-OPC", "AC-NPC"]}
    cols = [("Archetype", (d["check"].isin(["main", "root"])) & (d["weights"] == "archetype")),
            ("Marker", (d["check"].isin(["main", "root"])) & (d["weights"] == "marker")),
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
    save(fig, "Fig3_5_supported_connections")


def weights_sim_figure():
    real = lambda df: df[df["condition"].str.startswith("Real")]
    i2 = real(pd.read_csv(BASE / "Results/sim_improve/improve2_results.csv"))
    r2 = real(pd.read_csv(BASE / "Results/sim_refine/refine2_results.csv"))
    wf = real(pd.read_csv(BASE / "Results/sim_weights_fix/weights_fix_results.csv"))
    rows = [("UNMIX\n(archetypes)", i2["aa5_NJ"], i2["aa5_arb"]),
            ("Average of\n5 UNMIX runs", i2["aa_ensemble5_NJ"], i2["aa_ensemble5_arb"]),
            ("Marker-free\n(average + pure spots)", r2["ens_pure_NJ"], r2["ens_pure_arb"]),
            ("Marker-based\n(unscaled)", wf["none_NJ"], wf["none_arb"]),
            ("Marker-based\n(scaled)", wf["gene_sd_NJ"], wf["gene_sd_arb"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(rows))
    for i, (label, color) in enumerate([("NJ", "#2a78d6"), ("Most probable T\u0302 tree", "#eb6834")]):
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
    save(fig, "Fig3_6_weight_estimates_simulation")
    print("simulation (real purity, n =", len(wf), "):", {r[0].replace("\n", " "): (round(100 * r[1].mean(), 1), round(100 * r[2].mean(), 1)) for r in rows})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    W = {s: pd.read_csv(RES / f"{s}_weights.csv") for s in SAMPLES}
    weight_maps(W)
    dominant_maps(W)
    clone_figures(W)
    volcano_figure()
    pathway_figure()
    tree_figures(neftel_markers())
    support_figure()
    weights_sim_figure()
    print("Saved to", OUT_DIR)


if __name__ == "__main__":
    main()