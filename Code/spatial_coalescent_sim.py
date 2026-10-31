import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_compare import all_rooted_trees, case_name

BASE    = Path("/Users/arun/Desktop/Research")
OUT_DIR = BASE / "Results/sim_coalescent/examples"

STATES       = ["NPC", "AC", "OPC", "MES"]
ROOT         = "NPC"
STATE_COLORS = {"AC": "#C44E52", "MES": "#ED7D31", "NPC": "#70AD47", "OPC": "#4472C4"}

DEFAULTS = {
    "n_cells": 15000,
    "death_prob": 0.2,
    "switch_rate": 0.04,
    "sigma_space": 1.0,
    "n_genes": 2000,
    "sigma_tree": 0.6,
    "sigma_expr_drift": 0.03,
    "cells_per_spot": 8,
    "lib_size_log": float(np.log(6000)),
    "lib_size_sd": 0.3,
    "n_markers": 30,
    "mix": 0.0,
    "lineage": "forward",
    "coal_growth": None,
    "coal_depth": 21.0,
    "split_rate": False,
    "n_clones": 0,
    "clone_window": (3, 4),
    "clone_events": 3,
    "n_segments": 40,
}


def state_profiles(G, states, root, p, rng):
    idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((len(states), p["n_genes"]))
    C[idx[root]] = rng.normal(-1.0, 1.0, p["n_genes"])
    for parent, child in nx.bfs_edges(G, root):
        C[idx[child]] = C[idx[parent]] + rng.normal(0, p["sigma_tree"], p["n_genes"])
    return C


def branching_walk(G, states, root, p, rng):
    idx = {s: i for i, s in enumerate(states)}
    children = {idx[s]: [idx[c] for c in G.successors(s)] for s in states if G.out_degree(s) > 0}
    K = len(states)
    r = p["switch_rate"]
    while True:
        pos = np.zeros((1, 2))
        state = np.array([idx[root]])
        drift = np.zeros((1, p["n_genes"]), dtype=np.float32)
        clone = np.zeros(1, dtype=int)
        trans = np.zeros((K, K))
        found_at = np.sort(rng.integers(p["clone_window"][0], p["clone_window"][1] + 1, p["n_clones"] - 1)) if p["n_clones"] > 1 else []
        clone_parent = [-1]
        g = 0
        while 0 < len(state) < p["n_cells"]:
            g += 1
            for _ in range(int(np.sum(np.asarray(found_at) == g))):
                sizes = np.bincount(clone)
                ok = np.where(sizes[clone] > 1)[0]
                if len(ok) == 0:
                    continue
                f = int(rng.choice(ok))
                clone_parent.append(int(clone[f]))
                clone[f] = len(clone_parent) - 1
            pos = np.repeat(pos, 2, axis=0) + rng.normal(0, p["sigma_space"], (2 * len(pos), 2))
            par = np.repeat(state, 2)
            clone = np.repeat(clone, 2)
            drift = np.repeat(drift, 2, axis=0) + rng.normal(0, p["sigma_expr_drift"], (len(par), p["n_genes"])).astype(np.float32)
            new = par.copy()
            for s, cs in children.items():
                m = np.where(par == s)[0]
                u = rng.random(len(m))
                rc = r / len(cs) if p["split_rate"] else r
                for j, c in enumerate(cs):
                    hit = (u >= j * rc) & (u < (j + 1) * rc)
                    new[m[hit]] = c
            np.add.at(trans, (par, new), 1)
            keep = rng.random(len(new)) > p["death_prob"]
            n_old = len(state)
            pos, state, drift, clone = pos[keep], new[keep], drift[keep], clone[keep]
            if len(state) > 0:
                pos = pos * np.sqrt(len(state) / n_old)
        if len(state) >= p["n_cells"]:
            if p["n_clones"] > 1:
                return pos, state, drift, trans, g, clone, clone_parent
            return pos, state, drift, trans, g


def coalescent_walk(G, states, root, p, rng):
    import msprime
    idx = {s: i for i, s in enumerate(states)}
    children = {idx[s]: [idx[c] for c in G.successors(s)] for s in states}
    K = len(states)
    n = p["n_cells"]
    g = np.log(2 * (1 - p["death_prob"])) if p["coal_growth"] is None else p["coal_growth"]
    seed = int(rng.integers(1, 2**31 - 1))
    if g > 0:
        ts = msprime.simulate(population_configurations=[msprime.PopulationConfiguration(sample_size=n, initial_size=n, growth_rate=g)],
                              recombination_rate=0, random_seed=seed)
    else:
        ts = msprime.simulate(sample_size=n, Ne=n, recombination_rate=0, random_seed=seed)
    tree = ts.first()
    t = ts.tables.nodes.time.copy()
    if g <= 0:
        t *= p["coal_depth"] / t[tree.root]
    parent = tree.parent_array[:ts.num_nodes]
    r = p["switch_rate"]
    pos = np.zeros((ts.num_nodes, 2))
    state = np.zeros(ts.num_nodes, dtype=int)
    drift = np.zeros((ts.num_nodes, p["n_genes"]), dtype=np.float32)
    trans = np.zeros((K, K))
    time_in = np.zeros(K)
    state[tree.root] = idx[root]
    for u in np.argsort(-t, kind="stable"):
        v = parent[u]
        if v < 0:
            continue
        L = t[v] - t[u]
        var_space = (np.exp(g * t[v]) - np.exp(g * t[u])) / g if g > 0 else L
        pos[u] = pos[v] + rng.normal(0, p["sigma_space"] * np.sqrt(var_space), 2)
        drift[u] = drift[v] + rng.normal(0, p["sigma_expr_drift"] * np.sqrt(L), p["n_genes"]).astype(np.float32)
        s, elapsed = state[v], 0.0
        while children[s]:
            wait = rng.exponential(1 / r if p["split_rate"] else 1 / (r * len(children[s])))
            if elapsed + wait > L:
                break
            time_in[s] += wait
            elapsed += wait
            c = children[s][rng.integers(len(children[s]))]
            trans[s, c] += 1
            s = c
        time_in[s] += L - elapsed
        state[u] = s
    rate = trans / np.where(time_in > 0, time_in, 1)[:, None]
    T_true = rate + np.diag(1 - rate.sum(1))
    return pos[:n], state[:n], drift[:n], T_true, float(t[tree.root])


def spot_grid(pos, cells_per_spot):
    r = float(np.median(cKDTree(pos).query(pos, k=cells_per_spot + 1)[0][:, -1]))
    step = 2 * r * 0.9
    lo, hi = pos.min(0), pos.max(0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + step, step), np.arange(lo[1], hi[1] + step, step))
    gx[1::2] += step / 2
    return np.column_stack([gx.ravel(), gy.ravel()]), r


def clone_copy_numbers(clone_parent, p, rng):
    n_seg = p["n_segments"]
    cn = np.full((len(clone_parent), n_seg), 2, dtype=int)
    for c in range(1, len(clone_parent)):
        cn[c] = cn[clone_parent[c]]
        for seg in rng.choice(n_seg, p["clone_events"], replace=False):
            cn[c, seg] = max(1, cn[c, seg] + (1 if rng.random() < 0.5 else -1))
    return cn


def gene_copy_numbers(cn, n_genes):
    seg_of_gene = np.arange(n_genes) * cn.shape[1] // n_genes
    return cn[:, seg_of_gene]


def aggregate(pos, state, drift, C, K, p, rng, clone=None, gene_cn=None):
    centers, r = spot_grid(pos, p["cells_per_spot"])
    members = cKDTree(pos).query_ball_point(centers, r=r)
    keep = [i for i, m in enumerate(members) if len(m) >= 3]
    W = np.zeros((len(keep), K))
    counts = np.zeros((len(keep), p["n_genes"]), dtype=np.float32)
    ncell = np.zeros(len(keep), dtype=int)
    Q = np.zeros((len(keep), len(gene_cn))) if clone is not None else None
    for row, i in enumerate(keep):
        m = np.array(members[i])
        if p["mix"] > 0:
            extra = int(round(len(m) * p["mix"] / (1 - p["mix"])))
            m = np.concatenate([m, rng.integers(0, len(state), extra)])
        W[row] = np.bincount(state[m], minlength=K) / len(m)
        rate = np.exp(C[state[m]] + drift[m])
        if clone is not None:
            rate = rate * gene_cn[clone[m]] / 2.0
            Q[row] = np.bincount(clone[m], minlength=len(gene_cn)) / len(m)
        rate = rate.mean(0)
        lib = np.exp(rng.normal(p["lib_size_log"], p["lib_size_sd"]))
        counts[row] = rng.poisson(rate / rate.sum() * lib)
        ncell[row] = len(m)
    if clone is not None:
        return centers[keep], W, counts, ncell, Q
    return centers[keep], W, counts, ncell


def morans_i(xy, W, k=6):
    nb = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    out = []
    for col in W.T:
        z = col - col.mean()
        out.append(float((z * z[nb].sum(1)).sum() / (k * (z ** 2).sum())))
    return out


def marker_genes(C, n_markers):
    out = []
    for k in range(len(C)):
        diff = C[k] - np.delete(C, k, axis=0).mean(0)
        out += [f"g{j:04d}" for j in np.argsort(diff)[::-1][:n_markers]]
    return sorted(set(out))


def state_marker_genes(C, states, n_markers):
    out = {}
    for k, s in enumerate(states):
        diff = C[k] - np.delete(C, k, axis=0).mean(0)
        out[s] = [f"g{j:04d}" for j in np.argsort(diff)[::-1][:n_markers]]
    return out


def simulate(G, seed, states=STATES, root=ROOT, **overrides):
    p = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(seed)
    C = state_profiles(G, states, root, p, rng)
    if p["lineage"] == "coalescent":
        pos, state, drift, T_true, n_gen = coalescent_walk(G, states, root, p, rng)
    else:
        walk = branching_walk(G, states, root, p, rng)
        pos, state, drift, trans, n_gen = walk[:5]
        T_true = trans / np.where(trans.sum(1, keepdims=True) > 0, trans.sum(1, keepdims=True), 1)
    if p["n_clones"] > 1:
        clone, clone_parent = walk[5], walk[6]
        cn = clone_copy_numbers(clone_parent, p, rng)
        xy, W, counts, ncell, Q = aggregate(pos, state, drift, C, len(states), p, rng, clone, gene_copy_numbers(cn, p["n_genes"]))
    else:
        xy, W, counts, ncell = aggregate(pos, state, drift, C, len(states), p, rng)
    norm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    obs = pd.DataFrame(index=[f"spot_{i:05d}" for i in range(len(W))])
    for k in range(len(states)):
        obs[f"W{k}"] = W[:, k]
    obs["n_cells"] = ncell
    a = ad.AnnData(X=sp.csr_matrix(norm), obs=obs, var=pd.DataFrame(index=[f"g{j:04d}" for j in range(p["n_genes"])]))
    a.layers["counts"] = sp.csr_matrix(counts)
    a.layers["norm_counts"] = sp.csr_matrix(norm)
    a.obsm["spatial"] = xy.astype(np.float32)
    a.uns["states"] = list(states)
    a.uns["case"] = case_name(G, root)
    a.uns["tree_edges"] = json.dumps([list(e) for e in G.edges])
    a.uns["T_true"] = T_true
    a.uns["marker_genes"] = marker_genes(C, p["n_markers"])
    a.uns["state_markers"] = json.dumps(state_marker_genes(C, states, p["n_markers"]))
    a.uns["morans_i"] = morans_i(xy, W)
    a.uns["purity"] = float(W.max(1).mean())
    if p["n_clones"] > 1:
        for c in range(Q.shape[1]):
            a.obs[f"Q{c}"] = Q[:, c]
        a.uns["clone_cn"] = cn
        a.uns["clone_parent"] = json.dumps(clone_parent)
        a.uns["clone_fraction_cells"] = np.bincount(clone, minlength=len(clone_parent)) / len(clone)
    a.uns["params"] = json.dumps({"seed": seed, "n_generations": round(float(n_gen), 2), "n_cells_final": int(len(state)), **p})
    return a, pos, state


def plot_example(a, pos, state, name, states=STATES):
    fig, axes = plt.subplots(1, 2 + len(states), figsize=(4.4 * (2 + len(states)), 4.6))
    for k, s in enumerate(states):
        axes[0].scatter(*pos[state == k].T, s=0.3, c=STATE_COLORS[s], label=s, rasterized=True)
    axes[0].set_title("Cells (true state)")
    axes[0].legend(markerscale=15, fontsize=8, loc="upper right")
    xy = a.obsm["spatial"]
    W = a.obs[[f"W{k}" for k in range(len(states))]].values
    axes[1].scatter(*xy.T, s=6, c=[STATE_COLORS[states[k]] for k in W.argmax(1)])
    axes[1].set_title("Spots: dominant state")
    for k, s in enumerate(states):
        sc_ = axes[k + 2].scatter(*xy.T, s=6, c=W[:, k], cmap="viridis", vmin=0, vmax=1)
        axes[k + 2].set_title(f"W_{s}", color=STATE_COLORS[s])
        plt.colorbar(sc_, ax=axes[k + 2], shrink=0.8)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"Simulated tissue, true tree {a.uns['case']} | mean max W = {a.uns['purity']:.2f}", fontsize=13)
    fig.savefig(OUT_DIR / f"{name}_tissue.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for G in all_rooted_trees(STATES, ROOT):
        for mix in (0.0, 0.6):
            a, pos, state = simulate(G, seed=1, mix=mix)
            name = f"{case_name(G, ROOT)}__mix{int(mix * 100):02d}"
            plot_example(a, pos, state, name)
            print(f"{name}: spots={a.n_obs} purity={a.uns['purity']:.2f} Moran=" +
                  " ".join(f"{s}={m:.2f}" for s, m in zip(STATES, a.uns["morans_i"])))