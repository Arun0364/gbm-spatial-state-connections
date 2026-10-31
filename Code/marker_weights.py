import numpy as np
from scipy.optimize import nnls

STATES = ["NPC", "AC", "OPC", "MES"]
TOP_FRACTION = 0.05


def marker_scores(X, genes, markers, states):
    L = np.log1p(X)
    Z = (L - L.mean(0)) / (L.std(0) + 1e-12)
    gidx = {g: i for i, g in enumerate(genes)}
    return np.column_stack([Z[:, [gidx[g] for g in markers[s] if g in gidx]].mean(1) for s in states])


def references(X, score):
    n_top = max(5, int(TOP_FRACTION * len(X)))
    return np.column_stack([X[np.argsort(score[:, k])[::-1][:n_top]].mean(0) for k in range(score.shape[1])])


def solve(ref, X):
    W = np.array([nnls(ref, x)[0] for x in X])
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1
    return W / s


def weights_marker(X, genes, markers, states, scale="gene_sd"):
    ref = references(X, marker_scores(X, genes, markers, states))
    if scale == "none":
        return solve(ref, X)
    if scale == "gene_sd":
        d = 1.0 / (X.std(0) + 1e-12)
        return solve(ref * d[:, None], X * d)
    if scale == "log":
        return solve(np.log1p(ref), np.log1p(X))
    raise ValueError(scale)


def weights_gene_sd(X, genes, markers, states=STATES):
    return weights_marker(X, genes, markers, states, "gene_sd")