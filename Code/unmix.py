from math import factorial
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def pca_scores(M, n):
    X = M.T - M.mean(1)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    sign = np.sign(Vt[np.arange(n), np.abs(Vt[:n]).argmax(1)])
    return (U[:, :n] * s[:n]) * sign, Vt[:n].T * sign


def simplex_volume(V):
    E = V[1:] - V[0]
    g = np.linalg.det(E @ E.T)
    return np.sqrt(max(g, 0.0)) / factorial(len(E))


def pick_pair(P, k, rng):
    w = pdist(P) ** k
    idx = rng.choice(len(w), p=w / w.sum())
    n = len(P)
    i = int(n - 2 - np.floor(np.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2 - 0.5))
    j = int(idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
    return i, j


def grow_vertices(P, k, rng):
    i, j = pick_pair(P, k, rng)
    chosen = [i, j]
    while len(chosen) < k:
        V = P[chosen]
        E = V[1:] - V[0]
        Q, _ = np.linalg.qr(E.T)
        D = P - V[0]
        resid = D - (D @ Q) @ Q.T
        base = simplex_volume(V)
        vol = base * np.linalg.norm(resid, axis=1) / len(chosen)
        vol[chosen] = 0
        chosen.append(int(rng.choice(len(P), p=vol / vol.sum())))
    return P[chosen].copy()


def face_normal(H):
    d = H.shape[1]
    A = np.ones((d, d))
    A[1:] = H[1:] - H[0]
    b = np.zeros(d)
    b[0] = 1
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.linalg.norm(b - A @ x) > 1e-3:
        return None
    return x / np.linalg.norm(x)


def enclose(P, V, max_iter=1000, tol=1e-3):
    k = len(V)
    for _ in range(max_iter):
        best, best_i, best_shift = 0, None, None
        for i in range(k):
            H = np.delete(V, i, axis=0)
            n = face_normal(H)
            if n is None:
                continue
            side = (V[i] - H[0]) @ n
            if abs(side) < 1e-10:
                continue
            sgn = np.sign(side)
            dist = (P - H[0]) @ n
            out = dist * sgn < -tol
            if out.sum() > best:
                far = np.argmax(-dist * sgn)
                best, best_i, best_shift = out.sum(), i, dist[far] * n
        if best_i is None:
            break
        mask = np.arange(k) != best_i
        V[mask] += best_shift
    return V


def unmix(M, k, restarts=5, seed=0):
    rng = np.random.default_rng(seed)
    P, loadings = pca_scores(M, k - 1)
    hull_vol = ConvexHull(P, qhull_options="QJ").volume if k > 2 else np.ptp(P)
    best_V, best_vol = None, None
    for _ in range(restarts):
        V = enclose(P, grow_vertices(P, k, rng))
        vol = simplex_volume(V)
        if best_V is None or (best_vol > vol > hull_vol):
            best_V, best_vol = V, vol
    A = np.vstack([best_V.T, np.ones(k)])
    B = np.vstack([P.T, np.ones(len(P))])
    F = np.linalg.solve(A, B).T
    C = loadings @ best_V.T + M.mean(1, keepdims=True)
    return C, best_vol, F
