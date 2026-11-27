import itertools

import numpy as np
import pytest

from unmix import unmix


def synthetic_mixture(k=3, genes=30, spots=200, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.uniform(1.0, 10.0, size=(genes, k))
    W = rng.dirichlet(np.full(k, 0.5), size=spots)
    W[:k] = np.eye(k)
    return C, W, C @ W.T


def best_match(W, F):
    k = W.shape[1]
    return max(itertools.permutations(range(k)),
               key=lambda p: sum(np.corrcoef(W[:, j], F[:, p[j]])[0, 1] for j in range(k)))


@pytest.mark.parametrize("seed", range(5))
def test_fractions_are_proportions(seed):
    _, _, M = synthetic_mixture(seed=seed)
    _, _, F = unmix(M, 3, restarts=5, seed=0)
    np.testing.assert_allclose(F.sum(1), 1.0, atol=1e-8)
    assert F.min() > -0.01


@pytest.mark.parametrize("seed", range(5))
def test_fractions_track_the_true_mixture(seed):
    _, W, M = synthetic_mixture(seed=seed)
    _, _, F = unmix(M, 3, restarts=5, seed=0)
    p = best_match(W, F)
    for j in range(3):
        assert np.corrcoef(W[:, j], F[:, p[j]])[0, 1] > 0.95


@pytest.mark.parametrize("seed", range(5))
def test_archetypes_are_close_to_the_truth(seed):
    C, W, M = synthetic_mixture(seed=seed)
    C_hat, _, F = unmix(M, 3, restarts=5, seed=0)
    p = best_match(W, F)
    for j in range(3):
        assert np.linalg.norm(C_hat[:, p[j]] - C[:, j]) / np.linalg.norm(C[:, j]) < 0.2
