import numpy as np

from marker_weights import STATES, weights_gene_sd


def synthetic_spots(spots=300, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"{s}_m{i}" for s in STATES for i in range(5)] + [f"bg{i}" for i in range(20)]
    markers = {s: [f"{s}_m{i}" for i in range(5)] for s in STATES}
    P = np.ones((len(STATES), len(genes)))
    for k, s in enumerate(STATES):
        for g in markers[s]:
            P[k, genes.index(g)] = 20.0
    P[:, 20:] = rng.uniform(1.0, 5.0, size=(1, 20))
    W = rng.dirichlet(np.full(len(STATES), 0.7), size=spots)
    X = W @ P * rng.lognormal(0.0, 0.05, size=(spots, len(genes)))
    return X, genes, markers, W


def test_weights_are_proportions():
    X, genes, markers, _ = synthetic_spots()
    W_hat = weights_gene_sd(X, genes, markers)
    assert W_hat.shape == (len(X), len(STATES))
    assert W_hat.min() >= 0
    np.testing.assert_allclose(W_hat.sum(1), 1.0, atol=1e-9)


def test_weights_track_the_true_states():
    X, genes, markers, W = synthetic_spots()
    W_hat = weights_gene_sd(X, genes, markers)
    for k in range(len(STATES)):
        assert np.corrcoef(W[:, k], W_hat[:, k])[0, 1] > 0.9
