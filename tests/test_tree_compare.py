import networkx as nx
import numpy as np
import pytest

from tree_compare import (all_rooted_trees, neighbor_joining, nj_leaf_splits, nj_state_graph,
                          optimal_arborescence, patristic, rank_arborescences, rf_rooted, self_test)

STATES = ["NPC", "AC", "OPC", "MES"]


def tree_metric(edges, labels):
    G = nx.Graph()
    for a, b, w in edges:
        G.add_edge(a, b, length=w)
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    return np.array([[d[a][b] for b in labels] for a in labels])


def test_module_self_test():
    assert len(self_test()) == 16


def test_nj_recovers_four_leaf_additive_tree():
    labels = ["A", "B", "C", "D"]
    D = tree_metric([("A", "u", 1.0), ("B", "u", 2.0), ("u", "v", 1.5), ("C", "v", 0.5), ("D", "v", 3.0)], labels)
    T = neighbor_joining(D, labels)
    assert nj_leaf_splits(T, labels) == {frozenset({"A", "B"})}
    np.testing.assert_allclose(patristic(T, labels), D, atol=1e-9)


def test_nj_recovers_five_leaf_additive_tree():
    labels = ["A", "B", "C", "D", "E"]
    D = tree_metric([("A", "u", 1.0), ("B", "u", 1.0), ("u", "v", 2.0), ("C", "v", 1.0),
                     ("v", "w", 2.0), ("D", "w", 1.0), ("E", "w", 1.0)], labels)
    T = neighbor_joining(D, labels)
    assert nj_leaf_splits(T, labels) == {frozenset({"A", "B"}), frozenset({"D", "E"})}
    np.testing.assert_allclose(patristic(T, labels), D, atol=1e-9)


@pytest.mark.parametrize("index", range(16))
def test_nj_state_graph_recovers_every_four_state_tree(index):
    G = all_rooted_trees(STATES, "NPC")[index]
    D = tree_metric([(a, b, 1.0) for a, b in G.edges], STATES)
    H = nj_state_graph(neighbor_joining(D, STATES), STATES, "NPC")
    assert rf_rooted(H, G, "NPC") == 0


def test_optimal_arborescence_matches_best_ranked_tree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = rng.uniform(0.05, 1.0, size=(4, 4))
        np.fill_diagonal(T, 0)
        best = rank_arborescences(T, STATES, "NPC")[0][2]
        assert rf_rooted(optimal_arborescence(T, STATES, "NPC"), best, "NPC") == 0
