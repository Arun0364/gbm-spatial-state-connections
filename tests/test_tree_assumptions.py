import numpy as np

from three_point_test import four_point, three_point


def test_three_point_is_zero_for_an_ultrametric():
    D = np.array([[0, 2, 6, 6],
                  [2, 0, 6, 6],
                  [6, 6, 0, 4],
                  [6, 6, 4, 0]], dtype=float)
    mean, worst = three_point(D)
    assert mean == 0 and worst == 0


def test_four_point_is_zero_for_an_additive_metric():
    D = np.array([[0, 3, 4, 7],
                  [3, 0, 5, 8],
                  [4, 5, 0, 5],
                  [7, 8, 5, 0]], dtype=float)
    assert four_point(D) == 0


def test_four_point_detects_a_non_additive_metric():
    D = np.array([[0, 1, 5, 5],
                  [1, 0, 5, 1],
                  [5, 5, 0, 1],
                  [5, 1, 1, 0]], dtype=float)
    assert four_point(D) > 0
