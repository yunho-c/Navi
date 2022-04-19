import pytest

import numpy as np

from thesis.optim.population import TruncationSelection, UniformCrossover, GaussianMutation


def test_truncation_selection():
    k = 5
    selector = TruncationSelection(k)
    test_population = [16, 1, 0.2, 5, 10, 15, 20]
    res = selector.select(test_population)
    assert len(res) == len(test_population)
    assert len(res[0]) == 2


def test_uniform_crossover():
    method = UniformCrossover()
    a, b = [1, 1, 1, 1], [2, 2, 2, 2]
    res = method.crossover(a, b)
    assert len(res) == len(a) == len(b)
    for e in res:
        assert e in a or e in b


def test_gaussian_mutation():
    mut = GaussianMutation()
    child = np.array([1, 2, 3, 4])
    res = mut.mutate(child)
    assert len(child) == len(res)
    assert all(child != res)