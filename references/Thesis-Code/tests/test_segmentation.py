from hypothesis import given, strategies as st

from thesis.segmentation import *
from thesis.geometry import *


def test_iris_code_produces_result():
    ...


def test_segmentation_mask_draws_successfully():
    inner = [
        [0, 1],
        [0.8, 0.6],
        [1, 0],
        [0, -1],
        [-1, 0],
    ]

    outer = [
        [0, 2],
        [1.6, 1.2],
        [2, 0],
        [0, -2],
        [-2, 0],
    ]

    upper = [
        [-1, 1.8],
        [0, 1.7],
        [1, 1.8],
    ]

    lower = [
        [-1, -1.8],
        [0, -1.7],
        [1, -1.8],
    ]

    seg = IrisSegmentation(
        Ellipse.from_points(inner),
        Ellipse.from_points(outer),
        Quadratic.from_points_precise(*upper),
        Quadratic.from_points_precise(*lower))
    mask = seg.get_mask((5, 5))
