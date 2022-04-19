import os
import json
from typing import List

import numpy as np
import cv2
from thesis.data import GazeDataset, SegmentationDataset, PupilDataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import matplotlib.pyplot as plt


def load_json(directory, filename):
    """Load json file from subdirectory in "inputs/images" with the given filename
    - without .json extension!

    Returns: The json data as a dictionary or array (depending on the file).
    """
    with open(os.path.join('inputs', 'images', directory, f'{filename}.json')) as file:
        return np.array(json.load(file), dtype=np.float32)


def load_images(directory):
    """Load images from a subdirectory in "inputs/images" using OpenCV.

    Returns: The list of loaded images in order.
    """
    with open(os.path.join('inputs', 'images', directory, 'positions.json')) as file:
        screen_points = json.load(file)

    images = [cv2.imread(os.path.join(
        'inputs/images', directory, f'{i}.jpg')) for i in range(len(screen_points))]

    return np.array(images)


def dist(a, b):
    """Calculate the euclidean distance from a to b.
    """
    return np.linalg.norm(a - b)


def pupil_json_to_opencv(pupil):
    """Convert pupil loaded from json file to the format used by OpenCV,
    i.e. ((cx, cy), (ax, ay), angle)

    Returns: Tuple containing ((cx, cy), (ax, ay), angle)
    """
    p = pupil
    return ((p['cx'], p['cy']), (p['ax'], p['ay']), p['angle'])


def pupil_to_int(pupil):
    """Convert pupil parameters to integers. Useful when drawing.
    """
    p = pupil
    return ((int(p[0][0]), int(p[0][1])), (int(p[1][0]), int(p[1][1])), int(p[2]))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_moved_path(path, input_base, output_base):
    path_name, _ = os.path.splitext(path)
    end = os.path.relpath(path_name, start=input_base)
    return os.path.join(output_base, end)


def contour_plot(df, *, x: str, y: str, z: str, xlabel: str, ylabel: str, cmap='YlOrBr'):
    """
    Creates a triangulated contour plot from a data frame.

    Args:
        df: Source data frame.
        x: Column to use for x values.
        y: Column to use for y values.
        z: Column to use for z values.
        xlabel: Label for plot x-axis.
        ylabel: Label for plot y-axis.
        cmap (optional): Matplotlib color-map.
    """
    fig, ax = plt.subplots(1, 1)
    cp = ax.tricontourf(df[x], df[y], df[z], levels=3, cmap=cmap)
    ax.tricontour(
        df[x],
        df[y],
        df[z],
        linewidths=0.5,
        levels=3,
        colors='k',
        Nchunk=0,
    )
    plt.plot(df[x], df[y], 'ko', markersize=2)
    fig.colorbar(cp, ax=ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def load_gaze_data(datasets: List[str]) -> List[GazeDataset]:
    return list(map(GazeDataset.from_path, datasets))


def load_iris_data(datasets: List[str]) -> List[SegmentationDataset]:
    return list(map(SegmentationDataset.from_path, datasets))


def load_pupil_data(datasets: List[str]) -> List[PupilDataset]:
    return list(map(PupilDataset.from_path, datasets))
