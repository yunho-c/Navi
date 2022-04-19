from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import cv2 as cv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Protocol

from thesis.tracking import features


class FeatureTransformer(Protocol):
    @abstractmethod
    def fit_transform(self, X, y=None):
        ...


class Model(Protocol):
    @abstractmethod
    def fit(self, X, y=None):
        ...


@dataclass
class GazeModel:
    screen_height: int
    screen_width: int
    fov: float

    @abstractmethod
    def calibrate(self, images, gaze_positions):
        """Calibrate the model.

        Args:
            gaze_positions:
            images:
        """
        ...

    @abstractmethod
    def predict(self, images):
        """Predict gaze from an input image.

        Args:
            images (array-like): Single image or list of images to predict.

        Returns:
            Tuple[float, float]: ...
        """
        ...


def solve_similarity(source, target):
    mats = []
    for p in source:
        mats.append(np.array([
            [p[0], -p[1], 1, 0],
            [p[1], p[0], 0, 1]
        ]))

    mat = np.vstack(mats)

    b = target.reshape(-1)

    s: np.ndarray = np.linalg.inv(mat).dot(b)
    solution = np.array([
        [s[0], -s[1], s[2]],
        [s[1], s[0], s[3]],
        [0., 0., 1.]
    ])
    return solution


def solve_affine(source, target):
    mats = []
    for p in source:
        mats.append(np.array([
            [p[0], p[1], 1, 0, 0, 0],
            [0, 0, 0, p[0], p[1], 1]
        ]))

    mat = np.vstack(mats)

    b = target.reshape(-1)

    solution = np.linalg.inv(mat).dot(b)
    solution = solution.reshape(2, 3)
    solution = np.vstack((solution, [0, 0, 1]))
    return solution


def solve_homography(source, target):
    homography, mask = cv.findHomography(source, target, 0)
    return homography


def hom(points):
    points = np.array(points, dtype=np.float)
    if points.ndim == 1:
        points = points.reshape((points.shape[0], 1))
    z = np.ones((points.shape[1], 1))
    r = np.vstack((points, z))
    return r


class BasicGaze(GazeModel):
    def __init__(self, screen_height: int, screen_width: int, fov: float, glint_args=None, model: Pipeline = None,
                 pupil_detector=features.pupil_detector):
        super().__init__(screen_height, screen_width, fov)
        if glint_args is None:
            self.glint_args = {}
        else:
            self.glint_args = glint_args
        if model is None:
            model = Pipeline([
                ('design matrix', PolynomialFeatures(2)),
                ('model', LinearRegression())
            ])

        self.model = model
        self.pupil_detector = pupil_detector

    def _preprocess(self, images):
        # images = np.array([cv.GaussianBlur(img, (0, 0), 0.5) for img in images])
        images = np.array(images)
        if len(images.shape) == 2:
            images = [images]

        pupils = np.array([self.pupil_detector(img) for img in images])

        # print(pupils[0])
        centers = [[p[0], p[1]] for p in pupils]
        glints = [
            features.find_glints(img, center, **self.glint_args)
            for img, center in zip(images, centers)
        ]

        glints = [
            min(gs, key=lambda g: g[1]) if len(gs) > 0 else [] for gs in glints
        ]

        # nan_removed = np.array([g for g in glints if ~np.isnan(g).any()])
        # avg = nan_removed.mean(axis=0)
        normed = []
        for i, (c, g) in enumerate(zip(centers, glints)):
            # print(i)
            if len(g) == 0:
                normed.append([-1, -1])
            else:
                normed.append([c[0] - g[0], c[1] - g[1]])

        return np.array(normed)

    def calibrate(self, images, gaze_positions):
        pupil = self._preprocess(images)
        norm_pos = features.normalize_coordinates(gaze_positions, self.screen_height, self.screen_width)
        self.model.fit(pupil, norm_pos)

    def predict(self, images):
        pupil = self._preprocess(images)
        norm_predict = self.model.predict(pupil)
        return norm_predict  # features.unnormalize_coordinates(norm_predict, 2160, 3840)

    def score(self, images, gaze_positions):
        pupil = self._preprocess(images)
        norm_pos = features.normalize_coordinates(gaze_positions, 2160, 3840)
        return self.model.score(pupil, norm_pos)
