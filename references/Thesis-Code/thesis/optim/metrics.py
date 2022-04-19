from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List

import cv2 as cv
import numpy as np
from pupilfit import fit_else, fit_excuse

from thesis.data import GazeImage, PupilSample, deepeye_ref
from thesis.entropy import joint_gradient_histogram, entropy, mutual_information_grad, joint_gabor_histogram
from thesis.tracking.gaze import GazeModel
from thesis.tracking.features import normalize_coordinates, pupil_detector
from thesis.segmentation import SKImageIrisCodeEncoder

import sys

this = sys.modules[__name__]


class Logger:

    def __init__(self):
        self.data = defaultdict(list)

    def add(self, point: str, value: float):
        self.data[point].append(value)

    def list_form(self):
        res = []
        for i in range(len(next(iter(self.data)))):
            res.append({k: v[i] for k, v in self.data.items()})
        return res

    def columns(self):
        return list(self.data.keys())

    def means(self):
        return list(map(np.mean, self.data.values()))


class IrisMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, polar_image, polar_filtered, mask):
        ...


class ImageMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, image, filtered, mask):
        ...


class GazeMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, model: GazeModel, sample: GazeImage, filtered: np.ndarray):
        ...


class PupilMetric(ABC):
    @property
    def columns(self):
        ...

    @abstractmethod
    def log(self, results: Logger, pupil_sample: PupilSample, filtered: np.ndarray):
        ...


class GradientEntropyIris(IrisMetric):
    columns = ['gradient_entropy_iris_source', 'gradient_entropy_iris_filtered', 'gradient_mutual_information_iris']

    def __init__(self, histogram_divisions):
        self.histogram_divisions = histogram_divisions

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        hist_source, hist_filtered, hist_joint = joint_gradient_histogram(polar_image, polar_filtered, mask,
                                                                          self.histogram_divisions)
        entropy_source = entropy(hist_source)
        entropy_filtered = entropy(hist_filtered)
        mutual_information = mutual_information_grad(hist_source, hist_filtered, hist_joint)

        results.add('gradient_entropy_iris_source', entropy_source)
        results.add('gradient_entropy_iris_filtered', entropy_filtered)
        results.add('gradient_mutual_information_iris', mutual_information)


class GaborEntropyIris(IrisMetric):
    def __init__(self, scales, angles_per_scale, histogram_divisions):
        self.scales = scales
        self.angles_per_scale = angles_per_scale
        self.histogram_divisions = histogram_divisions

        self._columns = list(chain.from_iterable([
            [f'gabor_entropy_iris_source_{1 / 2 ** scale}x', f'gabor_entropy_iris_filtered_{1 / 2 ** scale}x',
             f'gabor_mutual_information_iris_{1 / 2 ** scale}x']
            for scale in range(self.scales)
        ]))

    @property
    def columns(self):
        return self._columns

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        angles = np.linspace(0, np.pi - np.pi / self.angles_per_scale,
                             self.angles_per_scale)  # TODO: Consider subtracting small amount
        for scale in range(self.scales):
            entropy_source = 0
            entropy_filtered = 0
            mutual_information = 0

            for theta in angles:
                hist_source, hist_filtered, hist_joint = joint_gabor_histogram(polar_image, polar_filtered, mask,
                                                                               theta, self.histogram_divisions)

                entropy_source += entropy(hist_source)
                entropy_filtered += entropy(hist_filtered)
                mutual_information += mutual_information_grad(hist_source, hist_filtered, hist_joint)

            # Hopefully fixed this to now be an actual AVERAGE of all the test angles at a given scale.
            entropy_source /= self.angles_per_scale
            entropy_filtered /= self.angles_per_scale
            mutual_information /= self.angles_per_scale

            results.add(f'gabor_entropy_iris_source_{1 / 2 ** scale}x', entropy_source)
            results.add(f'gabor_entropy_iris_filtered_{1 / 2 ** scale}x', entropy_filtered)
            results.add(f'gabor_mutual_information_iris_{1 / 2 ** scale}x', mutual_information)

            polar_image = cv.pyrDown(polar_image)
            polar_filtered = cv.pyrDown(polar_filtered)
            mask = cv.resize(mask, (polar_image.shape[1], polar_image.shape[0]), interpolation=cv.INTER_NEAREST)


class GradientEntropyImage(ImageMetric):
    columns = ['gradient_entropy_image_source', 'gradient_entropy_image_filtered', 'gradient_mutual_information_image']

    def __init__(self, histogram_divisions):
        self.histogram_divisions = histogram_divisions

    def log(self, results: Logger, image, filtered, mask):
        hist_source, hist_filtered, hist_joint = joint_gradient_histogram(image, filtered, mask,
                                                                          self.histogram_divisions)
        entropy_source = entropy(hist_source)
        entropy_filtered = entropy(hist_filtered)
        mutual_information = mutual_information_grad(hist_source, hist_filtered, hist_joint)

        results.add('gradient_entropy_image_source', entropy_source)
        results.add('gradient_entropy_image_filtered', entropy_filtered)
        results.add('gradient_mutual_information_image', mutual_information)


class GaborEntropyImage(ImageMetric):
    def __init__(self, scales, angles_per_scale, histogram_divisions):
        self.scales = scales
        self.angles_per_scale = angles_per_scale
        self.histogram_divisions = histogram_divisions

        self._columns = list(chain.from_iterable([
            [f'gabor_entropy_image_source_{1 / 2 ** scale}x', f'gabor_entropy_image_filtered_{1 / 2 ** scale}x',
             f'gabor_mutual_information_image_{1 / 2 ** scale}x']
            for scale in range(self.scales)
        ]))

    @property
    def columns(self):
        return self._columns

    def log(self, results: Logger, image, filtered, mask):
        angles = np.linspace(0, np.pi - np.pi / self.angles_per_scale,
                             self.angles_per_scale)  # TODO: Consider subtracting small amount
        for scale in range(self.scales):
            # Start at half scale
            image = cv.pyrDown(image)
            filtered = cv.pyrDown(filtered)
            mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)

            entropy_source = 0
            entropy_filtered = 0
            mutual_information = 0

            for theta in angles:
                hist_source, hist_filtered, hist_joint = joint_gabor_histogram(image, filtered, mask,
                                                                               theta, self.histogram_divisions)

                entropy_source += entropy(hist_source)
                entropy_filtered += entropy(hist_filtered)
                mutual_information += mutual_information_grad(hist_source, hist_filtered, hist_joint)

                results.add(f'gabor_entropy_image_source_{1 / 2 ** scale}x', entropy_source)
                results.add(f'gabor_entropy_image_filtered_{1 / 2 ** scale}x', entropy_filtered)
                results.add(f'gabor_mutual_information_image_{1 / 2 ** scale}x', mutual_information)

            # Hopefully fixed this to now be an actual AVERAGE of all the test angles at a given scale.
            entropy_source /= self.angles_per_scale
            entropy_filtered /= self.angles_per_scale
            mutual_information /= self.angles_per_scale




class GazeAccuracy(GazeMetric):
    columns = ['gaze_angle_error_source', 'gaze_aongle_error_filtered']

    def log(self, results: Logger, model: GazeModel, sample: GazeImage, filtered: np.ndarray):
        gaze_source = model.predict(sample.image)
        gaze_filtered = model.predict(filtered)
        screen = normalize_coordinates(np.array([sample.screen_position]), model.screen_height,
                                       model.screen_width)

        dist_source = np.linalg.norm(np.array(gaze_source) - np.array(screen))
        dist_filtered = np.linalg.norm(np.array(gaze_filtered) - np.array(screen))

        angle_error_source = model.fov * dist_source
        angle_error_filtered = model.fov * dist_filtered

        results.add('gaze_angle_error_source', angle_error_source)
        results.add('gaze_angle_error_filtered', angle_error_filtered)


class PupilDetector(ABC):
    name = ...

    @abstractmethod
    def __call__(self, image: np.ndarray) -> (float, float):
        ...


class BaseDetector(PupilDetector):
    name = 'base'

    def __call__(self, image: np.ndarray) -> (float, float):
        y, x = pupil_detector(image)[:2]
        return x, y


class ElseDetector(PupilDetector):
    name = 'else'

    def __call__(self, image: np.ndarray) -> (float, float):
        return fit_else(image)[0]


class ExcuseDetector(PupilDetector):
    name = 'excuse'

    def __call__(self, image: np.ndarray) -> (float, float):
        return fit_excuse(image)[0]


class DeepEyeDetector(PupilDetector):
    name = 'deep_eye'

    def __call__(self, image: np.ndarray) -> (float, float):
        image = np.uint8(image)
        cy, cx, _, _, _ = deepeye_ref(image)
        return cx, cy


class PupilDetectionError(PupilMetric):
    def __init__(self, detectors: List[type(PupilDetector)]):
        self.detectors = [getattr(this, x)() for x in detectors]
        self._columns = list(chain.from_iterable([
            [f'pupil_distance_{d.name}_pixel_error_source', f'pupil_distance_{d.name}_pixel_error_filtered']
            for d in self.detectors
        ]))

    @property
    def columns(self):
        return self._columns

    def log(self, results: Logger, pupil_sample: PupilSample, filtered: np.ndarray):
        for d in self.detectors:
            predicted_unmodified = d(pupil_sample.image)
            predicted_filtered = d(filtered)
            dist_unmodified = np.linalg.norm(np.array(predicted_unmodified) - np.array(pupil_sample.center))
            dist_filtered = np.linalg.norm(np.array(predicted_filtered) - np.array(pupil_sample.center))
            results.add(f'pupil_distance_{d.name}_pixel_error_source', dist_unmodified)
            results.add(f'pupil_distance_{d.name}_pixel_error_filtered', dist_filtered)


class IrisDistance(IrisMetric):
    columns = ['iris_code_distance', 'polar_image_normalized_similarity']

    def __init__(self, angles, scales, eps):
        self.encoder = SKImageIrisCodeEncoder(angles, -1, -1, scales, eps)

    def log(self, results: Logger, polar_image, polar_filtered, mask):
        code_source = self.encoder.encode_raw(polar_image, mask)
        code_filtered = self.encoder.encode_raw(polar_filtered, mask)
        iris_code_similarity = 1 - code_source.dist(code_filtered)
        results.add('iris_code_similarity', iris_code_similarity)

        source_masked = polar_image * mask
        source_masked = source_masked / np.linalg.norm(source_masked)
        filtered_masked = polar_filtered * mask
        norm = np.linalg.norm(filtered_masked)
        if norm == 0:
            return np.nan
        else:
            filtered_masked = filtered_masked / norm
            similarity = 1 - np.linalg.norm(source_masked - filtered_masked)
            results.add('polar_image_normalized_similarity', similarity)


class ImageSimilarity(ImageMetric):
    columns = ['image_normalized_similarity']

    def log(self, results: Logger, image, filtered, mask):
        norm = np.linalg.norm(filtered)
        if norm == 0:
            return np.nan
        else:
            filtered = filtered / norm
            similarity = 1 - np.linalg.norm(image - filtered)
            results.add('image_normalized_similarity', similarity)