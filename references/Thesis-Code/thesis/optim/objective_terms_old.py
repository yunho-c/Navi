from abc import ABC, abstractmethod
from typing import TypeVar, Callable

import cv2 as cv
import numpy as np

from thesis.data import SegmentationSample, GazeImage, PupilSample
from thesis.entropy import gradient_histogram, histogram, entropy, \
    mutual_information_grad
from thesis.tracking.gaze import GazeModel
from thesis.tracking.features import normalize_coordinates, pupil_detector
from thesis.segmentation import IrisCodeEncoder, IrisImage

from pupilfit import fit_else, fit_excuse

ANGULAR_RES = 30
RADIAL_RES = 18


def bilateral_filter(img, kernel_size, sigma_c, sigma_s):
    return cv.bilateralFilter(img, kernel_size, sigma_c, sigma_s)


def gradient_entropy(image, mask):
    hist = gradient_histogram(image, mask)
    return entropy(hist)


def intensity_entropy(image, mask):
    hist = histogram(image, mask)
    return entropy(hist)


T = TypeVar('T')


class SegmentationTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        ...


class GradientHistogramTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        ...


class GaborTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        ...


class GazeTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, model: GazeModel, sample: T, filtered: np.ndarray) -> float:
        ...


class PupilTerm(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        ...


class AbsoluteGradientEntropy(GradientHistogramTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return entropy_filtered


class AbsoluteOriginalGradientEntropy(GradientHistogramTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return entropy_source


class RelativeGradientEntropy(GradientHistogramTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return entropy_filtered / entropy_source


class AbsoluteGaborMutualS0(GaborTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return mutual_information[0]


class AbsoluteGaborSourceS0(GaborTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return entropy_source[0]


class AbsoluteGaborFilteredS0(GaborTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return entropy_filtered[0]


# TODO: Add masks to calculation!
def get_polar_images(sample: SegmentationSample, filtered: np.ndarray) -> (np.ndarray, np.ndarray):
    sample_polar, _ = sample.image.to_polar(ANGULAR_RES, RADIAL_RES)
    filtered_img = IrisImage(sample.image.segmentation, filtered)
    filtered_polar, _ = filtered_img.to_polar(ANGULAR_RES, RADIAL_RES)
    return sample_polar, filtered_polar


class AbsoluteMutualInformation(GradientHistogramTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return mutual_information


class RelativeMutualInformation(GradientHistogramTerm):
    def __call__(self, entropy_source, entropy_filtered, mutual_information) -> float:
        return mutual_information / entropy_source


# class AbsoluteGaborMutualAt1x(GradientHistogramTerm)


class IrisCodeSimilarity(SegmentationTerm):
    """Defined as (1-HD)"""

    def __init__(self):
        super().__init__()
        self.encoder = IrisCodeEncoder(scales=3,
                                       angles=5,
                                       angular_resolution=20,
                                       radial_resolution=18,
                                       wavelength_base=0.5,
                                       mult=1.41)

    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        code_sample = self.encoder.encode(sample.image)
        filtered_iris_image = IrisImage(sample.image.segmentation, filtered)
        code_filtered = self.encoder.encode(filtered_iris_image)
        return 1 - code_sample.dist(code_filtered)


class ImageSimilarity(SegmentationTerm):

    def __call__(self, sample: SegmentationSample, filtered: np.ndarray) -> float:
        sample_masked = sample.image.image * sample.image.mask
        sample_masked = sample_masked / np.linalg.norm(sample_masked)
        filtered_masked = filtered * sample.image.mask / 255
        norm = np.linalg.norm(filtered_masked)
        if norm == 0:
            filtered_masked = np.zeros(filtered_masked.shape)
        else:
            filtered_masked = filtered_masked / np.linalg.norm(filtered_masked)
        dist = np.linalg.norm(sample_masked - filtered_masked)
        return 1 - dist


class AbsoluteGazeAccuracy(GazeTerm):
    def __call__(self, model: GazeModel, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze = model.predict(filtered)
        true = normalize_coordinates(np.array([sample.screen_position]), 2160, 3840)
        return np.linalg.norm(np.array(gaze) - np.array(true))


class RelativeGazeAccuracy(GazeTerm):
    def __call__(self, model: GazeModel, sample: GazeImage, filtered: np.ndarray) -> float:
        gaze_a = model.predict(sample.image)
        gaze_b = model.predict(filtered)
        screen = normalize_coordinates(np.array([sample.screen_position]), 2160, 3840) # TODO: Set resolution!
        dist_a = np.linalg.norm(np.array(gaze_a) - np.array(screen))
        dist_b = np.linalg.norm(np.array(gaze_b) - np.array(screen))

        return dist_b / dist_a


def _pupil_detector(image):
    y, x = pupil_detector(image)[:2]
    return x, y


def pupil_distance_absolute(detector: Callable, sample: PupilSample, filtered: np.ndarray) -> float:
    predicted = detector(filtered)
    return np.linalg.norm(np.array(predicted) - np.array(sample.center))


def pupil_distance_relative(detector: Callable, sample: PupilSample, filtered: np.ndarray) -> float:
    predicted_unmodified = detector(sample.image)
    predicted_filtered = detector(filtered)
    dist_unmodified = np.linalg.norm(np.array(predicted_unmodified) - np.array(sample.center))
    dist_filtered = np.linalg.norm(np.array(predicted_filtered) - np.array(sample.center))

    return dist_filtered / (dist_unmodified + 10e-6)


def fit_else_center(image: np.ndarray) -> (float, float):
    return fit_else(image)[0]


def fit_excuse_center(image: np.ndarray) -> (float, float):
    return fit_excuse(image)[0]


class AbsolutePupilDistanceBaseAlgorithm(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_absolute(_pupil_detector, sample, filtered)


class RelativePupilDistanceBaseAlgorithm(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_relative(_pupil_detector, sample, filtered)


class AbsolutePupilDistanceElse(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_absolute(fit_else_center, sample, filtered)


class RelativePupilDistanceElse(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_relative(fit_else_center, sample, filtered)


class AbsolutePupilDistanceExcuse(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_absolute(fit_excuse_center, sample, filtered)


class RelativePupilDistanceExcuse(PupilTerm):
    def __call__(self, sample: PupilSample, filtered: np.ndarray) -> float:
        return pupil_distance_relative(fit_excuse_center, sample, filtered)
