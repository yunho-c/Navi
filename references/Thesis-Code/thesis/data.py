from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import json

import numpy as np
import pandas as pd
import cv2 as cv

from thesis.geometry import Ellipse
from thesis.segmentation import IrisImage
from thesis.tracking.gaze import GazeModel, BasicGaze
from thesis.tracking.features import normalize_coordinates, pupil_detector

from thesis.tools.st_utils import fit_else_ref #, create_deepeye_func
deepeye_ref = fit_else_ref #create_deepeye_func()


@dataclass
class GazeImage:
    image_path: str
    # pupil: Ellipse
    # glints: List[(float, float)]
    screen_position: (int, int)

    @property
    def image(self):
        return cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

    @staticmethod
    def from_json(path: str, data: dict):
        # pupil = Ellipse.from_dict(data['pupil'])
        # glints = data['glints']
        screen_position = data['position']
        return GazeImage(os.path.join(path, data['image']), screen_position)


@dataclass
class GazeDataset:
    name: str
    calibration_samples: List[GazeImage]
    test_samples: List[GazeImage]
    model: GazeModel

    def __len__(self):
        return len(self.test_samples)

    @staticmethod
    def from_path(path: str):
        with open(os.path.join(path, 'data.json')) as file:
            data = json.load(file)
            calibration_samples = list(map(lambda d: GazeImage.from_json(path, d), data['calibration']))
            test_samples = list(map(lambda d: GazeImage.from_json(path, d), data['test']))

            model = BasicGaze(data['screen']['res-y'], data['screen']['res-x'], data['fov'],
                              pupil_detector=deepeye_ref)
            print(data['fov'])
            images = [s.image for s in calibration_samples]
            gaze_positions = [s.screen_position for s in calibration_samples]
            model.calibrate(images, gaze_positions)

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            # print(normalize_coordinates(gaze_positions, 2160, 3840))
            # print(model.predict(images))

            return GazeDataset(name, calibration_samples, test_samples, model)

    def __repr__(self):
        return f'calibration samples: {len(self.calibration_samples)}, test samples: {len(self.test_samples)}'


@dataclass
class SegmentationSample:
    image: IrisImage
    user_id: str
    eye: str
    image_id: str
    session_id: str

    @staticmethod
    def from_dict(data: dict):
        image = IrisImage.from_dict(data)
        info = data['info']
        return SegmentationSample(image, **info)


@dataclass
class SegmentationDataset:
    name: str
    samples: List[SegmentationSample]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def from_path(path: str) -> SegmentationDataset:
        with open(path) as file:
            data = json.load(file)
            images = map(SegmentationSample.from_dict, data['data'])

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            return SegmentationDataset(name, list(images))


@dataclass
class PupilSample:
    image_path: str
    center: (int, int)

    @property
    def image(self):
        return cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

    @staticmethod
    def from_json(data: dict) -> PupilSample:
        screen_position = data['position']
        return PupilSample(data['image'], screen_position)


@dataclass
class PupilDataset:
    name: str
    samples: List[PupilSample]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def from_path(path: str) -> PupilDataset:
        with open(path) as file:
            data = json.load(file)
            images = map(PupilSample.from_json, data['data'])
            images = filter(lambda x: os.path.isfile(x.image_path), images)

            if 'name' in data:
                name = data['name']
            else:
                name = 'unnamed'

            return PupilDataset(name, list(images))
