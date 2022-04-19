import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import click
from glob2 import glob
from tqdm import tqdm
import random


class EyeSide(str, Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass
class ImageInfo:
    user_id: str
    eye: EyeSide
    image_id: str
    session_id: str


class DataFormat(ABC):
    @staticmethod
    @abstractmethod
    def get_info(name) -> ImageInfo:
        ...


class UbirisFormat(DataFormat):

    @staticmethod
    def get_info(name) -> ImageInfo:
        matches = re.match(r'C(?P<user_id>[0-9]+)_S(?P<session_id>[0-9]+)_I(?P<image_id>[0-9]+)', name)
        # This conversion is done to preserve IDs but keep ID equality between left and right eyes
        user_id = str(int(matches.group('user_id')) // 2 * 2)
        eye = EyeSide.RIGHT if int(matches.group('user_id')) % 2 else EyeSide.LEFT

        return ImageInfo(
            user_id, eye,
            image_id=matches.group('image_id'),
            session_id=matches.group('session_id'))


class CasiaIVFormat(DataFormat):
    @staticmethod
    def get_info(name) -> ImageInfo:
        matches = re.match(r'S(?P<user_id>[0-9]+)(?P<eye_side>[LR])(?P<image_id>[0-9]+)', name)
        # This conversion is done to preserve IDs but keep ID equality between left and right eyes
        eye = EyeSide.RIGHT if matches.group('eye_side') == 'R' else EyeSide.LEFT
        return ImageInfo(
            user_id=matches.group('user_id'),
            eye=eye,
            image_id=matches.group('image_id'),
            session_id='')


def read_points_from_file(path):
    with open(path) as f:
        lines = f.readlines()
        tokens = [line.split() for line in lines]
        coords = [[float(t) for t in line] for line in tokens]
        return coords


def get_segmented_images(dataset: str, path: str):
    image_folder = os.path.join(path, 'images', dataset)
    segment_folder = os.path.join(path, 'segmentation/IRISSEG-EP/dataset', dataset)

    extensions = ('.jpg', '.png', '.tiff', '.bmp')
    images = set()
    for ext in extensions:
        images |= set(glob(os.path.join(image_folder, f'**/**{ext}'), recursive=True))
    segmentations = set(glob(os.path.join(segment_folder, '**/**.txt'), recursive=True))

    res = []

    formatter = None
    if 'casia4i' in dataset:
        formatter = CasiaIVFormat()
    elif dataset == 'ubiris':
        formatter = UbirisFormat()

    for img_path in tqdm(images):
        img_base = os.path.basename(img_path)
        img_name = os.path.splitext(img_base)[0]

        files = filter(lambda x: img_name in x, segmentations)
        points = {
            os.path.basename(f).split(os.extsep)[1]: read_points_from_file(f)
            for f in files
        }

        if len(points) == 0:  # Only append files that have corresponding annotations
            continue

        res.append({
            'image': os.path.abspath(img_path),
            'points': points,
            'info': formatter.get_info(img_name).__dict__
        })

    return {
        'dataset': dataset,
        'data': res
    }


@click.group()
def data():
    pass


@data.group()
def iris():
    pass


@iris.command()
@click.argument('path')
@click.argument('dataset')
@click.argument('output')
@click.option('-l', '--limit', type=int, default=0)
def create(path, dataset, output, limit):
    """Create json file containing dataset points and image file paths.

    PATH is the base path to the "iris" dataset.

    DATASET is the subfolder in the "images" folder to create a json file for.

    OUTPUT is the path to the resulting json file.
    """
    dataset = get_segmented_images(dataset, path)
    if limit > 0:
        dataset['data'] = random.sample(dataset['data'], limit)
    with open(output, 'w') as f:
        json.dump(dataset, f)


@data.group()
def track():
    pass


def combine_json(positions: list, calibration_samples: int) -> dict:
    combined = [{
        'image': f'{i}.jpg',
        'position': position
    } for i, position in enumerate(positions)]

    return {
        'calibration': combined[:calibration_samples],
        'test': combined[calibration_samples:]
    }


@track.command()
@click.argument('path')
@click.argument('calibration_samples', type=int)
@click.option('--res-x', type=int, default=3840)
@click.option('--res-y', type=int, default=2160)
@click.option('--fov', type=float, default=87.0)
def create(path: str, calibration_samples: int, res_x: int, res_y: int, fov: float):
    """Create single json file for gaze data.

    PATH: Path to folder containing images and json files.
    OUTPUT: Filename of output json file.
    CALIBRATION_SAMPLES: Number of images used for calibration.
    """
    positions_path = os.path.join(path, 'positions.json')
    with open(positions_path) as positions_file:
        positions = json.load(positions_file)

        output_data = combine_json(positions, calibration_samples)
        output_data['screen'] = {
            'res-x': res_x,
            'res-y': res_y
        }
        output_data['fov'] = fov

        with open(os.path.join(path, 'data.json'), 'w') as output_file:
            json.dump(output_data, output_file, indent=4)

        print("Created data.json")


@data.group()
def pupil():
    ...


def get(paths, limit):
    images = []
    n = 0
    for file_path in tqdm(paths):
        directory = os.path.splitext(file_path)[0]
        with open(file_path) as positions_file:
            lines = positions_file.readlines()
            positions = [map(int, line.strip().split()[1:]) for line in lines[1:]]
            for image, x, y in positions:
                image_path = os.path.join(directory, f'{image:010d}.png')
                images.append({
                    'image': os.path.abspath(image_path),
                    'position': (x / 2, 288 - y / 2)
                })
                n += 1

    if 0 < limit < n:
        return random.sample(images, limit)
    else:
        return images


@pupil.command()
@click.argument('path')
@click.option('--limit', type=int, default=0)
def create(path, limit):
    """Create json file for data.

    PATH: Path to base pupil folder.
    """

    paths = glob(os.path.join(path, '**/data set*.txt'), recursive=True)
    print('Found the following datasets:')
    for p in paths:
        print(f'\t{p}')

    images = get(paths, limit)

    with open(os.path.join(path, 'info.json'), 'w') as output_file:
        json.dump({
            'name': 'pupil_std',
            'data': images
        }, output_file)


if __name__ == '__main__':
    data()
