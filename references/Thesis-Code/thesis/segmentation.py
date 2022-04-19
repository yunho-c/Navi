from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json

import cv2 as cv
import numpy as np
import math
from skimage.filters import gabor, gabor_kernel
from scipy import ndimage as ndi

from numba import jit, vectorize, prange

from thesis.geometry import Quadratic, Ellipse, Mask, Vec2


@jit(nopython=True)
def radius_at_angle(ellipse, theta):
    _, (a, b), angle = ellipse
    return (a * b) / np.sqrt(a ** 2 * np.sin(theta - angle) ** 2 + b ** 2 * np.cos(theta - angle) ** 2)


@jit(nopython=True)
def intersect_angle(ellipse, theta):
    (cx, cy), _, _ = ellipse
    r = radius_at_angle(ellipse, theta)
    return cx + r * np.sin(theta), cy + r * np.cos(theta)


@jit(nopython=True)
def polar_to_cartesian(cx, cy, radius, theta):
    return cx + radius * np.sin(theta), cy + radius * np.cos(theta)


@jit(nopython=True)
def linear_interpolation(start: (float, float), stop: (float, float), num: int) -> (np.ndarray, np.ndarray):
    return np.linspace(start[0], stop[0], num), np.linspace(start[1], stop[1], num)


@jit(nopython=True, parallel=True, error_model='numpy')
def polar_base_loop(img, mask, radial_resolution, angular_resolution, angle_steps, radii, cx, cy, n_samples=100):
    output = np.zeros((radial_resolution, angular_resolution), np.uint8)
    output_mask = np.zeros((radial_resolution, angular_resolution), np.uint8)

    for j in prange(radial_resolution):
        for i in prange(angular_resolution):
            r_left_1, r_left_2 = radii[j, i], radii[j + 1, i]
            r_right_1, r_right_2 = radii[j, i + 1], radii[j + 1, i + 1]

            min_radius = min(r_left_1, r_right_1)
            max_radius = max(r_left_2, r_right_2)

            side_len = int(np.sqrt(n_samples))
            # random_thetas_seed = np.random.uniform(0, 1, n_samples)
            # random_radii_seed = np.random.uniform(0, 1, n_samples)
            random_thetas_seed = np.linspace(0, 1, side_len).repeat(side_len)
            random_radii_seed = np.zeros(side_len ** 2)
            for x in range(side_len):
                random_radii_seed[x:x + 1] = x / side_len

            # random_thetas_seed = np.random.uniform(0, 1, n_samples)
            random_thetas = angle_steps[i] + random_thetas_seed * (angle_steps[i + 1] - angle_steps[i])
            min_radii = min_radius + random_thetas_seed * abs(r_right_1 - r_left_1)
            max_radii = max_radius + random_thetas_seed * abs(r_right_2 - r_left_2)
            random_radii = min_radii + random_radii_seed * (max_radii - min_radii)
            # random_radii = np.random.uniform(min_radii, max_radii, n_samples)

            # coords = np.zeros((2, n_samples), np.uint8)
            val_img = 0.
            val_mask = 0.
            num = 0
            for k in range(n_samples):
                x, y = polar_to_cartesian(cx, cy, random_radii[k], random_thetas[k])
                x, y = round(x), round(y)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    val_img += img[y, x]
                    val_mask += mask[y, x]
                    num += 1

            val_img /= num
            val_mask /= num
            # coords = [polar_to_cartesian(inner[0], r, t) for (r, t) in zip(random_radii, random_thetas)]
            # coords = [(round(x), round(y)) for (x, y) in coords]
            # coords = np.array([[x, y] for (x, y) in coords if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]])
            # coords = list(filter(lambda c: 0 <= c[0] < img.shape[1] and 0 <= c[1] < img.shape[0], coords))

            # val_img = img[coords].mean()
            # val_mask = mask[coords].mean()
            # val_img = np.array([img[y, x] for (x, y) in coords]).mean()
            # val_mask = np.array([mask[y, x] for (x, y) in coords]).mean()
            if math.isnan(val_img):
                continue
            output[j, i] = val_img
            output_mask[j, i] = 1 if val_mask > 0.2 else 0

    return output, output_mask


@jit(nopython=True, parallel=True, error_model='numpy')
def polar_from_ellipses(img: np.ndarray, mask: np.ndarray, angular_resolution: int, radial_resolution: int, inner,
                        outer, start_angle=0, n_samples=100) -> (
        np.ndarray, np.ndarray):
    """Create polar image.

    Args:
        img: input image
        mask: image mask
        radial_resolution:
        angular_resolution:
        inner: inner boundary ellipse
        outer: outer boundary ellipse
        start_angle:

    Returns:

    """

    angle_steps = np.linspace(start_angle, start_angle + 2 * np.pi, angular_resolution + 1)
    radii = np.zeros((radial_resolution + 1, angular_resolution + 1))

    for i, theta in enumerate(angle_steps):
        start = radius_at_angle(inner, theta)
        stop = radius_at_angle(outer, theta)
        margin = radial_resolution // 8
        radial_steps = np.linspace(start, stop, radial_resolution + 1 + margin * 2)
        for j, step in enumerate(radial_steps[margin:][:-margin]):
            radii[j, i] = step

    # for i, theta in enumerate(angle_steps):
    #     start = intersect_angle(inner, theta)
    #     stop = intersect_angle(outer, theta)
    #     margin = radial_resolution // 4
    #     x_coord, y_coord = linear_interpolation(start, stop, radial_resolution + margin * 2)
    #
    #     for j, (x, y) in enumerate(zip(x_coord[margin:][:-margin], y_coord[margin:][:-margin])):
    #         if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
    #             continue
    #
    #         diff_y = y - int(y)
    #         diff_x = x - int(x)
    #         y1, y2 = int(y), int(y)+1
    #         x1, x2 = int(x), int(x)+1
    #         fxy1 = (1-diff_x)*img[y1, x1] + diff_x*img[y1, x2]
    #         fxy2 = (1-diff_x)*img[y2, x1] + diff_x*img[y2, x2]
    #         fxy = (1-diff_y)*fxy1 + diff_y*fxy2
    #         output[j, i] = fxy
    #         # output[j, i] = img[int(y), int(x)]
    #         output_mask[j, i] = mask[round(y), round(x)]
    #

    return polar_base_loop(img, mask, radial_resolution, angular_resolution, angle_steps, radii, inner[0][0],
                           inner[0][1], n_samples)


@dataclass
class IrisSegmentation(Mask):
    inner: Ellipse
    outer: Ellipse
    upper_eyelid: Quadratic
    lower_eyelid: Quadratic

    @staticmethod
    def from_dict(obj: dict) -> IrisSegmentation:
        return IrisSegmentation(
            inner=Ellipse.from_points(obj['inner']),
            outer=Ellipse.from_points(obj['outer']),
            upper_eyelid=Quadratic.from_points_least_sq(obj['upper']),
            lower_eyelid=Quadratic.from_points_least_sq(obj['lower'])
        )

    def get_mask(self, size: (int, int)) -> np.ndarray:
        mask_inner = self.inner.get_mask(size)
        mask_outer = self.outer.get_mask(size)
        mask_upper = self.upper_eyelid.get_mask(size)
        mask_lower = 1 - self.lower_eyelid.get_mask(size)

        base = mask_outer - mask_inner
        with_eyelids = base * mask_upper * mask_lower
        return with_eyelids

    def intersect_angle(self, theta: float) -> (Vec2, Vec2):
        p1 = self.inner.intersect_angle(theta)
        p2 = self.outer.intersect_angle(theta)
        return p1, p2


@dataclass
class IrisImage:
    segmentation: IrisSegmentation
    mask: np.ndarray
    image: np.ndarray

    def __init__(self, segmentation: IrisSegmentation, image: np.ndarray):
        self.segmentation = segmentation
        if len(image.shape) > 2:
            self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.mask = segmentation.get_mask((image.shape[1], image.shape[0]))

        self.polar = None
        self.saved_angular_resolution = 0
        self.saved_radial_resolution = 0

    @staticmethod
    def from_dict(data: dict) -> IrisImage:
        segmentation = IrisSegmentation.from_dict(data['points'])
        image = cv.imread(data['image'])
        return IrisImage(segmentation, image)

    def to_polar(self, angular_resolution, radial_resolution, start_angle=0, n_samples=100) -> (np.ndarray, np.ndarray):
        """Create polar image.

        Args:
            angular_resolution: Number of angular stops.
            radial_resolution: Number of stops from pupil to iris.
            start_angle:

        Returns:

        """
        if self.saved_angular_resolution != angular_resolution \
                or self.saved_radial_resolution != radial_resolution \
                or self.polar is None:
            self.polar = polar_from_ellipses(self.image, self.mask, angular_resolution, radial_resolution,
                                             self.segmentation.inner.as_tuple(), self.segmentation.outer.as_tuple(),
                                             start_angle, n_samples)

        return self.polar


@jit(nopython=True)
def cdist(a_code, a_mask, b_code, b_mask):
    mask = a_mask | b_mask
    n = mask.sum()
    if n == a_code.size:
        return 1
    else:
        return (a_code != b_code)[mask == 0].sum() / (a_code.size - n)


@dataclass
class IrisCode:
    code: np.ndarray
    mask: np.ndarray

    def dist(self, other):
        return cdist(self.code, self.mask, other.code, other.mask)

    def __len__(self):
        return self.code.size

    def shift(self, n_bits):
        return IrisCode(np.concatenate((self.code[n_bits:], self.code[:n_bits])),
                        np.concatenate((self.mask[n_bits:], self.mask[:n_bits])))

    def masked_image(self):
        c = np.array(self.code)
        c[self.mask == 1] = 0
        c = c / 2 + 0.5
        return c


class IrisCodeEncoder:
    def __init__(self, scales: int = 3,
                 angles: int = 3,
                 angular_resolution=20,
                 radial_resolution=10,
                 wavelength_base=0.5,
                 mult=1.41,
                 eps=0.01,
                 n_samples=100):
        self.kernels = []
        self.angular_resolution = angular_resolution
        self.radial_resolution = radial_resolution
        self.eps = eps
        self.n_samples = 100
        wavelength = wavelength_base
        for s in range(scales):
            sigma = wavelength / 0.5
            k = max(3, int(sigma // 2 * 2 + 1))
            # print(sigma, k)
            for t in np.pi / np.arange(1, angles + 1):
                kernel = cv.getGaborKernel((k, k), sigma, theta=t, lambd=wavelength, gamma=1, psi=np.pi * 0.5,
                                           ktype=cv.CV_64F)
                self.kernels.append(kernel)
                # kernel = cv.getGaborKernel((k, k), sigma, theta=t + np.pi/4, lambd=wavelength, gamma=1, psi=np.pi * 0.5,
                #                            ktype=cv.CV_64F)
                # self.kernels.append(kernel)

            wavelength *= mult

    def encode(self, image, start_angle=0):
        polar, polar_mask = image.to_polar(self.angular_resolution, self.radial_resolution, start_angle, self.n_samples)
        polar = cv.equalizeHist(polar)
        polar = np.float64(polar)
        res = []
        mask = []
        for k in self.kernels:
            f = cv.filter2D(polar, cv.CV_64F, k)
            m = np.zeros(f.shape, np.uint8)
            m[np.abs(f) < self.eps] = 1
            f = np.sign(f)
            m[polar_mask == 0] = 1
            res.extend(f.reshape(-1))
            mask.extend(m.reshape(-1))

        return IrisCode(np.array(res), np.array(mask))


class SKImageIrisCodeEncoder:
    def __init__(self, angles: int = 3,
                 angular_resolution=20,
                 radial_resolution=10,
                 scales=6,
                 eps=0.01,
                 n_samples=100):
        self.angular_resolution = angular_resolution
        self.radial_resolution = radial_resolution
        self.eps = eps
        self.scales = scales
        self.kernels = []
        self.n_samples = n_samples
        for theta in range(0, angles):
            a = theta / angles * np.pi / 2
            kernel = gabor_kernel(1 / 3, a, bandwidth=1)
            self.kernels.append(kernel)

    def encode_raw(self, polar, polar_mask):
        p_next = np.float64(polar) / 255
        # p_next = cv.pyrDown(p_next)
        pyramid = [(p_next, polar_mask)]
        for _ in range(self.scales):
            p_next = cv.pyrDown(p_next)
            m_next = cv.resize(polar_mask, (p_next.shape[1], p_next.shape[0]), cv.INTER_NEAREST)
            pyramid.append((p_next, m_next))

        res = []
        mask = []
        for k in self.kernels:
            for pl, pmask in pyramid:
                # f_real = ndi.convolve(pl, k.real, mode='wrap')
                # f_imag = ndi.convolve(pl, k.imag, mode='wrap')
                f_real = cv.filter2D(pl, cv.CV_64F, k.real)
                f_imag = cv.filter2D(pl, cv.CV_64F, k.imag)
                m_real = np.zeros(f_real.shape, np.uint8)

                f_complex = np.dstack((f_real, f_imag)).view(dtype=np.complex128)[:, :, 0]

                m_real[np.abs(f_complex) < self.eps] = 1
                f_real = np.sign(f_real)
                m_real[pmask == 0] = 1
                res.extend(f_real.reshape(-1))
                mask.extend(m_real.reshape(-1))

                m_imag = np.zeros(f_imag.shape, np.uint8)
                m_imag[np.abs(f_complex) < self.eps] = 1
                f_imag = np.sign(f_imag)
                m_imag[pmask == 0] = 1
                res.extend(f_imag.reshape(-1))
                mask.extend(m_imag.reshape(-1))

        return IrisCode(np.array(res), np.array(mask))

    def encode(self, image, start_angle=0):
        polar, polar_mask = image.to_polar(self.angular_resolution, self.radial_resolution, start_angle, self.n_samples)
        return self.encode_raw(polar, polar_mask)
