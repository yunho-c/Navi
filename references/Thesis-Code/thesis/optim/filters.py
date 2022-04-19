import cv2 as cv
import numpy as np
from scipy import stats
from medpy.filter import smoothing

rng = np.random.default_rng()


def super_filter(img, sigma_c: float, sigma_s: float, scale: float, steps: int=1):
    img = bilateral_filter(img, sigma_c, sigma_s, steps)
    img = cauchy_noise(img, scale)
    return img


def super_filter_reverse(img, sigma_c: float, sigma_s: float, scale: float, steps: int=1):
    img = cauchy_noise(img, scale)
    img = bilateral_filter(img, sigma_c, sigma_s, steps)
    return img


def mean_filter(img, size):
    size = int(size)
    return cv.blur(img, (size, size))


def median_filter(img, size):
    size = int(size)
    return cv.medianBlur(img, size)


def anisotropic_diffusion(img, kappa: float, gamma: float, iterations: int = 1):
    img = img / 255
    img = smoothing.anisotropic_diffusion(img, iterations, kappa, gamma)
    return np.uint8(img * 255)


def non_local_means(img, h):
    return cv.fastNlMeansDenoising(img, h=h, templateWindowSize=7, searchWindowSize=21)


def bilateral_filter(img, sigma_c: float, sigma_s: float, steps: int = 1):
    steps = int(steps)
    for _ in range(steps):
        img = cv.bilateralFilter(img, 0, sigma_c, sigma_s)
    return img


def gaussian_filter(img, sigma: float):
    k = (int(sigma * 3) // 2) * 2 + 1
    return cv.GaussianBlur(img, (k, k), sigma)


def uniform_noise(img, intensity: float):
    return np.uint8(np.clip(img + np.random.uniform(-intensity // 2, intensity // 2, img.shape), 0, 255))


def gaussian_noise(img, loc, scale: float):
    return np.uint8(np.clip(img + np.random.normal(loc, scale, img.shape), 0, 255))


def cauchy_noise(img, scale: float):
    return np.uint8(np.clip(img + rng.standard_cauchy(img.shape) * scale, 0, 255))


def laplacian_noise(img, scale: float):
    return np.uint8(np.clip(img + rng.laplace(scale=scale, size=img.shape), 0, 255))


def salt_and_pepper(img, intensity: float, density: float):
    mask = np.random.rand(*img.shape)
    mask[mask > (1 - density)] = 1
    mask[mask <= (1 - density)] = 0
    return np.uint8(np.clip(img + mask * np.random.uniform(intensity), 0, 255))


def snow(img, density: float):
    img = np.copy(img)
    img[np.random.rand(*img.shape) > (1 - density)] = 127
    return img
