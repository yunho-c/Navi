import math

import eyeinfo

import numpy as np
import cv2 as cv
from skimage.filters import gabor, gabor_kernel
from sklearn.neighbors import KernelDensity

from collections import defaultdict
from itertools import product

GRAD_RESOLUTION = 255 * 2 + 1


def dx(img):
    return cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    # return cv.Scharr(img, cv.CV_64F, 1, 0)


def dy(img):
    return cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    # return cv.Scharr(img, cv.CV_64F, 0, 1)


def gradient_histogram(img, mask=None):
    img = cv.equalizeHist(img)
    xm = np.int16(dx(img) / 4)  # important for histogram calculation!
    ym = np.int16(dy(img) / 4)

    if mask is None:
        mask = np.ones(img.shape, np.uint8)

    hist = eyeinfo.gradient_histogram(xm, ym, mask)
    return hist


def joint_gabor_histogram(img_a, img_b, mask=None, theta=0, divisions=4):
    img_a = img_a.copy()
    img_b = img_b.copy()

    img_a = img_a / 255.0
    img_b = img_b / 255.5

    kernel = gabor_kernel(frequency=0.3, theta=theta, bandwidth=1)

    real_a = cv.filter2D(img_a, cv.CV_64F, kernel.real)
    imag_a = cv.filter2D(img_a, cv.CV_64F, kernel.imag)
    real_b = cv.filter2D(img_b, cv.CV_64F, kernel.real)
    imag_b = cv.filter2D(img_b, cv.CV_64F, kernel.imag)

    # real_a, imag_a = gabor(img_a, frequency=0.3, theta=theta, bandwidth=1)
    # real_b, imag_b = gabor(img_b, frequency=0.3, theta=theta, bandwidth=1)

    all_max = max(np.abs(real_a).max(), np.abs(imag_a).max(), np.abs(real_b).max(), np.abs(imag_b).max())
    eps = 10e-6

    real_a = np.int16((real_a / all_max) * (divisions // 2 - eps))
    imag_a = np.int16((imag_a / all_max) * (divisions // 2 - eps))
    real_b = np.int16((real_b / all_max) * (divisions // 2 - eps))
    imag_b = np.int16((imag_b / all_max) * (divisions // 2 - eps))

    if mask is None:
        mask = np.ones(real_a.shape, dtype=np.uint8)
    # else:
    #     mask = cv.resize(mask, (0, 0), fx=scale, fy=scale)

    hist_a, hist_b, hist_joint = eyeinfo.joint_gradient_histogram(real_a, imag_a, real_b, imag_b, mask,
                                                                  divisions)

    return hist_a, hist_b, hist_joint


def kde_histogram(img, mask=None):
    img = img.copy()
    img = cv.equalizeHist(img)

    xm = dx(img).reshape(-1)
    ym = dy(img).reshape(-1)

    print(xm[110])

    X = np.vstack((xm, ym)).T

    kde = KernelDensity(bandwidth=5)

    kde.fit(X)

    size = 32
    hist = np.zeros((size, size))

    coords_x = np.arange(0, size, 1)
    coords_y = np.arange(0, size, 1)
    coords = np.array(list(product(coords_x, coords_y)))

    # print(coords.shape)

    hist[coords] = kde.score(coords - size//2)
    # hist = np.exp(hist)

    # print('min', hist.min(), hist.max())
    #
    hist -= hist.min()
    hist /= hist.max()

    return hist


def joint_gradient_histogram(img_a, img_b, mask=None, divisions=4):
    img_a = img_a.copy()
    img_b = img_b.copy()
    img_a[mask == 0] = img_a.min()
    img_b[mask == 0] = img_b.min()
    img_a = cv.equalizeHist(img_a)
    img_b = cv.equalizeHist(img_b)

    xm_a = dx(img_a)
    ym_a = dy(img_a)
    xm_b = dx(img_b)
    ym_b = dy(img_b)

    # all_max = max(np.abs(xm_a).max(), np.abs(ym_a).max(), np.abs(xm_b).max(), np.abs(ym_b).max())
    all_max = 1024

    eps = 10e-6
    xm_a = np.int16(xm_a / all_max * (divisions // 2 - eps))
    ym_a = np.int16(ym_a / all_max * (divisions // 2 - eps))
    xm_b = np.int16(xm_b / all_max * (divisions // 2 - eps))
    ym_b = np.int16(ym_b / all_max * (divisions // 2 - eps))

    if mask is None:
        mask = np.ones(xm_a.shape, dtype=np.uint8)

    hist_a, hist_b, hist_joint = eyeinfo.joint_gradient_histogram(xm_a, ym_a, xm_b, ym_b, mask, divisions)

    return hist_a, hist_b, hist_joint


def mutual_information_grad(hist_a, hist_b, hist_joint):
    return eyeinfo.mutual_information_grad(hist_a, hist_b, hist_joint)


def mutual_information(hist_a, hist_b, hist_joint):
    e = 0
    n = len(next(iter(hist_joint))) // 2
    for pos, v in hist_joint.items():
        pos_left = pos[:n]
        pos_right = pos[n:]
        base_v = hist_a[pos_left]
        filt_v = hist_b[pos_right]
        if base_v > 0 and filt_v > 0:
            d = base_v * filt_v
            t = np.log2(v / d)
            r = v * t
            e += r
    return e


def joint__gabor_1d_histogram(img_a, img_b, mask=None, divisions=32):
    img_a = img_a.copy()
    img_b = img_b.copy()

    scale = 0.08
    img_a = cv.resize(img_a, (0, 0), fx=scale, fy=scale)
    img_b = cv.resize(img_b, (0, 0), fx=scale, fy=scale)

    # img_a = cv.equalizeHist(img_a)
    # img_b = cv.equalizeHist(img_b)

    # k = cv.getGaborKernel((0, 0), 2, 0, 2, 1)
    # real_r = cv.filter2D(img_a, cv.CV_64F, k)

    img_a = img_a / 255.0
    img_b = img_b / 255.5

    real_a, imag_a = gabor(img_a, frequency=0.2, theta=np.pi / 5, bandwidth=1)
    real_b, imag_b = gabor(img_b, frequency=0.2, theta=np.pi / 5, bandwidth=1)

    all_max = max(np.abs(real_a).max(), np.abs(imag_a).max(), np.abs(real_b).max(), np.abs(imag_b).max())
    eps = 10e-6

    real_a = np.int16((real_a / all_max) * (divisions // 2 - eps))
    real_b = np.int16((real_b / all_max) * (divisions // 2 - eps))

    if mask is None:
        mask = np.ones(real_a.shape, dtype=np.uint8)
    else:
        mask = cv.resize(mask, (0, 0), fx=scale, fy=scale)

    hist_a = np.zeros(divisions)
    hist_b = np.zeros(divisions)

    hist_joint = defaultdict(float)

    offset = (divisions // 2) - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                hist_joint[(real_a[y, x] + offset,
                            real_b[y, x] + offset)] += 1
                hist_a[real_a[y, x] + offset] += 1
                hist_b[real_b[y, x] + offset] += 1

    joint_sum = hist_a.sum()
    assert hist_a.sum() == hist_b.sum()
    hist_joint = defaultdict(float, {k: v / joint_sum for k, v in hist_joint.items()})
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def joint_histogram(img_a, img_b, mask=None, divisions=32):
    # img_a = cv.equalizeHist(img_a)
    # img_b = cv.equalizeHist(img_b)
    img_a = np.int32(img_a / img_a.max() * (divisions // 2))
    img_b = np.int32(img_b / img_b.max() * (divisions // 2))

    hist_a = np.zeros(divisions)
    hist_b = np.zeros(divisions)

    if mask is None:
        mask = np.zeros(img_a.shape)

    hist_joint = defaultdict(float)

    offset = (divisions // 2) - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                hist_joint[(img_a[y, x] + offset,
                            img_b[y, x] + offset)] += 1
                hist_a[img_a[y, x] + offset] += 1
                hist_b[img_b[y, x] + offset] += 1

    joint_sum = hist_a.sum()
    hist_joint = defaultdict(float, {k: v / joint_sum for k, v in hist_joint.items()})
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def joint_img_code_histogram(img, code, img_mask=None, code_mask=None, img_divisions=32):
    img_a = cv.equalizeHist(img)
    img = np.int32(img / img.max() * (img_divisions // 2))

    xm = dx(img)
    ym = dy(img)

    # all_max = max(np.abs(xm_a).max(), np.abs(ym_a).max(), np.abs(xm_b).max(), np.abs(ym_b).max())
    all_max = 1024

    eps = 10e-6
    xm = np.int16(xm / all_max * (img_divisions // 2 - eps))
    ym = np.int16(ym / all_max * (img_divisions // 2 - eps))

    hist_a = np.zeros((img_divisions, img_divisions))
    hist_b = np.zeros(2)

    if mask is None:
        mask = np.zeros(img_a.shape)

    hist_joint = defaultdict(float)

    offset = (divisions // 2) - 1
    height, width = img_a.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:
                hist_joint[(img_a[y, x] + offset,
                            img_b[y, x] + offset)] += 1
                hist_a[img_a[y, x] + offset] += 1
                hist_b[img_b[y, x] + offset] += 1

    joint_sum = hist_a.sum()
    hist_joint = defaultdict(float, {k: v / joint_sum for k, v in hist_joint.items()})
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return hist_a, hist_b, hist_joint


def entropy(hist):
    r = -eyeinfo.entropy(hist)
    return r


def histogram(img, mask=None):
    return eyeinfo.histogram(img, mask)


def kl_divergence(h1, h2):
    e = 0
    for j in range(-255, 256):
        for i in range(-255, 256):
            v1 = h1[i, j]
            v2 = h2[i, j]

            if v2 != 0:
                vt = v1 / v2
                if vt != 0:
                    e += v1 * np.log2(vt)

    return e
