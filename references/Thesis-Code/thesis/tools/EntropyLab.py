import streamlit as st

import os
import json
import math
from glob2 import glob
from collections import defaultdict

import numpy as np
import cv2 as cv
import pandas as pd
from skimage.filters import gabor
from medpy.filter import smoothing

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import time

import altair as alt

from thesis.entropy import gradient_histogram, histogram, entropy
from thesis.segmentation import IrisSegmentation, IrisImage, SKImageIrisCodeEncoder
from thesis.entropy import *

"""
# Entropy Test Lab
"""

# base = '/home/anton/data/eyedata/iris'
base = '/Users/Anton/Desktop/data/iris'

files = glob(os.path.join(base, '*.json'))
names = [os.path.basename(p).split('.')[0] for p in files]

dataset = st.sidebar.selectbox('Dataset', names)

with open(os.path.join(base, f'{dataset}.json')) as f:
    data = json.load(f)

id_map = defaultdict(list)
for i, x in enumerate(data['data']):
    id_map[x['info']['user_id']].append((i, x['info']))

num_images = len(data['data'])

user = st.sidebar.selectbox('User ID', list(id_map.keys()))
index, val = st.sidebar.selectbox('Index A', id_map[user])

sample = data['data'][index]

seg = IrisSegmentation.from_dict(sample['points'])
img = cv.imread(sample['image'], cv.IMREAD_GRAYSCALE)
ints = 100
filtered = img.copy()

radius_inner = np.sqrt(seg.inner.axes.x ** 2 + seg.inner.axes.y ** 2)
radius_outer = np.sqrt(seg.outer.axes.x ** 2 + seg.outer.axes.y ** 2)

table = pd.DataFrame([radius_inner, radius_outer], ['Radius inner', 'Radius outer'])
st.write(table)
# filtered = np.uint8(np.random.uniform(0, 255, img.shape))

st.sidebar.markdown('# Filters')
# num = 5000
# coords = np.random.randint(0, filtered.size, num)
# height, width = filtered.shape
# filtered[coords // width, coords % width] = 255
# ints = 100
# s = 30
# filtered[tmp_img.mask == 1] += np.uint8(np.random.uniform(-ints // 2, ints//2, filtered[tmp_img.mask == 1].shape))
# filtered = np.int32(filtered)
# for x in range(0, filtered.shape[1] // s, 2):
#     filtered[:, x * s:(x + 1) * s] += 35
# filtered = np.uint8(np.clip(filtered, 0, 255))
if st.sidebar.checkbox('Uniform noise'):
    intensity = st.sidebar.number_input('Intensity', 0, 1000, 10)
    filtered = np.uint8(np.clip(img + np.random.uniform(-intensity // 2, intensity // 2, img.shape), 0, 255))

rng = np.random.default_rng()
if st.sidebar.checkbox('Cauchy noise'):
    scale = st.sidebar.number_input('Scale', 0, 255, 10)
    filtered = np.uint8(np.clip(img + rng.standard_cauchy(img.shape) * scale, 0, 255))

if st.sidebar.checkbox('Gaussian filter'):
    sigma = st.sidebar.number_input('Sigma', 0, 1000, 3)
    filtered = cv.GaussianBlur(filtered, (0, 0), sigma)

if st.sidebar.checkbox('Bilateral filter'):
    sigma_c = st.sidebar.number_input('Sigma Color', 0, 255, 3)
    sigma_s = st.sidebar.number_input('Sigma Space', 0, 100, 3)
    filtered = cv.bilateralFilter(filtered, 0, sigma_c, sigma_s)

if st.sidebar.checkbox('Non local means'):
    h = st.sidebar.number_input('H (filter strength)', 0., 1000., 10.)
    template = st.sidebar.slider('Template window size', 3, 31, 7, 2)
    search = st.sidebar.slider('Search window size', 3, 31, 21, 2)
    filtered = cv.fastNlMeansDenoising(filtered, h=h, templateWindowSize=template, searchWindowSize=search)

if st.sidebar.checkbox('Anisotropic diffusion'):
    iterations = st.sidebar.number_input('Iterations', 1, 10000, 1)
    kappa = st.sidebar.number_input('Kappa', 1, 100, 50)
    gamma = st.sidebar.number_input('Gamma', 0., 1., 0.1, 0.001)
    fg = filtered / filtered.max()
    filtered = np.uint8(smoothing.anisotropic_diffusion(fg, iterations, kappa, gamma, option=3) * 255)

# filtered = cv.medianBlur(img, 55)
# filtered = np.uint8(np.random.uniform(0, 255, img.shape))

st.image([img, filtered])

img2 = img.copy()
filtered2 = filtered.copy()

iris_img = IrisImage(seg, img)
filter_img = IrisImage(seg, filtered)
encoder = SKImageIrisCodeEncoder(6, 100, 10, 4, 0.001, n_samples=100)
base_code = encoder.encode(iris_img)
filter_code = encoder.encode(filter_img)

st.write(base_code.code.size)

bins = 50
div = 64

st.sidebar.markdown('# Functions')

tmp_img = IrisImage(seg, img)
tmp_fil = IrisImage(seg, filtered)

i_img, _ = tmp_img.to_polar(100, 10, n_samples=100)
i_filtered, _ = tmp_fil.to_polar(100, 10, n_samples=10000)
st.image([i_img, i_filtered], ['base', 'filtered'])

if st.sidebar.checkbox('Entropy'):

    half_img = i_img.copy()
    half_filtered = i_filtered.copy()
    # half_img = cv.pyrDown(img)
    # half_filtered = cv.pyrDown(filtered)
    # half_img = cv.pyrDown(half_img)
    # half_filtered = cv.pyrDown(half_filtered)
    # half_img = cv.pyrDown(half_img)
    # half_filtered = cv.pyrDown(half_filtered)
    # half_img = cv.pyrDown(half_img)
    # half_filtered = cv.pyrDown(half_filtered)

    bcode = (base_code.code + 1) // 2
    fcode = (filter_code.code + 1) // 2

    cm = np.uint8(~((base_code.mask == 1) | (filter_code.mask == 1)).reshape((-1, 1)))

    mask = cv.resize(iris_img.mask, (half_img.shape[1], half_img.shape[0]), interpolation=cv.INTER_NEAREST)
    # ha, hb, hjoint = joint_histogram(bcode.reshape((-1, 1)), fcode.reshape((-1, 1)), mask=cm, divisions=2)
    ha, hb, hjoint = joint_histogram(half_img, half_filtered, mask=mask, divisions=256)
    # st.write(ha)
    # st.write(hb)
    # st.write(str(hjoint))
    f'ha: {entropy(ha)}, hb: {entropy(hb)}, hjoint: {mutual_information(ha, hb, hjoint)}'

    # st.image([img, half_img, mask * 255], ['img', 'half', 'mask'])

    d = st.slider('Number of Divisions', 4, 512, 16)
    t = time.perf_counter()
    # img_grad, fil_grad, joint_grad = joint_gabor_histogram(half_img, half_filtered, mask=mask, theta=0,
    #                                                        divisions=d)
    img_grad, fil_grad, joint_grad = joint_gradient_histogram(i_img, i_filtered, divisions=d)
    # st.write(img_grad)
    # st.write(fil_grad)
    # st.write({str(k): v for k, v in joint_grad.items()})
    # st.write(sum(joint_grad.values()))
    elapsed = time.perf_counter() - t
    f'Elapsed time for gradient histogram: {elapsed:.4f} seconds'

    eps = 10e-5

    log_norm_img = LogNorm(vmin=img_grad.min() + eps, vmax=img_grad.max())
    log_norm_fil = LogNorm(vmin=fil_grad.min() + eps, vmax=fil_grad.max())
    cbar_ticks_img = [math.pow(10, i) for i in
                      range(math.floor(math.log10(img_grad.min() + eps)), 1 + math.ceil(math.log10(img_grad.max())))]
    cbar_ticks_fil = [math.pow(10, i) for i in
                      range(math.floor(math.log10(fil_grad.min() + eps)), 1 + math.ceil(math.log10(fil_grad.max())))]

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    lbs = np.arange(-d//2, d//2, 1)
    sns.heatmap(img_grad,
                ax=ax[0],
                norm=log_norm_img,
                cbar_kws={'ticks': cbar_ticks_img},
                xticklabels=lbs, yticklabels=np.flip(lbs),
                # linewidths=1,
                square=True)
    sns.heatmap(fil_grad,
                ax=ax[1],
                norm=log_norm_fil,
                cbar_kws={'ticks': cbar_ticks_img},
                xticklabels=lbs, yticklabels=np.flip(lbs),
                # linewidths=1,
                square=True)
    st.pyplot(fig)

    e = entropy(img_grad)
    f'Gradient entropy of original: {e}'

    e2 = entropy(fil_grad)
    f'Gradient entropy of filtered: {e2}'

    t = time.perf_counter()
    m2 = mutual_information_grad(img_grad, fil_grad, joint_grad)
    elapsed = time.perf_counter() - t
    f'Elapsed time for entropy calculation: {elapsed:.4f} seconds'

    thetas = np.linspace(0, np.pi, 4)
    base = []
    ft = []
    m = []
    for theta in thetas:
        img_grad, fil_grad, joint_grad = joint_gabor_histogram(half_img, half_filtered, mask=mask, theta=theta,
                                                               divisions=d)

        m.append(mutual_information(img_grad, fil_grad, joint_grad))
        base.append(entropy(img_grad))
        ft.append(entropy(fil_grad))

    f'Multiple angle MI: {np.mean(m)}'
    f'Multiple angle base entropy: {np.mean(base)}'
    f'Multiple angle filtered entropy: {np.mean(ft)}'

    f'Gradient mutual: {m2}'

    f'Ratio: {(e - m2) / e * 100:.2f} %'
    f'Direct fraction: {m2 / e}'

if st.sidebar.checkbox('Gabor responses'):
    '# Gabor responses'
    scale_steps = st.number_input('Scale steps', 0, 8, 0)

    input_img = img.copy()
    for _ in range(scale_steps):
        input_img = cv.pyrDown(input_img)

    input_mask = cv.resize(iris_img.mask, (input_img.shape[1], input_img.shape[0]))

    f'Size: {input_img.shape}'

    n_angles = st.number_input('Angles', 1, 100, 4)
    start_angle = st.slider('Start angle', 0.0, 2 * np.pi, 0., 2 * np.pi / 100)
    frequency = st.slider('Frequency', 0.01, 0.5, 0.3, 0.01)
    thetas = np.linspace(start_angle, start_angle + np.pi - np.pi / n_angles, n_angles)

    amplitude_image = np.zeros((*input_img.shape, 1), dtype=np.float64)
    phase_image = np.zeros((*input_img.shape, 1), dtype=np.float64)

    f'Size: {phase_image.shape}'

    input_img = input_img / 255.0
    for theta in thetas:
        real, imag = gabor(input_img, frequency, theta)
        response = np.dstack((real, imag)).view(dtype=np.complex128)
        response[input_mask == 0] = 0
        amplitude = np.abs(response)
        phase = np.angle(response)

        amplitude_image += amplitude
        phase_image += phase

    amplitude_image -= amplitude_image.min()
    amplitude_image /= amplitude_image.max()
    phase_image -= phase_image.min()
    phase_image /= phase_image.max()

    comb = amplitude_image * phase_image

    f'Size: {phase_image.shape}'
    input_img = cv.resize(input_img, (320, 240))
    amplitude_image = cv.resize(amplitude_image, (320, 240))
    phase_image = cv.resize(phase_image, (320, 240))
    comb = cv.resize(comb, (320, 240))
    st.image([input_img, amplitude_image, phase_image, comb], ['input', 'amplitude', 'phase_image', 'comb'])

if st.sidebar.checkbox('Effectiveness'):
    '# Effectiveness'
    divisions = st.number_input('Histogram divisions', 1, 1048576, 128)

    angular = st.number_input('Angular resolution', 1, 1000, 100)
    radial = st.number_input('Radial resolution', 1, 1000, 50)

    iris_img = IrisImage(iris_img.segmentation, img2)
    pimg, pmask = iris_img.to_polar(angular, radial)
    iris_filtered = IrisImage(iris_img.segmentation, filtered2)
    pfil, _ = iris_filtered.to_polar(angular, radial)

    '## Polar images'
    st.image([pimg, pfil, pmask * 255], ['original', 'filtered', 'mask'])

    hist_pimg, hist_pfil, hist_joint = joint_gradient_histogram(pimg, pfil, pmask, divisions=divisions)

    pimg_ent = entropy(hist_pimg)
    pfil_ent = entropy(hist_pfil)
    joint_ent = mutual_information_grad(hist_pimg, hist_pfil, hist_joint)
    table = pd.DataFrame([pimg_ent, pfil_ent, joint_ent, joint_ent / pimg_ent],
                         ['Original entropy', 'Filtered entropy', 'Mutual information', 'Ratio'])
    st.write(table)
