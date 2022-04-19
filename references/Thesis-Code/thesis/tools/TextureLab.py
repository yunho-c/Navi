    import streamlit as st

from collections import defaultdict
import os
from glob2 import glob
import json

import cv2 as cv
import numpy as np

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from scipy.signal import butter, filtfilt
import PIL

import matplotlib.pyplot as plt

from thesis.segmentation import IrisImage, IrisSegmentation

st.set_option('deprecation.showfileUploaderEncoding', False)

shrink = (slice(0, None, 3), slice(0, None, 3))
brick = img_as_float(data.brick())[shrink]
grass = img_as_float(data.grass())[shrink]
gravel = img_as_float(data.gravel())[shrink]

"""
# Texture Lab
"""

st.sidebar.markdown('# Data setup')
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
iris_img = IrisImage(seg, img)


# angles = np.linspace(0, np.pi, 4)
# wavelengths = np.linspace(1/0.05, 1/0.25, 2)
# sigma = (1, 3, 5)
#
# kernels = []
# for a in angles:
#     for w in wavelengths:
#         for s in sigma:
#             k = cv.getGaborKernel((30, 30), sigma=s, theta=a, lambd=w, gamma=1, psi=0, ktype=cv.CV_64F)
#             kernels.append(k)


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    # return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap') ** 2 +
    #                ndi.convolve(image, np.imag(kernel), mode='wrap') ** 2)
    return ndi.convolve(image, np.imag(kernel), mode='constant')


angle_steps = st.number_input('Angular divisions', 1, 1000, 3)
frequency_steps = st.number_input('Frequency steps', 1, 30, 8)

freqs = np.logspace(0.05, 1.0, frequency_steps) / 10


# freqs = np.linspace(0.1, 1.0, frequency_steps)


@st.cache
def create_filters():
    kernels = []
    for theta in range(0, angle_steps):
        theta = theta / angle_steps * np.pi / 2
        for frequency in freqs:
            kernel = gabor_kernel(frequency, theta=theta,
                                  bandwidth=1)
            kernels.append(kernel)
    return kernels


kernels = create_filters()

fig, ax = plt.subplots(1, len(kernels))
for i, k in enumerate(kernels):
    ax[i].axis('off')
    ax[i].imshow(np.real(k))

st.pyplot(fig)


@st.cache(suppress_st_warning=True)
def apply_filters(image, kernels):
    bar = st.progress(0)
    res = []
    for i, k in enumerate(kernels):
        res.append(power(image, k))
        bar.progress(i / len(kernels))
    bar.progress(100)
    return res


image, mask = iris_img.to_polar(40, 20)
# image = np.uint8(np.random.uniform(0, 255, image.shape))


res = apply_filters(image, kernels)
cols = frequency_steps
rows = max(int(np.ceil(len(res) / cols)), 2)
# res = cv.filter2D(image, cv.CV_64F, kernels[filter_num])
fig, ax = plt.subplots(rows, cols)
for i, r in enumerate(res):
    ax[i // cols, i % cols].axis('off')
    ax[i // cols, i % cols].imshow(r)

st.pyplot(fig)
