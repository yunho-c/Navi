import streamlit as st

import os
import json
import math
import inspect
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
from thesis.segmentation import IrisSegmentation, IrisImage, IrisCodeEncoder
from thesis.entropy import *
from thesis.optim.filters import *

"""
# Filter configuration
"""
filter = st.selectbox('Filter', (
mean_filter, anisotropic_diffusion, non_local_means, bilateral_filter, gaussian_filter, uniform_noise, gaussian_noise,
cauchy_noise, salt_and_pepper), format_func=lambda x: x.__name__)

f'### Arguments for {filter.__name__}'
args = {v: st.number_input(v) for v in inspect.getfullargspec(filter).args[1:]}

"""
# Sampling
"""
n_samples = st.number_input('Number of samples', 1, value=10)
height = st.number_input('Sample height', 1, value=10)
width = st.number_input('Sample width', 1, value=100)


if st.checkbox('Run test'):
    results = []
    bar = st.progress(0)
    for i in range(n_samples):
        img = np.uint8(np.random.uniform(0, 255, (height, width)))
        filtered = filter(img, **args)
        ha, hb, h_joint = joint_gradient_histogram(img, filtered, divisions=16)

        mi = mutual_information(ha, hb, h_joint)
        results.append(mi)
        bar.progress(i/n_samples)
    bar.progress(100)

    capacity = np.max(results)
    st.write(capacity)