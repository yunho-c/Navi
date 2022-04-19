import json

import streamlit as st
from glob2 import glob
import os

import numpy as np

from thesis.optim import sampling
from pupilfit import fit_else, fit_excuse
#from thesis.deepeye import deepeye


def fit_else_ref(img, debug=False):
    center, axes, angle = fit_else(img)
    thresh = np.zeros(img.shape)
    if debug:
        return [center[1], center[0], axes[1], axes[0], angle], thresh
    else:
        return [center[1], center[0], axes[1], axes[0], angle]


def fit_excuse_ref(img, debug=False):
    center, axes, angle = fit_excuse(img)
    thresh = np.zeros(img.shape)
    if debug:
        return [center[1], center[0], axes[1], axes[0], angle], thresh
    else:
        return [center[1], center[0], axes[1], axes[0], angle]


# def create_deepeye_func():
#     path = '/Users/Anton/Documents/git/thesis/Thesis-Code/thesis/deepeye/models/default.ckpt'
#     # path = '/home/anton/git/thesis/Thesis-Code/thesis/deepeye/models/default.ckpt'
#     deepeye_model = deepeye.DeepEye(model=path)

#     def deepeye_ref(img, debug=False):
#         coords = deepeye_model.run(img)
#         thresh = np.zeros(img.shape)
#         if debug:
#             return [coords[1], coords[0], 0, 0, 0], thresh
#         else:
#             return [coords[1], coords[0], 0, 0, 0]

#     return deepeye_ref


def file_select(label, pattern):
    file_list = glob(pattern)
    return st.selectbox(label, file_list)


def file_select_sidebar(label, pattern):
    file_list = glob(pattern)
    return st.sidebar.selectbox(label, file_list)


def type_name(x):
    return x.__name__


def obj_type_name(x):
    return type(x).__name__


def json_to_strategy(data):
    params, generators = [], []
    for k, v in data.items():
        params.append(k)
        generators.append(getattr(sampling, v['type'])(**v['params']))
    return params, generators


def progress(iterator, total):
    bar = st.progress(0)
    for i, v in enumerate(iterator):
        bar.progress(i / total)
        yield v
    bar.progress(100)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
