import os
import click
import json
from tqdm import tqdm
from numba import njit, prange

import numpy as np
import pandas as pd


@njit
def find_gt(sorted_array, value, start=0):
    i = start
    while i < len(sorted_array):
        if sorted_array[i] >= value:
            break
        i += 1
    return i


def process_file(data):
    res = []
    for spec in tqdm(data['results'].keys()):
        intra = np.array(data['results'][spec]['intra_distance'])
        inter = np.array(data['results'][spec]['inter_distance'])
        ls = line(intra, inter)
        ls = list(map(dict, ls))
        for l in ls:
            l['name'] = spec
        res.extend(ls)

    return res


@njit
def line(intra, inter):
    res = []

    sorted_intra = np.sort(intra)
    sorted_inter = np.sort(inter)
    n_intra = len(intra)
    n_inter = len(inter)
    intra_idx = 0
    inter_idx = 0
    N = 10000
    for i in range(N):
        x = i / N

        intra_idx = find_gt(sorted_intra, x, intra_idx)
        inter_idx = find_gt(sorted_inter, x, inter_idx)

        tp = intra_idx
        tn = n_inter - inter_idx
        fp = inter_idx
        fn = n_intra - intra_idx

        m = {
            'threshold': x,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        }
        res.append(m)
    return res


filter_name_map = {
    'bilateral_filter': 'Bilateral filter',
    'gaussian_filter': 'Gaussian filter',
    'mean_filter': 'Mean filter',
    'median_filter': 'Median filter',
    'non_local_means': 'Non-local means',
    'uniform_noise': 'Uniform noise',
    'gaussian_noise': 'Gaussian noise',
    'cauchy_noise': 'Cauchy noise',
    'laplacian_noise': 'Laplacian noise',
    'snow': 'Snow noise',
    'salt_and_pepper': 'Salt-and-pepper noise',
    'baseline': 'Baseline'
}


@click.command()
@click.argument('path')
@click.argument('scale')
def convert(path, scale):
    in_path = f'{path}.json'
    out_path = f'{path}.pkl'
    with open(in_path) as f:
        data = json.load(f)
        r = pd.DataFrame(process_file(data))
        r['far'] = r['fp'] / (r['fp'] + r['tn'])
        r['frr'] = r['fn'] / (r['fn'] + r['tp'])
        r['precision'] = r['tp'] / (r['tp'] + r['fp'])
        r['recall'] = r['tp'] / (r['tp'] + r['fn'])
        r['f1'] = (r['precision'] * r['recall']) / (r['precision'] + r['recall'])
        r['accuracy'] = (r['tp'] + r['tn']) / (r['tp'] + r['tn'] + r['fp'] + r['fn'])
        r['filter'] = r['name'].apply(filter_name_map.get)

        r['scale'] = scale

        r.to_pickle(out_path)