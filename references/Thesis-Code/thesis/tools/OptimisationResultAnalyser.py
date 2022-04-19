import streamlit as st

import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from thesis import colorblind

from tools.st_utils import file_select
from thesis.optim.pareto import pareto_frontier, pareto_set

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
sns.set_context('paper')

st.title("Obfuscation result analysis")
"""
This tool provides simple visualisations for the analysis of the obfuscation experimental results.
"""

file_path = file_select('Result file', os.path.join('results', '*.json'))
with open(file_path) as file:
    results = json.load(file)

frame = pd.DataFrame(results['results'])


# frame['pupil_relative_error'] = frame[]

def aggregate(df, method: str):
    d = {c: method for c in df.columns}
    d['filter'] = 'first'
    return df.groupby(['filter', 'group']).agg(d)

method = st.selectbox('Aggregate method', ('mean', 'min', 'max', 'std', 'median'))

gaze_t = st.number_input('Relative gaze error threshold', 0.0, value=1.0)

agg = aggregate(frame, method)

agg['gaze_relative_error'] = agg['gaze_angle_error_filtered'] / agg['gaze_angle_error_source']
agg['pupil_relative_error_else'] = agg['pupil_distance_else_pixel_error_filtered'] / agg[
    'pupil_distance_else_pixel_error_source']
agg['pupil_relative_error_deep_eye'] = agg['pupil_distance_deep_eye_pixel_error_filtered'] / agg[
    'pupil_distance_deep_eye_pixel_error_source']

agg = agg[agg['gaze_relative_error'] < gaze_t]

frame['gaze_relative_error'] = frame['gaze_angle_error_filtered'] / frame['gaze_angle_error_source']
frame['pupil_relative_error_else'] = frame['pupil_distance_else_pixel_error_filtered'] / frame[
    'pupil_distance_else_pixel_error_source']
frame['pupil_relative_error_deep_eye'] = frame['pupil_distance_deep_eye_pixel_error_filtered'] / frame[
    'pupil_distance_deep_eye_pixel_error_source']

st.sidebar.write("# Display settings")
x = st.sidebar.selectbox('X-axis', agg.columns, index=0)
y = st.sidebar.selectbox('Y-axis', agg.columns, index=1)

do_pareto = st.sidebar.checkbox('Enable pareto')

# st.write(agg)

if results['optimizer']['method'] == 'PopulationMultiObjectiveOptimizer':
    k = st.sidebar.number_input('Choose iteration', 0, max(frame['k']), 0)
else:
    k = 0

types = {
    'blur': 'Destructive',
    'noise': 'Additive',
    'combo': 'Combined'
}

type_map = {k: types[v] for k, v in {
    'bilateral_filter': 'blur',
    'gaussian_filter': 'blur',
    'mean_filter': 'blur',
    'median_filter': 'blur',
    'non_local_means': 'blur',
    'uniform_noise': 'noise',
    'gaussian_noise': 'noise',
    'cauchy_noise': 'noise',
    'laplacian_noise': 'noise',
    'snow': 'noise',
    'salt_and_pepper': 'noise',
    'super_filter': 'combo',
    'super_filter_reverse': 'combo'
}.items()}

agg['Type'] = agg['filter'].apply(type_map.get)

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
    'super_filter': 'Super filter',
    'super_filter_reverse': 'Reverse super filter'
}

pretty_name_map = {
    'gradient_entropy_iris_source': 'Entropy of source  - gradient method',
    'gradient_entropy_iris_filtered': 'Entropy of result - gradient method',
    'gradient_mutual_information_iris': 'Mutual information - gradient method',
    'gabor_entropy_iris_source_1.0x': 'Entropy of source - gabor method (3px)',
    'gabor_entropy_iris_source_0.5x': 'Entropy of source - gabor method (6px)',
    'gabor_entropy_iris_source_0.25x': 'Entropy of source - gabor method (12px)',
    'gabor_entropy_iris_source_0.125x': 'Entropy of source - gabor method (24px)',
    'gabor_entropy_iris_source_0.0625x': 'Entropy of source - gabor method (48px)',
    'gabor_entropy_iris_filtered_1.0x': 'Entropy of result - gabor method (3px)',
    'gabor_entropy_iris_filtered_0.5x': 'Entropy of result - gabor method (6px)',
    'gabor_entropy_iris_filtered_0.25x': 'Entropy of result - gabor method (12px)',
    'gabor_entropy_iris_filtered_0.125x': 'Entropy of result - gabor method (24px)',
    'gabor_entropy_iris_filtered_0.0625x': 'Entropy of result - gabor method (48px)',
    'gabor_mutual_information_iris_1.0x': 'Mutual information - gabor method (3px)',
    'gabor_mutual_information_iris_0.5x': 'Mutual information - gabor method (6px)',
    'gabor_mutual_information_iris_0.25x': 'Mutual information - gabor method (12px)',
    'gabor_mutual_information_iris_0.125x': 'Mutual information - gabor method (24px)',
    'gabor_mutual_information_iris_0.0625x': 'Mutual information - gabor method (48px)',
    'gradient_entropy_image_source': 'Entropy of source  - gradient method',
    'gradient_entropy_image_filtered': 'Entropy of result - gradient method',
    'gradient_mutual_information_image': 'Mutual information - gradient method',
    'gabor_entropy_image_source_1.0x': 'Entropy of source - gabor method (3px)',
    'gabor_entropy_image_source_0.5x': 'Entropy of source - gabor method (6px)',
    'gabor_entropy_image_source_0.25x': 'Entropy of source - gabor method (12px)',
    'gabor_entropy_image_source_0.125x': 'Entropy of source - gabor method (24px)',
    'gabor_entropy_image_source_0.0625x': 'Entropy of source - gabor method (48px)',
    'gabor_entropy_image_filtered_1.0x': 'Entropy of result - gabor method (3px)',
    'gabor_entropy_image_filtered_0.5x': 'Entropy of result - gabor method (6px)',
    'gabor_entropy_image_filtered_0.25x': 'Entropy of result - gabor method (12px)',
    'gabor_entropy_image_filtered_0.125x': 'Entropy of result - gabor method (24px)',
    'gabor_entropy_image_filtered_0.0625x': 'Entropy of result - gabor method (48px)',
    'gabor_mutual_information_image_1.0x': 'Mutual information - gabor method (3px)',
    'gabor_mutual_information_image_0.5x': 'Mutual information - gabor method (6px)',
    'gabor_mutual_information_image_0.25x': 'Mutual information - gabor method (12px)',
    'gabor_mutual_information_image_0.125x': 'Mutual information - gabor method (24px)',
    'gabor_mutual_information_image_0.0625x': 'Mutual information - gabor method (48px)',
    'iris_code_similarity': 'Iris code similarity',
    'image_normalized_similarity': 'Image similarity',
    'gaze_angle_error_source': 'Gaze error source',
    'gaze_angle_error_filtered': 'Gaze error result',
    'gaze_relative_error': 'Gaze error relative',
    'pupil_distance_else_pixel_error_source': 'Pupil pixel distance error of source - ELSE method',
    'pupil_distance_else_pixel_error_filtered': 'Pupil pixel distance error of result - ELSE method',
    'pupil_distance_deep_eye_pixel_error_source': 'Pupil pixel distance error of source - DeepEye method',
    'pupil_distance_deep_eye_pixel_error_filtered': 'Pupil pixel distance error of result - DeepEye method',
    'pupil_relative_error_else': 'Pupil pixel distance relative - ELSE method',
    'pupil_relative_error_deep_eye': 'Pupil pixel distance relative - DeepEye method',
    'filter': 'Filter',
    'k': 'Kernel size',
    'h': '$h$',
    'sigma': 'Variance',
    'scale': 'Variance',
    'intensity': 'Intensity',
    'density': 'Density',
    'sigma_s': 'Spatial variance',
    'sigma_c': 'Colour variance',

}

for k, v in pretty_name_map.items():
    agg[v] = agg[k]

agg['Filter'] = agg['filter'].apply(lambda k: filter_name_map[k])

color = colorblind.qualitative_colors(12)
sns.set_palette(color)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=x, y=y, hue='Filter', data=agg, hue_order=list(filter_name_map.values()), style='Type', s=20, ax=ax,
                x_bins=20)
plt.xlabel(pretty_name_map[x])
plt.ylabel(pretty_name_map[y])
# fig._legend.set_title('Filter')
labels = ['Bilateral filter', 'Cauchy noise', 'Gaussian ']
st.pyplot(fig)

file = st.sidebar.text_input('Plot output')
path = os.path.join('results', 'plots', file)

if st.button('Export'):
    # fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
elif st.button('Export tikz'):
    # fig.tight_layout()
    tikzplotlib.save(path, fig)
    #fig.savefig(path, bbox_inches="tight")

# if st.checkbox('c2'):
#     agg['ELSE'] = agg['pupil_distance_else_pixel_error_filtered']
#     agg['DeepEye'] = agg['pupil_distance_deep_eye_pixel_error_filtered']
#     p = sns.pairplot(data=agg,
#                  x_vars=['ELSE', 'DeepEye'],
#                  y_vars=['gaze_angle_error_filtered', 'gradient_mutual_information'],
#                  hue='Filter')
#     st.pyplot(p)

if st.checkbox('comparisons'):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    # fig.suptitle('blah')

    sns.scatterplot(x='gaze_angle_error_filtered', y='pupil_distance_else_pixel_error_filtered', data=agg,
                    hue='Filter', style='Type', hue_order=list(filter_name_map.values()), s=20, ax=ax[0],
                    legend=False)
    sns.scatterplot(x='gaze_angle_error_filtered', y='pupil_distance_deep_eye_pixel_error_filtered', data=agg,
                    hue='Filter', style='Type', hue_order=list(filter_name_map.values()), s=20, ax=ax[1],
                    legend=False)
    # sns.scatterplot(x='gradient_mutual_information', y='pupil_distance_else_pixel_error_filtered', data=agg,
    #                 hue='Filter', style='Type', hue_order=list(filter_name_map.values()), s=20, ax=ax[0, 1],
    #                 legend=False)
    # sns.scatterplot(x='gradient_mutual_information', y='pupil_distance_deep_eye_pixel_error_filtered', data=agg,
    #                 hue='Filter', style='Type', hue_order=list(filter_name_map.values()), s=20, ax=ax[1, 1],
    #                 legend=False)

    ax[0].set_title('ELSE')
    ax[1].set_title('DeepEye')
    ax[0].set_ylabel('Pixel error')
    ax[1].set_ylabel('Pixel error')
    ax[1].set_xlabel('Gaze error (result)')
    st.pyplot(fig)

    if st.button('Export comp'):
        fig.savefig(path, bbox_inches="tight")

if st.checkbox('Pareto'):
    fig, ax = plt.subplots(2, 1, figsize=(6, 10), sharex=True)

    ax[0].set_title('All results')
    ax[1].set_title('Pareto frontier')
    sns.scatterplot(x=x, y=y, hue='Filter', data=agg,
                    hue_order=list(filter_name_map.values()), style='Type', s=20,
                    ax=ax[0], x_bins=20)
    plt.xlabel(pretty_name_map[x])
    plt.ylabel(pretty_name_map[y])
    # fig._legend.set_title('Filter')
    # labels = ['Bilateral filter', 'Cauchy noise', 'Gaussian ']
    type_markers = {types[k]: v for k, v in {
        'blur': 'o',
        'noise': 'X',
        'combo': '*'
    }.items()}

    rows = []
    for f in list(filter_name_map.values()):
        bfilter = agg[agg['Filter'] == f]
        if len(bfilter.index) == 0:
            continue
        s = pareto_set(np.array(bfilter[[x, y]]))
        row = bfilter.iloc[s]
        rows.append(row)
        sns.lineplot(x=x, y=y, data=row, ax=ax[1],
                     marker=type_markers[row['Type'].iloc[0]])
    # plt.legend(agg['Filter'].unique())
    st.pyplot(fig)

    if st.button('Export performance'):
        fig.savefig(path, bbox_inches="tight")

    # if st.button('Export latex'):
    #     tikzplotlib.save(path, figure=fig, standalone=True)

    pareto_optimal = pd.concat(rows)
# st.write(pareto_optimal)

f_names = ('gaussian_filter', 'non_local_means', 'bilateral_filter', 'bilateral_filter', 'mean_filter',
           'median_filter', 'uniform_noise', 'gaussian_noise', 'cauchy_noise', 'salt_and_pepper',
           'salt_and_pepper')
params = ('sigma', 'h', 'sigma_s', 'sigma_c', 'size', 'size', 'intensity', 'scale', 'scale', 'intensity',
          'density')

titles = (
    'Gaussian filter - Sigma',
    'Non-local means filter - H',
    'Bilateral filter - Spatial sigma',
    'Bilateral filter - Colour sigma',
    'Mean filter - Kernel size',
    'Median filter - Kernel size',
    'Uniform noise - Intensity',
    'Gaussian noise - Sigma',
    'Cauchy noise - Sigma',
    'Salt-and-pepper noise - Intensity',
    'Salt-and-pepper noise - Density',
)

if st.checkbox('Combinations'):
    # agg.reset_index(drop=True, inplace=True)
    # window = agg.groupby('filter').rolling(10).mean()
    # window.reset_index(level=0, inplace=True)
    # st.write(window)
    vars = ('gaze_relative_error', 'gradient_mutual_information', 'gradient_entropy_filtered')
    g = sns.pairplot(
        x_vars=vars,
        y_vars=vars,
        hue='filter',
        data=agg)
    st.pyplot(g)

if st.checkbox('Estimates'):
    # g = sns.PairGrid(frame)
    # g.map_upper(sns.lineplot)
    g = sns.lmplot(x=x, y=y, col='filter', data=agg, col_wrap=4, size=4,
                   fit_reg=False)
    st.pyplot(g)
    # n = len(f_names)
    # ncols = 4
    # nrows = int(np.ceil(n / ncols))
    # st.write(nrows, ncols)
    # fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10))
    # for i, (f, p, t) in enumerate(zip(f_names, params, titles)):
    #     brows = agg[agg['filter'] == f]
    #     points = sns.regplot(x=x, y=y, ax=ax[i // ncols, i % ncols], data=brows, logx=True)
    #     # points = ax[i // ncols, i % ncols].scatter(brows[x], brows[y], c=brows[p],
    #     #                                            cmap='viridis', vmin=0, vmax=brows[p].max(),
    #     #                                            s=5)
    #     ax[i // ncols, i % ncols].set_title(t)
    #     # fig.colorbar(points, ax=ax[i // ncols, i % ncols])
    #
    # for i in range(n, nrows * ncols):
    #     ax[i // ncols, i % ncols].axis('off')
    #
    # fig.tight_layout()
    # st.pyplot(fig)

if st.checkbox('Individual bilaterals'):
    n = len(f_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    st.write(nrows, ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10), sharex=True, sharey=True)
    for i, (f, p, t) in enumerate(zip(f_names, params, titles)):
        brows = agg[agg['filter'] == f]
        points = ax[i // ncols, i % ncols].scatter(brows['gradient_mutual_information'], brows['gaze_relative_error'], c=brows[p],
                                                   cmap='magma', vmin=0, vmax=brows[p].max(),
                                                   s=20)
        ax[i // ncols, i % ncols].set_title(t)
        ax[i // ncols, i % ncols].set_xlabel('Mutual information (gradient method)')
        ax[i // ncols, i % ncols].set_ylabel('Relative gaze error')
        fig.colorbar(points, ax=ax[i // ncols, i % ncols])

    for i in range(n, nrows * ncols):
        ax[i // ncols, i % ncols].axis('off')

    # plt.xlabel('Relative gaze error')
    # plt.ylabel('Mutual information (gradient method)')
    fig.tight_layout()
    st.pyplot(fig)

    if st.button('Export bil'):
        fig.savefig(path, bbox_inches="tight")

if st.checkbox('Metrics'):
    n = len(f_names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    st.write(nrows, ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10), sharey=True)
    for i, (f, p, t) in enumerate(zip(f_names, params, titles)):
        brows = agg[agg['filter'] == f]
        points = ax[i // ncols, i % ncols].scatter(brows[p], brows[y], s=5)
        ax[i // ncols, i % ncols].set_title(t)

    for i in range(n, nrows * ncols):
        ax[i // ncols, i % ncols].axis('off')

    fig.tight_layout()
    st.pyplot(fig)


# '# 3D Stuff'
# filt = agg[agg['filter'] == 'bilateral_filter']
#
# z_axis = st.selectbox('Z axis metric', filt.columns)
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(xs=filt['sigma_s'], ys=filt['sigma_c'], zs=filt[z_axis])
# ax.set_xlabel('Sigma spatial')
# ax.set_ylabel('Sigma color')
# ax.set_zlabel(z_axis)
# st.pyplot(fig)

#
cols = ['gaze_relative_error', 'iris_code_similarity', 'gradient_mutual_information_iris', 'gabor_mutual_information_iris_1.0x',
        'gabor_mutual_information_iris_0.5x',
        'gabor_mutual_information_iris_0.25x', 'gabor_mutual_information_iris_0.125x', 'gabor_mutual_information_iris_0.0625x']
#cols = map(pretty_name_map.get, cols)
vars = frame[cols]

corr = vars.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

fig, ax = plt.subplots(figsize=(5, 5))

lbs = ['Gaze', 'IrisCode', 'Gradient', '3px', '6px', '12px', '24px', '48px']
sns.heatmap(corr, ax=ax, cmap='vlag', vmin=-1, vmax=1, annot=True, square=True, cbar=False,
            xticklabels=lbs, yticklabels=lbs)
st.pyplot(fig)

if st.button('export'):
    # fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")

cols = ['gaze_angle_error_filtered', 'pupil_distance_deep_eye_pixel_error_filtered',
        'pupil_distance_else_pixel_error_filtered']
cols = map(pretty_name_map.get, cols)
vars = agg[cols]

corr = vars.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

fig, ax = plt.subplots(figsize=(4, 4))

lbs = ['Gaze', 'DeepEye', 'ELSE']
sns.heatmap(corr, ax=ax, cmap='viridis', vmin=0, vmax=1, annot=True, square=True, cbar=False,
            xticklabels=lbs, yticklabels=lbs)
st.pyplot(fig)

if st.button('export n2'):
    # fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")

if st.checkbox('Optimal values'):
    "# Optimal values"
    agg.reset_index(drop=True, inplace=True)
    m = agg.loc[agg.groupby('filter')['iris_code_similarity'].idxmin()]

    params = {
        'bilateral_filter': ['sigma_c', 'sigma_s'],
        'non_local_means': ['h'],
        'gaussian_filter': ['sigma'],
        'mean_filter': ['size'],
        'median_filter': ['size'],
        'uniform_noise': ['intensity'],
        'gaussian_noise': ['loc', 'scale'],
        'cauchy_noise': ['scale'],
        'laplacian_noise': ['scale'],
        'snow': ['density'],
        'salt_and_pepper': ['intensity', 'density'],
        'super_filter': ['sigma_c', 'sigma_s', 'scale'],
        'super_filter_reverse': ['sigma_c', 'sigma_s', 'scale']
    }

    out_params = {}
    info = []
    for f, p in params.items():
        info.append(m[m['filter'] == f].iloc[0])
        out_params[f] = {p_name: m[m['filter'] == f].iloc[0].at[p_name] for p_name in p}
    st.write(out_params)
    info = pd.concat(info, axis=1).T
    st.write(info[['filter', 'gaze_relative_error']])
