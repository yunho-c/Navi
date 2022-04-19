import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy import stats

from thesis.tools.st_utils import file_select

"""
# Iris Recognition analysis
"""

file_path = file_select('Result file', os.path.join('results', 'recognition', '*.json'))
with open(file_path) as file:
    data = json.load(file)

if 'parameters' in data:
    st.write(data['parameters'])

if not 'intra_distances' in data['results']:
    selected = st.selectbox('Results to show', list(data['results'].keys()))
    intra_distances = np.array(data['results'][selected]['intra_distance'])
    inter_distances = np.array(data['results'][selected]['inter_distance'])
else:
    intra_distances = np.array(data['results']['intra_distances'])
    inter_distances = np.array(data['results']['inter_distances'])

clip = st.slider('Clip margin', 0.0, 0.5, 0.01)

intra_distances = intra_distances[intra_distances < 1-clip]
intra_distances = intra_distances[intra_distances > clip]
inter_distances = inter_distances[inter_distances < 1-clip]
inter_distances = inter_distances[inter_distances > clip]

intra_distances.sort()
inter_distances.sort()
# intra_distances = inter_distances

threshold = st.slider('Threshold', 0.3, 0.5, 0.35)

false_accepts = (inter_distances <= threshold).sum() / len(inter_distances)
false_rejects = (intra_distances > threshold).sum() / len(intra_distances)

start = st.sidebar.slider('Minimum', 0.01, 0.5, 0.1)
stop = st.sidebar.slider('Maximum', 0.2, 0.9, 0.5)

"""
# Threshold test
"""
t = st.slider('Threshold', 0.1, 0.9, 0.3, 0.001)
tp = (intra_distances <= t).sum()
tn = (inter_distances > t).sum()
fp = (inter_distances <= t).sum()
fn = (intra_distances > t).sum()

far = (inter_distances <= t).sum() / len(inter_distances)
frr = (intra_distances > t).sum() / len(intra_distances)
st.write(f'FAR: {far}, FRR: {frr}')
st.write(f'Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}')


"""## Distribution and properties."""

inter_mean = np.nanmean(inter_distances)
inter_std = np.nanstd(inter_distances)
intra_mean = np.nanmean(intra_distances)
intra_std = np.nanstd(intra_distances)

daugman_measure = np.abs(inter_mean-intra_mean)/np.sqrt(0.5 * (inter_std**2 + intra_std**2))
st.write(f'Daugman d\': {daugman_measure}')

inter_dist = stats.norm(loc=inter_mean, scale=inter_std)
intra_dist = stats.norm(loc=intra_mean, scale=intra_std)

fig, ax = plt.subplots()

sns.distplot(inter_distances, kde=False, norm_hist=True, ax=ax)
sns.distplot(intra_distances, kde=False, norm_hist=True, ax=ax)

xs = np.linspace(0.0, 1.0, 200)
inter_y = []
intra_y = []
for x in xs:
    inter_y.append(inter_dist.pdf(x))
    intra_y.append(intra_dist.pdf(x))

ax.plot(xs, inter_y)
ax.plot(xs, intra_y)
ax.vlines([t], 0, 20)
st.pyplot(fig)

thresholds = np.linspace(start, stop, 100)
fp = []
p = []
far = []
frr = []
far_est = []
frr_est = []

for x in thresholds:
    fp.append(int((inter_distances <= x).sum()))
    p.append(int((inter_distances <= x).sum() + (intra_distances < x).sum()))
    far.append((inter_distances <= x).sum() / len(inter_distances))
    frr.append((intra_distances > x).sum() / len(intra_distances))
    far_est.append(inter_dist.cdf(x))
    frr_est.append(1 - intra_dist.cdf(x))

# st.write(np.array(fp)/np.array(p))
# st.write(p)
"""## Acceptable FAR consequences
The following shows the resulting FRR given a specified acceptable FAR."""
max_far = st.number_input('Acceptable FAR', 0.0, 1.0, 0.1, format='%f', step=10e-8)
count = len(intra_distances)
n = 0

while far[n] < max_far:
    n += 1



f'FRR: {frr[n]}'

"""## ROC curve"""
log_scale = st.sidebar.checkbox('Log scale')
display_estimate = st.sidebar.checkbox('Show estimate')

fig, ax = plt.subplots()
ax.plot(far, frr, label='Data')
if display_estimate:
    plt.plot(far_est, frr_est, label='Estimated distribution')
# plt.vlines([max_far], 0, 1)
ax.set_xlabel('FAR')
ax.set_ylabel('FRR')
ax.grid()

if log_scale:
    ax.set_xscale('log')
    ax.set_yscale('log')
    # plt.axis(xmin=10 ** -4, xmax=1, ymin=10 ** -4, ymax=1)
# st.write(far)
# st.write(frr)
ax.legend()
st.pyplot(fig)