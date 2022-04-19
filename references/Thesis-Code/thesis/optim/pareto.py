import numpy as np
from numba import jit
from tqdm import tqdm


def dominates(y, y_mark):
    y = np.array(y)
    y_mark = np.array(y_mark)
    return np.all(y <= y_mark) and np.any(y < y_mark)


def pareto_set(costs: np.ndarray):
    is_efficient = np.ones(len(costs), dtype=np.bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient


def pareto_frontier(df, columns: list):
    base = df[columns + ['filter']]
    comparison_sets = {filt: base[base['filter'] == filt] for filt in base['filter'].unique()}

    tqdm.pandas()

    def each_row(row):
        other_points = comparison_sets[row['filter']]
        return not any([dominates(comp, row) for _, comp in other_points.iterrows()])

    df[f'pareto'] = base.progress_apply(each_row, axis=1)
