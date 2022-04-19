from typing import Dict

import click
import os
import yaml

from tqdm import tqdm

import pandas as pd
import json

from thesis.optim.sampling import GridSearch, Sampler
from thesis.optim.multi_objective import MultiObjectiveOptimizer, NaiveMultiObjectiveOptimizer, ObfuscationObjective
from thesis.optim import multi_objective
from thesis.optim import filters
from thesis.optim import metrics
from thesis.tools.cli.utilities import load_gaze_data, load_iris_data, load_pupil_data
from thesis.tools.st_utils import json_to_strategy, obj_type_name, NpEncoder


def progress_tqdm(iterator, total):
    for v in tqdm(iterator, total=total):
        yield v


@click.command()
@click.argument('config')
@click.argument('name')
def main(config, name):
    with open(os.path.join('configs/filter_experiment/', config)) as file:
        data = yaml.safe_load(file)

        with open(data['data_config']) as data_config_file, \
                open(data['strategy_config']) as strategy_config_file, \
                open(data['metrics_config']) as metrics_config_file:
            data_config = yaml.safe_load(data_config_file)
            strategy_config = yaml.safe_load(strategy_config_file)
            metrics_config = yaml.safe_load(metrics_config_file)

        def load_data():
            return load_gaze_data(data_config['gaze_data']), load_iris_data(data_config['iris_data']), \
                   load_pupil_data(data_config['pupil_data'])

        loaded = load_data()
        gaze_data, iris_data, pupil_data = loaded

        dmetrics = data['metrics']

        ag = metrics_config['constructor_args']

        def do(x):
            return getattr(metrics, x)(**ag[x])

        iris_terms = list(map(do, dmetrics['iris_metrics']))
        image_terms = list(map(do, dmetrics['image_metrics']))
        gaze_terms = list(map(do, dmetrics['gaze_metrics']))
        pupil_terms = list(map(do, dmetrics['pupil_metrics']))

        optimizers: Dict[str, MultiObjectiveOptimizer] = {}
        projected_iterations = 0

        def make_strategy(data, num):
            parameters, generators = [], []
            for k, v in data.items():
                parameters.append(k)
                generators.append(getattr(sampling, v['type'])(**v['params'], num=num))
            return parameters, generators

        samples = data['samples']
        iris_samples = samples['iris']
        gaze_samples = samples['gaze']
        pupil_samples = samples['pupil']

        method = getattr(multi_objective, data['method'])

        params = {}
        if method == NaiveMultiObjectiveOptimizer:
            params['configuration'] = strategy_config
            sampling = GridSearch

            for f in map(lambda f: getattr(filters, f), data['filters']):
                objective = ObfuscationObjective(f, iris_data, gaze_data, pupil_data, iris_terms, image_terms,
                                                 gaze_terms, pupil_terms, iris_samples, gaze_samples, pupil_samples,
                                                 data['polar_resolution'])
                params, generators = json_to_strategy(strategy_config[f.__name__])
                sampler: Sampler = sampling(params, generators)
                projected_iterations += len(sampler)
                optimizers[f.__name__] = method([], objective, sampler)
        else:
            raise NotImplementedError("Only NaiveMultiObjectiveOptimizer currently supported.")

        results = []

        for filter_name, o in optimizers.items():
            f'Running optimizer for {filter_name}'
            o.run(wrapper=tqdm, parallel=False)

        'Results computed!'

        for filter_name, o in optimizers.items():
            dmetrics = o.metrics()
            # pareto = [o.pareto_frontier(k) for k in range(max([m[2] for m in dmetrics]) + 1)]

            dmetrics_df = []
            for i, (a, b, generations) in enumerate(dmetrics):
                for line in b.list_form():
                    dmetrics_df.append({**a, **line, 'filter': filter_name, 'k': generations, 'group': i})
            results.extend(dmetrics_df)
            # results[filter_name] = metrics_df
            dmetrics = pd.DataFrame(dmetrics_df)

        with open(os.path.join('results', f'{name}.json'), 'w') as f2:
            json.dump({
                'name': name,
                'optimizer': {
                    'method': method.__name__,
                    'params': params
                },
                # 'metrics': metrics,
                'results': results
            }, f2, cls=NpEncoder)


if __name__ == '__main__':
    main()
