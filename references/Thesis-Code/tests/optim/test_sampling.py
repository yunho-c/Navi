import pytest
import numpy as np

from eyelab.optim.sampling import uniform_sampler, samples_step


class TestSampling:
    def test_stratification(self):
        step = 2
        seq = np.arange(0, 10, step)
        stratified_seq = samples_step(0, 10, step)
        for regular, stratified in zip(seq, stratified_seq):
            assert stratified > regular-step/2
            assert stratified < regular+step/2
            assert stratified != regular


class TestUniformSampler:
    def test_different_lengths(self):
        generators = [
            samples_step(0, 10, 1),
            samples_step(20, 25, 1)
        ]
        params = ['first', 'second']
        res = list(uniform_sampler(params, generators))
        assert len(res) == 10

    def test_all_values_used(self):
        generators = [
            samples_step(0, 10, 1, stratified=False),
            samples_step(20, 30, 1, stratified=False)
        ]
        params = ['first', 'second']
        res = list(uniform_sampler(params, generators))
        first = [e['first'] for e in res]
        second = [e['second'] for e in res]
        for e in res:
            for i, v in enumerate(e.values()):
                assert v in generators[i]
