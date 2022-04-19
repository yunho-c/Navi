import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SelectionMethod(ABC):
    k: int

    @abstractmethod
    def select(self, y):
        ...


class TruncationSelection(SelectionMethod):
    def select(self, y):
        p = np.argsort(y)
        a = np.random.randint(0, self.k, (len(y), 2))
        return p[a]


class TournamentSelection(SelectionMethod):
    def select(self, y):
        pass


class CrossoverMethod(ABC):
    @abstractmethod
    def crossover(self, a, b):
        ...


class UniformCrossover(CrossoverMethod):
    def crossover(self, a, b):
        out = np.copy(a)
        for i in range(len(b)):
            if random.random() > 0.5:
                out[i] = b[i]
        return out


class MutationMethod(ABC):
    @abstractmethod
    def mutate(self, child):
        ...


@dataclass
class GaussianMutation(MutationMethod):
    sigma: np.ndarray
    mean: np.ndarray

    def mutate(self, child):
        return child + self.sigma * np.random.randn(*child.shape) + self.mean
