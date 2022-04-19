from abc import ABC, abstractmethod

from tqdm import tqdm
from streamlit import progress


class ProgressBar(ABC):
    @abstractmethod
    def __init__(self, total):
        ...

    @abstractmethod
    def update(self, i):
        ...

    @abstractmethod
    def close(self):
        ...


class TQDMBar(ProgressBar):
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def update(self, i):
        self.pbar.update(i)

    def close(self):
        self.pbar.close()


class StreamLitBar(ProgressBar):
    def __init__(self, total):
        self.bpar = progress(total)

    def update(self, i):
        self.bpar.progress(i)

    def close(self):
        pass