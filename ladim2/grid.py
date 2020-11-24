import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib

import numpy as np  # type: ignore


class Grid(ABC):

    xmin: float = 4
    xmax: float
    ymin: float
    ymax: float

    @abstractmethod
    def metric(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def atsea(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass


def makegrid(**args) -> Grid:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    grid_module = importlib.import_module(module)
    return grid_module.makegrid(**args)    # type: ignore
