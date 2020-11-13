import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib

import numpy as np  # type: ignore


class Grid(ABC):

    @abstractmethod
    def metric(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
    gmod = importlib.import_module(module)
    return gmod.makegrid(**args)    # type: ignore
