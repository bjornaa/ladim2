"""Abstract base class for LADiM grid"""


import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib
from typing import Tuple

import numpy as np  # type: ignore


class BaseGrid(ABC):
    """Abstract Base Class for LADiM grid"""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @abstractmethod
    def metric(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates grid spacing at particle positions"""

    @abstractmethod
    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Tests if particles are inside the grid"""

    @abstractmethod
    def atsea(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Tests if particles are at sea"""


def init_grid(module, **args) -> BaseGrid:
    """Creates an instance of the Grid class

    Args:
        module:
            Name of the module defining the Grid class
        args:
            Keyword arguments passed on to the Grid instance

    Returns:
        A Gridinstance

    The module should be in the LADiM source directory or in the working directory.
    The working directory takes priority.
    The Grid class in the module should be named "Forcing".
    """

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    grid_module = importlib.import_module(module)
    return grid_module.init_grid(**args)  # type: ignore
