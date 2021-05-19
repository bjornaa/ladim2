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

    @abstractmethod
    def __init__(self, modules: dict, **kwargs):
        self.modules = modules
        self.xmin: float
        self.xmax: float
        self.ymin: float
        self.ymax: float

    @abstractmethod
    def depth(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Estimates bottom depth at particle positions"""

    @abstractmethod
    def metric(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates grid spacing at particle positions"""

    @abstractmethod
    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Tests if particles are inside the grid"""

    @abstractmethod
    def atsea(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Tests if particles are at sea"""
