"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------

import sys
import os
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseForce(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def update(self, step: int, X: float, Y: float, Z: float) -> None:
        pass

    @abstractmethod
    def velocity(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        fractional_step: float = 0,
        method: str = "bilinear",
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


def init_force(**args) -> BaseForce:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    forcing_module = importlib.import_module(module)
    return forcing_module.init_force(**args)  # type: ignore
