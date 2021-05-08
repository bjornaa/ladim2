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

import numpy as np  # type: ignore


class BaseForce(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def update(self, step: int, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        """Update the forcing to the next time step"""

    @abstractmethod
    def velocity(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        fractional_step: float = 0,
        method: str = "bilinear",
    ) -> Tuple[np.ndarray, np.ndarray]:
        "Estimate velocity at particle positions"

    @abstractmethod
    def close(self) -> None:
        """Close the (last) forcing file"""


def init_force(module, **args) -> BaseForce:
    """Creates a Forcing class with an instance

    Args:
        module:
            Name of the module defining the Forcing class
        args:
            Keyword arguments passed on to the Forcing instance

    Returns:
        A Forcing instance

    The module should be in the LADiM source directory or in the working directory.
    The working directory takes priority.
    The Forcing class in the module should be named "Forcing".
    """
    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    forcing_module = importlib.import_module(module)
    return forcing_module.init_force(**args)  # type: ignore
