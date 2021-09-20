"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------

# import sys
# import os
# import importlib
# from pathlib import Path
from abc import ABC, abstractmethod
# from typing import Tuple, Dict, NewType

import numpy as np  # type: ignore

# Type aliases
Field = np.ndarray  # 3D or 2D gridded field
ParticleArray = np.ndarray  # 1D array, one element per particle
Velocity = tuple[ParticleArray, ParticleArray]


class BaseForce(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def __init__(self, modules: dict[str, str], **kwargs):
        self.modules = modules
        self.variables: dict[str, np.ndarray]

    @abstractmethod
    def update(self) -> None:
        """Update the forcing to the next time step"""

    @abstractmethod
    def velocity(
        self,
        X: ParticleArray,
        Y: ParticleArray,
        Z: ParticleArray,
        fractional_step: float = 0,
        method: str = "bilinear",
    ) -> Velocity:
        """Estimate velocity at particle positions"""

    @abstractmethod
    def close(self) -> None:
        """Close the (last) forcing file"""
