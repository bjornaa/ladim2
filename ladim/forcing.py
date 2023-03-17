"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# Type aliases
Field = np.ndarray  # 3D or 2D gridded field
ParticleArray = np.ndarray  # 1D array, one element per particle
Velocity = tuple[ParticleArray, ParticleArray]


class BaseForce(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def __init__(self, modules: dict[str, Any], **kwargs: dict[str, Any]):
        self.modules = modules
        self.variables: dict[str, ParticleArray]

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
