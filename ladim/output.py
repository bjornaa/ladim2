"""Abstract base class for LADiM forcing"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# November 2020
# ----------------------------------
from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ladim.state import State


class BaseOutput(ABC):
    """Abstract base class for LADiM forcing"""

    output_period = np.timedelta64(0, "s")

    @abstractmethod
    def __init__(self, modules: dict[str, Any], **kwargs: dict[str, Any]) -> None:
        self.modules = modules

    @abstractmethod
    def update(self) -> None:
        """Write data from instance variables to output file"""

    @abstractmethod
    def write_particle_variables(self, state: State) -> None:
        """Write data from particle variables to output file"""

    @abstractmethod
    def close(self) -> None:
        """Close (last) output file"""


def init_output(module: str, **args: dict[str, Any]) -> BaseOutput:
    """Initiates an Output class

    Args:
        module:
            Name of the module defining the Output class
        args:
            Keyword arguments passed on to the Output instance

    Returns:
        An Output instance

    The module should be in the LADiM source directory or in the working directory.
    The working directory takes priority.
    The Output class in the module should be named "Output".
    """

    # System path for ladim.ladim
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, Path.cwd())  # type: ignore

    # Import correct module
    output_module = importlib.import_module(module)
    return output_module.Output(**args)  # type: ignore
