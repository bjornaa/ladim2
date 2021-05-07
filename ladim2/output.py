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

import numpy as np  # type: ignore

from .state import State


class BaseOutput(ABC):
    """Abstract base class for LADiM forcing"""

    output_period: np.timedelta64(0, 's')

    @abstractmethod
    def write(self, state: State) -> None:
        pass

    @abstractmethod
    def write_particle_variables(self, state: State) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        """Close (last) output file"""




def init_output(**args) -> BaseOutput:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    output_module = importlib.import_module(module)
    return output_module.init_output(**args)  # type: ignore
