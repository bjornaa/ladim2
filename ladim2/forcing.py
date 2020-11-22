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


class Forcing(ABC):
    """Abstract base class for LADiM forcing"""

    @abstractmethod
    def update(self, step: int) -> None:
        pass


def init_forcing(**args) -> Forcing:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    forcing_module = importlib.import_module(module)
    return forcing_module.init_forcing(**args)    # type: ignore
