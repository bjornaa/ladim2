import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib

# import numpy as np  # type: ignore


class IBM(ABC):

    @abstractmethod
    def update(self, grid, state, forcing) -> None:
        pass


def initIBM(**args) -> IBM:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2
    p = Path(__file__).parent
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module
    ibm_mod = importlib.import_module(module)
    return ibm_mod.initIBM(**args)    # type: ignore
