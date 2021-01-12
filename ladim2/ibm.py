import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib


class BaseIBM(ABC):

    @abstractmethod
    def update(self, grid, state, forcing) -> None:
        pass


def init_IBM(**args) -> BaseIBM:

    args = args.copy()
    module = args.pop("module")

    # System path for ladim2.ladim2.ibms
    p = Path(__file__).parent // "ibms"
    sys.path.insert(0, str(p))
    # Working directory
    sys.path.insert(0, os.getcwd())

    # Import correct module and return the IBM class
    ibm_mod = importlib.import_module(module)
    return ibm_mod.init_IBM(**args)    # type: ignore
