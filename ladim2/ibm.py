"""Base class and import system for ndividual Based Module"""

import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
import importlib


class BaseIBM(ABC):
    """Base class for Individual Based Model"""

    @abstractmethod
    def update(self) -> None:
        """Updates the IBM to the next time step"""


def init_IBM(module, **args) -> BaseIBM:
    """Initiates an IBM class

    Args:
        module:
            Name of the module containing the IBM
        args:
            Keyword arguments passed on to the IBM

    Returns:
        An IBNM instance

    The working directory takes priority.
    The IBM class in the module should be named "IBM".
    """

    # Temporarily modify sys.path
    syspath = sys.path
    # Add ibms directory
    p = Path(__file__).parent / "ibms"
    sys.path.insert(0, str(p))
    # Put working directory first
    sys.path.insert(0, os.getcwd())

    # Import correct module and return an instance of the IBM class
    ibm_mod = importlib.import_module(module)
    sys.path = syspath
    return ibm_mod.IBM(**args)  # type: ignore
