"""
Class for the state of the model
"""

# import sys
# import os
# import importlib
# import logging
from typing import List, Dict, Union, Sized, Sequence

import numpy as np  # type: ignore


# ------------------------

# Kan bruke Sequence[Scalar] for 1D arrays o.l.
# kan sette dette som Vector?
Scalar = Union[float, int, bool]
#Arraylike = Union[np.ndarray, List[Scalar], Scalar]
#Variables = Dict[str, Arraylike]
DType = Union[str, type]
Vector = Union[Scalar, Sequence[Scalar]]



class State(Sized):
    """The model variables at a given time"""

    # def __init__(self, **args: Dict[str, DType]) -> None:
    def __init__(self, **args: DType) -> None:
        """Initialize the state with dictionary of extra variables

        These extra variables should be given by their dtype

        Mandatory state variables: pid, alive, X, Y, Z
        with predefined dtypes int, bool and float
        should not be initialized.

        attributes:
          npid: Number of particles released so far
          variables: List of names of state variables
          default_values: Default values for initializing state variables

        """

        # Start with empty state, with correct variables of correct type
        self.npid: int = 0
        self.variables = ["pid", "alive", "X", "Y", "Z"]
        self.pid = np.array([], int)
        self.alive = np.array([], bool)
        self.X = np.array([], float)
        self.Y = np.array([], float)
        self.Z = np.array([], float)

        for var, dtype in args.items():
            self.variables.append(var)
            setattr(self, var, np.array([], dtype=dtype))

        # self.variables: List(str) = list(dtypes)
        self.default_values = {"alive": np.array(True, dtype=bool)}
        # Kan sette alle andre default = 0

    def set_default_values(self, **args: Union[float, int, bool]) -> None:
        """Set default values for state variables"""
        for var, value in args.items():
            if var not in self.variables:
                raise ValueError(f"No such variable: ", var)
            if var == "pid":
                raise ValueError("Can not set default for pid")
            if not np.isscalar(value):
                raise TypeError(f"Default value for {var} should be scalar")
            self.default_values[var] = np.array(value, dtype=getattr(self, var).dtype)

    def append(self, **args: Vector):
        """Append particles to the State object"""

        # Accept only state variables (except pid)
        state_vars = set(self.variables)
        state_vars.remove("pid")
        for name in args:
            if name not in state_vars:
                raise ValueError(f"Invalid argument {name}")

        # ok_vars = arguments and variables with defaults
        # arguments override the defaults
        ok_vars = self.default_values.copy()
        ok_vars.update(args)

        # All state variables (except pid) should be ok
        for name in state_vars:
            if name not in set(ok_vars):
                raise TypeError(f"Variable {name} has no value")

        # All input should be scalars or broadcastable 1D arrays
        values = list(args)
        b = np.broadcast(*ok_vars.values())
        if b.ndim > 1:
            raise ValueError("Arguments must be 1D or scalar")
        if b.ndim == 0:  # All arguments are scalar
            # values[0] = np.array([values[0]])  # Make first argument 1D
            b = np.broadcast([0])  # Make b.size = 1
        nparticles = b.size

        # Make all values 1D of correct shape
        values = [np.broadcast_to(v, shape=(nparticles,)) for v in ok_vars.values()]

        # pid
        self.pid = np.concatenate(
            (self.pid, np.arange(self.npid, self.npid + nparticles, dtype=int))
        )
        self.npid = self.npid + nparticles

        # Set the state variables
        for name, value in zip(list(ok_vars), values):
            setattr(self, name, np.concatenate((getattr(self, name), value)))

    def compactify(self):
        """Remove dead particles"""
        alive = self.alive.copy()
        for var in self.variables:
            A = getattr(self, var)
            setattr(self, var, A[alive])

    def __len__(self) -> int:
        return len(self.X)
