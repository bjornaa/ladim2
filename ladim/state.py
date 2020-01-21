"""
Class for the state of the model
"""

# import sys
# import os
# import importlib
# import logging
#from typing import Any, Dict, Sized, Union, Optional, List, Mapping  # mypy
from typing import List, Dict, Union, Sized

import numpy as np  # type: ignore

# from .tracker import Tracker
# from .gridforce import Grid, Forcing

# ------------------------

Arraylike = Union[np.ndarray, List[float], float]
Variables = Dict[str, Arraylike]
DType = Union[str, type]


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
        self.variables = ['pid', 'alive', 'X', 'Y', 'Z']
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

    def append(self, **args: Arraylike):
        """Append particles to the State object"""

        # Accept only state variables (except pid)
        names = set(args.keys())
        state_vars = set(self.variables)
        state_vars.remove("pid")
        assert names.issubset(state_vars)

        # All state variables (except pid) should be included
        # or have a default value
        # (evt. få default = null)
        vars_with_value = names.union(self.default_values)
        assert state_vars.issubset(vars_with_value)

        # All input should be scalars or broadcastable 1D arrays
        values = list(args.values())
        b = np.broadcast(*values)
        if b.ndim > 1:
            raise ValueError("Arguments must be 1D or scalar")
        if b.ndim == 0:  # All arguments are scalar
            values[0] = np.array([values[0]])  # Make first argument 1D
            b = np.broadcast([0])  # Make b.size = 1
        nparticles = b.size
        values = np.broadcast_arrays(*values)

        # pid
        self.pid = np.concatenate(
            (self.pid, np.arange(self.npid, self.npid + nparticles, dtype=int))
        )
        self.npid = self.npid + nparticles

        # alive
        self.alive = np.concatenate((self.alive, nparticles * [True]))

        # Rest of the variables
        for name, value in zip(names, values):
            setattr(self, name, np.concatenate((getattr(self, name), value)))

    def compactify(self):
        """Remove dead particles"""
        alive = self.alive.copy()
        for var in self.variables:
            A = getattr(self, var)
            setattr(self, var, A[alive])

    def __len__(self) -> int:
        return len(self.X)


if __name__ == "__main__":

    X = np.array([10.0, 10.1])
    Y = np.array([0.0, 20.0])
    Z = np.array([10.0, 5.0])
    weight = np.array([10, 20], dtype=int)
    extra_variables = dict(weight=weight)

    S = State(weight="float")

    S.append(X=X, Y=Y, Z=Z, weight=[100, 101])

    S.alive[1] = False

    D = dict(Y=np.array([11, 12]), Z=5, weight=[200], X=[1, 2])
    S.append(**D)

    print("len   :", len(S))
    print("pid   :", S.pid)
    print("alive :", S.alive)
    print("weight:", S.weight)

    S.compactify()

    print("")
    print("len   :", len(S))
    print("pid   :", S.pid)
    print("alive :", S.alive)
    print("weight:", S.weight)  # type: ignore

    S.append(X=1, Y=2, Z=3, weight=4)

    print("")
    print("len   :", len(S))
    print("pid   :", S.pid)
    print("alive :", S.alive)
    print("weight:", S.weight)  # type: ignore
