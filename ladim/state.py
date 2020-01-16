"""
Class for the state of the model
"""

import sys
import os
import importlib
import logging
from typing import Any, Dict, Sized, Union, Optional, List, Mapping  # mypy

import numpy as np  # type: ignore

# from .tracker import Tracker
# from .gridforce import Grid, Forcing

# ------------------------

Arraylike = Union[np.ndarray, List[float], float]
Variables = Dict[str, Arraylike]
DType = Union[str, type]


class State(Sized):
    """The model variables at a given time"""

    def __init__(self, **args: DType) -> None:

        # Start with empty state, with correct variables of correct type
        self.npid = 0  # Number of particles involved
        vars: Dict[str, DType] = dict(
            pid=int, alive=bool, X=float, Y=float, Z=float
        )
        vars.update(args)
        for var, dtype in vars.items():
            setattr(self, var, np.array([], dtype=dtype))

        self.variables = list(vars.keys())
        self.dtypes = vars
        self.default_values = {'alive': np.array(True, dtype=bool)}
        # Kan sette alle andre default = 0

    def set_default_value(self, variable, value) -> None:
        """Set default values for a variable"""
        if variable not in self.variables:
            raise ValueError(f"No such variable: ", variable)
        if variable == 'pid':
            raise ValueError("Can not set default for pid")
        self.default_values[variable] = np.array(value, dtype=self.dtypes[variable])


    def append(self, **args: Arraylike):
        """Append particles to the State object"""

        # Accept only state variables (except pid)
        names = set(args.keys())
        state_vars = set(self.variables)
        state_vars.remove('pid')
        assert(names.issubset(state_vars))

        # All state variables (except pid) should be included
        # or have a default value
        # (evt. få default = null)
        vars_with_value = names.union(self.default_values)
        assert(state_vars.issubset(vars_with_value))

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
    print("weight:", S.weight)  # type: ignore

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
