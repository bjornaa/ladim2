"""
Class for the LADiM model state
"""

# import sys
# import os
# import importlib
import logging
from typing import Any, Dict, Sized, Union, Optional, List, Mapping  # mypy

import numpy as np  # type: ignore

# ------------------------

Arraylike = Union[np.ndarray, List[float], float]
Variables = Dict[str, Arraylike]
DType = Union[str, type]


class State(Sized):
    """The model variables at a given time"""

    def __init__(self, **args):

        # Start with empty state, with correct variables of correct type
        self.npid = 0  # Number of particles involved
        # vars: Dict[str, DType] = dict(pid=int, alive=bool, X=float, Y=float, Z=float)
        vars = dict(pid=int, alive=bool, X=float, Y=float, Z=float)
        vars.update(args)
        for var, dtype in vars.items():
            setattr(self, var, np.array([], dtype=dtype))

        self.variables = list(vars.keys())
        self.dtypes = vars
        self.default_values = {"alive": np.array(True, dtype=bool)}
        # Kan sette alle andre default = 0

    def set_default_value(self, variable, value):
        """Set default values for a variable"""
        if variable not in self.variables:
            raise ValueError("No such variable: ", variable)
        if variable == "pid":
            raise ValueError("Can not set default for pid")
        self.default_values[variable] = np.array(value, dtype=self.dtypes[variable])

    # Refaktorer denne
    def append(self, **args):
        """Append particles to the State object"""

        # Accept only state variables except pid
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

        # print(f"{state_vars=}")
        # print(f"{ok_vars=}")

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
        print("broadcast: ", b.shape)

        # Make all values 1D of correct shape
        values = [np.broadcast_to(v, shape=(nparticles,)) for v in ok_vars.values()]

        # pid
        self.pid = np.concatenate(
            (self.pid, np.arange(self.npid, self.npid + nparticles, dtype=int))
        )
        self.npid = self.npid + nparticles

        # Set the state variables
        for name, value in zip(list(ok_vars), values):
            print("+++", name, value)
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
    # S.append(X=1, Y=2, Z=3, weight=4)

    S.append(X=1, Y=2, Z=3)

    print("")
    print("len   :", len(S))
    print("pid   :", S.pid)
    print("alive :", S.alive)
    print("X :", S.X)
    print("Y :", S.Y)
    print("Z :", S.Z)
    print("weight:", S.weight)  # type: ignore

    # S.alive[1] = False

    # D = dict(Y=np.array([11, 12]), Z=5, weight=[200], X=[1, 2])
    # S.append(**D)

    # print("")
    # print("len   :", len(S))
    # print("pid   :", S.pid)
    # print("alive :", S.alive)
    # print("X :", S.X)
    # print("weight:", S.weight)  # type: ignore

    # S.compactify()

    # print("")
    # print("len   :", len(S))
    # print("pid   :", S.pid)
    # print("alive :", S.alive)
    # print("X :", S.X)
    # print("weight:", S.weight)  # type: ignore

    # S.append(X=1, Y=2, Z=3, weight=4)

    # print("")
    # print("len   :", len(S))
    # print("pid   :", S.pid)
    # print("alive :", S.alive)
    # print("X :", S.X)
    # print("weight:", S.weight)  # type: ignore
