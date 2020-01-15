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
        variables: Dict[str, DType] = dict(
            pid=int, alive=bool, X=float, Y=float, Z=float
        )
        variables.update(args)
        for var, dtype in variables.items():
            setattr(self, var, np.array([], dtype=dtype))

        # Trenger self.variables?
        # Sjekke at append har alle variable som trengs
        # Evt, ha en default (alder=0) f.eks.
        # Hvordan angi denne default
        # age = (float, 0.0) f.eks. mens weigth =

    def append(self, X: Arraylike, Y: Arraylike, Z: Arraylike, **args: Arraylike):
        """Append particles to the State object"""

        # All input should be scalars or broadcastable 1D arrays
        names = ["X", "Y", "Z"] + list(args.keys())
        values = [X, Y, Z] + list(args.values())
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

    S.append(X, Y, Z, weight=[100, 101])

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

    S.append(1, 2, 3, weight=4)

    print("")
    print("len   :", len(S))
    print("pid   :", S.pid)
    print("alive :", S.alive)
    print("weight:", S.weight)  # type: ignore
