"""
Class for the state of the model
"""

# ----------------------------------
# Bjørn Ådlandsvik
# Institute of Marine Research
# October 2020
# ----------------------------------

from numbers import Number
from typing import List, Dict, Union, Sized, Sequence, Optional

import numpy as np  # type: ignore


# From State point-of-view the only difference between
# instance and particle variables is that instance variables
# are affected by the compactify operation

# ------------------------

# Define some types
Scalar = Number
Arraylike = Union[np.ndarray, Sequence[Scalar], Scalar]

# State has no internal difference between particle and instance variable.
# With append also particle variables needs a value
# Have an undef value?, nan for float types,
# Should not compactify particle variables


class State(Sized):
    """
    The model variables at a given time

    Mandatory state variables: pid, alive, active, X, Y, Z
    with predefined dtypes int, 2*bool, and 3*float

    attributes:
        npid: Number of particles released so far
        variables: Dictionary of state variables
        default_values: Default values for initializing state variables

    methods:
        set_default_values: Define default values for state variables
        append: Append new particles to the state
        compactify: Throw out dead particles

    """

    # The data are kept in the dictionary variables
    # instance_variables and particle_variables are sets

    # The length of the state should only be allowed to change by
    # append and compactify (rename the last to kill?)

    def __init__(
        self,
        variables: Optional[Dict[str, type]] = None,
        particle_variables: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the state

        variables: dictionary of extra state variables with their dtype,
                   mandatory variables should not be given
        particle_variables: list of names of variables that are time independent

        """

        # print("State.__init__")

        # Make dtypes dictionary
        mandatory_variables = dict(
            pid=int, X=float, Y=float, Z=float, active=bool, alive=bool
        )
        if variables is None:
            variables = dict()
        self.dtypes = dict(mandatory_variables, **variables)

        # Make empty arrays of correct dtype
        self.variables = {
            var: np.array([], dtype) for var, dtype in self.dtypes.items()
        }

        # Sets of particle and instance variables
        if particle_variables is None:
            self.particle_variables = set()
        else:
            self.particle_variables = set(particle_variables)
        self.instance_variables = set(self.variables) - self.particle_variables

        # Predefined default values
        self.default_values = dict(
            alive=np.array(True, dtype=bool), active=np.array(True, dtype=bool)
        )

        self.npid: int = 0  # Total number of pids used

    def set_default_values(self, **args: Scalar) -> None:
        """Set default values for state variables"""
        for var, value in args.items():
            if var not in self.variables:
                raise ValueError("No such variable: ", var)
            if var == "pid":
                raise ValueError("Can not set default for pid")
            if not np.isscalar(value):
                raise TypeError(f"Default value for {var} should be scalar")
            self.default_values[var] = np.array(value, dtype=self.dtypes[var])

    def append(self, **args: Arraylike) -> None:
        """Append particles to the State object"""

        # Howto handle particle variables?

        # state_vars = instance_variables (without pid)
        state_vars = set(self.variables) - {"pid"}

        # Accept only state_vars
        for name in args:
            if name not in state_vars:
                raise ValueError(f"Invalid argument {name}")

        # Variables must have a value
        value_vars = dict(self.default_values, **args)
        for name in state_vars:
            if name not in set(value_vars):
                raise TypeError(f"Variable {name} has no value")

        # Broadcast all variables to 1D arrays
        #    Raise ValueError if not compatible
        ### values = list(args)
        b = np.broadcast(*value_vars.values())
        if b.ndim > 1:
            raise ValueError("Arguments must be 1D or scalar")
        # if b.ndim == 0:  # All arguments are scalar
        #    b = np.broadcast([0])  # Make b.size = 1
        num_new_particles = b.size
        values = {
            var: np.broadcast_to(v, shape=(num_new_particles,))
            for var, v in value_vars.items()
        }

        # pid must be handles separately
        self.variables["pid"] = np.concatenate(
            (
                self.variables["pid"],
                np.arange(self.npid, self.npid + num_new_particles, dtype=int),
            )
        )
        self.npid = self.npid + num_new_particles

        # Concatenate the rest of the variables
        for var in state_vars:
            self.variables[var] = np.concatenate((self.variables[var], values[var]))

    def compactify(self) -> None:
        """Remove dead particles from the instance variables"""
        alive = self.alive.copy()
        for var in self.instance_variables:
            self.variables[var] = self.variables[var][alive]

    def __len__(self) -> int:
        return len(self.pid)

    # Allow item notation, state[var]
    def __getitem__(self, var: str) -> np.ndarray:
        return self.variables[var]

    def __setitem__(self, var: str, item: Arraylike) -> None:
        value = np.array(item, dtype=self.dtypes[var])
        # The size of item should be unchanged
        if np.size(value) != len(self.variables[var]):
            raise KeyError("Size of data should be unchanged")
        else:
            self.variables[var] = value

    # Allow attribute notation, (should be read-only?)
    def __getattr__(self, var: str) -> np.ndarray:
        return self.variables[var]

    # Disallow
    # def __setattr__(self, var: str, item: Arraylike) -> None:
    # self.__setitem__(var, item)
    # self.variables[var] = np.array(item, dtype=self.dtypes[var])
