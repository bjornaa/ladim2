"""
Class for the state of the model
"""

from typing import List, Dict, Union, Sized, Sequence, Tuple

import numpy as np  # type: ignore


# From State point-of-view the only difference between
# instance anc particle variables is that instance variables
# are affected by the compactify operation

# ------------------------

# Kan bruke Sequence[Scalar] for 1D arrays o.l.
# kan sette dette som Vector?
Scalar = Union[float, int, bool]
# Arraylike = Union[np.ndarray, List[Scalar], Scalar]
# Variables = Dict[str, Arraylike]
DType = Union[str, type]
Vector = Union[Scalar, Sequence[Scalar]]


class State(Sized):
    """The model variables at a given time"""

    # def __init__(self, **args: Dict[str, DType]) -> None:
    def __init__(self, variables=None, particle_variables=None) -> None:
        """Initialize the state with dictionary of extra variables

        These extra variables should be given by their dtype

        Mandatory state variables: pid, alive, active, X, Y, Z
        with predefined dtypes int, 2*bool, and 3*float
        should not be initialized.

        attributes:
          npid: Number of particles released so far
          variables: List of names of state variables
          default_values: Default values for initializing state variables

        """

        print("State.__init__")

        # Make variables dictionary (with values = dtype)
        mandatory_variables = dict(
            pid=int, X=float, Y=float, Z=float, active=bool, alive=bool
        )
        if variables is None:
            variables = dict()
        self.variables = dict(mandatory_variables, **variables)

        # Replace the dtypes by empty arrays
        self.variables = {
            var: np.array([], dtype) for var, dtype in self.variables.items()
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

        self.npid: int = 0   # Total number of pids used

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

    def append(self, **args: Vector) -> None:
        """Append particles to the State object"""

        # Accept only instance variables (except pid)
        state_vars = set(self.variables)
        state_vars.remove("pid")
        for name in args:
            if name not in state_vars:
                raise ValueError(f"Invalid argument {name}")

        # ok_vars = arguments and variables with defaults
        # arguments override the defaults
        ok_vars = dict(self.default_values, **args)
        #ok_vars = self.default_values.copy()
        #ok_vars.update(args)

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
        self.variables['pid'] = np.concatenate(
            (self.variables['pid'], np.arange(self.npid, self.npid + nparticles, dtype=int))
        )
        self.npid = self.npid + nparticles

        # Set the state variables
        for name, value in zip(list(ok_vars), values):
            #setattr(self, name, np.concatenate((getattr(self, name), value)))
            self.variables[name] = np.concatenate((self.variables[name], value))


    def compactify(self) -> None:
        """Remove dead particles"""
        alive = self.alive.copy()
        for var in self.instance_variables:
            A = getattr(self, var)
            setattr(self, var, A[alive])

    def __len__(self) -> int:
        return len(self.pid)



    # Allow item notation, state[var]
    def __getitem__(self, name: str):
        return getattr(self, name)

    # Allow attribute notation, (should be read-only)
    def __getattr__(self, var: str):
        return self.variables[var]
