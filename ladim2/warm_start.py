"""Perform initialization for warm start in LADiM"""

from typing import List

import numpy as np
from netCDF4 import Dataset

from .state import State


def warm_start(
    warm_start_file: str, warm_start_variables: List[str], state: State
) -> None:
    """Initiate the state from a warm start"""


    print("wwarm start: variables = ", warm_start_variables)

    wvars = warm_start_variables.copy() + ["pid"]

    # Open warm start file
    try:
        f = Dataset(warm_start_file)
        f.set_auto_mask(False)
    except FileNotFoundError:
        # logging.critical(f"Can not open warm start file: {warm_start_file}")
        raise SystemExit(1)

    # Use last record in file
    pstart = f.variables["particle_count"][:-1].sum()
    pcount = f.variables["particle_count"][-1]
    pend = pstart + pcount
    pid_max = np.max(f.variables["pid"][:]) + 1

    state.npid = pid_max

    for var in wvars:
        print(" -- ", var)
        # print(var, f.variables[var].dtype)
        ncvar = f.variables[var]
        if var in state.instance_variables:
            values = ncvar[pstart:pend]
        else:  # Particle variable
            values = ncvar[:pid_max]
        # Check for time
        if "units" in ncvar.ncattrs() and "since" in ncvar.units:
            units = ncvar.units
            reftime = np.datetime64(units.split("since")[1])
            print(reftime)
            print(values[0], values[-1], values.dtype)
            print(units[0])
            # print(np.timedelta64(values, units[0]))
            values = reftime + values * np.timedelta64(1, units[0])
            print(values[0], values[-1], values.dtype)

        if var in state.instance_variables:
            state.variables[var] = values
        else:  # particle variable
            state.variables[var] = values

        # Instance variables with default
        if "alive" not in wvars:
            state.variables["alive"] = np.ones(pcount).astype("bool")
        if "active" not in wvars:
            state.variables["active"] = np.ones(pcount).astype("bool")
