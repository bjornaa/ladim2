"""Time related information for LADiM"""

# ================================
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2020-10-29
# ================================

import datetime
from typing import Union, Optional
import numpy as np  # type: ignore

Time = Union[str, np.datetime64, datetime.datetime]
TimeDelta = Union[int, np.timedelta64, datetime.timedelta]

# TODO: Implement reasonable behaviour for backward tracking
#       stop before start


class TimeKeeper:
    """Time utilities for LADiM

    attributes:
        start_time: start time of simulation
        stop_time: stop time for simulation
        reference_time: reference time for cf-standard output
        dt: np.timedelta64, model time step
        Nsteps: Total number of time steps in simulation

    methods:
        time2step: time -> time step number
        step2isotime: time step number -> yyyy-mm-ddThh:mm:ss
        step2nctime: time -> seconds since reference time (or hours/days)
        cfunits: reference time -> "seconds since reference time" (or hours/days)

    """

    unit_table = dict(s="seconds", m="minutes", h="hours", d="days")

    def __init__(
        self, start: Time, stop: Time, dt: TimeDelta, reference: Optional[Time] = None
    ) -> None:
        """
        start      start time
        stop       stop time
        dt         duration of time step
        reference  reference time (for netcdf), optional, default=start

        start, stop, reference given as iso-strings or datetime instances
        dt given i seconds

        """

        # print("TimeControl.__init__")

        self.start_time = np.datetime64(start, "s")
        self.stop_time = np.datetime64(stop, "s")
        if reference:
            self.reference_time = np.datetime64(reference, "s")
        else:
            self.reference_time = self.start_time
        self._dt = np.timedelta64(dt, "s")
        self.dt = self._dt.astype("int")

        self.Nsteps = (self.stop_time - self.start_time) // self._dt
        self.simulation_time = self.Nsteps * self._dt

    def time2step(self, time_: Time) -> int:
        """Timestep from time

        time can be datetime instance or an iso time string
        """
        # Raise exception if not an integer??
        return (np.datetime64(time_) - self.start_time) // self._dt

    def step2isotime(self, stepnr: int) -> str:
        """Return time in iso 8601 format from a time step number"""
        return str(self.start_time + stepnr * self._dt)

    def step2nctime(self, stepnr: int, unit: str = "s") -> float:
        """
        Return value from a time step following the netcdf standard

        unit should be a single character, "s", "m", or "h", default = "s"
        """
        delta = self.start_time + stepnr * self._dt - self.reference_time
        value = delta / np.timedelta64(1, unit)
        return value

    def cf_units(self, unit="s"):
        """Return string with units for time following the CF standard"""
        return f"{self.unit_table[unit]} since {self.reference_time}"
