"""Time related information for LADiM"""

# ================================
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# 2020-10-29
# ================================

import datetime
import re
from typing import Union, Optional, Sequence
import numpy as np  # type: ignore

Time = Union[str, np.datetime64, datetime.datetime]
TimeDelta = Union[int, np.timedelta64, datetime.timedelta, Sequence, str]

# TODO: Implement reasonable behaviour for backward tracking
#       stop before start


class TimeKeeper:
    """Time utilities for LADiM

    attributes:
        start_time: start time of simulation
        stop_time: stop time for simulation
        reference_time: reference time for cf-standard output
        dt: seconds, model time step
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
        self._dt = normalize_period(dt)  # np.datetime64(-,"s")
        self.dt = self._dt.astype("int")  # seconds

        # Number of time steps (excluding initial)
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


def normalize_period(per: TimeDelta) -> np.timedelta64:
    """Normalize different time period formats to np.timedelta64(-,"s")

    Accepted formats:
       int:  numbe of seconds
       np.timedelta64
       [value, unit]:  np.timedelta(value, unit), unit = "h", "m", "s"
       ISO 8601 format: "PTxHyMzS", x hours, y minutes, z seconds
    """

    if isinstance(per, (int, np.timedelta64, datetime.timedelta)):
        return np.timedelta64(per, "s")
    # if isinstance(per, np.timedelta64):
    #    return per.astype("m8[s]")
    if isinstance(per, (list, tuple)):
        value, unit = per
        try:
            return np.timedelta64(np.timedelta64(value, unit), "s")
        except (TypeError, ValueError):
            raise ValueError(f"{per} is not a valid time period")

    if isinstance(per, str):  # ISO 8601 standard PTxHyMzS
        pattern = r"^PT(\d+H)?(\d+M)?(\d+S)?$"
        m = re.match(pattern, per)
        if m is None:
            raise ValueError(f"{per} is not a valid time period")
        td = np.timedelta64(0, "s")
        if not any(m.groups()):
            raise ValueError(f"{per} is not a valid time period")
        for item in m.groups():
            if item:
                value = int(item[:-1])
                unit = item[-1].lower()
                td = td + np.timedelta64(value, unit)
        return td

    # None of the above
    raise ValueError(f"{per} is not a valid time period")


# def parse_isoperiod(s: str) -> np.timedelta64:
#     """Parse an ISO 8601 time period

#     Valid input format examples:
#         PT3H:     3 hours
#         PT30M:    30 minutes
#         PT600S:   600 seconds = 10 minutes
#         PT16M40S: 1000 seconds

#     """

#     pattern = r"^PT(\d+H)?(\d+M)?(\d+S)?$"
#     m = re.match(pattern, s)
#     if m is None:
#         raise ValueError(f"{s} is not recognized as an ISO 8601 time period")
#     td = np.timedelta64(0, "s")
#     if not any(m.groups()):
#         raise ValueError(f"{s} is not recognized as an ISO 8601 time period")
#     for item in m.groups():
#         if item:
#             value = int(item[:-1])
#             unit = item[-1].lower()
#             td = td + np.timedelta64(value, unit)
#     return td
