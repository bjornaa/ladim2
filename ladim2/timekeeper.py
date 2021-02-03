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
        time_reversal: flag for time reversal
        initial_time: usually start_time, stop_time if time_reversal
        reference_time: reference time for cf-standard output
        time_reversed: switch for time reversal
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
        self,
        start: Time,
        stop: Time,
        dt: TimeDelta,
        reference: Optional[Time] = None,
        time_reversal: bool = False,
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
        self.time_reversal = time_reversal
        self.time = self.start_time  # Running clock

        # Quality control
        duration = self.stop_time - self.start_time
        if time_reversal != (duration < np.timedelta64(0)):
            if time_reversal:
                print("ERROR: Backwards time and start before stop")
            else:
                print("ERROR: Forward time and stop before start")
            raise SystemExit(3)

        self.min_time = min(self.start_time, self.stop_time)
        self.max_time = max(self.start_time, self.stop_time)

        if reference:
            self.reference_time = np.datetime64(reference, "s")
        else:
            self.reference_time = self.min_time

        self._dt = normalize_period(dt)  # np.timedelta64(-,"s")
        self.dt = self._dt.astype("int")  # seconds

        # Number of time steps (excluding initial)
        self.Nsteps = abs(duration) // self._dt
        self.simulation_time = self.Nsteps * self._dt

    def update(self) -> None:
        """Update the clock"""
        if self.time_reversal:
            self.time = self.time - self._dt
        else:
            self.time = self.time + self._dt

    def reset(self) -> None:
        """Reset the clock"""
        self.time = self.start_time

    def nctime(self, unit: str = "s") -> float:
        """Get float value of model time"""
        delta = self.time - self.reference_time
        return delta / np.timedelta64(1, unit)

    def time2step(self, time_: Time) -> int:
        """Timestep from time

        time can be datetime instance or an iso time string
        """
        if self.time_reversal:
            return (self.start_time - np.datetime64(time_)) // self._dt
        return (np.datetime64(time_) - self.start_time) // self._dt

    def step2isotime(self, stepnr: int) -> str:
        """Return time in iso 8601 format from a time step number"""
        if self.time_reversal:
            return str(self.start_time - stepnr * self._dt)
        return str(self.start_time + stepnr * self._dt)

    def step2nctime(self, stepnr: int, unit: str = "s") -> float:
        """
        Return value from a time step following the netcdf standard

        unit should be a single character, "s", "m", or "h", default = "s"
        """
        if self.time_reversal:
            delta = self.start_time - stepnr * self._dt - self.reference_time
        else:
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
