# A class for all time related info

# Formål: slippe ad-hoc konvertering ulike steder i koden
# Internt: Bruke tidsteg konsekvent som klokke
# Eksternt: Bruke np.datetime64[s] konsekvent
# Vurdere cftime? F.eks. ved klimasimulering 360 dagers år,
#   det er i alle fall eksternt og kan håndteres med egen timer klasse

# 1) Holde reder på start, stopp og referanse-tid
# 2) Lagre dt
# 3) Konvertere mellom tidsteg og tid
# 4) Konvertere mellom tidsteg og nctime

# Timer = dårlig navn?
# Vanligvis brukt for å ta tid for å utføre en oppgave
#
# Ta det inn i grid (4D grid)?
# grid blir da en mer generell diskretiserings-klasse
#
# Fordel: En klasse mindre å importere
# Ulempe: Dårligere modularisering

import datetime
from typing import Union, Optional, Tuple
import numpy as np

Time = Union[str, np.datetime64, datetime.datetime]
TimeDelta = Union[int, np.timedelta64, datetime.timedelta]

# TODO: Implement reasonable behaviour for backward tracking
#       stop before start

class Timer:

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
        dt given i seconds (Være mer generell?)

        """
        self.start_time = np.datetime64(start, "s")
        self.stop_time = np.datetime64(stop, "s")
        if reference:
            self.reference_time = np.datetime64(reference, "s")
        else:
            self.reference_time = self.start_time
        self._dt = np.timedelta64(dt, "s")
        self.dt = self._dt.astype("int")
        # Eller self.dt = np.timedelta64(dt, 's')

        self.Nsteps = (self.stop_time - self.start_time) // self._dt

    def time2step(self, time_: Time) -> int:
        """Timestep from time

        time can be datetime instance or an iso time string
        """
        # Raise exception if not an integer??
        return (np.datetime64(time_) - self.start_time) // self._dt

    def step2isotime(self, n: int) -> str:
        """Return time in iso 8601 format from a time step"""
        return str(self.start_time + n * self._dt)

    def step2nctime(self, n: int, unit: str = "s") -> Tuple[float, str]:
        """
        Return value and time from a time step following the netcdf standard

        unit should be a single character, "s", "m", or "h"
        """
        delta = self.start_time + n * self._dt - self.reference_time
        value = delta / np.timedelta64(1, unit)
        unit_string = f"{Timer.unit_table[unit]} since {self.reference_time}"
        return value, unit_string
