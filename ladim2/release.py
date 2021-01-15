"""Particle release class"""

# -------------------
# release.py
# part of LADiM
# --------------------

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# ----------------------------------

from collections.abc import Iterator
from pathlib import Path
import logging
from typing import List, Optional, Union, Dict, Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .timekeeper import TimeKeeper, normalize_period
from .grid import BaseGrid

# from .utilities import ingrid

DEBUG = False


class ParticleReleaser(Iterator):
    """Particle Release Class"""

    def __init__(
        self,
        release_file: Union[Path, str],
        timer: TimeKeeper,
        datatypes: Dict[str, Any],
        names: Optional[List[str]] = None,
        grid: Optional[BaseGrid] = None,
        # dtype: Dict = None,
        continuous: bool = False,
        release_frequency: int = 0,  # frequnency in seconds
        # warm_start=False,
        **args,
    ) -> None:

        self.start_time = timer.start_time
        self.stop_time = timer.stop_time
        self.time_reversal = timer.time_reversal

        # # logging.info("Initializing the particle releaser")

        self._df = self.read_release_file(release_file, datatypes, names)

        # If no mult column, add a column of ones
        if "mult" not in self._df.columns:
            self._df["mult"] = 1

        self.clean_position(grid)

        # Remove everything after simulation stop time
        if timer.time_reversal:
            self._df = self._df[self._df.index >= self.stop_time]  # Use < ?
        else:
            self._df = self._df[self._df.index <= self.stop_time]  # Use < ?
        if len(self._df) == 0:  # All release after simulation time
            logging.critical("All particles released after similation stop")
            raise SystemExit(3)

        # Make the dataframe explicitly discrete
        # REMARK: continuous flag is unnecessary,
        # could use release_frequency == True (not None, > 0)
        if continuous:
            self.release_frequency = normalize_period(release_frequency)
            self.discretize()

        # Now discrete, remove everything before start
        if timer.time_reversal:
            self._df = self._df[self._df.index <= self.start_time]
        else:
            self._df = self._df[self._df.index >= self.start_time]

        if len(self._df) == 0:  # All release before start
            logging.critical("All particles released before simulation start")
            raise SystemExit(3)

        # Add a release_time column if requested by the datatypes
        if "release_time" in datatypes.keys():
            self._df["release_time"] = self._df.index

        # # Optionally, remove everything outside a subgrid
        # try:
        #     subgrid: List[int] = config["grid_args"]["subgrid"]
        # except KeyError:
        #     subgrid = []
        # if subgrid:
        #     lenA = len(A)
        #     A = A[ingrid(A["X"], A["Y"], subgrid)]
        #     if len(A) < lenA:
        #         logging.warning("Ignoring particle release outside subgrid")

        # # If warm start, no new release at start time (already accounted for)
        # if config["start"] == "warm":
        #     A = A[A.index > start_time]

        # Total number of particles released
        self.total_particle_count = self._df.mult.sum()
        # logging.info("Total particle count = {}".format(self.total_particle_count))

        # Release times
        self.times = self._df.index.unique()
        self.steps = [timer.time2step(t) for t in self.times]
        # logging.info("Number of release times = {}".format(len(self.times)))

        # # Make dataframes for each timeframe
        # # self._B = [x[1] for x in A.groupby('release_time')]
        self._B = [x[1] for x in self._df.groupby(self._df.index)]

        # # Read the particle variables
        self._index = 0  # Index of next release
        self._particle_count = 0  # Particle counter

        # # Handle the particle variables initially
        # # TODO: Need a test to check that this iw working properly
        # pvars = dict()
        # for name in config["particle_variables"]:
        #     dtype = config["release_dtype"][name]
        #     if dtype == np.datetime64:
        #         dtype = np.float64
        #     pvars[name] = np.array([], dtype=dtype)

        # TODO: Move warm start to separate function
        # # Get particle data from  warm start
        # if config["start"] == "warm":
        #     with Dataset(config["warm_start_file"]) as f:
        #         # warm_particle_count = len(f.dimensions['particle'])
        #         warm_particle_count = np.max(f.variables["pid"][:]) + 1
        #         for name in config["particle_variables"]:
        #             pvars[name] = f.variables[name][:warm_particle_count]
        # else:
        #     warm_particle_count = 0

        # # initital number of particles
        # if config["start"] == "warm":
        #     particles_released = [warm_particle_count]
        # else:
        #     particles_released = [0]

        # # Loop through the releases, collect particle variable data
        # for t in self.times:
        #     V = next(self)
        #     particles_released.append(particles_released[-1] + len(V))
        #     for name in config["particle_variables"]:
        #         dtype = config["release_dtype"][name]
        #         if dtype == np.datetime64:
        #             g = np.array(V[name]).astype("M8[s]")
        #             rtimes = g - config["reference_time"]
        #             rtimes = rtimes.astype(np.float64)
        #             pvars[name] = np.concatenate((pvars[name], rtimes))
        #         else:
        #             pvars[name] = np.concatenate((pvars[name], V[name]))

        # self.total_particle_count = warm_particle_count + self._particle_count
        # self.particle_variables = pvars

        # self.particles_released = particles_released

        # # Reset the counter after the particle counting
        # self._index = 0  # Index of next release
        # self._particle_count = warm_particle_count

        # self.df = A

    def __next__(self) -> pd.DataFrame:
        """Perform the next particle release

           Return a DataFrame with the release info,
           repeated mult times

        """

        if DEBUG:
            print("particle release")

        # This should not happen
        if self._index >= len(self.times):
            raise StopIteration

        # Skip first release if warm start (should be present in start file)
        # Not always, make better test
        # Moving test to state.py
        # if self._index == 0 and self._particle_count > 0:  # Warm start
        #    return

        # rel_time = self.times[self._index]
        # file_time = self._file_times[self._file_index]

        V = self._B[self._index]
        # nnew = V.mult.sum()
        # Workaround, missing repeat method for pandas DataFrame
        V0 = V.to_records(index=False)
        V0 = V0.repeat(V.mult)
        V = pd.DataFrame(V0)
        # Do not need the mult column any more
        V.drop("mult", axis=1, inplace=True)
        # Buffer the new values

        # Update the counters
        self._index += 1
        self._particle_count += len(V)

        # print(V)

        return V

    @staticmethod
    def read_release_file(
        rls_file: Union[Path, str],
        datatypes: Dict[str, Any],
        names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read the release file into a pandas DataFrame"""

        # Handle time separately
        time_vars = ["release_time"]
        dtypes = dict()
        for var, dtype in datatypes.items():
            if dtype in ["M8[s]", np.dtype("M8[s]"), np.dtype("M8[ns]")]:
                time_vars.append(var)
            else:
                dtypes[var] = dtype

        df = pd.read_csv(
            rls_file,
            parse_dates=list(set(time_vars)),
            names=names,
            dtype=dtypes,
            delim_whitespace=True,
            index_col="release_time",
        )

        return df

    def clean_position(self, grid: Optional[BaseGrid] = None) -> None:
        """Make sure the release data have mult, X, and Y columns

        X and Y may be inferred from lon and lat using grid.ll2xy
        """

        df = self._df
        # # If no mult column, add a column of ones
        # if "mult" not in df.columns:
        #     df["mult"] = 1

        # Conversion from longitude, latitude to grid coordinates
        if "X" not in df.columns or "Y" not in df.columns:
            if "lon" not in df.columns or "lat" not in df.columns:
                # logging.critical("Particle release must include a position")
                raise SystemExit(3)
            try:
                X, Y = grid.ll2xy(df["lon"], df["lat"])  # type: ignore
            except AttributeError:
                print("""Can not convert from lon/lat to grid coordinates""")
                raise SystemExit(3)

            df["lon"] = X
            df["lat"] = Y
            df.rename(columns={"lon": "X", "lat": "Y"}, inplace=True)

        self._df = df

    def discretize(self) -> None:
        """Make a continuous release sequence discrete"""

        df = self._df

        # Find last release time <= start_time
        if not self.time_reversal:
            n = np.sum(df.index <= self.start_time)
        else:
            n = np.sum(df.index >= self.start_time)

        if n == 0:
            logging.warning("No particles released at simulation start")
            n = 1  # Use first release entry
        release_time0 = df.index[n - 1]
        if DEBUG:
            print("release_time0 =", release_time0)
        # Remove the early entries
        # NOTE: Makes a new DataFrame
        if not self.time_reversal:
            df = df[df.index >= release_time0]
        else:
            df = df[df.index <= release_time0]

        # Find first effective release time
        # i.e. the last time <= start_time
        #   and remove too early releases
        # Can be moved out of if-block?
        # n = np.sum(df.index <= self.start_time)
        # if n == 0:
        #    logging.warning("No particles released at simulation start")
        #    n = 1

        file_times = df.index.unique()

        # time0 = file_times[0]
        # time1 = stop_time
        # times = np.arange(time0, time1, release_frequency)

        if not self.time_reversal:
            freq = self.release_frequency
        else:
            freq = -self.release_frequency

        times = np.arange(file_times[0], self.stop_time, np.timedelta64(freq, "s"))

        # Reindex the index
        J = pd.Series(file_times, index=file_times).reindex(times, method="ffill")
        num_entries_per_time = {i: mylen(df.loc[i]) for i in file_times}
        df = df.loc[J]

        # Set non-unique index
        S: List[int] = []
        for t in times:
            S.extend(num_entries_per_time[J[t]] * [t])
        df.index = S

        self._df = df


def mylen(df: pd.DataFrame) -> int:
    """Number of rows in a DataFrame,

    A workaround for len() which does not
    have the expected behaviour with itemizing,
    """
    return df.shape[0] if df.ndim > 1 else 1
