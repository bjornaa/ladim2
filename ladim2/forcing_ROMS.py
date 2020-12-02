"""
Forcing for LADiM from the Regional Ocean Model System (ROMS)

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2017-03-01
# -----------------------------------

from pathlib import Path
import logging
from typing import Union, Optional, List, Tuple, Dict

import numpy as np        # type: ignore
from netCDF4 import Dataset, num2date   # type: ignore

# from ladim2.state import State
from ladim2.grid_ROMS import Grid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import BaseForce


DEBUG = False


def init_force(**args) -> BaseForce:
    return Force_ROMS(**args)


class Force_ROMS(BaseForce):
    """
    Class for ROMS forcing

    Public methods:
        __init__
        update
        velocity
        force_particles (bedre interpolate2particles)

    Public attributes:
        ibm_forcing
        variables
        steps
        # Legg til ny = fields, fields["u"] = 3D field



    """

    def __init__(
        self,
        grid: Grid,
        timer: TimeKeeper,
        filename: Union[Path, str],
        ibm_forcing: Optional[List[str]] = None,
    ) -> None:

        logging.info("Initiating forcing")
        print("Forcing.__init__")

        self.grid = grid  # Get the grid object.
        self.timer = timer

        self.ibm_forcing = ibm_forcing if ibm_forcing else []

        # 3D forcing fields
        self.fields = {
            var: np.array([], float) for var in ["u", "v"] + self.ibm_forcing
        }
        # Forcing interpolated to particle positions
        self.variables = {
            var: np.array([], float) for var in ["u", "v"] + self.ibm_forcing
        }

        # Input files and times

        files = find_files(filename)
        numfiles = len(files)
        if numfiles == 0:
            logging.error("No input file: {}".format(filename))
            raise SystemExit(3)
        logging.info("Number of forcing files = {}".format(numfiles))

        self.files = files

        self.time_reversal = timer.time_reversal
        steps, file_at_step, recordnr_at_step = forcing_steps(files, timer)
        # self.stepdiff = np.diff(steps)
        self.file_at_step = file_at_step
        self.recordnr_at_step = recordnr_at_step
        self._first_read = True  # True until first file is opened
        # self._nc = None  # Not opened yet
        self.steps = steps

        # Read old input
        # requires at least one input before start
        # to get Runge-Kutta going
        # --------------
        # prestep = last forcing step < 0
        #

        V = [step for step in steps if step < 0]
        prestep = max(V) if V else 0
        i = steps.index(prestep)
        nextstep = steps[i - 1] if timer.time_reversal else steps[i + 1]
        stepdiff0 = nextstep - prestep

        self.fields["u"], self.fields["v"] = self._read_velocity(prestep)
        self.fields["u_new"], self.fields["v_new"] = self._read_velocity(nextstep)
        self.fields["dU"] = (self.fields["u_new"] - self.fields["u"]) / stepdiff0
        self.fields["dV"] = (self.fields["v_new"] - self.fields["v"]) / stepdiff0

        # Interpolate to time step = -1
        self.fields["u"] = self.fields["u"] - (prestep + 1) * self.fields["dU"]
        self.fields["v"] = self.fields["v"] - (prestep + 1) * self.fields["dV"]

        # Other forcing
        for name in self.ibm_forcing:
            self.fields[name] = self._read_field(name, prestep)

    # ===================================================

    # Turned off time interpolation of scalar fields
    # TODO: Implement a switch for turning it on again if wanted
    def update(self, step: int, X: float, Y: float, Z: float) -> None:
        """Update the fields to given time step t"""

        # Read from config?
        interpolate_velocity_in_time = True
        # interpolate_ibm_forcing_in_time = False

        logging.debug("Updating forcing, time step = {}".format(step))
        if step in self.steps:  # No time interpolation
            self.fields["u"] = self.fields["u_new"]
            self.fields["v"] = self.fields["v_new"]
            # Read other fields
            for name in self.ibm_forcing:
                self.fields[name] = self._read_field(name, step)

        else:
            if step - 1 in self.steps:  # Need new fields
                i = self.steps.index(step - 1)
                nextstep = (
                    self.steps[i - 1] if self.time_reversal else self.steps[i + 1]
                )
                stepdiff = nextstep - step
                self.fields["u_new"], self.fields["v_new"] = self._read_velocity(
                    nextstep
                )

                if interpolate_velocity_in_time:
                    self.fields["dU"] = (
                        self.fields["u_new"] - self.fields["u"]
                    ) / stepdiff
                    self.fields["dV"] = (
                        self.fields["v_new"] - self.fields["v"]
                    ) / stepdiff

            # "Ordinary" time step (including self.steps+1)
            if interpolate_velocity_in_time:
                self.fields["u"] += self.fields["dU"]
                self.fields["v"] += self.fields["dV"]

        # Update forcing values at particles
        self.force_particles(X, Y, Z)

    # ==============================================

    def open_forcing_file(self, time_step: int) -> None:

        """Open forcing file and get scaling info given time step"""
        # Open the correct forcing file
        if DEBUG:
            print("Opening forcing file: ", self.file_at_step[time_step])
        nc = Dataset(self.file_at_step[time_step])
        nc.set_auto_maskandscale(False)
        self._nc = nc

        # Get scaling info per variable
        self.scaled = dict()
        self.scale_factor = dict()
        self.add_offset = dict()
        forcing_variables = ["u", "v"] + self.ibm_forcing
        for key in forcing_variables:
            if hasattr(nc.variables[key], "scale_factor"):
                self.scaled[key] = True
                self.scale_factor[key] = np.float32(nc.variables[key].scale_factor)
                self.add_offset[key] = np.float32(nc.variables[key].add_offset)
            else:
                self.scaled[key] = False

    def _read_velocity(self, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read velocity fields at given time step"""
        # Need a switch for reading W
        # T = self._nc.variables['ocean_time'][n]  # Read new fields

        # Handle file opening/closing
        # Always read velocity before other fields
        logging.info("Reading velocity for time step = {}".format(time_step))

        if self._first_read:
            self.open_forcing_file(time_step)  # Open first file
            self._first_read = False
        elif str(self.file_at_step[time_step]) != self._nc.filepath():
            # self._nc out of sync, open next file
            self._nc.close()
            self.open_forcing_file(time_step)

        frame = self.recordnr_at_step[time_step]

        if DEBUG:
            print("_read_velocity")
            print("   model time step =", time_step)
            timevar = self._nc.variables["ocean_time"]
            time_origin = np.datetime64(timevar.units.split("since")[1])
            data_time = time_origin + np.timedelta64(int(timevar[frame]), "s")
            print("   data file:   ", self.file_at_step[time_step])
            print("   data record: ", frame)
            print("   data time:   ", data_time)

        # Read the velocity
        U = self._nc.variables["u"][frame, :, self.grid.Ju, self.grid.Iu]
        V = self._nc.variables["v"][frame, :, self.grid.Jv, self.grid.Iv]

        # Scale if needed
        # Assume offset = 0 for velocity
        if self.scaled["u"]:
            U = self.scale_factor["u"] * U
            V = self.scale_factor["v"] * V
            # U = self.add_offset['u'] + self.scale_factor['u']*U
            # V = self.add_offset['v'] + self.scale_factor['v']*V

        # If necessary put U,V = zero on land and land boundaries
        # Stay as float32
        np.multiply(U, self.grid.Mu, out=U)
        np.multiply(V, self.grid.Mv, out=V)
        return U, V

    def _read_field(self, name, n):
        """Read a 3D field"""
        frame = self.recordnr_at_step[n]
        F = self._nc.variables[name][frame, :, self.grid.J, self.grid.I]
        if self.scaled[name]:
            F = self.add_offset[name] + self.scale_factor[name] * F
        return F

    # Allow item notation
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    # ------------------

    def close(self) -> None:
        self._nc.close()

    def force_particles(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    ):
        """Interpolate forcing to particle positions"""
        i0 = self.grid.i0
        j0 = self.grid.j0
        K, A = z2s(self.grid.z_r, X - i0, Y - j0, Z)
        for name in self.ibm_forcing:
            self.variables[name] = sample3D(
                self.fields[name], X - i0, Y - j0, K, A, method="nearest"
            )
        self.variables["u"], self.variables["v"] = sample3DUV(
            self.fields["u"], self.fields["v"], X - i0, Y - j0, K, A, method="bilinear",
        )
        if self.time_reversal:
            self.variables["u"] = -self.variables["u"]
            self.variables["v"] = -self.variables["v"]

    def velocity(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        fractional_step: float = 0,
        method: str = "bilinear",
    ) -> Tuple[np.ndarray, np.ndarray]:

        i0 = self.grid.i0
        j0 = self.grid.j0
        K, A = z2s(self.grid.z_r, X - i0, Y - j0, Z)
        if fractional_step < 0.001:
            U = self.fields["u"]
            V = self.fields["v"]
        else:
            U = self.fields["u"] + fractional_step * self.fields["dU"]
            V = self.fields["v"] + fractional_step * self.fields["dV"]
        if self.time_reversal:
            return sample3DUV(-U, -V, X - i0, Y - j0, K, A, method=method)
        return sample3DUV(U, V, X - i0, Y - j0, K, A, method=method)

    def field(self, X, Y, Z, name):
        """A do-nothing function for backwards compability for IBMs"""
        return self.variables[name]


# ------------------------
#   Sampling routines
# ------------------------


def z2s(
    z_rho: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find s-level and coefficients for vertical interpolation

    input:
        z_rho  3D array with vertical s-coordinate structure at rho-points
        X, Y   1D arrays, horizontal position in grid coordinates
        Z      1D array, particle depth, meters, positive

    Returns
        K      1D integer array
        A      1D float array

    With:
        1 <= K < kmax = z_rho.shape[0]
        z_rho[K-1] < -Z < z_rho[K] for 1 < K < kmax - 1
        -Z < z_rho[1] for K = 1
        z_rho[-1] < -Z for K = kmax - 1
        0.0 <= A <= 1
        Interior linear interpolation:
            A * z_rho[K - 1] + (1 - A) * z_rho[K] = -Z
            for z_rho[0] < -Z < z_rho[-1]
        Extend constant below lowest:
            A * z_rho[K - 1] + (1 - A) * z_rho[K] = z_rho[0]
            for -Z < z_rho[0]  (K=1, A=1)
        Extend constantly above highest:
            A * z_rho[K - 1] + (1 - A) * z_rho[K] = z_rho[-1]
            for -Z > z_rho[-1]  (K=kmax-1, A=0)

    """

    kmax = z_rho.shape[0]  # Number of vertical levels

    # Find rho-based horizontal grid cell (rho-point)
    I = np.around(X).astype("int")
    J = np.around(Y).astype("int")

    # Vectorized searchsorted
    K = np.sum(z_rho[:, J, I] < -Z, axis=0)
    K = K.clip(1, kmax - 1)

    A = (z_rho[K, J, I] + Z) / (z_rho[K, J, I] - z_rho[K - 1, J, I])
    A = A.clip(0, 1)  # Extend constantly

    return K, A


def sample3D(
    F: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    K: np.ndarray,
    A: np.ndarray,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Sample a 3D field on the (sub)grid

    F = 3D field
    S = depth structure matrix
    X, Y = 1D arrays of horizontal grid coordinates
    Z = 1D array of depth [m, positive downwards]

    Everything in rho-points

    F.shape = S.shape = (kmax, jmax, imax)
    S.shape = (kmax, jmax, imax)
    X.shape = Y.shape = Z.shape = (pmax,)

    # Interpolation = 'bilinear' for trilinear Interpolation
    # = 'nearest' for value in 3D grid cell

    """

    if method == "bilinear":
        # Find rho-point as lower left corner
        I = X.astype("int")
        J = Y.astype("int")
        P = X - I
        Q = Y - J
        W000 = (1 - P) * (1 - Q) * (1 - A)
        W010 = (1 - P) * Q * (1 - A)
        W100 = P * (1 - Q) * (1 - A)
        W110 = P * Q * (1 - A)
        W001 = (1 - P) * (1 - Q) * A
        W011 = (1 - P) * Q * A
        W101 = P * (1 - Q) * A
        W111 = P * Q * A

        return (
            W000 * F[K, J, I]
            + W010 * F[K, J + 1, I]
            + W100 * F[K, J, I + 1]
            + W110 * F[K, J + 1, I + 1]
            + W001 * F[K - 1, J, I]
            + W011 * F[K - 1, J + 1, I]
            + W101 * F[K - 1, J, I + 1]
            + W111 * F[K - 1, J + 1, I + 1]
        )

    # else:  method == 'nearest'
    I = X.round().astype("int")
    J = Y.round().astype("int")
    return F[K, J, I]


def sample3DUV(
    U: np.ndarray,
    V: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    K: np.ndarray,
    A: np.ndarray,
    method="bilinear",
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        sample3D(U, X + 0.5, Y, K, A, method=method),
        sample3D(V, X, Y + 0.5, K, A, method=method),
    )


# --------------------------
# File utility functions
# --------------------------


def find_files(
    file_pattern: Union[Path, str],
    first_file: Union[Path, str, None] = None,
    last_file: Union[Path, str, None] = None,
) -> List[Path]:
    """Find ordered sequence of files following a pattern

    The sequence can be limited by first_file and/or last_file

    """
    directory = Path(file_pattern).parent
    fname = Path(file_pattern).name
    files = sorted(directory.glob(fname))
    if first_file is not None:
        files = [f for f in files if f >= Path(first_file)]
    if last_file is not None:
        files = [f for f in files if f <= Path(last_file)]
    return files


def scan_file_times(files: List[Path]) -> Tuple[np.ndarray, Dict[Path, int]]:
    """Check netcdf files and scan the times

    Returns:
    all_frames: List of all time frames
    num_frames: Mapping: filename -> number of time frames in file

    """
    # print("scan starting")
    all_frames = []  # All time frames
    num_frames = {}  # Number of time frames in each file
    for fname in files:
        with Dataset(fname) as nc:
            new_times = nc.variables["ocean_time"][:]
            num_frames[fname] = len(new_times)
            units = nc.variables["ocean_time"].units
            new_frames = num2date(new_times, units)
            all_frames.extend(new_frames)

    # Check that time frames are strictly sorted
    all_frames = np.array([np.datetime64(tf) for tf in all_frames])
    I: np.ndarray = all_frames[1:] <= all_frames[:-1]
    if np.any(I):
        i = I.nonzero()[0][0] + 1  # Index of first out-of-order frame
        oooframe = str(all_frames[i]).split(".")[0]  # Remove microseconds
        logging.info(f"Time frame {i} = {oooframe} out of order")
        logging.critical("Forcing time frames not strictly sorted")
        raise SystemExit(4)

    logging.info(f"Number of available forcing times = {len(all_frames)}")
    # print("scan finished")
    return all_frames, num_frames


def forcing_steps(
    files: List[Path], timer: TimeKeeper
) -> Tuple[List[int], Dict[int, Path], Dict[int, int]]:
    """Return time step numbers of the forcing and pointers to the data"""

    all_frames, num_frames = scan_file_times(files)

    time0 = all_frames[0]
    time1 = all_frames[-1]
    logging.info(f"First forcing time = {time0}")
    logging.info(f"Last forcing time = {time1}")
    # start_time = self.start_time)
    # stop_time = self.stop_time)
    # dt = np.timedelta64(self.timer.dt, "s")

    # Check that forcing period covers the simulation period
    # ------------------------------------------------------

    if time0 > timer.min_time:
        error_string = "No forcing at minimum time"
        # logging.error(error_string)
        raise SystemExit(error_string)
    if time1 < timer.max_time:
        error_string = "No forcing at maximum time"
        # logging.error(error_string)
        raise SystemExit(error_string)

    # Make a list steps of the forcing time steps
    # --------------------------------------------
    steps = []  # Model time step of forcing
    for t in all_frames:
        steps.append(timer.time2step(t))

    file_at_step = dict()  # mapping step -> file name
    recordnr_at_step = dict()  # mapping step -> record number in file
    step_counter = -1
    # for i, fname in enumerate(files):
    for fname in files:
        for i in range(num_frames[fname]):
            step_counter += 1
            step = steps[step_counter]
            file_at_step[step] = fname
            recordnr_at_step[step] = i
    return steps, file_at_step, recordnr_at_step
