"""
Grid and Forcing for LADiM for the Regional Ocean Model System (ROMS)

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2017-03-01
# -----------------------------------

from pathlib import Path
import logging
from typing import Union, Optional, List

import numpy as np
from netCDF4 import Dataset, num2date

from ladim2.grid import Grid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import Forcing


def init_forcing(**args) -> Forcing:
    return Forcing_ROMS(**args)


class Forcing_ROMS(Forcing):
    """
    Class for ROMS forcing

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

        self.grid = grid  # Get the grid objec.
        self.ibm_forcing = ibm_forcing if ibm_forcing else []

        self.timer = timer

        files = self.find_files(filename)
        # print("files = ", files)
        numfiles = len(files)
        if numfiles == 0:
            logging.error("No input file: {}".format(filename))
            raise SystemExit(3)
        logging.info("Number of forcing files = {}".format(numfiles))

        # self.start_time = .start_time)
        # self.stop_time = np.datetime64(t.stop_time)

        # ---------------------------
        # Overview of all the files
        # ---------------------------

        all_frames, num_frames = self.scan_file_times(files)

        self.files = files

        steps, file_idx, frame_idx = self.forcing_steps(all_frames, num_frames)

        # print("all_frames = ", all_frames)
        # print("num_frames = ", num_frames)

        # self.stepdiff = stepdiff
        self.stepdiff = np.diff(steps)
        self.file_idx = file_idx
        self.frame_idx = frame_idx
        self._nc = None

        # Read old input
        # requires at least one input before start
        # to get Runge-Kutta going
        # --------------
        # prestep = last forcing step < 0
        #

        V = [step for step in steps if step < 0]
        if V:  # Forcing available before start time
            prestep = max(V)
            stepdiff = self.stepdiff[steps.index(prestep)]
            nextstep = prestep + stepdiff
            self.U, self.V = self._read_velocity(prestep)
            self.Unew, self.Vnew = self._read_velocity(nextstep)
            self.dU = (self.Unew - self.U) / stepdiff
            self.dV = (self.Vnew - self.V) / stepdiff
            # Interpolate to time step = -1
            self.U = self.U - (prestep + 1) * self.dU
            self.V = self.V - (prestep + 1) * self.dV
            # Other forcing
            for name in self.ibm_forcing:
                self[name] = self._read_field(name, prestep)
                self[name + "new"] = self._read_field(name, nextstep)
                self["d" + name] = (self[name + "new"] - self[name]) / prestep
                self[name] = self[name] - (prestep + 1) * self["d" + name]

        elif steps[0] == 0:
            # Simulation start at first forcing time
            # Runge-Kutta needs dU and dV in this case as well
            self.U, self.V = self._read_velocity(0)
            self.Unew, self.Vnew = self._read_velocity(steps[1])
            self.dU = (self.Unew - self.U) / steps[1]
            self.dV = (self.Vnew - self.V) / steps[1]
            # Synchronize with start time
            self.Unew = self.U
            self.Vnew = self.V
            # Extrapolate to time step = -1
            self.U = self.U - self.dU
            self.V = self.V - self.dV
            # Other forcing:
            # for name in self.ibm_forcing:
            #     self[name] = self._read_field(name, 0)
            #     self[name + "new"] = self._read_field(name, steps[1])
            #     self["d" + name] = (self[name + "new"] - self[name]) / steps[1]
            #     self[name] = self[name] - self["d" + name]

        else:
            # No forcing at start, should already be excluded
            raise SystemExit(3)

        self.steps = steps
        # self.files = files

        # print("Init: finished")

    # ===================================================
    @staticmethod
    def find_files(
        input_file: Union[Path, str],
        first_file: Union[Path, str, None] = None,
        last_file: Union[Path, str, None] = None,
    ) -> List[Path]:
        # def find_files(input_file: Union[Path, str], **args) -> List[Path]:
        """Find (and sort) the forcing file(s)"""
        datadir = Path(input_file).parent
        fname = Path(input_file).name
        files = sorted(datadir.glob(fname))
        # ffile = args.get("first_file", None)
        if first_file is not None:
            files = [f for f in files if f >= Path(first_file)]
        # lfile = args.get("last_file", None)
        if last_file is not None:
            files = [f for f in files if f <= Path(last_file)]
        return files

    @staticmethod
    def scan_file_times(files):
        """Check files and scan the times

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
        # all_frames = np.array(all_frames, dtype=np.datetime64)
        all_frames = np.array([np.datetime64(tf) for tf in all_frames])
        I = all_frames[1:] <= all_frames[:-1]
        if np.any(I):
            i = I.nonzero()[0][0] + 1  # Index of first out-of-order frame
            oooframe = str(all_frames[i]).split(".")[0]  # Remove microseconds
            logging.info(f"Time frame {i} = {oooframe} out of order")
            logging.critical("Forcing time frames not strictly sorted")
            raise SystemExit(4)

        logging.info(f"Number of available forcing times = {len(all_frames)}")
        # print("scan finished")
        return all_frames, num_frames

    # -----------------------------------------

    def forcing_steps(self, all_frames, num_frames):

        time0 = all_frames[0]
        time1 = all_frames[-1]
        logging.info(f"First forcing time = {time0}")
        logging.info(f"Last forcing time = {time1}")
        # start_time = self.start_time)
        # stop_time = self.stop_time)
        dt = np.timedelta64(self.timer.dt, "s")

        # Check that forcing period covers the simulation period
        # ------------------------------------------------------

        if time0 > self.timer.start_time:
            logging.error("No forcing at start time")
            raise SystemExit(3)
        if time1 < self.timer.stop_time:
            logging.error("No forcing at stop time")
            raise SystemExit(3)

        # Make a list steps of the forcing time steps
        # --------------------------------------------
        steps = []  # Model time step of forcing
        for t in all_frames:
            # dtime = np.timedelta64(t - self.start_time, "s")
            # steps.append(int(dtime / dt))
            steps.append(self.timer.time2step(t))

        file_idx = dict()  # Dårlig navn
        frame_idx = dict()
        step_counter = -1
        # for i, fname in enumerate(files):
        for fname in self.files:
            for i in range(num_frames[fname]):
                step_counter += 1
                step = steps[step_counter]
                file_idx[step] = fname
                frame_idx[step] = i
        return steps, file_idx, frame_idx

    # ==============================================

    # Turned off time interpolation of scalar fields
    # TODO: Implement a switch for turning it on again if wanted
    def update(self, t):
        """Update the fields to time step t"""

        # Read from config?
        interpolate_velocity_in_time = True
        interpolate_ibm_forcing_in_time = False

        logging.debug("Updating forcing, time step = {}".format(t))
        if t in self.steps:  # No time interpolation
            self.U = self.Unew
            self.V = self.Vnew
            # for name in self.ibm_forcing:
            #   self[name] = self[name + "new"]
        else:
            if t - 1 in self.steps:  # Need new fields
                stepdiff = self.stepdiff[self.steps.index(t - 1)]
                nextstep = t - 1 + stepdiff
                self.Unew, self.Vnew = self._read_velocity(nextstep)
                # for name in self.ibm_forcing:
                #    self[name + "new"] = self._read_field(name, nextstep)
                if interpolate_velocity_in_time:
                    self.dU = (self.Unew - self.U) / stepdiff
                    self.dV = (self.Vnew - self.V) / stepdiff
                # if interpolate_ibm_forcing_in_time:
                #    for name in self.ibm_forcing:
                #        self["d" + name] = (self[name + "new"] - self[name]) / stepdiff

            # "Ordinary" time step (including self.steps+1)
            if interpolate_velocity_in_time:
                self.U += self.dU
                self.V += self.dV
            # if interpolate_ibm_forcing_in_time:
            #    for name in self.ibm_forcing:
            #        self[name] += self["d" + name]

    # --------------

    def open_forcing_file(self, n):
        """Open forcing file at time step = n"""
        nc = self._nc
        nc = Dataset(self.file_idx[n])
        nc.set_auto_maskandscale(False)

        self.scaled = dict()
        self.scale_factor = dict()
        self.add_offset = dict()

        # Åpne for alias til navn
        forcing_variables = ["u", "v"] + self.ibm_forcing
        # forcing_variables = ["u", "v"]
        for key in forcing_variables:
            if hasattr(nc.variables[key], "scale_factor"):
                self.scaled[key] = True
                self.scale_factor[key] = np.float32(nc.variables[key].scale_factor)
                self.add_offset[key] = np.float32(nc.variables[key].add_offset)
            else:
                self.scaled[key] = False

        self._nc = nc

    def _read_velocity(self, n):
        """Read fields at time step = n"""
        # Need a switch for reading W
        # T = self._nc.variables['ocean_time'][n]  # Read new fields

        # Handle file opening/closing
        # Always read velocity before other fields
        logging.info("Reading velocity for time step = {}".format(n))

        # If finished a file or first read (self._nc == "")
        if not self._nc:  # First read
            self.open_forcing_file(n)
        elif self.frame_idx[n] == 0:  # Just finished a forcing file
            self._nc.close()
            self.open_forcing_file(n)

        frame = self.frame_idx[n]

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
        frame = self.frame_idx[n]
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

    def close(self):

        self._nc.close()

    def velocity(self, X, Y, Z, tstep=0, method="bilinear"):

        i0 = self.grid.i0
        j0 = self.grid.j0
        K, A = z2s(self.grid.z_r, X - i0, Y - j0, Z)
        if tstep < 0.001:
            U = self.U
            V = self.V
        else:
            U = self.U + tstep * self.dU
            V = self.V + tstep * self.dV
        return sample3DUV(U, V, X - i0, Y - j0, K, A, method=method)

    # Simplify to grid cell
    def field(self, X, Y, Z, name):
        # should not be necessary to repeat
        i0 = self.grid.i0
        j0 = self.grid.j0
        K, A = z2s(self.grid.z_r, X - i0, Y - j0, Z)
        F = self[name]
        return sample3D(F, X - i0, Y - j0, K, A, method="nearest")


# ------------------------
#   Sampling routines
# ------------------------


def z2s(z_rho, X, Y, Z):
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


def sample3D(F, X, Y, K, A, method="bilinear"):
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


def sample3DUV(U, V, X, Y, K, A, method="bilinear"):
    return (
        sample3D(U, X + 0.5, Y, K, A, method=method),
        sample3D(V, X, Y + 0.5, K, A, method=method),
    )
