"""
LADiM Grid og Forcing form the Regional Ocean Model System (ROMS)

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# February, 2021
# -----------------------------------

from pathlib import Path
import logging
from typing import Union, Optional, List, Tuple, Dict

import numpy as np  # type: ignore
from netCDF4 import Dataset, num2date  # type: ignore

from ladim2.grid import BaseGrid
from ladim2.forcing import BaseForce
from ladim2.sample import sample2D, bilin_inv
from ladim2.timekeeper import TimeKeeper

DEBUG = False

# ---------------------------
# Grid class
# ---------------------------


def init_grid(**args) -> BaseGrid:
    return Grid(**args)


class Grid(BaseGrid):
    """Simple ROMS grid object

    Possible grid arguments:
      subgrid = [i0, i1, j0, j1]
        Ordinary python style, start points included, not end points
        Each of the elements can be replaced with None, for no limitation
      Vinfo: dictionary with N, hc, theta_s and theta_b

    """

    # Lagrer en del unødige attributter

    def __init__(
        self,
        filename: Union[Path, str],
        subgrid: Optional[Tuple[int, int, int, int]] = None,
        Vinfo=None,
        **args,
    ) -> None:

        print("Grid.__init__")

        # logging.info("Initializing ROMS-type grid object")

        # Grid file
        # if "file" in config:
        #     filename = config["file"]
        # else:
        #     # logging.error("No grid file specified")
        #     print("No grid file specified")
        #     raise SystemExit(1)
        try:
            ncid = Dataset(filename)
        except OSError:
            # logging.error("Could not open grid file " + grid_file)
            print(f"Could not open grid file {filename}")
            raise SystemExit(1)
        ncid.set_auto_maskandscale(False)

        # Subgrid, only considers internal grid cells
        # 1 <= i0 < i1 <= imax-1, default=end points
        # 1 <= j0 < j1 <= jmax-1, default=end points
        # Here, imax, jmax refers to whole grid
        # imax: int
        # jmax: int
        jmax0, imax0 = ncid.variables["h"].shape
        limits = list(subgrid) if subgrid else [1, imax0 - 1, 1, jmax0 - 1]
        # Negative values are counting from right/upper end of model domain
        for i in range(2):
            if limits[i] < 0:
                limits[i] = imax0 + limits[i]
        for i in range(2, 4):
            if limits[i] < 0:
                limits[i] = jmax0 + limits[i]
        # Sanity check
        if (not 1 <= limits[0] < limits[1] <= imax0 - 1) or (
            not 1 <= limits[2] < limits[3] <= jmax0 - 1
        ):
            raise SystemExit("Illegal subgrid specification :", limits)

        self.i0, self.i1, self.j0, self.j1 = limits
        self.imax = self.i1 - self.i0
        self.jmax = self.j1 - self.j0
        # print('Grid : imax, jmax, size = ',
        #     self.imax, self.jmax, self.imax*self.jmax)

        # Limits for where velocities are defined
        self.xmin = float(self.i0)
        self.xmax = float(self.i1 - 1)
        self.ymin = float(self.j0)
        self.ymax = float(self.j1 - 1)

        # Slices
        #   rho-points
        self.I = slice(self.i0, self.i1)
        self.J = slice(self.j0, self.j1)
        #   U and V-points
        self.Iu = slice(self.i0 - 1, self.i1)
        self.Ju = self.J
        self.Iv = self.I
        self.Jv = slice(self.j0 - 1, self.j1)

        # Vertical grid

        if Vinfo is not None:
            self.N = Vinfo["N"]
            self.hc = Vinfo["hc"]
            self.Vstretching = Vinfo.get("Vstretching", 1)
            self.Vtransform = Vinfo.get("Vtransform", 1)
            self.Cs_r = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="rho",
                Vstretching=self.Vstretching,
            )
            self.Cs_w = s_stretch(
                self.N,
                Vinfo["theta_s"],
                Vinfo["theta_b"],
                stagger="w",
                Vstretching=self.Vstretching,
            )

        else:  # Verical info from the grid file
            # Burde ha en test om mangler vertikal informasjon
            self.hc = ncid.variables["hc"].getValue()
            self.Cs_r = ncid.variables["Cs_r"][:]
            self.Cs_w = ncid.variables["Cs_w"][:]
            self.N = len(self.Cs_r)
            # Vertical transform
            try:
                self.Vtransform = ncid.variables["Vtransform"].getValue()
            except KeyError:
                self.Vtransform = 1  # Default = old way

        # Read some variables
        self.H = ncid.variables["h"][self.J, self.I]
        self.M = ncid.variables["mask_rho"][self.J, self.I].astype(int)
        # self.Mu = ncid.variables['mask_u'][self.Ju, self.Iu]
        # self.Mv = ncid.variables['mask_v'][self.Jv, self.Iv]
        self.dx = 1.0 / ncid.variables["pm"][self.J, self.I]
        self.dy = 1.0 / ncid.variables["pn"][self.J, self.I]
        self.lon = ncid.variables["lon_rho"][self.J, self.I]
        self.lat = ncid.variables["lat_rho"][self.J, self.I]
        self.angle = ncid.variables["angle"][self.J, self.I]

        self.z_r = sdepth(
            self.H, self.hc, self.Cs_r, stagger="rho", Vtransform=self.Vtransform
        )
        self.z_w = sdepth(
            self.H, self.hc, self.Cs_w, stagger="w", Vtransform=self.Vtransform
        )

        # Land masks at u- and v-points
        M = self.M
        Mu = np.zeros((self.jmax, self.imax + 1), dtype=int)
        Mu[:, 1:-1] = M[:, :-1] * M[:, 1:]
        Mu[:, 0] = M[:, 0]
        Mu[:, -1] = M[:, -1]
        self.Mu = Mu

        Mv = np.zeros((self.jmax + 1, self.imax), dtype=int)
        Mv[1:-1, :] = M[:-1, :] * M[1:, :]
        Mv[0, :] = M[0, :]
        Mv[-1, :] = M[-1, :]
        self.Mv = Mv

        # Close the file(s)
        ncid.close()

    def metric(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample the metric coefficients

        Changes slowly, so using nearest neighbour
        """
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0

        # Metric is conform for PolarStereographic
        A = self.dx[J, I]
        return A, A

    def depth(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the depth of grid cells"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0
        return self.H[J, I]

    def lonlat(
        self, X: np.ndarray, Y: np.ndarray, method: str = "bilinear"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the longitude and latitude from grid coordinates"""
        if method == "bilinear":  # More accurate
            return self.xy2ll(X, Y)
        # else: containing grid cell, less accurate
        I = X.round().astype("int") - self.i0
        J = Y.round().astype("int") - self.j0
        return self.lon[J, I], self.lat[J, I]

    def ingrid(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Returns True for points inside the subgrid"""
        return (
            (self.xmin + 0.5 < X)
            & (X < self.xmax - 0.5)
            & (self.ymin + 0.5 < Y)
            & (Y < self.ymax - 0.5)
        )

    def onland(self, X, Y):
        """Returns True for points on land"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0
        return self.M[J, I] < 1

    # Error if point outside
    def atsea(self, X, Y):
        """Returns True for points at sea"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0
        return self.M[J, I] > 0

    def xy2ll(self, X, Y):
        return (
            sample2D(self.lon, X - self.i0, Y - self.j0),
            sample2D(self.lat, X - self.i0, Y - self.j0),
        )

    def ll2xy(self, lon, lat):
        Y, X = bilin_inv(lon, lat, self.lon, self.lat)
        return X + self.i0, Y + self.j0


# ------------------------------------------------
# ROMS utility functions, from the ROPPY package
# ------------------------------------------------


def s_stretch(N, theta_s, theta_b, stagger="rho", Vstretching=1):
    """Compute a s-level stretching array

    *N* : Number of vertical levels

    *theta_s* : Surface stretching factor

    *theta_b* : Bottom stretching factor

    *stagger* : "rho"|"w"

    *Vstretching* : 1|2|4

    """

    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N + 1)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vstretching == 1:
        cff1 = 1.0 / np.sinh(theta_s)
        cff2 = 0.5 / np.tanh(0.5 * theta_s)
        return (1.0 - theta_b) * cff1 * np.sinh(theta_s * S) + theta_b * (
            cff2 * np.tanh(theta_s * (S + 0.5)) - 0.5
        )

    if Vstretching == 2:
        a, b = 1.0, 1.0
        Csur = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        Cbot = np.sinh(theta_b * (S + 1)) / np.sinh(theta_b) - 1
        mu = (S + 1) ** a * (1 + (a / b) * (1 - (S + 1) ** b))
        return mu * Csur + (1 - mu) * Cbot

    if Vstretching == 4:
        C = (1 - np.cosh(theta_s * S)) / (np.cosh(theta_s) - 1)
        C = (np.exp(theta_b * C) - 1) / (1 - np.exp(-theta_b))
        return C

    # else:
    raise ValueError("Unknown Vstretching")


def sdepth(H, Hc, C, stagger="rho", Vtransform=1):
    """Return depth of rho-points in s-levels

    *H* : arraylike
      Bottom depths [meter, positive]

    *Hc* : scalar
       Critical depth

    *cs_r* : 1D array
       s-level stretching curve

    *stagger* : [ 'rho' | 'w' ]

    *Vtransform* : [ 1 | 2 ]
       defines the transform used, defaults 1 = Song-Haidvogel

    Returns an array with ndim = H.ndim + 1 and
    shape = cs_r.shape + H.shape with the depths of the
    mid-points in the s-levels.

    Typical usage::

    >>> fid = Dataset(roms_file)
    >>> H = fid.variables['h'][:, :]
    >>> C = fid.variables['Cs_r'][:]
    >>> Hc = fid.variables['hc'].getValue()
    >>> z_rho = sdepth(H, Hc, C)

    """
    H = np.asarray(H)
    Hshape = H.shape  # Save the shape of H
    H = H.ravel()  # and make H 1D for easy shape maniplation
    C = np.asarray(C)
    N = len(C)
    outshape = (N,) + Hshape  # Shape of output
    if stagger == "rho":
        S = -1.0 + (0.5 + np.arange(N)) / N  # Unstretched coordinates
    elif stagger == "w":
        S = np.linspace(-1.0, 0.0, N)
    else:
        raise ValueError("stagger must be 'rho' or 'w'")

    if Vtransform == 1:  # Default transform by Song and Haidvogel
        A = Hc * (S - C)[:, None]
        B = np.outer(C, H)
        return (A + B).reshape(outshape)

    if Vtransform == 2:  # New transform by Shchepetkin
        N = Hc * S[:, None] + np.outer(C, H)
        D = 1.0 + Hc / H
        return (N / D).reshape(outshape)

    # else:
    raise ValueError("Unknown Vtransform")


# -----------------------------------
# Force
# -----------------------------------


def init_force(**args) -> BaseForce:
    return Forcing(**args)


class Forcing(BaseForce):
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
        # self.timer = timer

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
        steps, file_idx, frame_idx = forcing_steps(files, timer)
        self.stepdiff = np.diff(steps)
        self.file_idx = file_idx
        self.frame_idx = frame_idx
        self._first_read = True  # True until first file is opened
        # self._nc = None  # Not opened yet

        # Read old input
        # requires at least one input before start
        # to get Runge-Kutta going
        # --------------
        # prestep = last forcing step < 0
        #

        print("steps = ", steps)
        V = [step for step in steps if step < 0]
        prestep = max(V) if V else 0
        # if V:  # Forcing available before start time
        if True:
            # prestep = max(V)
            print("prestep", prestep)

            i = steps.index(prestep)
            if timer.time_reversal:
                i = i - 1
            stepdiff0 = self.stepdiff[i]
            nextstep = prestep + stepdiff0
            # print("stediff, nextstep = ", stepdiff0, nextstep)
            if timer.time_reversal:
                nextstep = prestep - stepdiff0
            self.fields["u"], self.fields["v"] = self._read_velocity(prestep)
            self.fields["u_new"], self.fields["v_new"] = self._read_velocity(nextstep)
            self.fields["dU"] = (self.fields["u_new"] - self.fields["u"]) / stepdiff0
            self.fields["dV"] = (self.fields["v_new"] - self.fields["v"]) / stepdiff0
            # Interpolate to time step = -1
            # Skal virke i revers også
            self.fields["u"] = self.fields["u"] - (prestep + 1) * self.fields["dU"]
            self.fields["v"] = self.fields["v"] - (prestep + 1) * self.fields["dV"]
            # Other forcing
            for name in self.ibm_forcing:
                self.fields[name] = self._read_field(name, prestep)
                # self[name + "new"] = self._read_field(name, nextstep)
                # self["d" + name] = (self[name + "new"] - self[name]) / prestep
                # self[name] = self[name] - (prestep + 1) * self["d" + name]

        # elif steps[0] == 0:
        #     # Simulation start at first forcing time
        #     # Runge-Kutta needs dU and dV in this case as well
        #     self.fields["u"], self.fields["v"] = self._read_velocity(0)
        #     self.fields["u_new"], self.fields["v_new"] = self._read_velocity(steps[1])
        #     self.fields["dU"] = (self.fields["u_new"] - self.fields["u"]) / steps[1]
        #     self.fields["dV"] = (self.fields["v_new"] - self.fields["v"]) / steps[1]
        #     # Synchronize with start time
        #     self.fields["u_new"] = self.fields["u"]
        #     self.fields["v_new"] = self.fields["v"]
        #     # Extrapolate to time step = -1
        #     self.fields["u"] = self.fields["u"] - self.fields["dU"]
        #     self.fields["v"] = self.fields["v"] - self.fields["dV"]
        #     # Other forcing:
        #     for name in self.ibm_forcing:
        #         self[name] = self._read_field(name, 0)
        #     #     self[name + "new"] = self._read_field(name, steps[1])
        #     #     self["d" + name] = (self[name + "new"] - self[name]) / steps[1]
        #     #     self[name] = self[name] - self["d" + name]

        else:
            # No forcing at start, should already be excluded
            raise SystemExit(3)

        self.steps = steps
        # self.files = files

        # print("Init: finished")

    # ===================================================

    # Turned off time interpolation of scalar fields
    # TODO: Implement a switch for turning it on again if wanted
    def update(
        self, step: int, X: float, Y: float, Z: float
    ) -> None:
        """Update the fields to given time step t"""

        # Read from config?
        interpolate_velocity_in_time = True
        # interpolate_ibm_forcing_in_time = False

        logging.debug("Updating forcing, time step = {}".format(step))
        if step in self.steps:  # No time interpolation
            self.fields["u"] = self.fields["u_new"]
            self.fields["v"] = self.fields["v_new"]
            # for name in self.ibm_forcing:
            #   self[name] = self[name + "new"]
        else:
            if step - 1 in self.steps:  # Need new fields
                i = self.steps.index(step - 1)
                if self.time_reversal:
                    i = i - 1
                # print("i =", i)
                stepdiff = self.stepdiff[i]
                nextstep = step - 1 + stepdiff
                if self.time_reversal:
                    nextstep = step - 1 - stepdiff
                # print("stediff, nextstep = ", stepdiff, nextstep)
                self.fields["u_new"], self.fields["v_new"] = self._read_velocity(
                    nextstep
                )
                # for name in self.ibm_forcing:
                #    self[name + "new"] = self._read_field(name, nextstep)
                if interpolate_velocity_in_time:
                    self.fields["dU"] = (
                        self.fields["u_new"] - self.fields["u"]
                    ) / stepdiff
                    self.fields["dV"] = (
                        self.fields["v_new"] - self.fields["v"]
                    ) / stepdiff
                # if interpolate_ibm_forcing_in_time:
                #    for name in self.ibm_forcing:
                #        self["d" + name] = (self[name + "new"] - self[name]) / stepdiff

            # "Ordinary" time step (including self.steps+1)
            if interpolate_velocity_in_time:
                self.fields["u"] += self.fields["dU"]
                self.fields["v"] += self.fields["dV"]
            # if interpolate_ibm_forcing_in_time:
            #    for name in self.ibm_forcing:
            #        self[name] += self["d" + name]

        # Update forcing values at particles
        self.force_particles(X, Y, Z)

    # ==============================================

    def open_forcing_file(self, time_step: int) -> None:

        """Open forcing file and get scaling info given time step"""
        # Open the correct forcing file
        nc = Dataset(self.file_idx[time_step])
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
        elif self.frame_idx[time_step] == 0:  # Open next file
            self._nc.close()
            self.open_forcing_file(time_step)

        frame = self.frame_idx[time_step]

        if DEBUG:
            print("_read_velocity")
            print("   model time step =", time_step)
            timevar = self._nc.variables["ocean_time"]
            time_origin = np.datetime64(timevar.units.split("since")[1])
            data_time = time_origin + np.timedelta64(int(timevar[frame]), "s")
            print("   data file:   ", self.file_idx[time_step])
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
        else:
            return sample3DUV(U, V, X - i0, Y - j0, K, A, method=method)

    # Simplify to grid cell
    # def field(
    #     self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, name: str
    # ) -> np.ndarray:
    #     # should not be necessary to repeat
    #     i0 = self.grid.i0
    #     j0 = self.grid.j0
    #     K, A = z2s(self.grid.z_r, X - i0, Y - j0, Z)
    #     F = self[name]
    #     return sample3D(F, X - i0, Y - j0, K, A, method="nearest")
    def field(self, X, Y, Z, name):
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

    file_idx = dict()  # mapping step -> file name
    frame_idx = dict()  # mapping step -> record number in file
    step_counter = -1
    # for i, fname in enumerate(files):
    for fname in files:
        for i in range(num_frames[fname]):
            step_counter += 1
            step = steps[step_counter]
            file_idx[step] = fname
            frame_idx[step] = i
    return steps, file_idx, frame_idx
