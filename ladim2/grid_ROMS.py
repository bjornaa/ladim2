"""
LADiM Grid class for the Regional Ocean Model System (ROMS)

"""

# -----------------------------------
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
# Bergen, Norway
# 2020-04-01
# -----------------------------------

# import sys
import glob

# import logging
import numpy as np
from netCDF4 import Dataset

from ladim2.sample import sample2D, bilin_inv


class Grid:
    """Simple ROMS grid object

    Possible grid arguments:
      subgrid = [i0, i1, j0, j1]
        Ordinary python style, start points included, not end points
        Each of the elements can be replaced with None, for no limitation
      Vinfo: dictionary with N, hc, theta_s and theta_b

    """

    # Lagrer en del unødige attributter

    def __init__(self, **kwargs):

        config = kwargs

        # logging.info("Initializing ROMS-type grid object")

        # Grid file
        if "grid_file" in config:
            grid_file = config["grid_file"]
        else:
            # logging.error("No grid file specified")
            raise SystemExit(1)

        try:
            ncid = Dataset(grid_file)
        except OSError:
            # logging.error("Could not open grid file " + grid_file)
            raise SystemExit(1)
        ncid.set_auto_maskandscale(False)

        # Subgrid, only considers internal grid cells
        # 1 <= i0 < i1 <= imax-1, default=end points
        # 1 <= j0 < j1 <= jmax-1, default=end points
        # Here, imax, jmax refers to whole grid
        jmax, imax = ncid.variables["h"].shape
        whole_grid = [1, imax - 1, 1, jmax - 1]
        if "subgrid" in config:
            limits = list(config["subgrid"])
        else:
            limits = whole_grid

        if limits[0] < 1 or limits[0] > limits[1] or limits[1] > imax - 1:
            raise SystemExit("Illegal subgrid specification")
        if limits[2] < 1 or limits[2] > limits[3] or limits[3] > jmax - 1:
            raise SystemExit("Illegal subgrid specification")
        # Allow None if no imposed limitation
        for ind, val in enumerate(limits):
            if val is None:
                limits[ind] = whole_grid[ind]

        self.i0, self.i1, self.j0, self.j1 = limits
        self.imax = self.i1 - self.i0
        self.jmax = self.j1 - self.j0
        # print('Grid : imax, jmax, size = ',
        #      self.imax, self.jmax, self.imax*self.jmax)

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

        if "Vinfo" in config:
            Vinfo = config["Vinfo"]
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

    def metric(self, X, Y):
        """Sample the metric coefficients

        Changes slowly, so using nearest neighbour
        """
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0

        # Metric is conform for PolarStereographic
        A = self.dx[J, I]
        return A, A

    def depth(self, X, Y):
        """Return the depth of grid cells"""
        I = X.round().astype(int) - self.i0
        J = Y.round().astype(int) - self.j0
        return self.H[J, I]

    def lonlat(self, X, Y, method="bilinear"):
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
