"""Module for plotting curly vectors with LADiM"""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import ladim.state
import ladim.tracker
import ladim.ROMS


class Curly:
    """Class for curly vectors"""

    def __init__(
        self,
        data_file: str,
        rec: int,
        depth: float = 0.0,
        subgrid: Optional[Tuple[int, int, int, int]] = None,
        dt: Optional[int] = None,
        num_steps: int = 50,
        density: Optional[int] = None,
    ):

        # Define grid
        grid = ladim.ROMS.Grid(filename=data_file, subgrid=subgrid)
        self.grid = grid

        if density is None:
            # Value 60 determined by trial and error
            density = 1 + (grid.i1 - grid.i0 + grid.j1 - grid.j0) // 60
        self.density = density

        dx = grid.dx.mean()
        if dt is None:
            # Value 0.5 by trial and error
            dt = int(0.5 * dx)
        self.dt = dt

        self.num_steps = num_steps

        # Extract forcing
        with Dataset(data_file) as nc:
            U3D = nc.variables["u"][rec, :, grid.Ju, grid.Iu]
            V3D = nc.variables["v"][rec, :, grid.Jv, grid.Iv]
        U2D, V2D = zslice(grid, U3D, V3D, depth)
        # force = FixedForcing(grid, U2D, V2D)

        state = ladim.state.State()
        tracker = ladim.tracker.Tracker(
            advection="RK2",
            modules=dict(
                state=state,
                time=FixedTimeKeeper(dt, num_steps),
                grid=grid,
                forcing=FixedForcing(grid, U2D, V2D),
            ),
        )

        # Initial particle distribution in a rectangular grid
        X0, Y0 = np.meshgrid(
            # np.arange(i0 + density // 2, i1 - density // 2, density),
            # np.arange(j0 + density // 2, j1 - density // 2, density),
            np.arange(i0 + density // 2, i1 - 1, density),
            np.arange(j0 + density // 2, j1 - 1, density),
        )
        I0 = X0.round().astype(int) - i0
        J0 = Y0.round().astype(int) - j0
        atsea = grid.M[J0, I0] > 0
        X0 = X0[atsea]
        Y0 = Y0[atsea]

        # Initiate the LADiM state
        state.append(X=X0, Y=Y0)
        num_particles = len(state)
        # Predefine trajectory storage
        X = np.zeros((num_steps + 1, num_particles))
        Y = np.zeros((num_steps + 1, num_particles))
        X[0, :] = X0
        Y[0, :] = Y0
        # "Time" loop
        for step in range(1, num_steps + 1):
            tracker.update()
            X[step, :] = state.X
            Y[step, :] = state.Y
        self.X, self.Y = X, Y

    def trajectories(self):
        """The trajectories behind the curly vectors"""
        return self.X, self.Y

    def plot(self, color="red", width=0.5):
        """Plot the curly vectors with matplotlib"""
        # Plot trajectories
        plt.plot(self.X, self.Y, color=color, linewidth=width)
        # Mark start position
        plt.plot(self.X[0], self.Y[0], ".", color="black", markersize=1)


def zslice(
    grid: ladim.ROMS.Grid, U3D: np.ndarray, V3D: np.ndarray, Z: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal slice of 3D ROMS vector field, return fields are non-staggered"""
    i0, i1, j0, j1 = grid.i0, grid.i1, grid.j0, grid.j1
    Xc, Yc = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))  # Grid cell centers
    Xc = Xc.ravel()
    Yc = Yc.ravel()
    Z = np.zeros_like(Xc) + Z  # Make a 1D array
    K, A = ladim.ROMS.z2s(grid.z_r, Xc - grid.i0, Yc - grid.j0, Z)
    U2D, V2D = ladim.ROMS.sample3DUV(
        U3D, V3D, Xc - i0, Yc - j0, K, A, method="bilinear",
    )
    return U2D.reshape(j1 - j0, i1 - i0), V2D.reshape(j1 - j0, i1 - i0)


class FixedForcing:
    """Fixed forcing from non-staggered U and V fields"""

    def __init__(self, grid: ladim.ROMS.Grid, U2D: np.ndarray, V2D: np.ndarray):
        self.grid = grid
        self.U2D = U2D
        self.V2D = V2D

    def velocity(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, fractional_step: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample velocity at the particle positions

        Z and fractional_step are ignored, dummy input from tracker
        """
        i0, j0 = self.grid.i0, self.grid.j0
        U = ladim.ROMS.sample2D(self.U2D, X - i0, Y - j0)
        V = ladim.ROMS.sample2D(self.V2D, X - i0, Y - j0)
        return U, V


class FixedTimeKeeper:
    """A minimal TimeKeeper class for LADiM"""

    def __init__(self, dt: int, num_steps: int):
        self.dt = np.timedelta64(dt, "s")
        self.Nsteps = num_steps


if __name__ == "__main__":

    # --------------
    # Settings
    # --------------

    # ROMS file, record number and depth
    data_file = "../data/ocean_avg_0014.nc"
    rec = 8
    depth = 10

    # Subgrid
    i0, i1, j0, j1 = 110, 170, 40, 90
    # i0, i1, j0, j1 = 20, 170, 20, 170
    i0, i1, j0, j1 = 130, 158, 53, 70

    # Curly vector settings
    # dt = 3600  # seconds
    # num_steps = 50
    # density = 3

    c = Curly(data_file, rec, depth, subgrid=(i0, i1, j0, j1))

    # -----------
    # Plotting
    # -----------

    # Cell boundaries
    Xcb = np.arange(i0 - 0.5, i1)
    Ycb = np.arange(j0 - 0.5, j1)

    # Landmask
    constmap = plt.matplotlib.colors.ListedColormap([0.2, 0.6, 0.4])
    M = np.where(c.grid.M < 1, 0, np.nan)
    plt.pcolormesh(Xcb, Ycb, M, cmap=constmap)

    print("density = ", c.density)
    print("num_steps = ", c.num_steps)
    print("dt = ", c.dt)

    c.plot(color="blue")

# Trajectories with start points

plt.axis("image")
plt.show()
