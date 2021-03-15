# ------------------------------------
# tracker.py
# Part of the LADiM Model
#
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
#
# Licensed under the MIT license
# ------------------------------------

# import logging
from typing import Tuple

import numpy as np  # type:ignore

from .state import State
from .timekeeper import normalize_period
from .forcing import BaseForce
from .grid import BaseGrid


Velocity = Tuple[np.ndarray, np.ndarray]


class Tracker:
    """The physical particle tracking kernel"""

    # logging.info("Initiating the particle tracking")

    def __init__(
        self, dt, advection, diffusion=0.0, vertdiff=0.0, vertical_advection=False
    ):

        print("Tracker.__init__")

        self.dt = normalize_period(dt).astype("int")  # integer, unit = seconds
        self.advection = advection  # Name of advection method

        # advect <- requested advection method
        # advection = string "EF", "RK2", "RK4"
        # advect = the actual method
        if self.advection:
            self.advect = getattr(self, self.advection)

        self.vertical_advection = vertical_advection

        self.diffusion = bool(diffusion)
        self.D = diffusion
        self.vertdiff = bool(vertdiff)
        self.Dz = vertdiff

    def update(self, state: State, grid: BaseGrid, force: BaseForce) -> None:
        """Move the particles"""

        X, Y, Z = state.X, state.Y, state.Z

        self.dx, self.dy = grid.metric(X, Y)
        # dt = self.dt
        # self.num_particles = len(X)
        # Make more elegant, need not do every time
        # Works for C-grid
        self.xmin = grid.xmin + 0.01
        self.xmax = grid.xmax - 0.01
        self.ymin = grid.ymin + 0.01
        self.ymax = grid.ymax - 0.01

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        # --- Advection ---
        if self.advection:
            Uadv, Vadv = self.advect(X, Y, Z, force)
            U += Uadv
            V += Vadv

        # --- Diffusion ---
        if self.diffusion:
            Udiff, Vdiff = self.diffuse(num_particles=len(X))
            U += Udiff
            V += Vdiff

        # --- Move the particles

        # New position, if OK
        X1 = X + U * self.dt / self.dx
        Y1 = Y + V * self.dt / self.dy

        # Kill particles trying to move out of the grid
        out_of_grid = ~grid.ingrid(X1, Y1)
        state.alive[out_of_grid] = False
        state.active[out_of_grid] = False  # Not necessary if they are removed

        # Do not move inactive particles
        inactive = ~state.active
        X1[inactive] = X[inactive]
        Y1[inactive] = Y[inactive]

        # Land, boundary treatment. Do not move the particles onto land
        # Consider a sequence of different actions
        onland = ~grid.atsea(X1, Y1)
        X1[onland] = X[onland]
        Y1[onland] = Y[onland]

        # Update the particle positions
        state["X"] = X1
        state["Y"] = Y1

        # --- Vertical movement ---

        # Sample the depth level
        h = None
        if self.vertdiff or self.vertical_advection:
            if hasattr(grid, "sample_depth") and callable(grid.sample_depth):
                h = grid.sample_depth(X, Y)
            elif hasattr(grid, "depth") and callable(grid.depth):
                h = grid.depth(X, Y)

            # Diffusion
            if self.vertdiff:
                W = self.diffuse_vert(num_particles=len(X))
                Z += W * self.dt

            # Advection
            if self.vertical_advection:
                W = force.variables["w"]
                Z += W * self.dt

            # Reflexive boundary conditions at surface
            Z[Z < 0] *= -1

            # Reflexive boundary conditions at bottom
            if h is not None:
                below_seabed = Z > h
                Z[below_seabed] = 2 * h[below_seabed] - Z[below_seabed]

            # Update particle positions
            state["Z"] = Z

    def EF(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:
        """Euler-Forward advective velocity"""

        # dt = self.dt
        # pm, pn = grid.sample_metric(X, Y)

        U, V = force.velocity(X, Y, Z)

        return U, V

    # def RK2a(self, forcing: Forcing, state: State) -> Velocity:
    #     """Runge-Kutta second order = Heun scheme"""

    #     X, Y, Z = state["X"], state["Y"], state["Z"]
    #     dt = self.dt

    #     U, V = forcing.velocity(X, Y, Z)
    #     X1 = X + 0.5 * U * dt / self.dx
    #     Y1 = Y + 0.5 * V * dt / self.dy

    #     U, V = forcing.velocity(X1, Y1, Z, tstep=0.5)
    #     return U, V

    def RK2b(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:

        # def RK2b(self, forcing: Forcing, state: State) -> Velocity:
        """Runge-Kutta second order = Heun scheme

        This version does not sample velocities outside the grid
        """

        # X, Y, Z = state["X"], state["Y"], state["Z"]
        dt = self.dt

        U, V = force.velocity(X, Y, Z)
        X1 = X + 0.5 * U * dt / self.dx
        Y1 = Y + 0.5 * V * dt / self.dy
        X1.clip(self.xmin, self.xmax, out=X1)
        Y1.clip(self.ymin, self.ymax, out=Y1)

        U, V = force.velocity(X1, Y1, Z, fractional_step=0.5)
        return U, V

    RK2 = RK2b

    #     """Runge-Kutta fourth order advection"""

    #     X, Y, Z = state["X"], state["Y"], state["Z"]
    #     dt = self.dt
    #     dx, dy = self.dx, self.dy

    #     U1, V1 = forcing.velocity(X, Y, Z, tstep=0.0)
    #     X1 = X + 0.5 * U1 * dt / dx
    #     Y1 = Y + 0.5 * V1 * dt / dy

    #     U2, V2 = forcing.velocity(X1, Y1, Z, tstep=0.5)
    #     X2 = X + 0.5 * U2 * dt / dx
    #     Y2 = Y + 0.5 * V2 * dt / dy

    #     U3, V3 = forcing.velocity(X2, Y2, Z, tstep=0.5)
    #     X3 = X + U3 * dt / dx
    #     Y3 = Y + V3 * dt / dy

    #     U4, V4 = forcing.velocity(X3, Y3, Z, tstep=1.0)

    #     U = (U1 + 2 * U2 + 2 * U3 + U4) / 6.0
    #     V = (V1 + 2 * V2 + 2 * V3 + V4) / 6.0

    #     return U, V

    def RK4(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, force: BaseForce
    ) -> Velocity:
        """Runge-Kutta fourth order advection

        This version does not sample velocities outside the grid

        """

        dt = self.dt
        dx, dy = self.dx, self.dy
        dtdx = dt / dx
        dtdy = dt / dy
        xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

        U1, V1 = force.velocity(X, Y, Z, fractional_step=0.0)
        X1, Y1 = RKstep(X, Y, U1, V1, 0.5, dtdx, dtdy)

        # X1 = X + 0.5 * U1 * dt / dx
        # Y1 = Y + 0.5 * V1 * dt / dy
        # X1.clip(xmin, xmax, out=X1)
        # Y1.clip(ymin, ymax, out=Y1)

        U2, V2 = force.velocity(X1, Y1, Z, fractional_step=0.5)
        X2, Y2 = RKstep(X, Y, U2, V2, 0.5, dtdx, dtdy)

        # X2 = X + 0.5 * U2 * dt / dx
        # Y2 = Y + 0.5 * V2 * dt / dy
        # X2.clip(xmin, xmax, out=X2)
        # Y2.clip(ymin, ymax, out=Y2)

        U3, V3 = force.velocity(X2, Y2, Z, fractional_step=0.5)
        X3, Y3 = RKstep(X, Y, U3, V3, 1.0, dtdx, dtdy)
        # X3 = X + U3 * dt / dx
        # Y3 = Y + V3 * dt / dy
        # X3.clip(xmin, xmax, out=X3)
        # Y3.clip(ymin, ymax, out=Y3)

        U4, V4 = force.velocity(X3, Y3, Z, fractional_step=1.0)

        # U = (U1 + 2 * U2 + 2 * U3 + U4) / 6.0
        # V = (V1 + 2 * V2 + 2 * V3 + V4) / 6.0
        U = RK4avg(U1, U2, U3, U4)
        V = RK4avg(V1, V2, V3, V4)

        return U, V

    def diffuse(self, num_particles: int) -> Velocity:
        """Random walk diffusion"""

        # Diffusive velocity
        stddev = (2 * self.D / self.dt) ** 0.5
        U = stddev * np.random.normal(size=num_particles)
        V = stddev * np.random.normal(size=num_particles)

        return U, V

    def diffuse_vert(self, num_particles: int):
        """Random walk diffusion"""

        # Diffusive velocity
        stddev = (2 * self.Dz / self.dt) ** 0.5
        W = stddev * np.random.normal(size=num_particles)

        return W


def RKstep(X, Y, U, V, frac, dtdx, dtdy):
    Xp = X + frac * U * dtdx
    Yp = Y + frac * V * dtdy
    return Xp, Yp


def RK4avg(U1, U2, U3, U4):
    return (U1 + 2 * U2 + 2 * U3 + U4) / 6.0
