# ------------------------------------
# tracker.py
# Part of the LADiM Model
#
# Bjørn Ådlandsvik, <bjorn@imr.no>
# Institute of Marine Research
#
# Licenced under the MIT license
# ------------------------------------

# import logging
from typing import Any, Tuple, Dict
import numpy as np
from .state import State

from numba import njit

Velocity = Tuple[np.ndarray, np.ndarray]
# State = Any  # Could not find any better


class Tracker:
    """The physical particle tracking kernel"""

    def __init__(self, config: Dict[str, Any]) -> None:
        # logging.info("Initiating the particle tracking")
        # self.dt = grid["dt"]
        self.advection = config.get("advection", None)
        # advect is the advection method
        if self.advection:
            self.advect = getattr(self, self.advection)
        diffusion = config.get("diffusion", None)
        if diffusion and diffusion <= 0:
            diffusion = None
        self.diffusion = diffusion

        # #if config["advection"]:
        # #    self.advect = getattr(self, config["advection"])
        # #else:
        # #    self.advect = None
        # # Read from config:
        # self.diffusion = config["diffusion"]
        # if self.diffusion:
        #     self.D = config["diffusion_coefficient"]  # [m2.s-1]
        # self.active_check = 'active' in config['ibm_variables']

    # def move_particles(self, grid: Grid, force: Forcing, state: State) -> None:
    def update(self, state: State, grid, force) -> None:
        """Move the particles"""

        X, Y, Z = state.X, state.Y, state.Z
        dx, dy = grid.metric(X, Y)
        dt = grid.dt
        # self.num_particles = len(X)
        # Make more elegant, need not do every time
        # Works for C-grid
        # self.xmin = grid.xmin + 0.01
        # self.xmax = grid.xmax - 0.01
        # self.ymin = grid.ymin + 0.01
        # self.ymax = grid.ymax - 0.01

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        # --- Advection ---
        if self.advection:
            Uadv, Vadv = self.advect(X, Y, Z, force)
            U += Uadv
            V += Vadv

        # --- Diffusion ---
        # if self.diffusion:
        #    Udiff, Vdiff = self.diffuse(self.diffuse)
        #    U += Udiff
        #    V += Vdiff

        # --- Move the particles

        # New position, if OK
        X1 = X + U * dt / dx
        Y1 = Y + V * dt / dy

        # Do not move out of grid
        I = ~grid.ingrid(X1, Y1)
        X1[I] = X[I]
        Y1[I] = Y[I]
        # Kill particles trying to move out of the grid
        state.alive[I] = False
        state.active[I] = False  # Not necessary if they are removed

        # if self.active_check:
        # Do not move inactive particles
        I = state.active < 1
        X1[I] = X[I]
        Y1[I] = Y[I]

        # Land, boundary treatment. Do not move the particles
        # Consider a sequence of different actions
        # I = (grid.ingrid(X1, Y1)) & (grid.atsea(X1, Y1))
        I = grid.atsea(X1, Y1)
        # I = True
        X[I] = X1[I]
        Y[I] = Y1[I]

        state.X = X
        state.Y = Y

    # @njit
    def EF(self, X, Y, Z, force) -> Velocity:
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

    # def RK2b(self, forcing: Forcing, state: State) -> Velocity:
    #     """Runge-Kutta second order = Heun scheme

    #     This version does not sample velocities outside the grid
    #     """

    #     X, Y, Z = state["X"], state["Y"], state["Z"]
    #     dt = self.dt

    #     U, V = forcing.velocity(X, Y, Z)
    #     X1 = X + 0.5 * U * dt / self.dx
    #     Y1 = Y + 0.5 * V * dt / self.dy
    #     X1.clip(self.xmin, self.xmax, out=X1)
    #     Y1.clip(self.ymin, self.ymax, out=Y1)

    #     U, V = forcing.velocity(X1, Y1, Z, tstep=0.5)
    #     return U, V

    # RK2 = RK2b

    # def RK4a(self, forcing: Forcing, state: State) -> Velocity:
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

    # def RK4b(self, forcing: Forcing, state: State) -> Velocity:
    #     """Runge-Kutta fourth order advection

    #     This version does not sample velocities outside the grid

    #     """

    #     X, Y, Z = state["X"], state["Y"], state["Z"]
    #     dt = self.dt
    #     dx, dy = self.dx, self.dy
    #     xmin, xmax, ymin, ymax = self.xmin, self.xmax, self.ymin, self.ymax

    #     U1, V1 = forcing.velocity(X, Y, Z, tstep=0.0)
    #     X1 = X + 0.5 * U1 * dt / dx
    #     Y1 = Y + 0.5 * V1 * dt / dy
    #     X1.clip(xmin, xmax, out=X1)
    #     Y1.clip(ymin, ymax, out=Y1)

    #     U2, V2 = forcing.velocity(X1, Y1, Z, tstep=0.5)
    #     X2 = X + 0.5 * U2 * dt / dx
    #     Y2 = Y + 0.5 * V2 * dt / dy
    #     X2.clip(xmin, xmax, out=X2)
    #     Y2.clip(ymin, ymax, out=Y2)

    #     U3, V3 = forcing.velocity(X2, Y2, Z, tstep=0.5)
    #     X3 = X + U3 * dt / dx
    #     Y3 = Y + V3 * dt / dy
    #     X3.clip(xmin, xmax, out=X3)
    #     Y3.clip(ymin, ymax, out=Y3)

    #     U4, V4 = forcing.velocity(X3, Y3, Z, tstep=1.0)

    #     U = (U1 + 2 * U2 + 2 * U3 + U4) / 6.0
    #     V = (V1 + 2 * V2 + 2 * V3 + V4) / 6.0

    #     return U, V

    # RK4 = RK4b

    # def diffuse(self) -> Velocity:
    #     """Random walk diffusion"""

    #     # Diffusive velocity
    #     stddev = (2 * self.D / self.dt) ** 0.5
    #     U = stddev * np.random.normal(size=self.num_particles)
    #     V = stddev * np.random.normal(size=self.num_particles)

    #     return U, V
