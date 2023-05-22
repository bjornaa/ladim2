from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ladim.state import State
from ladim.tracker import Tracker
from numpy import cos, exp, pi, sin

# Stommel model parameters as global variables
km = 1000.0  # [m]
D = 200.0  # Depth [m]
lambda_ = 10000 * km  # West-east extent of domain   [m]
b = 6300 * km  # South-north extent of domain  [m]
r = 1.0e-6  # Bottom friction coefficient  [s-1]
beta = 1.0e-11  # Coriolis derivative  [m-1 s-1]
F = 0.1  # Wind stress amplitude  [N m-2]
rho = 1025.0  # Density  [kg/m3]
alfa = beta / r  # [m-1]
gamma = F * pi / (r * b)  # [kg m2 s-1]


def main():
    # --- Simulation ---
    num_steps = 1736

    # --- Setup ---
    grid = Grid()
    force = Forcing()
    timer = TimeKeeper(dt=np.timedelta64(1, "D"))
    state = State()
    modules = dict(grid=grid, forcing=force, time=timer, state=state)
    tracker = Tracker(advection="RK2", modules=modules)

    # Initialize
    X0, Y0 = initial_release(grid=grid)
    state.append(X=X0, Y=Y0, Z=5)

    # Time loop
    for _n in range(num_steps):
        tracker.update()

    # Plot results
    plot_particles(state, X0, Y0, forcing=force)


@dataclass
class Grid:
    xmin: float = 0
    xmax: float = lambda_
    ymin: float = 0
    ymax: float = b

    @staticmethod
    def metric(X, Y):
        return np.ones_like(X), np.ones_like(X)

    def ingrid(self, X, Y):
        return (0 < X) & (X < self.xmax) & (0 < Y) & (Y < self.ymax)

    def depth(self, X, Y):
        return D + np.zeros_like(X)

    @staticmethod
    def atsea(X, Y):
        return np.ones(X.shape, dtype="bool")


@dataclass
class Forcing:
    G: float = (1 / rho) * (1 / D) * gamma * (b / pi) ** 2  # [m2 s-1]
    A: float = -0.5 * alfa + np.sqrt(0.25 * alfa**2 + (pi / b) ** 2)  # [m-1]
    B: float = -0.5 * alfa - np.sqrt(0.25 * alfa**2 + (pi / b) ** 2)  # [m-1]
    p: float = (1.0 - exp(B * lambda_)) / (exp(A * lambda_) - exp(B * lambda_))
    q: float = 1 - p

    def velocity(self, X, Y, Z, fractional_step=0):
        # Unselfify: self.v -> v
        A, B, G, p, q = (getattr(self, v) for v in "A B G p q".split())
        U = G * (pi / b) * cos(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)
        V = -G * sin(pi * Y / b) * (p * A * exp(A * X) + q * B * exp(B * X))
        return U, V

    def psi(self, X, Y):
        """Stream function"""

        # Unselfify: self.v -> v
        A, B, G, p, q = (getattr(self, v) for v in "A B G p q".split())

        return G * sin(pi * Y / b) * (p * exp(A * X) + q * exp(B * X) - 1)


@dataclass
class TimeKeeper:
    dt: np.timedelta64


def initial_release(grid):
    """Initialize with particles in two concentric circles"""
    km = 1000
    x0 = lambda_ / 3.0
    y0 = b / 3.0
    r1 = 800 * km
    r2 = 1600 * km

    T = np.linspace(0, 2 * np.pi, 1000)
    X01 = x0 + r1 * cos(T)
    Y01 = y0 + r1 * sin(T)
    X02 = x0 + r2 * cos(T)
    Y02 = y0 + r2 * sin(T)

    X0 = np.concatenate((X01, X02))
    Y0 = np.concatenate((Y01, Y02))
    return X0, Y0


def plot_particles(state, X0, Y0, forcing):
    # Discetize and contour the streamfunction
    km = 1000
    imax, jmax = 101, 64
    dx = 100 * km

    # Plot stream function background
    I = np.arange(imax) * dx
    J = np.arange(jmax) * dx
    JJ, II = np.meshgrid(I, J)
    Psi = forcing.psi(JJ, II)
    plt.contour(I / km, J / km, Psi, colors="k", linestyles=":", linewidths=0.5)

    # Initial state
    plt.plot(X0 / km, Y0 / km, ".b")
    # Final state
    plt.plot(state.X / km, state.Y / km, ".r")

    plt.axis("image")
    plt.show()


# --------------------------------------
if __name__ == "__main__":
    main()
