"""
Move particle in a circle (clockwise)

Testbed for different numerical schemes

Example on the use of LADiM as a library
"""

from collections import namedtuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from ladim import analytical
from ladim.state import State

# Numerical method "EF", "RK2", or "RK4"
METHOD = "RK2"
match METHOD:
    case "EF":
        get_velocity = analytical.get_velocity1
    case "RK2":
        get_velocity = analytical.get_velocity2
    case "RK4":
        get_velocity = analytical.get_velocity4
    case _:
        raise SystemExit(f"{METHOD} is not a supported method")


A: float = 0.01  # Velocity scale factor
DT = 5  # seconds
NUM_STEPS = 2000

Trajectory = namedtuple("Trajectory", ["x", "y"])


def main():
    # Initialize the state and the trajectory
    state = State()
    X0, Y0 = 0, 100  # Initial position
    state.append(X=X0, Y=Y0, Z=5)
    traj = Trajectory([], [])  # Empty trajectory
    append_trajectory(traj, state)

    # Time loop
    for n in range(NUM_STEPS):
        advance(state, DT)
        append_trajectory(traj, state)

    plot_trajectory(traj)
    estimate_error(traj)


def sample_velocity(x, y):
    """Clockwise (A > 0) circular motion"""
    return A * y, -A * x


def advance(state, DT) -> None:
    """Move forward to the next time step"""
    velocity = get_velocity(state, sample_velocity, DT)  # type: ignore
    state["X"] += DT * velocity.U
    state["Y"] += DT * velocity.V


def append_trajectory(trajectory, state):
    """Append the state to the trajectory"""
    trajectory.x.append(state.X[0])
    trajectory.y.append(state.Y[0])


def plot_trajectory(trajectory):
    """Plot the trajectory with matplotlib"""
    plt.plot(trajectory.x, trajectory.y)
    # Mark initial position
    plt.plot(trajectory.x[0], trajectory.y[0], "ro")
    plt.axis("image")
    plt.show()


def estimate_error(trajectory):
    """Estimate the relative error in the radius"""
    r = np.hypot(trajectory.x, trajectory.y)
    print("Relative error [%]: ", 100 * (r[-1] / r[0] - 1))


if __name__ == "__main__":
    main()
