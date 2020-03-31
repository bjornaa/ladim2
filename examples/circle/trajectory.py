# Move particle in a circle (clockwise)

# from collections import namedtuple
from typing import Tuple, Sequence

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from ladim2.state import State
from ladim2 import analytical

a: float = 0.01
# dt: int = 0.3   # seconds    <1% error EF
dt: int = 7  # seconds       <1% error RK2
# dt: int = 30  # seconds      <1% error RK4
nsteps: int = 2000

Trajectory = Tuple[Sequence[float], Sequence[float]]


def main():

    state = State()

    # Initialize
    X0, Y0 = 0, 100
    state.append(X=X0, Y=Y0, Z=5)
    trajectory = [X0], [Y0]

    # Time loop
    for n in range(1, nsteps):
        advance(state, dt)
        append_trajectory(trajectory, state)

    plot_trajectory(trajectory)
    plot2(trajectory)


def sample_velocity(x: float, y: float) -> Tuple[float, float]:
    # Clockwise circular motion
    return a * y, -a * x


get_velocity = analytical.get_velocity2


def advance(state: State, dt: int) -> None:
    velocity = get_velocity(state, sample_velocity, dt)
    state.update(velocity, dt)


def append_trajectory(trajectory, state):
    trajectory[0].append(state.X[0])
    trajectory[1].append(state.Y[0])


def plot_trajectory(trajectory):
    plt.plot(trajectory[0], trajectory[1])
    plt.plot(trajectory[0][0], trajectory[1][0], "ro")
    plt.axis("image")
    plt.show()


def plot2(trajectory):
    x = np.array(trajectory[0])
    y = np.array(trajectory[1])
    r = np.sqrt(x * x + y * y)
    print(r[-1])
    # plt.plot(r)
    # plt.show()


if __name__ == "__main__":
    main()
