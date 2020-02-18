# Move particle in a circle (clockwise)

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from state import State

a = 0.01
dt = 1  # seconds
nsteps = 2000

state = State()
velocity = namedtuple("Velocity", "U V")
trajectory = namedtuple("Trajectory", "X Y")


def initialize():
    """Initialize"""
    # Release a particle
    X0, Y0 = 0, 100  # Release position
    state.append(X=[X0], Y=[Y0], Z=[5])

    # Initalize trajectory
    trajectory.X = [X0]
    trajectory.Y = [Y0]


def get_velocity(state):
    velocity.U = a * state.Y
    velocity.V = -a * state.X
    return velocity


def advance(state):
    velocity = get_velocity(state)
    state.update(velocity, dt)


def append_trajectory(state):
    trajectory.X.append(state.X[0])
    trajectory.Y.append(state.Y[0])


def main():

    initialize()

    # Time loop
    for n in range(1, nsteps):
        advance(state)
        append_trajectory(state)

    # Plot trajectory
    plt.plot(trajectory.X, trajectory.Y)
    plt.plot(trajectory.X[0], trajectory.Y[0], "ro")
    plt.axis("image")
    plt.show()


if __name__ == "__main__":
    main()
