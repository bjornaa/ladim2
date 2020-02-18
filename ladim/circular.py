# Move particle in a circle (clockwise)

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from state import State
# from velocity import Velocity


a = 0.01
dt = 1  # seconds
nsteps = 1000


def main():
    # Initialize
    state = State()
    velocity = namedtuple('Velocity', 'U V')


    # Release a particle
    X0, Y0 = 0, 100
    state.append(X=[X0], Y=[Y0], Z=[5])

    # Initalize trajectory
    X = [X0]
    Y = [Y0]

    # Time loop
    for n in range(1, nsteps):
        velocity.U = a * state.Y
        velocity.V = -a * state.X
        state.update(velocity, dt)
        X.append(state.X[0])
        Y.append(state.Y[0])

    # Plot trajectory
    plt.plot(X, Y)
    plt.axis('image')
    plt.show()

def update(state, U, V, dt):
    # Update the state
    print(state.X, state.Y)
    print("   ", U, V)
    state.X += U * dt
    state.Y += V * dt


if __name__ == '__main__':
    main()