import numpy as np
from ladim2.state import State
from ladim2.tracker import Tracker


class Grid:
    def __init__(self):
        self.imax = 100
        self.jmax = 100
        self.xmin = 0.0
        self.xmax = self.imax - 1.0
        self.ymin = 0.0
        self.ymax = self.jmax - 1.0
        self.dt = 600

    def metric(self, X, Y):
        return 100 * np.ones_like(X), 100 * np.ones_like(Y)

    def sample_depth(self, X, Y):
        return 50.0 * np.ones_like(X)

    def ingrid(self, X, Y):
        return (self.xmin <= X) & (X <= self.xmax) & (self.ymin <= Y) & (Y <= self.ymax)

    def atsea(self, X, Y):
        return np.ones(len(X), dtype="bool")

    def lonlat(self, X, Y, method=None):
        return 5.0 * np.ones(len(X)), 60.0 * np.ones(len(X))


class Forcing:
    def __init__(self, grid):
        pass

    def velocity(self, X, Y, Z):
        return 0.5 * np.ones_like(X), np.ones_like(Y)

    # def field(self, X, Y, Z, name):
    #     return 8.0 * np.ones_like(X)


def test_advection():
    """Check advection with constant velocity"""
    state = State()
    grid = Grid()
    forcing = Forcing(grid=grid)
    tracker = Tracker(config=dict(advection="EF"))

    X = [30, 22.2, 11.1]
    Y = [40, 42, 45]
    state.append(X=X, Y=Y, Z=5)

    tracker.update(state, grid, forcing)

    assert all(state.X == [x + 3 for x in X])
    assert all(state.Y == [y + 6 for y in Y])


def test_out_of_area():
    """Particles moving out of the area should be killed"""

    state = State()
    grid = Grid()
    forcing = Forcing(grid=grid)
    tracker = Tracker(config=dict(advection="EF"))

    X = [30, grid.imax - 2.1, 11.1]
    Y = [30, grid.jmax - 2.1, 22.2]

    state.append(X=X, Y=Y, Z=5)

    tracker.update(state, grid, forcing)

    # Killed particle 1, out-of-area
    # Question: should tracker.update do a compactify?
    assert np.all(state.alive == [True, False, True])
    assert np.all(state.active == [True, False, True])


if __name__ == "__main__":
    test_ok()
