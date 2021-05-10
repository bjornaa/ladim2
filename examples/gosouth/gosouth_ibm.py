"""IBM module for the gosouth example in LADiM version 2"""

# ---------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute of Marine Research
# ----------------------------------

# 2021-01-18 Modified for LADiM v.2

import numpy as np

from ladim2.ibm import BaseIBM
from ladim2.timekeeper import TimeKeeper
from ladim2.state import State
from ladim2.grid import BaseGrid


class IBM(BaseIBM):
    """Adding a constant horizontal velocity to the particle tracking"""

    def __init__(
        self,
        direction: float,  # clockwise degree from North
        speed: float,  # swimming speed [m/s]
        timer: TimeKeeper,
        state: State,
        grid: BaseGrid,
        forcing=None,  # This IBM does not use forcing
    ):
        self.dt = timer.dtsec
        self.state = state
        self.grid = grid

        # Compute swimming velocity in grid coordinates
        azimuth = direction * np.pi / 180
        angle = grid.angle  # type: ignore
        self.Xs = speed * np.sin(azimuth + angle)
        self.Ys = speed * np.cos(azimuth + angle)

    def update(self) -> None:

        state = self.state
        grid = self.grid

        # Compute new position
        I = np.round(state.X).astype("int")
        J = np.round(state.Y).astype("int")
        X1 = state.X + self.Xs[J, I] * self.dt / grid.dx[J, I]  # type: ignore
        Y1 = state.Y + self.Ys[J, I] * self.dt / grid.dy[J, I]  # type: ignore

        # Only move particles to sea positions inside the grid
        move = grid.ingrid(X1, Y1) & grid.atsea(X1, Y1)
        state.X[move] = X1[move]
        state.Y[move] = Y1[move]
