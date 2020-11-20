# Minimal IBM to kill old particles

import numpy as np

from ladim2.ibm import IBM
from ladim2.timekeeper import TimeKeeper

DAY = 24 * 60 * 60  # Number of seconds in a day


def initIBM(**args) -> IBM:
    return KillerIBM(**args)


class KillerIBM(IBM):
    def __init__(self, timer: TimeKeeper, lifetime: float) -> None:
        print("Initializing killer feature")
        self.lifetime = lifetime
        self.dt = timer.dt

    def update(self, grid, state, forcing):

        # Update the particle age
        state["age"] += self.dt

        # Mark particles older than 2 days as dead
        state["alive"] = state.age < self.lifetime * DAY
