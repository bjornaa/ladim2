# Minimal IBM to track particle age

from ladim2.ibm import BaseIBM
from ladim2.timekeeper import TimeKeeper


def init_IBM(**args) -> BaseIBM:
    return AgeIBM(**args)


class AgeIBM(BaseIBM):
    def __init__(self, timer: TimeKeeper) -> None:
        print("Initializing age IBM")
        self.dt = timer.dt

    def update(self, grid, state, forcing):

        # Update the particle age
        state["age"] += self.dt
