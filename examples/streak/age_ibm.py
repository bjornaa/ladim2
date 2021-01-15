# Minimal IBM to track particle age

from ladim2.ibm import BaseIBM
from ladim2.timekeeper import TimeKeeper
from ladim2.state import State


class IBM(BaseIBM):
    def __init__(
        self, timer: TimeKeeper, state: State, forcing=None, grid=None
    ) -> None:
        print("Initializing age IBM")
        self.timer = timer
        self.state = state
        self.dt = timer.dt

    def update(self):

        # Update the particle age
        self.state["age"] += self.timer.dt
