# Minimal IBM to kill old particles

from ladim2.ibm import BaseIBM
from ladim2.timekeeper import TimeKeeper
from ladim2.state import State

DAY = 24 * 60 * 60  # Number of seconds in a day


class IBM(BaseIBM):
    def __init__(
        self,
        lifetime: float,
        timer: TimeKeeper,
        state: State,
        forcing=None,    # This IBM does not use forcing
        grid=None,       # This IBM does not use grid
    ) -> None:

        print("Initializing killer feature")
        self.lifetime = lifetime
        self.state = state
        self.dt = timer.dt

    def update(self) -> None:

        state = self.state

        # Update the particle age
        state["age"] += self.dt

        # Mark particles older than 2 days as dead
        state["alive"] = state.age < self.lifetime * DAY
