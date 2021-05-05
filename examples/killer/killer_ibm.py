# Minimal IBM to kill old particles

import numpy as np

from ladim2.ibm import BaseIBM
from ladim2.timekeeper import TimeKeeper
from ladim2.state import State


class IBM(BaseIBM):
    def __init__(
        self,
        lifetime: float,  # Particle life time, units=days
        timer: TimeKeeper,
        state: State,
        **args,
    ) -> None:

        print("Initializing killer feature")
        self.lifetime = lifetime
        self.state = state
        self.dt = timer.dt / np.timedelta64(1, "D")

    def update(self) -> None:

        state = self.state

        # Update the particle age
        state["age"] += self.dt

        # Add particles older than prescribed lifetime to dead
        state["alive"] &= state.age < self.lifetime
