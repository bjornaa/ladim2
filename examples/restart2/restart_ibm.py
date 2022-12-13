# IBM for testing restart

import numpy as np


class IBM:
    """Minimal IBM for the killer example"""

    def __init__(self, modules, **kwargs) -> None:

        print("Initializing IBM: restart_ibm")
        self.state = modules["state"]
        self.forcing = modules["forcing"]

        self.dt = modules["time"].dt / np.timedelta64(1, "D")  # Unit = days
        self.lifetime = kwargs["lifetime"]

    def update(self) -> None:

        # Update the particle age
        self.state["age"] += self.dt

        # Update weight
        self.state["weight"] += 0.01 * self.state.temp

        # Kill particles older than prescribed lifetime
        self.state["alive"] &= self.state["age"] < self.lifetime

