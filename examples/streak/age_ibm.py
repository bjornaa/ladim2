# Minimal IBM to track particle age

from ladim2.ibm import IBM


def initIBM(**args) -> IBM:
    return AgeIBM(**args)


class AgeIBM(IBM):
    def __init__(self, dt) -> None:
        print("Initializing age IBM")
        self.dt = dt.astype(int)

    def update(self, grid, state, forcing):

        # Update the particle age
        state["age"] += self.dt
