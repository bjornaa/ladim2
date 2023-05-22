# Minimal IBM to track particle age


class IBM:
    def __init__(self, modules, **kwargs) -> None:
        self.state = modules["state"]
        self.dt = modules["time"].dtsec

    def update(self):
        # Update the particle age
        self.state["age"] += self.dt
