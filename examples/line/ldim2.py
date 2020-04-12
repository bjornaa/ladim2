import numpy as np

import yaml

import ladim2
from ladim2.state import State
from ladim2.grid_ROMS import Grid
from ladim2.timer import Timer
from ladim2.forcing_ROMS import Forcing
from ladim2.tracker import Tracker
from release import Release
from output import Output

# ----------------
# Configuration
# ----------------

with open("ladim2.yaml") as fid:
    config = yaml.safe_load(fid)

# -------------------
# Initialization
# -------------------

state = State(variables=dict(X0=float), particle_variables=['X0'])
timer = Timer(**config["time_control"])
grid = Grid(**config["grid"])
force = Forcing(grid=grid, timer=timer, **config["forcing"])
tracker = Tracker(dt=timer.dt, **config["tracker"])
release = Release(**config["release"])
output = Output(state=state, timer=timer, release=release, **config["output"])

# Initial particle release
print("Initial particle release")
V = release.df.drop(columns="release_time")
V["X0"] = V["X"]     # Add the particle variable
state.append(**V)

# print(state.X0)

# --------------
# Time loop
# --------------

print("Time loop")
for step in range(timer.Nsteps):
    tracker.update(state, grid=grid, force=force)
    if step % output.frequency == 0:
        output.write(step)

# --------------
# Finalisation
# --------------

print("Cleaning up")
output.save_particle_variables()
output.close()
