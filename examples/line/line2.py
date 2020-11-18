#from pathlib import Path
#from typing import Union

from ladim2.state import State
from ladim2.grid import makegrid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import Forcing
from ladim2.tracker import Tracker
from ladim2.release import ParticleReleaser
from ladim2.output import Output
from ladim2.configure import configure


# ----------------
# Configuration
# ----------------


configuration_file = "ladim2.yaml"
config = configure(configuration_file)

# -------------------
# Initialization
# -------------------

print("Initiating")
state = State(**config["state"])
timer = TimeKeeper(**config["time"])
grid = makegrid(**config["grid"])
force = Forcing(grid=grid, timer=timer, **config["forcing"])
tracker = Tracker(**config["tracker"])
release = ParticleReleaser(timer=timer, datatypes=state.dtypes, **config["release"])
output = Output(
    timer=timer, num_particles=release.total_particle_count, **config["output"]
)


# --------------
# Time loop
# --------------

# Number of time steps between output (have that computed in output.py)
output_period_step = output.output_period / timer._dt

# Initial particle release and output
step = 0
if 0 in release.steps:
    V = next(release)
    state.append(**V)
output.write(state)

print("Time loop")
while step < timer.Nsteps:

    # Update, forcing first, then state  (correct?)
    # --- Update forcing ---
    force.update(step)
    tracker.update(state, grid=grid, force=force)

    step += 1

    # --- Particle release and output
    if step in release.steps:
        V = next(release)
        state.append(**V)
    if step % output_period_step == 0:
        output.write(state)
