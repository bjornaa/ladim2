"""LADiM – the line example as a scipt using the yaml configuration"""

# ------------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institue of Marine Research
# 2020-04-04
# ------------------------------------

from ladim2.state import State
from ladim2.grid import init_grid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import init_force
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
grid = init_grid(**config["grid"])
force = init_force(grid=grid, timer=timer, **config["forcing"])
tracker = Tracker(**config["tracker"])
release = ParticleReleaser(timer=timer, datatypes=state.dtypes, **config["release"])
output = Output(
    timer=timer, num_particles=release.total_particle_count, **config["output"]
)

# Initialize the time loop

# Number of time steps between output (have that computed in output.py)
output_period_step = output.output_period / timer._dt

# Initial particle release and output
step = 0
if 0 in release.steps:
    V = next(release)
    state.append(**V)
output.write(state)

# --------------
# Time loop
# --------------

while step < timer.Nsteps:

    timer.update()

    # Update, forcing first, then state  (correct?)
    # --- Update forcing ---
    force.update(step, state.X, state.Y, state.Z)
    tracker.update(state, grid=grid, force=force)

    step += 1

    # --- Particle release and output
    if step in release.steps:
        V = next(release)
        state.append(**V)
    if step % output_period_step == 0:
        output.write(state)
