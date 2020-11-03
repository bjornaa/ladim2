from ladim2.configure import configure
from ladim2.state import State
from ladim2.grid_ROMS import Grid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing_ROMS import Forcing
from ladim2.tracker import Tracker
from ladim2.release import ParticleReleaser
from ladim2.output import Output
#from output import Output
#from configure import configure

# ----------------
# Configuration
# ----------------

config = configure("ladim2.yaml")

# -------------------
# Initialization
# -------------------

state = State(**config["state"])
timer = TimeKeeper(**config["time_control"])
grid = Grid(**config["grid"])
force = Forcing(grid=grid, timer=timer, **config["forcing"])
# tracker = Tracker(dt=timer.dt, **config["tracker"])
tracker = Tracker(**config["tracker"])
release = ParticleReleaser(time_control=timer, **config["release"])
output = Output(timer=timer, **config["output"])

# --------------------------
# Initial particle release
# --------------------------

# Skal automatisere tilpasning til state-variablene
# Også initiering av variable som ikke er i release-filen
# X0 er et eksempel.
print("Initial particle release")
V = next(release)
# TODO: Simplify release
## next provides pid, this is handled by state itself
# V = V.drop(columns='pid')
state.append(**V)

# --------------
# Time loop
# --------------

print("Time loop")
for step in range(timer.Nsteps+1):
    tracker.update(state, grid=grid, force=force)
    if step % output.output_period_steps == 0:
        output.write(state)

# --------------
# Finalisation
# --------------

print("Cleaning up")
# output.save_particle_variables()
# output.close()
