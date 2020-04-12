# The line example as a scipt
# Presently without storing results

# Bjørn Ådlandsvik
# 2020-04-04

import numpy as np

# import ladim2
from ladim2.state import State
from ladim2.grid_ROMS import Grid
from ladim2.timer import Timer
from ladim2.forcing_ROMS import Forcing
from ladim2.tracker import Tracker
from output import Output

# --------------
# Settings
# --------------

start_time = "1989-05-24 12"
stop_time = "1989-06-20"
reference_time = "1970-01-01"
# particle_variables=['release_X']
data_file = "../data/ocean_avg_0014.nc"
advection = "EF"
dt = 3600  # seconds
output_file = "out.nc"
output_frequency = [3, "h"]

# -------------------
# Initialization
# -------------------

state = State()
grid = Grid(grid_file=data_file, dt=dt)
timer = Timer(start=start_time, stop=stop_time, dt=dt, reference=reference_time)
force = Forcing(grid=grid, timer=timer, forcing_file=data_file)
tracker = Tracker(dt=dt, advection=advection)

# Make initial state, num_particles points along a line

# End points of line in grid coordinates
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90
num_particles = 1000
X0 = np.linspace(x0, x1, num_particles)
Y0 = np.linspace(y0, y1, num_particles)
Z0 = 5  # Fixed particle depth
state.append(X=X0, Y=Y0, Z=Z0)

# Tungvint måte å gi info til output
class Release():
    def __init__(self, num_particles):
        self.total_num_particles=num_particles
release = Release(num_particles)


# Define output format
output_instance_variables = dict(
    pid=dict(ncformat="i4", long_name="particle identifier"),
    X=dict(ncformat="f4", long_name="particle X-coordinate"),
    Y=dict(ncformat="f4", long_name="particle Y-coordinate"),
    Z=dict(
        ncformat="f4",
        long_name="particle depth",
        standard_name="depth_below_surface",
        units="m",
        positive="down",
    ),
)

output_particle_variables = dict(
    X0=dict(ncformat="f4", long_name="X-coordinate of particle release")
)


output = Output(state=state, timer=timer, release=release,
    filename=output_file,
    frequency=output_frequency,
    instance_variables=output_instance_variables,
    particle_variables=output_particle_variables,
)

# -------------
# Time loop
# -------------

# Mer konsistent å bruke forcing=f,
# alternativ: bruke force overalt i stedet for forcing
for step in range(timer.Nsteps):
    tracker.update(state, grid=grid, force=force)
    if step % output.frequency == 0:
        print(step)
        output.write(step)

# --------------
# Finalisation
# --------------

output.close()
