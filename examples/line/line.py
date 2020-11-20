# The line example as a scipt

# Bjørn Ådlandsvik
# 2020-04-04

import numpy as np

# import ladim2
from ladim2.state import State
from ladim2.grid_ROMS import makegrid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing_ROMS import init_forcing
from ladim2.tracker import Tracker
from ladim2.output import Output

# --------------
# Settings
# --------------

start_time = "1989-05-24 12"
stop_time = "1989-06-20"
reference_time = "1970-01-01"

data_file = "../data/ocean_avg_0014.nc"

advection = "EF"
dt = 3600  # seconds

output_file = "out.nc"
output_frequency = 3 * dt
grid_module = "ladim2.grid_ROMS"
num_particles = 1000
output_variables = dict(
    pid=dict(
        encoding=dict(datatype="i4", zlib=True),
        attributes=dict(long_name="particle identifier"),
    ),
    X=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(long_name="particle X-coordinate"),
    ),
    Y=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(long_name="particle Y-coordinate"),
    ),
    Z=dict(
        encoding=dict(datatype="f4", zlib=True),
        attributes=dict(
            long_name="particle_depth",
            standard_name="depth below surface",
            units="m",
            positive="down",
        ),
    ),
)

# Initiate LADiM

state = State()
grid = makegrid(module=grid_module, filename=data_file)
timer = TimeKeeper(start=start_time, stop=stop_time, dt=dt, reference=reference_time)
force = init_forcing(grid=grid, timer=timer, filename=data_file)
tracker = Tracker(dt=dt, advection=advection)
output = Output(
    timer=timer,
    filename=output_file,
    num_particles=num_particles,
    output_period=output_frequency,
    instance_variables=output_variables,
)


# Initiate particle distribution
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90
X0 = np.linspace(x0, x1, num_particles)
Y0 = np.linspace(y0, y1, num_particles)
Z0 = 5  # Fixed particle depth
state.append(X=X0, Y=Y0, Z=Z0)
# Write initial state
output.write(state)

# -------------
# Time loop
# -------------

print("Time loop")
for step in range(timer.Nsteps):
    force.update(step)
    tracker.update(state, grid=grid, force=force)
    if step % output.output_period_steps == 0:
        output.write(state)
