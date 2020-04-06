# The line example as a scipt
# Presently without storing results

# Bjørn Ådlandsvik
# 2020-04-04

import numpy as np
import matplotlib.pyplot as plt

import ladim2
from ladim2.state import State
from ladim2.grid_ROMS import Grid
from ladim2.timer import Timer
from ladim2.forcing_ROMS import Forcing
from ladim2.tracker import Tracker

# --------------
# Settings
# --------------

data_file = "../data/ocean_avg_0014.nc"
start_time = "1989-05-24 12"
stop_time = "1989-06-20"
dt = 3600  # seconds

# -------------------
# Initialization
# -------------------

g = Grid(grid_file=data_file, dt=dt)
t = Timer(start=start_time, stop=stop_time, dt=dt)
f = Forcing(grid=g, timer=t, forcing_file=data_file)

# Make initial state

# End points of line in grid coordinates
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90

Npart = 1000  # Number of particles along the line

# Initial particle positions
X0 = np.linspace(x0, x1, Npart)
Y0 = np.linspace(y0, y1, Npart)
Z0 = 5  # Fixed particle depth

# Vil slå de to under sammen, initiere State med partikkelfordeling
# uten å ødelegge for initiering med IBM-variable
state = State()
state.append(X=X0, Y=Y0, Z=Z0)

# Number of time steps
# period = np.datetime64(stop_time) - np.datetime64(start_time)
# nsteps = period // np.timedelta64(dt, "s")

tracker = Tracker(advection="EF")

# -------------
# Time loop
# -------------


# Mer konsistent å bruke forcing=f,
# alternativ: bruke force overalt i stedet for forcing
for n in range(t.Nsteps):
    # if n % 10 == 0: print(n)
    tracker.update(state, grid=g, timer=t, force=f)


# --------------
# Finalisation
# --------------

# Plot the initial particle distribution
plt.plot(X0, Y0, ".", color="cyan")
# Plot the final particle distribution
plt.plot(state.X, state.Y, ".", color="red")

plt.show()
