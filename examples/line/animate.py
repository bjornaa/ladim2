"""Animate particle tracking from LADiM"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from netCDF4 import Dataset
from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "out.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Subgrid definition
i0, i1 = 55, 150
j0, j1 = 55, 145

# ----------------

# ROMS grid, plot domain
with Dataset(grid_file) as f0:
    H = f0.variables["h"][j0:j1, i0:i1]
    M = f0.variables["mask_rho"][j0:j1, i0:i1]
    lon = f0.variables["lon_rho"][j0:j1, i0:i1]
    lat = f0.variables["lat_rho"][j0:j1, i0:i1]

# Cell centers and boundaries
Xcell = np.arange(i0, i1)
Ycell = np.arange(j0, j1)
Xb = np.arange(i0 - 0.5, i1)
Yb = np.arange(j0 - 0.5, j1)

# particle_file
pf = ParticleFile(particle_file)
num_times = pf.num_times

# Set up the plot area
fig = plt.figure(figsize=(9, 8))
ax = plt.axes(xlim=(i0 + 1, i1 - 1), ylim=(j0 + 1, j1 - 1), aspect="equal")

# Background bathymetry
cmap = plt.get_cmap("Blues")
ax.contourf(Xcell, Ycell, H, cmap=cmap, alpha=0.5)

# Lon/lat lines
ax.contour(
    Xcell, Ycell, lat, levels=range(55, 64), colors="black", linestyles=":", alpha=0.5
)
ax.contour(
    Xcell,
    Ycell,
    lon,
    levels=range(-4, 10, 2),
    colors="black",
    linestyles=":",
    alpha=0.5,
)

# Landmask
constmap = plt.matplotlib.colors.ListedColormap([0.2, 0.6, 0.4])
M = np.ma.masked_where(M > 0, M)
plt.pcolormesh(Xb, Yb, M, cmap=constmap)

# Plot initial particle distribution
X, Y = pf.position(0)
(particle_dist,) = ax.plot(X, Y, ".", color="red", markeredgewidth=0, lw=0.5)
timestamp = ax.text(
    0.02,
    0.95,
    pf.ftime(0),
    fontsize=15,
    backgroundcolor="white",
    transform=ax.transAxes,
)


# Update function
def animate(t):
    X, Y = pf.position(t)
    particle_dist.set_data(X, Y)
    timestamp.set_text(pf.ftime(t))
    return particle_dist, timestamp


# Start/stop function
# Using function attribute to avoid a global variable
def onClick(event):
    """Make an event halt or restart the animation"""
    if onClick.running:
        anim.event_source.stop()
        onClick.running = False
    else:
        anim.event_source.start()
        onClick.running = True


onClick.running = True
fig.canvas.mpl_connect("button_press_event", onClick)

# Do the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=num_times,
    interval=40,
    repeat=True,
    repeat_delay=500,
    blit=True,
)

# Uncomment the next lineto save the animation
# anim.save('line.gif',  writer='imagemagick')
plt.show()

pf.close()
