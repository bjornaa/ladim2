# import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from netCDF4 import Dataset

# from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "out_dense.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Subgrid definition
i0, i1 = 111, 121
j0, j1 = 96, 105

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
pf = Dataset(particle_file)
num_times = len(pf.dimensions["time"])

# Set up the plot area
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(xlim=(i0 + 1, i1 - 1), ylim=(j0 + 1, j1 - 1), aspect="equal")

# Background bathymetry
cmap = plt.get_cmap("Blues")
ax.contourf(Xcell, Ycell, H, cmap=cmap, alpha=0.3)

# Lon/lat lines
# ax.contour(Xcell, Ycell, lat, levels=range(57, 64), colors="black", linestyles=":")
# ax.contour(Xcell, Ycell, lon, levels=range(-4, 10, 2), colors="black", linestyles=":")

# Landmask
# constmap = plt.matplotlib.colors.ListedColormap([0.2, 0.6, 0.4])
# M[M > 0] = np.nan
# plt.pcolormesh(Xb, Yb, M, cmap=constmap)

# Plot initial particle distribution
X = pf.variables["X"][0, :]
Y = pf.variables["Y"][0, :]
(particle_dist,) = ax.plot(X, Y, ".", color="red", markeredgewidth=0, markersize=20)
timestamp = ax.text(
    0.01, 0.95, pf.variables["time"][0], fontsize=15, transform=ax.transAxes
)


# Update function
def animate(t):
    X = pf.variables["X"][t, :]
    Y = pf.variables["Y"][t, :]
    particle_dist.set_data(X, Y)
    timestamp.set_text(pf.variables["time"][t])
    return particle_dist, timestamp


# Do the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=num_times,
    interval=30,
    repeat=True,
    repeat_delay=500,
    blit=True,
)

plt.show()

pf.close()
