# import itertools
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "station.nc"
# particle_file = "/home/bjorn/ladim/examples/station/station.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Subgrid definition
i0, i1 = 100, 140
j0, j1 = 84, 133

# time step
t = 100

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
# num_times = pf.num_times

# Set up the plot area
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(xlim=(i0 + 1, i1 - 1), ylim=(j0 + 1, j1 - 1), aspect="equal")

# Background bathymetry
cmap = plt.get_cmap("Blues")
ax.contourf(Xcell, Ycell, H, cmap=cmap, alpha=0.3)

# Lon/lat lines
ax.contour(Xcell, Ycell, lat, levels=range(57, 64), colors="black", linestyles=":")
ax.contour(Xcell, Ycell, lon, levels=range(-4, 10, 2), colors="black", linestyles=":")

# Landmask
constmap = plt.matplotlib.colors.ListedColormap([0.2, 0.6, 0.4])
M = np.ma.masked_where(M > 0, M)
plt.pcolormesh(Xb, Yb, M, cmap=constmap)

# Plot particle distribution
X, Y = pf.position(time=t)
Z = pf["Z"][t]
particle_dist = ax.scatter(X, Y, c=Z, cmap=plt.get_cmap("plasma_r"))
# Colorbar
cb = plt.colorbar(particle_dist)
cb.ax.invert_yaxis()
cb.set_label("Particle depth", fontsize=14)
# Time stamp
timestamp = ax.text(0.01, 0.96, pf.time(t), fontsize=15, transform=ax.transAxes)

plt.show()
