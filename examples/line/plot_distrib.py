"""Plot the particle distribution at a given time

Plot the particle distribution at a given TIMESTEP
on a specified subgrid with a land mask and lon/lat-lines.

This version is low level, using only the netCDF library,
The higher level postladim package is not used.

"""

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# ---------------
# User settings
# ---------------

# Files
particle_file = "out.nc"
grid_file = "../data/ocean_avg_0014.nc"

# Time step
TIMESTEP = 50

# Subgrid definition
I0, I1 = 55, 150  # X range (grid cells)
J0, J1 = 55, 145  # Y range (grid cells)

# ----------------
# Data handling
# ----------------

# ROMS grid, plot domain
with nc.Dataset(grid_file) as gf:
    H = gf.variables["h"][J0:J1, I0:I1]
    M = gf.variables["mask_rho"][J0:J1, I0:I1]
    lon = gf.variables["lon_rho"][J0:J1, I0:I1]
    lat = gf.variables["lat_rho"][J0:J1, I0:I1]
M[M > 0] = np.nan  # Mask out sea cells

# Cell centers and cell boundaries
Xcell = np.arange(I0, I1)
Ycell = np.arange(J0, J1)

# Read the particle distribution
with nc.Dataset(particle_file) as pf:
    # Positions
    end = np.cumsum(pf.variables["particle_count"])
    start = end - pf.variables["particle_count"]
    X = pf.variables["X"][start[TIMESTEP] : end[TIMESTEP]]
    Y = pf.variables["Y"][start[TIMESTEP] : end[TIMESTEP]]
    # Time
    timevar = pf.variables["time"]
    timestamp = nc.num2date(timevar[TIMESTEP], timevar.units)

# Integer latitude levels, even integer longitude levels
lat_levels = np.arange(np.ceil(lat.min()), np.ceil(lat.max()))
lon_levels = np.arange(2 * np.ceil(0.5 * lon.min()), 2 * np.ceil(0.5 * lon.max()), 2)

# ------------
# --- Plot ---
# ------------

# Set up the plot area
fig = plt.figure(figsize=(8, 7))
ax = plt.axes(xlim=(I0, I1 - 1), ylim=(J0, J1 - 1), aspect="equal")

# Background bathymetry
cmap = plt.get_cmap("Blues")
ax.contourf(Xcell, Ycell, H, cmap=cmap, alpha=0.8)

# Lon/lat lines
ax.contour(
    Xcell,
    Ycell,
    lat,
    levels=lat_levels,
    colors="grey",
    linestyles=":",
)
ax.contour(
    Xcell,
    Ycell,
    lon,
    levels=lon_levels,
    colors="grey",
    linestyles=":",
)

# Landmask
constmap = mpl.colors.ListedColormap([0.2, 0.6, 0.4])
ax.pcolormesh(Xcell, Ycell, M, cmap=constmap)

# Particle distribution
ax.plot(X, Y, ".", color="red", lw=2)

# Timestamp
ax.text(
    0.02,
    0.95,
    timestamp,
    fontsize=15,
    backgroundcolor="white",
    transform=ax.transAxes,
)

plt.show()
