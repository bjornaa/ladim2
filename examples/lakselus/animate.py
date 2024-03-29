import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
from netCDF4 import Dataset  # type: ignore
from postladim import ParticleFile

# ---------------
# User settings
# ---------------

# Files
particle_file = "out.nc"
# grid_file = "/scratch/Data/NK800/file_0000.nc"
grid_file = "/home/bjorn/nk800/norkyst_800m_his.nc4_2017033001-2017033100"
# Subgrid definition
i0, i1 = 340, 440
j0, j1 = 490, 560

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

# Plot initial particle distribution
X, Y = pf.position(0)
(particle_dist,) = ax.plot(X, Y, ".", color="red", markeredgewidth=0, lw=0.5)
# title = ax.set_title(pf.time(0))
timestamp = ax.text(0.01, 0.97, pf.ftime(0), fontsize=15, transform=ax.transAxes)


# Update function
def animate(t):
    X, Y = pf.position(t)
    particle_dist.set_data(X, Y)
    timestamp.set_text(pf.ftime(t))
    return particle_dist, timestamp


# Do the animation
anim = FuncAnimation(
    fig,
    animate,
    frames=num_times,
    interval=20,
    repeat=True,
    repeat_delay=500,
    blit=True,
)

plt.show()

pf.close()
