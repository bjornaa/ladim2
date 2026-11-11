"""Make a coast line in grid coordinates"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute if Marine Research
# 2021-02-10
# ----------------------------------

import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import numpy as np
from ladim.ROMS import Grid
from shapely import geometry

# Choose between c, l, i, h, f resolutions
GSHHS_resolution = "i"

# Grid file
grid_file = "../data/ocean_avg_0014.nc"

# Name of output coast file
coast_file = "coast_grid.npy"


grid = Grid(grid_file)

# Boundary (grid coordinates)
RES = 20  # Resolution of subdivision of boundary rectangle
EPS = 12  # Stay away from interpolation problems near the boundary
xmin, xmax = grid.xmin + EPS, grid.xmax - EPS
ymin, ymax = grid.ymin + EPS, grid.ymax - EPS
xbd = np.concatenate(
    (
        np.linspace(xmin, xmax, RES),
        RES * [xmax],
        np.linspace(xmax, xmin, RES),
        RES * [xmin],
    )
)
ybd = np.concatenate(
    (
        RES * [ymin],
        np.linspace(ymin, ymax, RES),
        RES * [ymax],
        np.linspace(ymax, ymin, RES),
    )
)

# Convert boundary to polygon in longitude, latitude
bd_pol = geometry.Polygon(np.column_stack(grid.xy2ll(xbd, ybd)))

# Global coastline from GSHHS as shapely collection generator
path = shapereader.gshhs(scale=GSHHS_resolution)
coast = shapereader.Reader(path).geometries()

# Restrict the coastline to the regional domain
coast = (bd_pol.intersection(p) for p in coast if bd_pol.intersects(p))
# Filter out isolated points
coast = filter(
    lambda p: isinstance(p, geometry.MultiPolygon) or isinstance(p, geometry.Polygon),
    coast,
)

# Convert the polygons to grid coordinates, and put in large arrays seperated by NaNs
Xcoast, Ycoast = np.array([]), np.array([])
for mp in coast:
    if isinstance(mp, geometry.Polygon):  # Single polygon
        X, Y = grid.ll2xy(*mp.boundary.xy)
        Xcoast = np.concatenate((Xcoast, X, [np.nan]))
        Ycoast = np.concatenate((Ycoast, Y, [np.nan]))
    else:  # Multipolygon, handle each polygon separately
        for p in mp.geoms:
            X, Y = grid.ll2xy(*p.boundary.xy)
            Xcoast = np.concatenate((Xcoast, X, [np.nan]))
            Ycoast = np.concatenate((Ycoast, Y, [np.nan]))
# Remove the final NaNs
Xcoast, Ycoast = Xcoast[:-1], Ycoast[:-1]

# Save the coastline to a npy file
with open(coast_file, "wb") as f:
    np.save(f, Xcoast)
    np.save(f, Ycoast)

# Test, simple plot of the coast
plt.fill(Xcoast, Ycoast)
plt.show()
