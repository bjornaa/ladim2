"""Make a coast line in grid coordinates"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute if Marine Research
# 2021-02-10
# ----------------------------------

import numpy as np
import scipy
from netCDF4 import Dataset

from shapely import geometry
import cartopy.io.shapereader as shapereader

from ladim.ROMS import Grid

# Choose between c, l, i, h, f resolutions
GSHHS_resolution = "i"

# Grid file
grid_file = "../data/ocean_avg_0014.nc"

# Name of output coast file
coast_file = "coast_grid.npy"


grid = Grid(grid_file)

with Dataset(grid_file) as nc:
    glon = nc.variables["lon_rho"][:, :]
    glat = nc.variables["lat_rho"][:, :]
jmax, imax = glon.shape


# Boundary (grid coordinates)
res = 50
eps = 5  # Stay away from boundary errors
xmin, xmax = grid.xmin + eps, grid.xmax - eps
ymin, ymax = grid.ymin + eps, grid.ymax - eps
xbd = np.concatenate(
    (
        np.linspace(xmin, xmax, res),
        res * [xmax],
        np.linspace(xmax, xmin, res),
        res * [xmin],
    )
)
ybd = np.concatenate(
    (
        res * [ymin],
        np.linspace(ymin, ymax, res),
        res * [ymax],
        np.linspace(ymax, ymin, res),
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

# Xcoast, Ycoast = np.array([]), np.array([])
# for mp in coast:
#     if isinstance(mp, geometry.Polygon):
#         X, Y = grid.ll2xy(*mp.boundary.xy)
#         Xcoast = np.concatenate((Xcoast, X, [np.nan]))
#         Ycoast = np.concatenate((Ycoast, Y, [np.nan]))
#     else:  # MultiPolygon
#         for p in mp.geoms:
#             X, Y = grid.ll2xy(*p.boundary.xy)
#             Xcoast = np.concatenate((Xcoast, X, [np.nan]))
#             Ycoast = np.concatenate((Ycoast, Y, [np.nan]))
# # remove final nans
# Xcoast, Ycoast = Xcoast[:-1], Ycoast[:-1]

# Flatten to a list of polygons
A = []
for mp in coast:
    if isinstance(mp, geometry.Polygon):
        A.append(mp)
    else:
        A.extend(mp.geoms)

V = np.column_stack([glon.ravel(), glat.ravel()])
gx, gy = np.meshgrid(np.arange(imax), np.arange(jmax))

Xcoast, Ycoast = np.array([]), np.array([])
for p in A:
    # X, Y = grid.ll2xy(*p.boundary.xy)
    ll = np.column_stack((p.boundary.xy[0], p.boundary.xy[1]))
    X = scipy.interpolate.griddata(V, gx.ravel(), ll)
    Y = scipy.interpolate.griddata(V, gy.ravel(), ll)
    Xcoast = np.concatenate((Xcoast, X, [np.nan]))
    Ycoast = np.concatenate((Ycoast, Y, [np.nan]))
# remove final nans
Xcoast, Ycoast = Xcoast[:-1], Ycoast[:-1]


# Save the coastline to a npy file
with open(coast_file, "wb") as f:
    np.save(f, Xcoast)
    np.save(f, Ycoast)


# # Save to WKB file
# with open(fname, mode="wb") as fp:
#     wkb.dump(coast, fp, output_dimension=2)
