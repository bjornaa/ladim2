"""Save a regional part of GSHHS  to a wkb-file

"""

# ----------------------------------
# Bjørn Ådlandsvik <bjorn@imr.no>
# Institute if Marine Research
# 2021-01-15
# ----------------------------------

from itertools import chain

import cartopy.io.shapereader as shapereader
from shapely import geometry, wkb

# Choose between c, l, i, h, f resolutions
# Crude, Low, Intermediate, High, or Full
GSHHS_resolution = "i"

# Name of output coast file
fname = "coast.wkb"

# Define regional domain
lonmin, lonmax, latmin, latmax = -6, 12, 54, 63  # North Sea

# Global coastline from GSHHS as shapely collection generator
path = shapereader.gshhs(scale=GSHHS_resolution)
coast = shapereader.Reader(path).geometries()

# Restrict the coastline to the regional domain
bbox = geometry.box(lonmin, latmin, lonmax, latmax)
coast = (bbox.intersection(p) for p in coast if bbox.intersects(p))

# Filter out isolated points
coast = filter(
    lambda p: isinstance(p, (geometry.MultiPolygon, geometry.Polygon)),
    coast,
)

# The filtered intersection can consist of both polygons, multipolygons
# which may not be dumped correctly to file.
# First make a generator expression of lists of polygons
# and thereafter one large MultiPolygon

coast = (p.geoms if isinstance(p, geometry.MultiPolygon) else [p] for p in coast)
coast = geometry.MultiPolygon(chain.from_iterable(coast))

# Save to WKB file
with open(fname, mode="wb") as fp:
    wkb.dump(coast, fp, output_dimension=2)
