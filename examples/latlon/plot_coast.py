"""Simple script to check that the coast line from make_coast is OK"""

import matplotlib.pyplot as plt
import shapely.wkb

# Read the wkb file into a shapely MultiPolygon
with open("coast.wkb", mode="rb") as fid:
    mpoly = shapely.wkb.load(fid)

# Filled plots of the separate polygons
for polygon in mpoly.geoms:
    plt.fill(*polygon.boundary.xy)

plt.show()
