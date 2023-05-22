"""Make a particle release file for the line example"""

from pathlib import Path

import numpy as np

version = 2  # Use 1 for no header line

release_file = Path("line1.rls") if version == 1 else Path("line.rls")

# End points of line in grid coordinates
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90

# Number of particles along the line
Npart = 1000

# Fixed particle depth
Z = 5

X = np.linspace(x0, x1, Npart)
Y = np.linspace(y0, y1, Npart)

with release_file.open(mode="w") as f:
    if version != 1:
        f.write("release_time   X       Y         Z\n")
    for x, y in zip(X, Y):
        f.write(f"1989-05-24T12 {x:7.3f} {y:7.3f} {Z:6.1f}\n")
