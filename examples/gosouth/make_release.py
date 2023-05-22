# Make an particles.in file

import numpy as np

# End points of line in grid coordinates
x0, x1 = 63.55, 123.45
y0, y1 = 90.0, 90

# Number of particles along the line
Npart = 1000

# Fixed particle depth
Z = 5

X = np.linspace(x0, x1, Npart)
Y = np.linspace(y0, y1, Npart)

with open("line.rls", mode="w") as fid:
    for _i, (x, y) in enumerate(zip(X, Y)):
        fid.write(f"1989-05-24T12 {x:7.3f} {y:7.3f} {Z:6.1f}\n")
