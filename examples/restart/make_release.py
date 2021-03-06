# Make a particles.in file for a streak line
# Continuous release from single source

from numpy import datetime64

start_time = datetime64("1989-05-24 12")

# Release point in grid coordinates
x, y = 115, 100
z = 5

with open("restart.rls", mode="w") as f:
    f.write("{:s} {:7.3f} {:7.3f} {:6.1f}\n".format(start_time, x, y, z))
