# Make a particles.in file for a streak line
# Continuous release from single source

from numpy import datetime64

start_time = datetime64("1989-05-24 12")

# Release point in grid coordinates
x, y = 115, 100
# Release deptj
z = 5

with open("killer.rls", mode="w") as f:
    f.write(f"{start_time:s} {x:7.3f} {y:7.3f} {z:6.1f}\n")
