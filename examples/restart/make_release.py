# Make a particles.in file for a streak line
# Continuous release from single source

from numpy import datetime64

start_time = datetime64("1989-05-24 12")

# Release point in grid coordinates
x, y = 115, 100
z = 5
weight = 2.0

with open("restart.rls", mode="w") as f:
    f.write("mult release_time  X      Y        Z     weight\n")
    f.write(f"1    {start_time} {x:6.2f} {y:6.2f} {z:6.2f} {weight:4.1f}\n")
