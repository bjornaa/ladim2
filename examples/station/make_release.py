# Make a particles.in file for a streak line
# Continuous release from single source

from numpy import linspace

start_time = "1989-05-24T12"

num_particles = 1001
# Release point in grid coordinates
x, y = 115, 100
zmax = 200

Z = linspace(zmax, 0, num_particles)

with open("station.rls", mode="w") as f:
    for z in Z:
        f.write(f"{start_time:s} {x:7.3f} {y:7.3f} {z:6.2f}\n")
