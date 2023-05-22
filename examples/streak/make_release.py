# Make a particles.in file for a streak line
# Continuous release from single source

from pathlib import Path

from numpy import datetime64

release_file = Path("streak.rls")

start_time = datetime64("1989-05-24 12")

# Release point in grid coordinates
x, y = 115, 100
z = 5

with release_file.open(mode="w") as fid:
    fid.write(f"{start_time:s} {x:7.3f} {y:7.3f} {z:6.1f}\n")
