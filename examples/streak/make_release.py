# Make a particles.in file for a streak line
# Continuous release from single source

from numpy import datetime64

start_time = datetime64("1989-05-24 12")
# stop_time = datetime64('1989-06-15 13')   # Extra hour to get last time

# Release point in grid coordinates
x, y = 115, 100
z = 5

with open("streak.rls", mode="w") as fid:
    fid.write(f"{start_time:s} {x:7.3f} {y:7.3f} {z:6.1f}\n")
