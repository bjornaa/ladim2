"""Compare output from unsplit, split, and restart runs"""

from postladim import ParticleFile

print("\n --- unsplit ---")
t = 10
with ParticleFile("unsplit.nc") as pf:
    print("time = ", pf.time(t))
    print("X[4] ", float(pf.X[t][4]))

print("\n --- split ---")
t = 2
with ParticleFile("split_002.nc") as pf:
    print("time = ", pf.time(t))
    print("X[4] ", float(pf.X[t][4]))

print("\n --- restart ---")
t = 2
with ParticleFile("restart_002.nc") as pf:
    print("time = ", pf.time(t))
    print("X[4] ", float(pf.X[t][4]))
