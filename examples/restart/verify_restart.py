"""Verify that output split and warm start works correctly"""

from postladim import ParticleFile


# Note: t = 9, pid = 3 produces non-identical X-value for restart

# Choose time step number (0 <= t <= 12) in unsplit output
t = 9

# Choose particle number (0 <= pid <= t)
pid = 3

# Specify value used in split.yaml and restart.yaml
numrec = 4

# restart file number (0 <= n <= 3)
# i.e. warm_start_file = f"split_{n:03d}.nc" in restart.yaml
n = 1

# Reference values from unsplit simulation
with ParticleFile("unsplit.nc") as pf:
    time_ = pf.time(t)
    x = float(pf.X[t][pid])
    y = float(pf.Y[t][pid])

# Corresponding file number and record in split output
file_number, tsplit = divmod(t, numrec)

# Check splitting
with ParticleFile(f"split_{file_number:03d}.nc") as pf:
    assert pf.time(tsplit) == time_
    assert float(pf.X[tsplit][pid]) == x
    assert float(pf.Y[tsplit][pid]) == y

# Check restart
with ParticleFile(f"split_{file_number:03d}.nc") as pf:
    assert pf.time(tsplit) == time_
    assert float(pf.X[tsplit][pid]) == x
    assert float(pf.Y[tsplit][pid]) == y
