from postladim import ParticleFile

pf0 = ParticleFile("unsplit.nc")

# --- Split ---

pf1 = ParticleFile("split_002.nc")
assert pf1.time[1] == pf0.time[9]
assert all(pf1.X[1] == pf0.X[9])
assert all(pf1.weight[1] == pf0.weight[9])

# --- Restart ---

pf2 = ParticleFile("restart_002.nc")

# Timing
assert pf2.num_times == pf1.num_times
assert pf2.time[0] == pf1.time[0]

# Number of particles and pid
assert all(pf2.count == pf1.count)
assert all(pf2.pid[0] == pf1.pid[0])

# instance variables
assert all(abs(pf2.X[1] - pf1.X[1]) < 1e-6)
assert all(abs(pf1.temp[0] - pf1.temp[0]) < 1e-6)
assert all(abs(pf2.age[1] - pf1.age[1]) < 1e-6)
assert all(abs(pf1.weight[0] - pf1.weight[0]) < 1e-6)
assert all(abs(pf1.weight[3] - pf1.weight[3]) < 1e-6)
