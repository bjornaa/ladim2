# Continuous release,
# 1 particle per hour from 2 locations
# Fixed depth at 5 m

f0 = open("anlegg.dat")
f1 = open("salmon_lice.rls", mode="w")

mult = 300

next(f0)  # Skip initial line
f1.write("mult release_time      X      Y      Z  farmid   super\n")
for line in f0:
    w = line.split(",")
    farmid = int(w[0])
    x = float(w[1])
    y = float(w[2])
    z = 5
    super_ = float(w[4])
    timestamp = w[5]
    f1.write(
        f"{mult:4d}  {timestamp:s} {x:6.1f} {y:6.1f} {z:6.1f}   {farmid:5d} {super_:7.1f}\n"
    )

f1.close()
