"""Main function for running LADiM as an application"""

from ladim.configure import configure
from ladim.model import Model

# ----------------
# Configuration
# ----------------

configuration_file = "ladim.yaml"
config = configure(configuration_file)

# -------------------
# Initialization
# -------------------

model = Model(config)

# --------------
# Time loop
# --------------

for step in range(model.timer.Nsteps):
    model.update()

# --------------
# Finalisation
# --------------

model.finish()
