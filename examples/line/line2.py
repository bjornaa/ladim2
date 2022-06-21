"""Main function for running LADiM as an application"""

from ladim.configure import configure
from ladim.model import Model

# ----------------
# Configuration
# ----------------

configuration_file = "ladim.yaml"
config_version = 2
config = configure(configuration_file, config_version)

# -------------------
# Initialization
# -------------------

model = Model(config)

# --------------
# Time loop
# --------------

for step in range(model.timer.Nsteps + 1):
    model.update(step)

# --------------
# Finalisation
# --------------

model.finish()
