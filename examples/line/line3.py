"""Module containing the LADiM Model class definition"""

from contextlib import suppress

from ladim.configure import configure
from ladim.model import init_module

configuration_file = "ladim.yaml"
config = configure(configuration_file)

# The module names can be taken from the configuration file.
# But, the order matters
module_names = [
    "state",
    "time",
    "grid",
    "forcing",
    "release",
    "output",
    "tracker",
]

# Initiate
modules = dict()
for name in module_names:
    modules[name] = init_module(name, config[name], modules)

# Time stepping
Nsteps = modules["time"].Nsteps
for step in range(Nsteps + 1):
    if step > 0:
        modules["time"].update()
    modules["release"].update()
    modules["forcing"].update()
    modules["output"].update()
    if step < Nsteps:
        modules["tracker"].update()

# Clean up
with suppress(AttributeError, KeyError):
    for name in module_names:
        modules[name].close()
