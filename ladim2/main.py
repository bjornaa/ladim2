from pathlib import Path
from typing import Union

from .state import State
from .grid import init_grid
from .timekeeper import TimeKeeper
from .forcing import init_force
from .tracker import Tracker
from .release import ParticleReleaser
from .output import Output
from .configure import configure
from .ibm import init_IBM


def main(configuration_file: Union[Path, str]) -> None:
    """Main function for complete particle tracking model"""

    # ----------------
    # Configuration
    # ----------------

    config = configure(configuration_file)

    # -------------------
    # Initialization
    # -------------------

    print("Initiating")
    state = State(**config["state"])
    timer = TimeKeeper(**config["time"])
    grid = init_grid(**config["grid"])
    force = init_force(grid=grid, timer=timer, **config["forcing"])
    tracker = Tracker(**config["tracker"])
    release = ParticleReleaser(timer=timer, datatypes=state.dtypes, **config["release"])
    output = Output(
        timer=timer, num_particles=release.total_particle_count, **config["output"]
    )
    if config["ibm"]:
        ibm = init_IBM(timer=timer, **config["ibm"])

    # --------------------------
    # Initial particle release
    # --------------------------

    # Skal automatisere tilpasning til state-variablene
    # Også initiering av variable som ikke er i release-filen
    # X0 er et eksempel.
    print("Initial particle release")

    # --------------
    # Time loop
    # --------------

    # Number of time steps between output (have that computed in output.py)
    output_period_step = output.output_period / timer._dt

    # Initial particle release and output
    step = 0
    if 0 in release.steps:
        V = next(release)
        state.append(**V)
    if not output.skip_initial:
        output.write(state)

    print("Time loop")
    while step < timer.Nsteps:

        # Update
        # --- Update forcing ---  (forcing først)
        force.update(step, state.X, state.Y, state.Z)

        tracker.update(state, grid=grid, force=force)
        if config["ibm"]:
            ibm.update(grid, state, force)
            state.compactify()

        step += 1

        # --- Particle release and output
        if step in release.steps:
            V = next(release)
            state.append(**V)
        if step % output_period_step == 0:
            output.write(state)

    # --------------
    # Finalisation
    # --------------

    print("Cleaning up")
    # output.write_particle_variables(state)
    # output.close()
