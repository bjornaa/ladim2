import sys
import os
import platform
from pathlib import Path
import logging
import datetime
import subprocess
from typing import Union

from . import __version__, __file__
from .state import State
from .grid import init_grid
from .timekeeper import TimeKeeper, duration2iso

from .forcing import init_force
from .tracker import Tracker
from .release import ParticleReleaser
from .output import init_output
from .configure import configure
from .ibm import init_IBM
from .warm_start import warm_start


def main(configuration_file: Union[Path, str], loglevel: int = logging.INFO) -> None:
    """Main function for complete particle tracking model"""

    wall_clock_start = datetime.datetime.now()

    # ---------------------
    # Logging
    # ---------------------

    # set master log level
    logging.basicConfig(level=loglevel, format="%(levelname)s:%(module)s - %(message)s")
    logger = logging.getLogger("main")

    # ----------------
    # Start message
    # ----------------

    # Set log level at least to INFO for the main program
    logger.setLevel(min(logging.INFO, loglevel))

    logger.debug(f"Host machine: {platform.node()}")
    logger.debug(f"Platform: {platform.platform()}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if conda_env:
        logger.info(f"conda environment: {conda_env}")
    logger.info(f"python executable: {sys.executable}")
    logger.info(f"python version:  {sys.version.split()[0]}")
    logger.info(f"LADiM version: {__version__}")
    logger.debug(f"LADiM path: {Path(__file__).parent}\n")
    logger.info(f"Configuration file: {configuration_file}")
    logger.info(f"Loglevel: {logging.getLevelName(loglevel)}")
    logger.info(f"Wall clock start time: {wall_clock_start:%Y-%m-%d %H:%M:%S}")

    # ----------------
    # Configuration
    # ----------------

    config = configure(configuration_file)

    # -------------------
    # Initialization
    # -------------------

    logger.info("Initiating")
    state = State(**config["state"])
    timer = TimeKeeper(**config["time"])
    grid = init_grid(**config["grid"])
    force = init_force(grid=grid, timer=timer, **config["forcing"])
    tracker = Tracker(**config["tracker"])
    release = ParticleReleaser(
        timer=timer, datatypes=state.dtypes, grid=grid, **config["release"]
    )
    output = init_output(
        timer=timer,
        grid=grid,
        num_particles=release.total_particle_count,
        **config["output"],
    )
    if config["ibm"]:
        ibm = init_IBM(
            timer=timer, state=state, forcing=force, grid=grid, **config["ibm"]
        )

    # --------------------------
    # Initial particle release
    # --------------------------

    logger.debug("Initial particle release")

    # Number of time steps between output (have that computed in output.py)
    output_period_step = output.output_period / timer.dt

    # Warm start?
    if config["warm_start"]:
        D = config["warm_start"]
        warm_start(D["filename"], D["variables"], state)

    # --------------
    # Time loop
    # --------------

    logger.info("Starting time loop")
    for step in range(timer.Nsteps + 1):

        # Update
        # -- Update clock ---
        if step > 0:
            timer.update()
        logger.debug(f"step, model time: {step:4d}, {timer.time}")

        # --- Particle release
        if step in release.steps:
            V = next(release)
            state.append(**V)

        # --- Update forcing ---
        force.update(step, state.X, state.Y, state.Z)

        if config["ibm"]:
            ibm.update()  # type: ignore
            state.compactify()

        # --- Output
        if step % output_period_step == 0:
            output.write(state)

        # --- Update state to next time step
        # Improve: no need to update after last write
        tracker.update(state, grid=grid, force=force)

    # --------------
    # Finalisation
    # --------------

    logger.setLevel(logging.INFO)

    # logger.info("Cleaning up")
    # output.write_particle_variables(state)
    # output.close()
    force.close()
    wall_clock_stop = datetime.datetime.now()
    logger.info(f"Wall clock stop time:  {wall_clock_stop:%Y-%m-%d %H:%M:%S}")
    delta = wall_clock_stop - wall_clock_start
    logger.info(f"Wall clock running time: {duration2iso(delta)}")


def script():
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="LADiM 2.0 — Lagrangian Advection and Diffusion Model"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Show more information",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Show less information",
        action="store_const",
        dest="loglevel",
        const=logging.WARNING,
        default=logging.INFO,
    )
    parser.add_argument("config_file", nargs="?", default="ladim2.yaml")

    args = parser.parse_args()

    # Start up message
    print(" ========================================================")
    print(" === LADiM – Lagrangian Advection and Diffusion Model ===")
    print(" ========================================================\n")

    main(args.config_file, loglevel=args.loglevel)
