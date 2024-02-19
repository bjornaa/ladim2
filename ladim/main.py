"""Main function for running LADiM as an application"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import platform
import sys
import typing
from pathlib import Path
from typing import Union

import rich.highlighter
import rich.logging

from ladim import __file__, __version__
from ladim.configure import configure
from ladim.model import Model

# from ladim.warm_start import warm_start
from ladim.timekeeper import duration2iso


def main(
    configuration_file: Union[Path, str],
    loglevel: int = logging.INFO,
    # config_version: Literal[1, 2] = 2,
) -> None:
    """Main function for LADiM

    args:
        configuration_file:
            Path to yaml configuration

        loglevel:
            Log level

    """

    wall_clock_start = datetime.datetime.now()

    # ---------------------
    # Logging
    # ---------------------

    loghandler = rich.logging.RichHandler(
        show_time=False,
        show_path=False,
        markup=True,
        highlighter=rich.highlighter.NullHighlighter(),
    )
    loghandler.setFormatter(LadimLogFormatter())

    logging.basicConfig(
        level=loglevel,
        handlers=[loghandler],
    )

    logger = logging.getLogger("main")

    # ----------------
    # Start info
    # ----------------

    # Set log level at least to INFO for the main function
    logger.setLevel(min(logging.INFO, loglevel))

    logger.debug("Host machine: %s", platform.node())
    logger.debug("Platform: %s", platform.platform())
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", None)
    if conda_env:
        logger.info("conda environment: %s", conda_env)
    logger.info("python executable: %s", sys.executable)
    logger.info("python version: %s", sys.version.split()[0])
    logger.info("LADiM version: %s", __version__)
    logger.debug("LADiM path: %s\n", Path(__file__).parents[1])
    logger.info("Configuration file: %s", configuration_file)
    logger.info("Loglevel: %s", logging.getLevelName(loglevel))
    logger.info("Wall clock start time: %s\n", str(wall_clock_start)[:-7])

    # ----------------
    # Configuration
    # ----------------

    config = configure(configuration_file)

    # -------------------
    # Initialization
    # -------------------

    logger.info("Initiating")
    model = Model(config)

    # --------------------------
    # Initial particle release
    # --------------------------

    logger.debug("Initial particle release")

    # --------------
    # Time loop
    # --------------

    logger.info("Starting time loop")
    # for step in range(model.timer.Nsteps + 1):
    for _step in range(model.timer.Nsteps):
        model.update()

    # --------------
    # Finalisation
    # --------------

    logger.setLevel(logging.INFO)

    logger.info("Cleaning up")
    model.finish()

    wall_clock_stop = datetime.datetime.now()
    logger.info("Wall clock stop time: %s", str(wall_clock_stop)[:-7])
    delta = wall_clock_stop - wall_clock_start
    logger.info("Wall clock running time: %s", duration2iso(delta))


def script() -> None:
    """Function for running LADiM as a command line application"""

    parser = argparse.ArgumentParser(
        description="LADiM 2.0 — Lagrangian Advection and Diffusion Model"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Show debug information",
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
    parser.add_argument(
        "-v", "--version", help="Configuration format version", type=int, default=2
    )
    parser.add_argument(
        "config_file", nargs="?", help="Configuration file", default="ladim.yaml"
    )

    args = parser.parse_args()

    # Start up message
    print("")
    print(" ========================================================")
    print(" === LADiM – Lagrangian Advection and Diffusion Model ===")  # noqa: RUF001
    print(" ========================================================")
    print("")

    main(args.config_file, loglevel=args.loglevel)


class LadimLogFormatter(logging.Formatter):
    """Set up coloured log messages for LADiM"""

    logformat = "%(module)s - %(message)s"

    formats: typing.ClassVar = {
        logging.DEBUG: "[green]" + logformat,
        logging.INFO: logformat,
        logging.WARNING: "[magenta]" + logformat,
        logging.ERROR: "[bold bright_red]" + logformat,
        logging.CRITICAL: "[bold white on red]" + logformat,
    }

    def format(self, record):
        log_fmt = self.formats[record.levelno]
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
