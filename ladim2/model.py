"""Module containing the LADiM Model class definition"""

import logging
from typing import Dict, Any, Optional


# from ladim2 import __version__, __file__
from ladim2.state import State
from ladim2.grid import init_grid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import init_force
from ladim2.tracker import Tracker
from ladim2.release import ParticleReleaser
from ladim2.warm_start import warm_start
from ladim2.output import init_output

# from ladim2.configure import configure
from ladim2.ibm import init_IBM, BaseIBM


DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Model:
    """A complete LADiM model"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.state = State(**config["state"])
        self.timer = TimeKeeper(**config["time"])
        self.grid = init_grid(**config["grid"])
        self.force = init_force(grid=self.grid, timer=self.timer, **config["forcing"])
        self.tracker = Tracker(**config["tracker"])
        self.release = ParticleReleaser(
            timer=self.timer,
            datatypes=self.state.dtypes,
            grid=self.grid,
            **config["release"],
        )
        self.output = init_output(
            timer=self.timer,
            grid=self.grid,
            num_particles=self.release.total_particle_count,
            **config["output"],
        )
        if config["ibm"]:
            self.ibm: Optional[BaseIBM] = init_IBM(
                timer=self.timer,
                state=self.state,
                forcing=self.force,
                grid=self.grid,
                **config["ibm"],
            )
        else:
            self.ibm = None

        if config["warm_start"]:
            D = config["warm_start"]
            warm_start(D["filename"], D["variables"], self.state)

    def update(self, step):
        """Update the model to the next time step"""
        if step > 0:
            self.timer.update()
        logger.debug("step, model time: %4d %s", step, self.timer.time)

        # --- Particle release
        if step in self.release.steps:
            V = next(self.release)
            self.state.append(**V)

        # --- Update forcing ---
        self.force.update(step, self.state.X, self.state.Y, self.state.Z)

        if self.ibm is not None:
            self.ibm.update()  # type: ignore
            self.state.compactify()

        # --- Output
        if step % self.output.output_period_step == 0:
            self.output.write(self.state)

        # --- Update state to next time step
        # Improve: no need to update after last write
        self.tracker.update(self.state, grid=self.grid, force=self.force)

    def finish(self):
        """Clean-up after the model run"""
        self.force.close()
        # output.close()
