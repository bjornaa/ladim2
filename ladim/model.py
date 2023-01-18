"""Module containing the LADiM Model class definition"""

from pathlib import Path
import importlib

import logging
import types
from typing import Any, Optional
import numpy as np

from ladim.state import State
from ladim.grid import BaseGrid
from ladim.timekeeper import TimeKeeper
from ladim.forcing import BaseForce
from ladim.tracker import Tracker
from ladim.release import ParticleReleaser
from ladim.warm_start import warm_start
from ladim.output import BaseOutput
from ladim.ibm import IBM

DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Model:
    """A complete LADiM model"""

    def __init__(self, config: dict[str, Any]) -> None:
        # Initialize submodules
        self.modules: dict[str, Any] = dict()
        module_names = [
            "state",
            "time",
            "grid",
            "forcing",
            "release",
            "tracker",
            "ibm",
            "output",
        ]
        for name in module_names:
            self.modules[name] = init_module(name, config[name], self.modules)

        # Define shorthand for individual modules
        self.state: State = self.modules["state"]
        self.timer: TimeKeeper = self.modules["time"]
        self.grid: BaseGrid = self.modules["grid"]
        self.force: BaseForce = self.modules["forcing"]
        self.tracker: Tracker = self.modules["tracker"]
        self.release: ParticleReleaser = self.modules["release"]
        self.output: BaseOutput = self.modules["output"]
        self.ibm: IBM = self.modules["ibm"]

        self.skip_first_output = False
        if config["warm_start"]:
            logger.info("Executing warm start")
            D = config["warm_start"]
            warm_start(D["filename"], D["variables"], self.state)
            self.timer.step = 0
            self.timer.time = self.timer.step2time(self.timer.step)
            self.release.update()
            self.force.update()
            self.tracker.update()
            self.ibm.update()

    def update(self) -> None:
        """Update the model to the next time step"""

        self.timer.update()
        step = self.timer.step

        logger.info("step, model time: %4d %s", step, self.timer.time)

        self.release.update()
        self.force.update()

        # self.state.compactify()
        if step >= 0:
            self.output.update()

        # --- Update state to next time step
        # Improve: no need to update after last write
        self.tracker.update()
        self.ibm.update()

    def finish(self) -> None:
        """Clean-up after the model run"""
        module_names = ["grid", "forcing", "release", "tracker", "ibm", "output"]
        for name in module_names:
            module = self.modules[name]
            if hasattr(module, "close") and callable(module.close):
                module.close()


def init_module(
    module_name: str,
    conf_dict: dict[str, Any],
    all_modules_dict: Optional[dict[str, Any]] = None,
) -> Any:
    """Initiate the main class in one of the modules"""
    default_module_names = dict(
        output="ladim.out_netcdf",
        release="ladim.release",
        grid="ladim.ROMS",
        time="ladim.timekeeper",
        forcing="ladim.ROMS",
        tracker="ladim.tracker",
        state="ladim.state",
        ibm="ladim.ibm",
    )
    default_module_name = default_module_names[module_name]

    main_class_names = dict(
        output="Output",
        time="TimeKeeper",
        release="ParticleReleaser",
        grid="Grid",
        forcing="Forcing",
        tracker="Tracker",
        state="State",
        ibm="IBM",
    )
    main_class_name = main_class_names[module_name]

    module_name = conf_dict.get("module", default_module_name)
    module_object = load_module(module_name)
    MainClass = getattr(module_object, main_class_name)

    if "module" in conf_dict:
        del conf_dict["module"]

    return MainClass(modules=all_modules_dict, **conf_dict)


def load_module(module_name: str) -> types.ModuleType:
    """Load LADiM module

    Modules are given as paths (absolute or relative to the working directory) to
    python modules (with or without .py extension) or modules on the ordinary python
    search path. The former takes precedence.

    """

    module_name = module_name.removesuffix(".py")
    file_name = Path(module_name + ".py")

    # First try to load the module from a file
    if file_name.exists():

        basename = file_name.stem
        internal_name = "ladim_custom_" + basename  # To avoid naming collisions
        spec = importlib.util.spec_from_file_location(internal_name, file_name)  # type: ignore
        module_object = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module_object)  # type: ignore
        return module_object

    # Secondly, try ordinary importing module from sys.path
    try:
        return importlib.import_module(module_name)

    # Nothing worked
    except ModuleNotFoundError as err:
        logging.critical("Can not find module %s", module_name)
        raise SystemExit from err
