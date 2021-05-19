"""Module containing the LADiM Model class definition"""

import logging
from typing import Dict, Any, Optional


# from ladim2 import __version__, __file__
from ladim2.state import State
from ladim2.grid import BaseGrid
from ladim2.timekeeper import TimeKeeper
from ladim2.forcing import BaseForce
from ladim2.tracker import Tracker
from ladim2.release import ParticleReleaser
from ladim2.warm_start import warm_start
from ladim2.output import BaseOutput

# from ladim2.configure import configure
from ladim2.ibm import IBM


DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Model:
    """A complete LADiM model"""

    def __init__(self, config: Dict[str, Any]) -> None:
        # Initialize submodules
        self.modules = dict()
        module_names = ['state', 'time', 'grid', 'forcing', 'release', 'tracker', 'ibm', 'output']
        for name in module_names:
            self.modules[name] = init_module(name, config[name], self.modules)

        # Define shorthand for individual modules
        self.state: State = self.modules['state']
        self.timer: TimeKeeper = self.modules['time']
        self.grid: BaseGrid = self.modules['grid']
        self.force: BaseForce = self.modules['forcing']
        self.tracker: Tracker = self.modules['tracker']
        self.release: ParticleReleaser = self.modules['release']
        self.output: BaseOutput = self.modules['output']
        self.ibm: IBM = self.modules['ibm']

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
        self.force.update()

        self.ibm.update()  # type: ignore

        # self.state.compactify()

        # --- Output
        if step % self.output.output_period_step == 0:
            self.output.write(self.state)

        # --- Update state to next time step
        # Improve: no need to update after last write
        self.tracker.update()

    def finish(self):
        """Clean-up after the model run"""
        self.force.close()
        # output.close()


def init_module(module_name, conf_dict, all_modules_dict: dict = None) -> Any:
    default_module_names = dict(
        output='ladim2.out_netcdf',
        release='ladim2.release',
        grid='ladim2.ROMS',
        time='ladim2.timekeeper',
        forcing='ladim2.ROMS',
        tracker='ladim2.tracker',
        state='ladim2.state',
        ibm='ladim2.ibm',
    )
    default_module_name = default_module_names[module_name]

    main_class_names = dict(
        output='Output',
        time='TimeKeeper',
        release='ParticleReleaser',
        grid='Grid',
        forcing='Forcing',
        tracker='Tracker',
        state='State',
        ibm='IBM',
    )
    main_class_name = main_class_names[module_name]

    from ladim2.configure import load_module
    module_name = conf_dict.get('module', default_module_name)
    module_object = load_module(module_name)
    MainClass = getattr(module_object, main_class_name)

    if 'module' in conf_dict:
        del conf_dict['module']

    return MainClass(modules=all_modules_dict, **conf_dict)
