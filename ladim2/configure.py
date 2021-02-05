""""

Configuration reader for LADiM version 2
with compability wrapper for LADiM version 1 configuration

"""

# -----------------------------------
# Bjørn Ådlandsvik <bjorn@hi.no>
# Institute of Marine Research
# December 2020
# -----------------------------------

from pathlib import Path
from pprint import pprint

import numpy as np
from netCDF4 import Dataset, num2date

from typing import Union, Dict, Any
import yaml

# from .timekeeper import normalize_period

DEBUG = False


def configure(config_file: Union[Path, str]) -> Dict[str, Any]:
    """Main configuration function"""

    with open(config_file) as fid:
        config: Dict[str, Any] = yaml.safe_load(fid)

    # Assume version >= 2 has explicit version tag
    # Consider alternative: using ladim2 -v1 xxx.yaml
    if "version" not in config:
        config = configure_v1(config)
        # pprint(config)

    # Some sections may be missing
    if "state" not in config:
        config["state"] = dict()
    if "grid" not in config:
        config["grid"] = dict()
    if "ibm" not in config:
        config["ibm"] = dict()
    if "warm_start" not in config:
        config["warm_start"] = dict()

    # Handle non-orthogonality

    # Use time step from time
    config["tracker"]["dt"] = config["time"]["dt"]
    # if config["ibm"]:
    #    config["ibm"]["dt"] = normalize_period(config["time"]["dt"])

    # If missing grid["filename"] use forcing["filename"]
    if "module" not in config["grid"]:
        config["grid"]["module"] = config["forcing"]["module"]
    if "filename" not in config["grid"]:
        filename = Path(config["forcing"]["filename"])
        # glob if necessary and use first file
        if ("*" in str(filename)) or ("?" in str(filename)):
            directory = filename.parent
            filename = sorted(directory.glob(filename.name))[0]
        config["grid"]["filename"] = filename

    # Warm start
    if "filename" in config["warm_start"]:
        warm_start_file = config["warm_start"]["filename"]
        # Warm start overrides start time
        try:
            nc = Dataset(warm_start_file)
        except (FileNotFoundError, OSError):
            # logging.error(f"Could not open warm start file,{warm_start_file}")
            raise SystemExit(1)
        tvar = nc.variables["time"]
        # Use last record in restart file
        warm_start_time = np.datetime64(num2date(tvar[-1], tvar.units))
        warm_start_time = warm_start_time.astype("M8[s]")
        config["time"]["start"] = warm_start_time
        # logging.info(f"    Warm start at {warm_start_time}")

        # warm start -> release
        config["release"]["warm_start_file"] = config["warm_start"]["filename"]

    # Possible improvement: write a yaml-file
    if DEBUG:
        pprint(config)

    return config


# --------------------------------------------------------------------


def configure_v1(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert version 1 configuration file

    It is possible to have configuration files that works with both ladim1 and ladim2.
    Unfortunately, it is also possible to have configuration files that works with
    ladim1 and not ladim2.
    Ordinary use cases at IMR should work

    """

    conf: Dict[str, Any] = dict()  # output version 2 configuration
    # conf["version"] = 1.0

    # time
    conf["time"] = dict(
        start=config["time_control"]["start_time"],
        stop=config["time_control"]["stop_time"],
        dt=config["numerics"]["dt"],
    )
    if "reference_time" in config["time_control"]:
        conf["time"]["reference"] = config["time_control"]["reference_time"]

    # grid and forcing
    conf["grid"] = dict()
    conf["forcing"] = dict()
    if "ladim.gridforce.ROMS" in config["gridforce"]["module"]:
        conf["grid"]["module"] = "ladim2.grid_ROMS"
        if "gridfile" in config["gridforce"]:
            conf["grid"]["filename"] = config["gridforce"]["gridfile"]
        elif "gridfile" in config["files"]:
            conf["grid"]["filename"] = config["files"]["gridfile"]
        conf["forcing"]["module"] = "ladim2.forcing_ROMS"
        if "input_file" in config["gridforce"]:
            conf["forcing"]["filename"] = config["gridforce"]["input_file"]
        elif "input_file" in config["files"]:
            conf["forcing"]["filename"] = config["files"]["input_file"]
    if "subgrid" in config["gridforce"]:
        conf["grid"]["subgrid"] = config["gridforce"]["subgrid"]
    if "ibm_forcing" in config["gridforce"]:
        conf["forcing"]["ibm_forcing"] = config["gridforce"]["ibm_forcing"]

    # state
    conf["state"] = dict()
    instance_variables = dict()
    particle_variables = dict()
    if "ibm" in config and "variables" in config["ibm"]:
        for var in config["ibm"]["variables"]:
            instance_variables[var] = "float"
    for var in config["particle_release"]:
        if var in ["mult", "X", "Y", "Z"]:  # Ignore
            continue
        if (
            "particle_variables" in config["particle_release"]
            and var in config["particle_release"]["particle_variables"]
        ):
            particle_variables[var] = config["particle_release"].get(var, "float")
    conf["state"]["instance_variables"] = instance_variables
    conf["state"]["particle_variables"] = particle_variables
    conf["state"]["default_values"] = dict()
    for var in conf["state"]["instance_variables"]:
        conf["state"]["default_values"][var] = 0

    # tracker
    conf["tracker"] = dict(advection=config["numerics"]["advection"])
    if config["numerics"]["diffusion"]:
        conf["tracker"]["diffusion"] = config["numerics"]["diffusion"]

    # release
    conf["release"] = dict(
        release_file=config["files"]["particle_release_file"],
        names=config["particle_release"]["variables"],
    )
    if "release_type" in config["particle_release"]:
        if config["particle_release"]["release_type"] == "continuous":
            conf["release"]["continuous"] = True
            conf["release"]["release_frequency"] = config["particle_release"][
                "release_frequency"
            ]
    # ibm
    if "ibm" in config:
        conf["ibm"] = dict()
        for var in config["ibm"]:
            if var == "ibm_module":
                conf["ibm"]["module"] = config["ibm"][var]
                continue
            if var != "variables":
                conf["ibm"][var] = config["ibm"][var]

    # output
    conf["output"] = dict(
        filename=config["files"]["output_file"],
        output_period=config["output_variables"]["outper"],
        instance_variables=dict(),
        particle_variables=dict(),
        ncargs=dict(
            data_model=config["output_variables"].get("format", "NETCDF3_CLASSIC")
        ),
    )
    for var in config["output_variables"]["instance"]:
        conf["output"]["instance_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf["output"]["instance_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf["output"]["instance_variables"][var]["attributes"] = D
    for var in config["output_variables"]["particle"]:
        conf["output"]["particle_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf["output"]["particle_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf["output"]["particle_variables"][var]["attributes"] = D

    return conf
