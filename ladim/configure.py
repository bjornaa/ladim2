""""

Configuration reader for LADiM version 2
with compability wrapper for LADiM version 1 configuration

"""

# -----------------------------------
# Bjørn Ådlandsvik <bjorn@hi.no>
# Institute of Marine Research
# December 2020
# -----------------------------------

import sys
from pathlib import Path
import logging
from typing import Union, Literal, Any

import numpy as np
from netCDF4 import Dataset, num2date  # type: ignore
import yaml  # type: ignore


DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


def configure(config_file: Union[Path, str]) -> dict[str, Any]:
    """Main configuration function of LADiM

    Args:
        config_file:
            Name of configuration file
        version:
            configuration format version,
                = 1 for LADiM version 1.x
                = 2 for LADiM version 2.x

    Returns:
        2-level configuration dictionary

    """

    logger.info("Configuration")
    logger.info("  Configuration file %s:", config_file)

    if not Path(config_file).exists():
        logger.critical("No configuration file %s:", config_file)
        raise SystemExit(3)

    try:
        with open(config_file, encoding="utf-8") as fid:
            config: dict[str, Any] = yaml.safe_load(fid)
    except yaml.parser.ParserError as err:
        logger.critical("Not a valid yaml file: %s", config_file)
        raise SystemExit(3) from err

    # Determine configuration version
    version = config.get("version", 0)  # zero for undetermined
    if "time_control" in config:
        version = 1
    if "time" in config:
        version = 2


    logger.info("  Configuration file version: %s", version)

    if version == 2:
        configure_v2(config)
    elif version == 1:
        config = configure_v1(config)
    else:
        logger.critical("Not a valid configuration version")
        raise(SystemExit(3))


    # Possible improvement: write a yaml-file
    if DEBUG:
        yaml.dump(config, stream=sys.stdout)

    return config


# --------------------------------------------------------------------


def configure_v2(config: dict[str, Any]) -> None:
    """Read version 2 configuration file"""

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

    # If grid["filename"] is missing, use forcing["filename"]
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
        except (FileNotFoundError, OSError) as err:
            logging.critical("Could not open warm start file:%s", warm_start_file)
            raise SystemExit(1) from err
        tvar = nc.variables["time"]
        # Use last record in restart file
        warm_start_time = np.datetime64(num2date(tvar[-1], tvar.units))
        warm_start_time = warm_start_time.astype("M8[s]")
        config["time"]["start"] = warm_start_time
        logging.info("    Warm start at %s", warm_start_time)

        if "variables" not in config["warm_start"]:
            config["warm_start"]["variables"] = []

        # warm start -> release
        config["release"]["warm_start_file"] = config["warm_start"]["filename"]

    # skip_initial is default with warm start
    if "filename" in config["warm_start"] and "skip_initial" not in config["output"]:
        config["output"]["skip_initial"] = True



def configure_v1(config: dict[str, Any]) -> dict[str, Any]:
    """Tries to read version 1 configuration files

    This function may fail for valid configuration files for LADiM version 1.
    Ordinary use cases at IMR should work.

    """

    conf2: dict[str, Any] = dict()  # output version 2 configuration

    # time
    conf2["time"] = dict(
        start=config["time_control"]["start_time"],
        stop=config["time_control"]["stop_time"],
        dt=config["numerics"]["dt"],
    )
    if "reference_time" in config["time_control"]:
        conf2["time"]["reference"] = config["time_control"]["reference_time"]

    # grid and forcing
    conf2["grid"] = dict()
    conf2["forcing"] = dict()
    if "ladim1.gridforce.ROMS" in config["gridforce"]["module"]:
        conf2["grid"]["module"] = "ladim.ROMS"
        conf2["forcing"]["module"] = "ladim.ROMS"
    else:
        conf2["grid"]["module"] = config["gridforce"]["module"]
        conf2["forcing"]["module"] = config["gridforce"]["module"]

    if "input_file" in config["gridforce"]:
        conf2["forcing"]["filename"] = config["gridforce"]["input_file"]
    elif "input_file" in config["files"]:
        conf2["forcing"]["filename"] = config["files"]["input_file"]
    else:
        conf2["forcing"]["filename"] = ""

    if "gridfile" in config["gridforce"]:
        conf2["grid"]["filename"] = config["gridforce"]["gridfile"]
    elif "gridfile" in config["files"]:
        conf2["grid"]["filename"] = config["files"]["gridfile"]
    else:
        conf2["grid"]["filename"] = ""

    if not conf2["grid"]["filename"] and conf2["forcing"]["filename"]:
        filename = Path(conf2["forcing"]["filename"])
        # glob if necessary and use first file
        if ("*" in str(filename)) or ("?" in str(filename)):
            directory = filename.parent
            filename = sorted(directory.glob(filename.name))[0]
            print("--", filename)
        conf2["grid"]["filename"] = filename


    if "subgrid" in config["gridforce"]:
        conf2["grid"]["subgrid"] = config["gridforce"]["subgrid"]
    if "ibm_forcing" in config["gridforce"]:
        conf2["forcing"]["ibm_forcing"] = config["gridforce"]["ibm_forcing"]

    # state
    conf2["state"] = dict()
    instance_variables = dict()
    particle_variables = dict()
    if "ibm" in config and "variables" in config["ibm"]:
        for var in config["ibm"]["variables"]:
            instance_variables[var] = "float"
    for var in config["particle_release"]["variables"]:
        if var in ["mult", "X", "Y", "Z"]:  # Ignore
            continue
        if var in ["lon", "lat"]:
            instance_variables[var] = "float"
        if (
            "particle_variables" in config["particle_release"]
            and var in config["particle_release"]["particle_variables"]
        ):
            particle_variables[var] = config["particle_release"].get(var, "float")
    conf2["state"]["instance_variables"] = instance_variables
    conf2["state"]["particle_variables"] = particle_variables
    conf2["state"]["default_values"] = dict()
    for var in conf2["state"]["instance_variables"]:
        conf2["state"]["default_values"][var] = 0

    # tracker
    conf2["tracker"] = dict(advection=config["numerics"]["advection"])
    if config["numerics"]["diffusion"]:
        conf2["tracker"]["diffusion"] = config["numerics"]["diffusion"]

    # release
    conf2["release"] = dict(
        release_file=config["files"]["particle_release_file"],
        names=config["particle_release"]["variables"],
    )
    if "release_type" in config["particle_release"]:
        if config["particle_release"]["release_type"] == "continuous":
            conf2["release"]["continuous"] = True
            conf2["release"]["release_frequency"] = config["particle_release"][
                "release_frequency"
            ]
    # ibm
    if "ibm" in config:
        conf2["ibm"] = dict()
        for var in config["ibm"]:
            if var == "ibm_module":
                conf2["ibm"]["module"] = config["ibm"][var]
                continue
            if var != "variables":
                conf2["ibm"][var] = config["ibm"][var]
    else:
        conf2["ibm"] = dict()


    if "warm_start" not in config:
        conf2["warm_start"] = dict()

    # output
    conf2["output"] = dict(
        filename=config["files"]["output_file"],
        output_period=config["output_variables"]["outper"],
        instance_variables=dict(),
        particle_variables=dict(),
        ncargs=dict(
            data_model=config["output_variables"].get("format", "NETCDF3_CLASSIC")
        ),
    )
    for var in config["output_variables"]["instance"]:
        conf2["output"]["instance_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf2["output"]["instance_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf2["output"]["instance_variables"][var]["attributes"] = D
    for var in config["output_variables"]["particle"]:
        conf2["output"]["particle_variables"][var] = dict()
        D = config["output_variables"][var].copy()
        conf2["output"]["particle_variables"][var]["encoding"] = dict(
            datatype=D.pop("ncformat")
        )
        conf2["output"]["particle_variables"][var]["attributes"] = D

    return conf2
