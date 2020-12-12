from pathlib import Path

from typing import Union, Dict, Any
import yaml

# from .timekeeper import normalize_period


def configure(config_file: Union[Path, str]) -> Dict[str, Any]:
    with open(config_file) as fid:
        config: Dict[str, Any] = yaml.safe_load(fid)

    if "version" not in config:
        config = configure_v1(config)

    # Some sections may be missing
    if "state" not in config:
        config["state"] = dict()
    if "grid" not in config:
        config["grid"] = dict()
    if "ibm" not in config:
        config["ibm"] = dict()

    # Handle non-orthogonality

    # If no grid file use the forcing file
    # Is this correct, should "file" be "filename"?
    # if "filename" not in config["grid"]:
    #     config["grid"]["filename"] = config["forcing"]["filename"]
    #     print(config["output"])

    # Use time step from time_control
    config["tracker"]["dt"] = config["time"]["dt"]

    # Missing grid.filename
    if "filename" not in config["grid"]:
        filename = Path(config["forcing"]["filename"])
        # glob if necessary
        if ("*" in str(filename)) or ("?" in str(filename)):
            directory = filename.parent
            filename = sorted(directory.glob(filename.name))[0]
        config["grid"]["filename"] = filename

    # if config["ibm"]:
    #    config["ibm"]["dt"] = normalize_period(config["time"]["dt"])

    return config


def configure_v1(config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle version 1 configuration file"""
    conf: Dict[str, Any] = dict()  # version 2 configuration
    # conf["version"] = 1.0

    conf["time"] = dict(
        start=config["time_control"]["start_time"],
        stop=config["time_control"]["stop_time"],
        dt=config["numerics"]["dt"],
    )
    if "reference_time" in config["time_control"]:
        conf["time"]["reference"] = config["time_control"]["reference_time"]

    conf["grid"] = dict()
    conf["forcing"] = dict()
    if "ladim.gridforce.ROMS" in config["gridforce"]["module"]:
        conf["grid"]["module"] = "ladim2.grid_ROMS"
        if "gridfile" in config["gridforce"]:
            conf["grid"]["filename"] = config["gridforce"]["gridfile"]
        elif "gridfile" in config["files"]:
            conf["grid"]["filename"] = config["files"]["gridfile"]
        conf["forcing"]["module"] = "ladim2.forcing_ROMS"
        conf["forcing"]["filename"] = config["gridforce"]["input_file"]

    conf["tracker"] = dict(advection=config["numerics"]["advection"])
    # mangler diffusjon

    conf["release"] = dict(
        release_file=config["files"]["particle_release_file"],
        names=config["particle_release"]["variables"],
    )

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

    return conf
