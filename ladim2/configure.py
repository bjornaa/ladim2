from pathlib import Path

from typing import Union, Dict, Any
import yaml


def configure(config_file: Union[Path, str]) -> Dict[str, Any]:
    with open(config_file) as fid:
        config: Dict[str, Any] = yaml.safe_load(fid)

    # Some sections may be missing
    if "state" not in config:
        config["state"] = dict()
    if "grid" not in config:
        config["grid"] = dict()

    # Handle non-orthogonality

    # If no grid file use the forcing file
    # Is this correct, should "file" be "filename"?
    if "file" not in config["grid"]:
        config["grid"]["filename"] = config["forcing"]["filename"]

    # Use time step from time_control
    config["tracker"]["dt"] = config["time"]["dt"]

    return config
