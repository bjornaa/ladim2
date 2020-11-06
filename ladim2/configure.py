import yaml


def configure(config_file):
    with open("ladim2.yaml") as fid:
        config = yaml.safe_load(fid)

    # Some sections may be missing
    if "state" not in config:
        config["state"] = dict()
    if "grid" not in config:
        config["grid"] = dict()

    # Handle non-orthogonality

    print(config["grid"])

    # If no grid file use the forcing file
    if "file" not in config["grid"]:
        config["grid"]["filename"] = config["forcing"]["filename"]

    # Use time step from time_control
    config["tracker"]["dt"] = config["time_control"]["dt"]

    config["release"]["timekeeper"] = config["time_control"]



    return config
