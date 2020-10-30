# Trenger:
#
# time_keeper
# output_period
#   kan beregne num_output
# numrec = number of records per file
# nparticles
# ncargs: generelt, noe per variabel også?
#    Kalt encoding, bruke ncargs per variabal
#    og global_ncargs analogt med ncattrs


# import os
import re
from pathlib import Path
from typing import Dict, Generator, Union, Optional, Any

import numpy as np             # type: ignore
from netCDF4 import Dataset    # type: ignore

# from .timekeeper import TimeKeeper  # For typing
# from .state import State  # For typing
from .timekeeper import TimeKeeper  # For typing
from .state import State  # For typing


class Output:
    """Class for writing LADiM results to netCDF

    Attributes:
        filename:  name of netcdf-file
        num_particles:  Total number of particles in the simulation
        instance_variables
        particle_variables
        ncargs


    """

    def __init__(
        self,
        timer: TimeKeeper,
        filename: Union[Path, str],
        # num_output: int,  # Number of times with output
        output_period: np.timedelta64,  # Time interval between outputs
        num_particles: int,  # Total number of particles
        instance_variables: Dict[str, Any],
        particle_variables: Optional[Dict[str, Any]] = None,
        ncargs: Optional[Dict[str, Any]] = None,
        numrec: int = 0,  # Number of records per file, no multfile if zeo
        skip_initial: Optional[bool] = False,
        global_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:

        # logging.info("Initializing output")

        self.timer = timer
        self.filename = filename
        # self.num_output = num_output
        self.num_particles = num_particles
        self.instance_variables = instance_variables
        self.particle_variables = particle_variables if particle_variables else dict()
        self.skip_initial = skip_initial
        self.numrec = numrec
        self.ncargs = ncargs if ncargs else dict()
        if "mode" not in self.ncargs:
            self.ncargs["mode"] = "w"  # Default = write access

        self.global_attributes = global_attributes

        self.num_records = 1 + (timer.stop_time - timer.start_time) // output_period
        if skip_initial:
            self.num_records -= 1

        if self.numrec:
            self.multifile = True
            self.filenames = fname_gnrt(Path(filename))
            self.filename = next(self.filenames)
        else:
            self.multifile = False
            self.filename = Path(filename)
            self.numrec = 999999

        self.record_count = 0
        self.instance_count = 0

        self.nc = self.create_netcdf()
        self.local_instance_count = 0
        self.local_record_count = 0

        self.step2nctime = timer.step2nctime
        self.time_unit = "s"
        self.nctime = 0  # juster hvis skip_initial
        self.cf_units = timer.cf_units(self.time_unit)
        # Use output period in time steps
        self.output_period = np.timedelta64(output_period, "s")
        # self.num_records = timer.Nsteps * self.timer._dt // self.output_period

        # self.filenames = fname_gnrt(self.filename)

        # self.record_count = 0  # Record number to write (start at zero)
        # self.instance_count = 0  # Count of written particle instances
        # self.outcount = -1  # No output yet
        # self.file_counter = -1  # No file yer
        # self.skip_output = config["skip_initial"]
        # self.dt = config["dt"]
        # self.release = release
        # # Indicator for lon/lat output
        # self.lonlat = (
        #     "lat" in self.instance_variables or "lon" in self.instance_variables
        # )

    def create_netcdf(self) -> Dataset:
        """Create a LADiM output netCDF file"""

        # print(locals())

        # Handle netcdf args
        ncargs = self.ncargs
        nc = Dataset(self.filename, **ncargs)

        # self.offset = self.record_count  # record_count at start of file

        # Number of records in the file (last file may be smaller)
        int(self.numrec)
        self.local_num_records = min(self.numrec, self.num_records - self.record_count)

        nc.createDimension("time", self.local_num_records)
        nc.createDimension("particle_instance", None)  # Unlimited
        nc.createDimension("particle", self.num_particles)

        v = nc.createVariable("time", "f8", ("time",))
        v.long_name = "time"
        v.standard_name = "time"
        v.units = f"seconds since {self.timer.reference_time}"
        v = nc.createVariable("particle_count", "i", ("time",))
        v.long_name = "Number of particles"
        v.ragged_row_count = "particle count at nth timestep"

        if self.instance_variables is not None:
            for var, conf in self.instance_variables.items():
                v = nc.createVariable(
                    var, conf["encoding"]["datatype"], ("particle_instance",)
                )
                for att, value in conf["attributes"].items():
                    setattr(v, att, value)

        if self.particle_variables is not None:
            for var, conf in self.particle_variables.items():
                v = nc.createVariable(var, conf["encoding"]["datatype"], ("particle",))
                for att, value in conf["attributes"].items():
                    setattr(v, att, value)

        if self.global_attributes is not None:
            for att, value in self.global_attributes.items():
                setattr(nc, att, value)

        return nc

    def write(self, state: State) -> None:
        """Multifile write

        Arguments:
          state: Model state
          step: Time step number

        """

        count = len(state)  # Present number of particles
        start = self.local_instance_count
        end = start + count

        print(self.local_record_count, self.local_num_records)

        # rec_count = self.record_count % self.numrec  # record count *in* the file

        self.nc.variables["time"][self.local_record_count] = self.nctime
        self.nc.variables["particle_count"][self.local_record_count] = count

        for var in self.instance_variables:
            self.nc.variables[var][start:end] = getattr(state, var)

        # Flush to file
        self.nc.sync()

        self.instance_count += count
        self.record_count += 1
        self.local_instance_count += count
        self.local_record_count += 1
        self.nctime += self.output_period / np.timedelta64(1, self.time_unit)

        # File finished? (beregn fra rec_count)
        if self.local_record_count == self.local_num_records:
            self.nc.close()
            # New file?
            if self.record_count < self.num_records:
                self.filename = next(self.filenames)
                self.nc = self.create_netcdf()
                self.local_instance_count = 0
                self.local_record_count = 0


def fname_gnrt(filename: Path) -> Generator[Path, None, None]:
    """Generate file names based on prototype

    Examples:
    output/cake.nc -> output/cake_000.nc, output/cake_001.nc, ...
    cake_04.nc -> cake_04.nc, cake_05.nc, ....
    """

    # fname0, ext = splitext(filename)
    stem = filename.stem  # filename without extension
    pattern = r"_(\d+)$"  # _digits at end of string
    m = re.search(pattern, stem)

    if m:  # Start from a number (or trailing underscore)
        ddd = m.group(1)
        filenumber = int(ddd)
        number_width = len(ddd)
        xxxx = stem[: -number_width - 1]  # remove _ddd
    else:  # Start from zero
        filenumber = 0
        number_width = 3
        xxxx = stem
    filename_template = f"{xxxx}_{{:0{number_width}d}}{filename.suffix}"

    while True:
        yield filename.parent / filename_template.format(filenumber)
        filenumber += 1
