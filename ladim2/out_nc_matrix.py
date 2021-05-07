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
from datetime import date
from typing import Dict, Union, Optional, Sequence, Any, Generator

import numpy as np  # type: ignore
from netCDF4 import Dataset  # type: ignore

from ladim2.timekeeper import TimeKeeper, normalize_period
from ladim2.state import State  # For typing
from ladim2.grid import BaseGrid
from ladim2.output import BaseOutput

Variable = Dict[str, Any]


def init_output(**args) -> BaseOutput:
    return Output(**args)


class Output(BaseOutput):
    """LADiM output, orhogonal array representation in NetCDF

    """

    def __init__(
        self,
        timer: TimeKeeper,
        filename: Union[Path, str],
        output_period: Union[int, np.timedelta64, Sequence],
        num_particles: int,  # Total number of particles
        instance_variables: Dict[str, Variable],
        particle_variables: Optional[Dict[str, Variable]] = None,
        grid: Optional[BaseGrid] = None,
        ncargs: Optional[Dict[str, Any]] = None,
        numrec: int = 0,  # Number of records per file, no multfile if zero
        skip_initial: Optional[bool] = False,
        global_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:

        # logging.info("Initializing output")

        self.timer = timer
        self.filename = filename
        self.num_particles = num_particles
        self.instance_variables = instance_variables
        self.particle_variables = particle_variables if particle_variables else dict()

        self.skip_initial = skip_initial
        # self.numrec = numrec if numrec else 0
        self.numrec = numrec
        self.ncargs = ncargs if ncargs else dict()
        if "mode" not in self.ncargs:
            self.ncargs["mode"] = "w"  # Default = write access

        if global_attributes:
            self.global_attributes = global_attributes
        else:
            self.global_attributes = dict()
        self.global_attributes["type"] = "LADiM output, netcdf orthogonal array"
        self.global_attributes["history"] = f"Created by LADiM, {date.today()}"

        self.output_period = normalize_period(output_period)
        self.output_period_step = self.output_period // timer.dt
        if timer.time_reversal:
            self.output_period = -self.output_period

        self.num_records = abs(
            (timer.stop_time - timer.start_time) // self.output_period
        )
        if not skip_initial:  # Add an initial record
            self.num_records += 1

        if self.numrec:
            self.multifile = True
            self.filenames = fname_gnrt(Path(filename))
            self.filename = next(self.filenames)
        else:
            self.multifile = False
            self.filename = Path(filename)
            self.numrec = 999999

        self.record_count = 0
        # self.instance_count = 0

        self.nc = self.create_netcdf()
        # self.local_instance_count = 0
        self.local_record_count = 0

        self.step2nctime = timer.step2nctime
        self.time_unit = "s"
        self.nctime = timer.step2nctime(0, "s")
        self.cf_units = timer.cf_units(self.time_unit)

        if "lon" in self.instance_variables or "lat" in self.instance_variables:
            self.lonlat = True
        else:
            self.lonlat = False
        if self.lonlat:
            try:
                self.xy2ll = grid.xy2ll  # type: ignore
            except AttributeError:
                self.xy2ll = lambda x, y: (x, y)

    def create_netcdf(self) -> Dataset:
        """Create a LADiM output netCDF file"""

        # Handle netcdf args
        ncargs = self.ncargs
        nc = Dataset(self.filename, **ncargs)

        # self.offset = self.record_count  # record_count at start of file

        # Number of records in the file (last file may be smaller)
        self.local_num_records = min(self.numrec, self.num_records - self.record_count)

        nc.createDimension("time", self.local_num_records)
        nc.createDimension("particle", self.num_particles)

        v = nc.createVariable("time", "f8", ("time",))
        v.long_name = "time"
        v.standard_name = "time"
        v.units = f"seconds since {self.timer.reference_time}"

        if self.instance_variables is not None:
            for var, conf in self.instance_variables.items():
                if var == "pid":
                    continue
                v = nc.createVariable(
                    var, conf["encoding"]["datatype"], ("time", "particle")
                )
                for att, value in conf["attributes"].items():
                    setattr(v, att, value)

        if self.particle_variables is not None:
            for var, conf in self.particle_variables.items():
                # xarray requires nan as fillvalue to interpret time
                if conf["encoding"]["datatype"] in ["f4", "f8"]:
                    v = nc.createVariable(
                        var,
                        conf["encoding"]["datatype"],
                        ("particle",),
                        fill_value=np.nan,
                    )
                else:
                    v = nc.createVariable(
                        var, conf["encoding"]["datatype"], ("particle",),
                    )
                for att, value in conf["attributes"].items():
                    # Replace string "reference_time" with actual reference time
                    if "reference_time" in value:
                        value = value.replace(
                            "reference_time", str(self.timer.reference_time)
                        )
                    setattr(v, att, value)

        if self.global_attributes is not None:
            for att, value in self.global_attributes.items():
                setattr(nc, att, value)

        return nc

    def write(self, state: State) -> None:
        """Write output instance variables to a (multi-)file

        Arguments:
          state: Model state

        """

        # print("Write, time = ", self.timer.time)

        # May skip initial output
        if self.skip_initial:
            self.skip_initial = False
            return

        # count = len(state)  # Present number of particles
        # start = self.local_instance_count
        # end = start + count

        # rec_count = self.record_count % self.numrec  # record count *in* the file

        self.nc.variables["time"][self.local_record_count] = self.timer.nctime()
        # self.nc.variables["particle_count"][self.local_record_count] = count

        for var in self.instance_variables:
            if var == "pid":
                continue
            self.nc.variables[var][self.local_record_count, state.pid] = getattr(
                state, var
            )

        # Compute lon, lat if needed
        if self.lonlat:
            lon, lat = self.xy2ll(state.X, state.Y)
            self.nc.variables["lon"][self.local_record_count, :] = lon
            self.nc.variables["lat"][self.local_record_count, :] = lat

        # Flush to file
        self.nc.sync()

        # self.instance_count += count
        self.record_count += 1
        # self.local_instance_count += count
        self.local_record_count += 1
        self.nctime += self.output_period / np.timedelta64(1, self.time_unit)

        # File finished? (beregn fra rec_count)
        if self.local_record_count == self.local_num_records:
            self.write_particle_variables(state)
            self.nc.close()
            # New file?
            if self.record_count < self.num_records:
                self.filename = next(self.filenames)
                self.nc = self.create_netcdf()
                # self.local_instance_count = 0
                self.local_record_count = 0

    def write_particle_variables(self, state: State) -> None:
        """Write all output particle variables"""
        npart = state.pid.max() + 1  # Total number of particles so far
        for var in self.particle_variables:
            if state.dtypes[var] == np.dtype("datetime64[s]"):
                unit = self.time_unit
                delta = state[var].astype("M8[s]") - self.timer.reference_time
                self.nc.variables[var][:npart] = delta[:npart] / np.timedelta64(1, unit)
            else:
                self.nc.variables[var][:npart] = state[var][:npart]

    def close(self) -> None:
        self.nc.close()


def fname_gnrt(filename: Path) -> Generator[Path, None, None]:
    """Generate file names based on prototype

    Examples:
    output/cake.nc -> output/cake_000.nc, output/cake_001.nc, ...
    cake_04.nc -> cake_04.nc, cake_05.nc, ....
    """

    # fname0, ext = splitext(filename)
    stem = filename.stem  # filename without parent and extension
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
