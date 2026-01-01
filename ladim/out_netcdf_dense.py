"""Output module for NetCDF dense array"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from netCDF4 import Dataset

from ladim.out_netcdf import filename_generator
from ladim.output import BaseOutput
from ladim.timekeeper import TimeDelta, normalize_period

if TYPE_CHECKING:
    from ladim.state import State


Variable = dict[str, Any]

DEBUG = False
logger = logging.getLogger(__name__)
if DEBUG:
    logger.setLevel(logging.DEBUG)


class Output(BaseOutput):
    """LADiM output to NetCDF"""

    def __init__(
        self,
        modules: dict[str, Any],
        filename: Path | str,
        output_period: TimeDelta,
        instance_variables: dict[str, Variable],
        particle_variables: dict[str, Variable] | None = None,
        ncargs: dict[str, Any] | None = None,
        numrec: int = 0,  # Number of records per file, no multifile if zero
        skip_initial: bool | None = False,
        global_attributes: dict[str, Any] | None = None,
    ) -> None:
        logger.info("Initializing output")
        super().__init__(modules)
        timer = modules["time"]
        grid = modules["grid"]
        self.filename = filename
        self.timer = timer
        self.num_particles = modules["release"].total_particle_count
        self.instance_variables = instance_variables
        # No need to save pid in orthogonal layout
        self.pid = self.instance_variables.pop("pid", None)
        self.particle_variables = particle_variables if particle_variables else dict()
        logger.info("  Filename: %s", filename)
        logger.info("  Dense format")
        logger.info("  Instance variables: %s", list(instance_variables))
        logger.info("  Particle variables: %s", list(self.particle_variables))

        self.skip_initial = skip_initial
        if skip_initial:
            logger.info("  Skipping initial output")
        # self.numrec = numrec if numrec else 0
        self.numrec = numrec
        self.ncargs = ncargs if ncargs else dict()
        self.ncargs["format"] = "NETCDF4"  # Only accepted format
        if "mode" not in self.ncargs:
            self.ncargs["mode"] = "w"  # Default = (over)write

        if global_attributes:
            self.global_attributes = global_attributes
        else:
            self.global_attributes = dict()
        self.global_attributes["type"] = "LADiM output, dense = netcdf orthogonal array"
        self.global_attributes["history"] = f"Created by LADiM, {date.today()}"

        self.output_period = normalize_period(output_period)
        self.output_period_step = self.output_period // timer.dt
        if timer.time_reversal:
            self.output_period = -self.output_period
        logger.info("  Output period: %s", str(self.output_period))

        self.num_records = int(
            abs((timer.stop_time - timer.start_time) // self.output_period)
        )
        # if not skip_initial:  # Add an initial record
        #     self.num_records += 1
        logger.info("  Number of records: %s", self.num_records)

        if self.numrec:
            self.multifile = True
            self.filenames = filename_generator(Path(filename))
            self.filename = next(self.filenames)
            logger.info("  Multifile output")
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
                self.xy2ll = grid.xy2ll
            except AttributeError:
                self.xy2ll = lambda x, y: (x, y)

    def update(self) -> None:
        step = self.modules["time"].step
        if step % self.output_period_step == 0:
            logger.info("writing, time = %s", self.modules["time"].time)
            self.write(self.modules["state"])

    def create_netcdf(self) -> Dataset:
        """Create a LADiM output netCDF file, sparse (default) or dense layout

        Returns:
            An open NetCDF Dataset
        """

        logging.info("Creating new output file: %s", self.filename)

        # Handle netcdf args
        ncargs = self.ncargs
        nc = Dataset(self.filename, **ncargs)

        # self.offset = self.record_count  # record_count at start of file

        # Number of records in the file (the final file may be smaller)
        self.local_num_records = min(self.numrec, self.num_records - self.record_count)

        # Dimensions
        # nc.createDimension("time", self.local_num_records)
        nc.createDimension("time", None)
        nc.createDimension("particle", self.num_particles)
        instance_dim: tuple[str, ...] = ("time", "particle")

        # Variables
        v = nc.createVariable("time", "f8", ("time",))
        v.long_name = "time"
        v.standard_name = "time"
        v.units = f"seconds since {self.timer.reference_time}"

        if self.instance_variables is not None:
            for var, conf in self.instance_variables.items():
                v = nc.createVariable(var, conf["encoding"]["datatype"], instance_dim)
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
                        var,
                        conf["encoding"]["datatype"],
                        ("particle",),
                    )

                for att, value in conf["attributes"].items():
                    # Replace string "reference_time" with actual reference time
                    if "reference_time" in value:
                        new_value = value.replace(
                            "reference_time", str(self.timer.reference_time)
                        )
                        setattr(v, att, new_value)

        if self.global_attributes is not None:
            for att, value in self.global_attributes.items():
                setattr(nc, att, value)

        return nc

    def write(self, state: State) -> None:
        """Write output instance variables at specific time to a (multi-)file

        Arguments:
          state: Model state

        """

        # May skip initial output
        self.skip_initial = False
        if self.skip_initial:
            self.skip_initial = False
            return

        self.nc.variables["time"][self.local_record_count] = self.timer.nctime()

        # Fill out state.alive to total number of particles
        alive = np.full(self.num_particles, False)
        if len(alive) > 0:
            alive[state.pid] = state.alive

        for var in self.instance_variables:
            full_data = np.zeros(self.num_particles, dtype=getattr(state, var).dtype)
            full_data[alive] = getattr(state, var)[state.alive]
            full_data = np.ma.array(full_data, mask=~alive)
            self.nc.variables[var][self.local_record_count, :] = full_data
        # Compute and save lon, lat if requested
        if self.lonlat:
            lon, lat = self.xy2ll(state.X, state.Y)

            full_data = np.zeros(self.num_particles, dtype=lon.dtype)
            full_data[alive] = lon[state.alive]
            full_data = np.ma.array(full_data, mask=~alive)
            self.nc.variables["lon"][self.local_record_count, :] = full_data

            full_data = np.zeros(self.num_particles, dtype=lat.dtype)
            full_data[alive] = lat[state.alive]
            full_data = np.ma.array(full_data, mask=~alive)
            self.nc.variables["lat"][self.local_record_count, :] = full_data

        # Flush to file
        self.nc.sync()

        # Prepare for next time
        self.record_count += 1
        self.local_record_count += 1
        self.nctime += float(self.output_period / np.timedelta64(1, self.time_unit))

        # File finished?
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
        """Write all output particle variables

        Args:
            state: A Ladim State instance
        """
        npart = int(state.pid.max()) + 1  # Total number of particles so far
        for var in self.particle_variables:
            if state.dtypes[var] == np.dtype("datetime64[s]"):
                unit = self.time_unit
                delta = state[var].astype("M8[s]") - self.timer.reference_time
                self.nc.variables[var][:npart] = delta[:npart] / np.timedelta64(1, unit)
            else:
                self.nc.variables[var][:npart] = state[var][:npart]

    def close(self) -> None:
        if self.nc.isopen():
            self.nc.close()


#     while True:
#         yield filename.parent / filename_template.format(filenumber)
#         filenumber += 1
