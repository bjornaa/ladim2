import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from ladim.out_netcdf_dense import Output

# import pytest
from ladim.state import State
from ladim.timekeeper import TimeKeeper
from netCDF4 import Dataset

NCFILE = Path("output_test.nc")
total_particle_count = 0


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


config0: dict[str, Any] = dict(
    modules=dict(
        time=TimeKeeper(start="2020-01-01 12", stop="2020-01-03 12", dt=1800),
        release=Dummy(total_particle_count=0),
        grid=None,
    ),
    filename=NCFILE,
    output_period=np.timedelta64(12, "h"),
    instance_variables=dict(
        pid=dict(
            encoding=dict(datatype="i"),
            attributes=dict(long_name="particle_identifier"),
        ),
        X=dict(
            encoding=dict(datatype="f4", zlib=True),
            attributes=dict(long_name="particle X-coordinate"),
        ),
    ),
    particle_variables=dict(
        X0=dict(
            encoding=dict(datatype="f4"),
            attributes=dict(long_name="initial X-position"),
        ),
    ),
    global_attributes=dict(institution="Institute of Marine Research", source="LADiM"),
)


def test_output_init():
    """Test module initialization"""
    out = Output(**config0)

    # Check some attributes of the output class
    assert out.filename == NCFILE
    assert set(out.instance_variables) == {"X"}
    assert out.output_period_step == 24  # 12 h / 0.5 h
    out.close()

    # Check that the file exist and is recognized by ncdump as a netcdf file
    assert NCFILE.exists()
    assert (
        subprocess.run(
            ["ncdump", "-h", str(NCFILE)], stdout=subprocess.DEVNULL, shell=False
        ).returncode
        == 0
    )

    # Check some of the file content
    with Dataset(NCFILE) as nc:
        assert nc.data_model == "NETCDF4"
        assert set(nc.dimensions) == {"time", "particle"}
        assert set(nc.variables.keys()) == {"time", "X", "X0"}
        assert nc.variables["time"].dimensions == ("time",)
        assert nc.variables["X"].dimensions == ("time", "particle")
        assert nc.variables["X0"].dimensions == ("particle",)
        assert set(nc.ncattrs()) == {"institution", "source", "type", "history"}
        assert nc.getncattr("source") == "LADiM"

    # NCFILE.unlink()


def test_reference_time():
    """Explicit reference time"""

    timer = TimeKeeper(
        start="2020-01-01 12",
        stop="2020-01-03 12",
        reference="2000-01-01",
        dt=1800,
    )
    timer.update()  # Update to time zero = start
    state = State()
    config = config0.copy()
    config["modules"] = config["modules"].copy()
    config["modules"]["time"] = timer
    out = Output(**config)
    state.append(X=100, Y=10, Z=5)
    out.write(state)
    out.close()
    with Dataset(NCFILE) as nc:
        tvar = nc.variables["time"]
        assert tvar.units == "seconds since 2000-01-01T00:00:00"
        assert (
            timer.reference_time + np.timedelta64(int(tvar[0]), "s") == timer.start_time
        )
    NCFILE.unlink()


def test_write():
    """Test writing a sequence of states"""
    state = State(particle_variables={"X0": float})
    config = config0.copy()
    config["modules"]["release"] = Dummy(total_particle_count=3)
    out = Output(**config)
    timer = out.timer
    outper = out.output_period // timer.dt

    assert out.record_count == 0

    # Initially one particle
    state.append(X=100, Y=10, Z=5, X0=100)
    timer.reset()
    out.write(state)
    assert out.record_count == 1

    # Update position
    state["X"] += 1
    for _ in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 2

    # Update first particle and add two new particles
    state["X"] += 1
    state.append(
        X=np.array([200, 300]), Y=np.array([20, 30]), Z=5, X0=np.array([200, 300])
    )
    for _ in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 3

    # Update particle positions and kill the first particle,
    state["X"] = state["X"] + 1.0
    state["alive"][0] = False
    for _ in range(outper):
        timer.update()
    out.write(state)
    assert out.record_count == 4

    # assert out.instance_count == 7
    # Update positions
    # state["X"] += 1
    # for _i in range(outper):
    #     timer.update()
    # out.write(state)
    # assert out.record_count == 5
    # assert out.instance_count == 9

    # Write particle variable
    # out.write_particle_variables(state)
    # out.close()

    # Check some of the content
    h = 3600
    with Dataset(NCFILE) as nc:
        Xvar = nc.variables["X"]
        Tvar = nc.variables["time"]

        assert Xvar.shape == (4, 3)  # Correct shape

        # Time
        assert all(Tvar[:] == 12 * h * np.arange(4))

        # X data
        assert np.all(Xvar[0, :] == [100, np.nan, np.nan])
        assert np.all(Xvar[1, :] == [101, np.nan, np.nan])
        assert np.all(Xvar[2, :] == [102, 200, 300])
        assert np.all(Xvar[3, :] == [np.nan, 201, 301])

        # Start positions
        assert all(nc.variables["X0"][:] == [100, 200, 300])

    NCFILE.unlink()  # clean up


def test_multifile():
    """Test the multifile functionality"""
    # Missing: Test with particle variables (when implemented)

    h = 3600

    config = dict(config0, filename="b.nc", numrec=2, particle_variables=dict())
    config["modules"]["release"] = Dummy(total_particle_count=3)
    out = Output(**config)
    state = State()
    timer = out.timer
    outper = out.output_period // timer.dt

    # First file
    state.append(X=100, Y=10, Z=5)
    timer.reset()
    out.write(state)
    state["X"] += 1
    for _ in range(outper):
        timer.update()
    out.write(state)
    with Dataset("b_000.nc") as nc:
        Xvar = nc.variables["X"]
        assert all(nc.variables["time"][:] == [0, 12 * h])
        assert np.all(Xvar[0, :] == [100, np.nan, np.nan])
        assert np.all(Xvar[1, :] == [101, np.nan, np.nan])

    # Second file
    # Update first particle and add two new particles
    state["X"] += 1
    state.append(X=np.array([200, 300]), Y=np.array([20, 30]), Z=5)
    for _ in range(outper):
        timer.update()
    out.write(state)
    # Update all particles and kill the first
    state["X"] = state["X"] + 1.0
    state["alive"][0] = False
    for _ in range(outper):
        timer.update()
    out.write(state)
    with Dataset("b_001.nc") as nc:
        Xvar = nc.variables["X"]
        assert all(nc.variables["time"][:] == [24 * h, 36 * h])
        assert np.all(Xvar[0, :] == [102, 200, 300])
        assert np.all(Xvar[1, :] == [np.nan, 201, 301])

    #     assert all(nc.variables["particle_count"][:] == [2])
    #     assert all(nc.variables["pid"][:] == [1, 2])

    # Clean up, remove the netcdf files
    for file_ in Path(".").glob("b_0??.nc"):
        pass
        # file_.unlink()
