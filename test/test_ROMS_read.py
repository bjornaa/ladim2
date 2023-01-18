from pathlib import Path

import numpy as np
from netCDF4 import Dataset
import pytest

from ladim.timekeeper import TimeKeeper
from ladim import ROMS


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setitem__(self, var, value):
        pass


filename = Path("examples/data/ocean_avg_0014.nc")
# Can be run from test directory or ladim root
cwd = Path.cwd()
if cwd.stem == "test":
    filename = cwd.parent / filename
pytestmark = pytest.mark.skipif(not filename.exists(), reason="Missing data file")


@pytest.fixture(scope="module")
def setup():
    # Take a position at sea
    x, y = 80.2, 90.0
    i, j = int(x + 0.5), int(y + 0.5)  # Grid cell containing the point
    # The rho-point at the center
    X, Y = np.array([x]), np.array([y])
    Z = np.array([0.01])  # Near surface

    # Define a subgrid
    subgrid = [50, 110, 60, 120]

    # Access the file directly
    with Dataset(filename) as nc:
        # Spatial interpolation for velocity
        U_dir = (
            0.3 * nc.variables["u"][:, -1, j, i - 1]
            + 0.7 * nc.variables["u"][:, -1, j, i]
        )
        # Nearest neighbour for scalar fields
        temp_dir = nc.variables["temp"][:, -1, j, i]
    return X, Y, Z, subgrid, U_dir, temp_dir


def test_prestep0(setup):
    """Start = time of first record"""

    X, Y, Z, subgrid, U_dir, temp_dir = setup

    timer = TimeKeeper(start="1989-05-24T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    state = Dummy(X=X, Y=Y, Z=Z)
    modules = dict(grid=grid, time=timer, state=state)
    force = ROMS.Forcing(modules, filename, extra_forcing=["temp"])

    # First field
    force.update()
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[0])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[0])

    # Do some updating
    while timer.step < force.steps[2]:
        timer.update()
        force.update()

    # Check third field
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[2])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[2])


def test_midstep(setup):
    """Start = half way to next step"""

    X, Y, Z, subgrid, U_dir, temp_dir = setup

    timer = TimeKeeper(start="1989-05-26T00", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    state = Dummy(X=X, Y=Y, Z=Z)
    modules = dict(grid=grid, time=timer, state=state)
    force = ROMS.Forcing(modules, filename, extra_forcing=["temp"])

    # First field
    force.update()

    # Time interpolation for velocity
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(0.5 * (U_dir[0] + U_dir[1]))
    # No time interpolation for scalar fields
    assert force.variables["temp"][0] == pytest.approx(temp_dir[0])

    # Do some updating
    while timer.step < force.steps[2]:
        timer.update()
        force.update()

    # Check third field
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[2])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[2])


def test_start_second(setup):
    """Start = time of second record"""

    X, Y, Z, subgrid, U_dir, temp_dir = setup

    timer = TimeKeeper(start="1989-05-27T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    state = Dummy(X=X, Y=Y, Z=Z)
    modules = dict(grid=grid, time=timer, state=state)
    force = ROMS.Forcing(modules, filename, extra_forcing=["temp"])

    # First field
    timer.update()
    force.update()
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[1])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[1])

    # Do some updating
    while timer.step < force.steps[3]:
        timer.update()
        force.update()

    # Check third field (4th on file)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[3])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[3])
