import numpy as np
from netCDF4 import Dataset
import pytest

from ladim2.timekeeper import TimeKeeper
from ladim2 import ROMS


filename = "../examples/data/ocean_avg_0014.nc"

# Take a U-point at sea
x, y = 80.5, 90

X = np.array([x])
Y = np.array([y])
Z = np.array([0.01])  # Near surface
i = int(x)
j = int(y)

# Access the file directly
with Dataset(filename) as nc:
    U_dir = nc.variables["u"][:, -1, j, i]
print(U_dir)


def test_prestep0():
    """Start = time of first record"""

    timer = TimeKeeper(start="1989-05-24T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename)
    force = ROMS.Forcing(grid, timer, filename)

    # First field
    force.update(force.steps[0], X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]

    #assert U_field == pytest.approx(U_dir[0])
    #assert U_vel == pytest.approx(U_dir[0])

    # Do some updating
    force.update(force.steps[0] + 1, X, Y, Z)
    force.update(force.steps[0] + 2, X, Y, Z)
    force.update(force.steps[1], X, Y, Z)
    force.update(force.steps[1] + 1, X, Y, Z)

    # Check third field
    force.update(force.steps[2], X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]
    assert U_field == pytest.approx(U_dir[2])
    assert U_vel == pytest.approx(U_dir[2])


def test_midstep():
    """Start = half way to next step"""

    timer = TimeKeeper(start="1989-05-26T00", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename)
    force = ROMS.Forcing(grid, timer, filename)

    # First field
    force.update(0, X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]

    assert U_field == pytest.approx(0.5*(U_dir[0]+U_dir[1]))
    assert U_vel == pytest.approx(0.5*(U_dir[0]+U_dir[1]))

    # Do some updating
    force.update(force.steps[1], X, Y, Z)
    force.update(force.steps[1] + 1, X, Y, Z)

    # Check third field
    force.update(force.steps[2], X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]
    assert U_field == pytest.approx(U_dir[2])
    assert U_vel == pytest.approx(U_dir[2])


def test_start_second():
    """Start = time of second record"""

    timer = TimeKeeper(start="1989-05-27T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename)
    force = ROMS.Forcing(grid, timer, filename)

    # First field
    force.update(0, X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]

    assert U_field == pytest.approx(U_dir[1])
    assert U_vel == pytest.approx(U_dir[1])

    # Do some updating
    force.update(1, X, Y, Z)
    force.update(72, X, Y, Z)
    force.update(73, X, Y, Z)     # Read new field

    # Check third field
    force.update(144, X, Y, Z)
    U_field = force.fields["u"][-1, j - grid.j0, i - grid.i0 + 1]
    U_vel = force.velocity(X, Y, Z)[0][0]
    assert U_field == pytest.approx(U_dir[3])
    assert U_vel == pytest.approx(U_dir[3])
