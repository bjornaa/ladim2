import numpy as np
from netCDF4 import Dataset
import pytest

from ladim2.timekeeper import TimeKeeper
from ladim2 import ROMS


filename = "../examples/data/ocean_avg_0014.nc"

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
        0.3 * nc.variables["u"][:, -1, j, i - 1] + 0.7 * nc.variables["u"][:, -1, j, i]
    )
    # Nearest neighbour for scalar fields
    temp_dir = nc.variables["temp"][:, -1, j, i]


def test_prestep0():
    """Start = time of first record"""

    timer = TimeKeeper(start="1989-05-24T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    force = ROMS.Forcing(grid, timer, filename, ibm_forcing=["temp"])

    # First field
    force.update(force.steps[0], X, Y, Z)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[0])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[0])

    # Do some updating
    force.update(force.steps[0] + 1, X, Y, Z)
    force.update(force.steps[1], X, Y, Z)
    force.update(force.steps[1] + 1, X, Y, Z)

    # Check third field
    force.update(force.steps[2], X, Y, Z)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[2])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[2])


def test_midstep():
    """Start = half way to next step"""

    timer = TimeKeeper(start="1989-05-26T00", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    force = ROMS.Forcing(grid, timer, filename, ibm_forcing=["temp"])

    # First field
    force.update(0, X, Y, Z)

    # Time interpolation for velocity
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(0.5 * (U_dir[0] + U_dir[1]))
    # No time interpolation for scalar fields
    assert force.variables["temp"][0] == pytest.approx(temp_dir[0])

    # Do some updating
    force.update(force.steps[1], X, Y, Z)
    force.update(force.steps[1] + 1, X, Y, Z)

    # Check third field
    force.update(force.steps[2], X, Y, Z)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[2])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[2])


def test_start_second():
    """Start = time of second record"""

    timer = TimeKeeper(start="1989-05-27T12", stop="1989-05-30T15", dt=3600)
    grid = ROMS.Grid(filename, subgrid=subgrid,)
    force = ROMS.Forcing(grid, timer, filename, ibm_forcing=["temp"])

    # First field
    force.update(force.steps[1], X, Y, Z)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[1])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[1])

    # Do some updating
    force.update(force.steps[1] + 1, X, Y, Z)
    force.update(force.steps[2], X, Y, Z)
    force.update(force.steps[2] + 1, X, Y, Z)

    # Check third field (4th on file)
    force.update(force.steps[3], X, Y, Z)
    assert force.velocity(X, Y, Z)[0][0] == pytest.approx(U_dir[3])
    assert force.variables["temp"][0] == pytest.approx(temp_dir[3])


if __name__ == "__main__":
    test_midstep()
