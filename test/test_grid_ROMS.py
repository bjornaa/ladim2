import numpy as np
from netCDF4 import Dataset
import pytest

import ladim2
from ladim2.grid_ROMS import Grid

grid_file = "../examples/data/ocean_avg_0014.nc"


def test_ok():
    """Test minimal correct configuration"""
    g = Grid(grid_file=grid_file, dt=180)
    assert g.xmin == 1.0
    assert g.xmax == 180.0
    assert g.ymin == 1.0
    assert g.ymax == 190.0
    assert g.N == 32
    assert g.H.shape == (190, 180)


def test_nofile():
    """Capture wrong file name"""
    with pytest.raises(SystemExit):
        g = Grid(grid_file="does_not_exist.nc", dt=180)


def test_subgrid():
    """Test subgrid specification"""
    i0, i1, j0, j1 = 20, 150, 30, 170
    g = Grid(grid_file=grid_file, subgrid=(i0, i1, j0, j1), dt=180)
    assert g.xmin == i0
    assert g.xmax == i1 - 1
    assert g.ymin == j0
    assert g.ymax == j1 - 1
    assert g.imax == i1 - i0
    assert g.jmax == j1 - j0
    assert g.H.shape == (j1 - j0, i1 - i0)
    assert g.z_r.shape == (32, j1 - j0, i1 - i0)
    with Dataset(grid_file) as ncid:
        g.H[10, 20] = ncid.variables["h"][j0 + 10, i0 + 20]


def test_subgrid_error():
    """Capture errors in the subgrid specification"""
    with pytest.raises(SystemExit):
        g = Grid(grid_file=grid_file, subgrid=(20, 150, 30, 222), dt=180)
    with pytest.raises(SystemExit):
        g = Grid(grid_file=grid_file, subgrid=(1200, 20, 30, 180), dt=180)
