import os
from pathlib import Path
from contextlib import contextmanager
import shutil
import runpy

import yaml  # type: ignore
from netCDF4 import Dataset

from ladim.main import main as run_ladim


EXAMPLE_DIR = Path(__file__).parents[1] / "examples"


def test_streak():
    name = "streak"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "ladim.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_streak_ibm():
    name = "streak"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "age_ibm.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        shutil.copy(example_dir / "age_ibm.py", temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_station():
    name = "station"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "ladim.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


# Improve this
def test_restart():
    name = "restart"
    example_dir = EXAMPLE_DIR / name
    yaml_file0 = "unsplit.yaml"
    yaml_file1 = "split.yaml"
    yaml_file2 = "restart.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file0, temp_dir)
        shutil.copy(example_dir / yaml_file1, temp_dir)
        shutil.copy(example_dir / yaml_file2, temp_dir)
        shutil.copy(example_dir / "verify_restart.py", temp_dir)
        runpy.run_path("make_release.py")
        # Unsplit
        run_ladim(yaml_file0)
        verify_output(yaml_file0)
        # Split and Restart
        run_ladim(yaml_file1)
        run_ladim(yaml_file2)
        runpy.run_path("verify_restart.py")




def test_gosouth():
    name = "gosouth"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "ladim.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        shutil.copy(example_dir / "gosouth_ibm.py", temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_latlon():
    name = "latlon"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "ladim.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_killer():
    name = "killer"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "ladim.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        shutil.copy(example_dir / "killer_ibm.py", temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_killer_matrix():
    name = "killer"
    example_dir = EXAMPLE_DIR / name
    yaml_file = "dense.yaml"
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file, temp_dir)
        shutil.copy(example_dir / "killer_ibm.py", temp_dir)
        runpy.run_path("make_release.py")
        run_ladim(yaml_file)
        verify_output(yaml_file)


def test_line():
    name = "line"
    example_dir = EXAMPLE_DIR / name
    yaml_file1 = "ladim.yaml"  # sparse output
    yaml_file2 = "dense.yaml"  # dense output
    with create_tempdir(name) as temp_dir:
        shutil.copy(example_dir / "make_release.py", temp_dir)
        shutil.copy(example_dir / yaml_file1, temp_dir)
        shutil.copy(example_dir / yaml_file2, temp_dir)
        os.chdir(temp_dir)
        runpy.run_path("make_release.py")
        # Ordinary sparse output test
        run_ladim(yaml_file1)
        verify_output(yaml_file1)
        # Dense output test
        run_ladim(yaml_file2)
        verify_output(yaml_file2)

# ----------------------------------------------------


@contextmanager
def create_tempdir(name):
    curdir = Path.cwd()
    temp_dir = EXAMPLE_DIR / (name + "_temp")

    try:
        Path.mkdir(temp_dir)
        os.chdir(temp_dir)
        yield temp_dir

    finally:
        os.chdir(curdir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        assert not os.path.exists(temp_dir)


def verify_output(yaml_file):
    """Verify that LADiM has created a valid output file"""
    # Get name of ladim output file
    with open(yaml_file) as fid:
        d = yaml.safe_load(fid)
        out_file = Path(d["output"]["filename"])

    # Assert that the result_file exists
    assert out_file.exists()

    # Assert that the file is netcdf and probably OK
    with Dataset(out_file) as nc:
        # Common tests for sparse and dense
        assert "time" in nc.dimensions
        assert ("X" in nc.variables) or ("lon" in nc.variables)
        assert ("Y" in nc.variables) or ("lat" in nc.variables)
        # The dense file format is recognized by a global attribute "type"
        # containing the substring "dense"
        type_attribute = getattr(nc, "type", "")
        if "dense" in type_attribute:
            assert "particle" in nc.dimensions
        else:  # Default sparse format
            assert "particle_instance" in nc.dimensions
            assert "particle_count" in nc.variables
            assert "pid" in nc.variables
