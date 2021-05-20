import os
import pytest
from contextlib import contextmanager
import shutil
from ladim2.main import main
import importlib.util
from unittest.mock import patch


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
EXAMPLE_NAMES = ['killer']


@contextmanager
def create_tempdir(name):
    curdir = os.getcwd()
    example_dir = os.path.join(EXAMPLE_DIR, name)
    temp_dir = os.path.join(EXAMPLE_DIR, name + '_temp')

    try:
        shutil.copytree(example_dir, temp_dir)
        os.chdir(temp_dir)
        yield temp_dir

    finally:
        os.chdir(curdir)
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_module(path):
    from uuid import uuid4
    internal_name = 'module_' + uuid4().hex
    spec = importlib.util.spec_from_file_location(internal_name, path)
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_example(name):
    with create_tempdir(name):
        # Create release file
        if os.path.exists('make_release.py'):
            run_module('make_release.py')

        # Create output file
        if os.path.exists('ladim2.yaml'):
            main('ladim2.yaml')

        # Create animation, except for showing it on screen
        if os.path.exists('animate.py'):
            import matplotlib.pyplot as plt
            with patch.object(plt, 'show', return_value=None) as mock_method:
                run_module('animate.py')
                mock_method.assert_called_once()
