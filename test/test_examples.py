import os
from contextlib import contextmanager
import shutil
from ladim2.main import main as run_ladim
import importlib.util
from unittest.mock import patch


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')


def test_killer_matrix():
    with create_tempdir('killer'):
        run_module('make_release.py')
        run_ladim('matrix.yaml')
        run_animate('animate_matrix.py')


def test_killer():
    with create_tempdir('killer'):
        run_module('make_release.py')
        run_ladim('ladim2.yaml')
        run_animate('animate.py')


def test_line_matrix():
    with create_tempdir('line'):
        run_module('make_release.py')
        run_ladim('matrix.yaml')
        run_animate('animate_matrix.py')


def test_line():
    with create_tempdir('line'):
        run_module('make_release.py')
        run_ladim('ladim2.yaml')
        run_animate('animate.py')


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
        shutil.rmtree(temp_dir, ignore_errors=False)
        assert not os.path.exists(temp_dir)


def run_module(path):
    from uuid import uuid4
    internal_name = 'module_' + uuid4().hex
    spec = importlib.util.spec_from_file_location(internal_name, path)
    module_object = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_object)


def run_animate(path):
    import matplotlib.pyplot as plt
    with patch.object(plt, 'show', return_value=None) as mock_method:
        run_module(path)
        mock_method.assert_called_once()
