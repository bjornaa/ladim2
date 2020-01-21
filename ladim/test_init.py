import numpy as np
import pytest
from state import State


def test_minimal_init():
    """Init State with no arguments"""
    S = State()
    assert S.npid == 0
    assert set(S.variables) == {"pid", "alive", "X", "Y", "Z"}
    assert S.pid.dtype == int
    assert S.alive.dtype == bool
    assert S.X.dtype == np.float64
    assert S.Y.dtype == float
    assert S.Z.dtype == "float"
    assert S.default_values["alive"] == True


def test_init_args():
    """Init State with extra variables"""
    S = State(age=float, stage=int)
    assert S.age.dtype == float


def test_init_args_dict():
    """Init state with extra variables in a dictionary"""
    D = {"age": np.float64, "stage": int}
    S = State(**D)
    assert S.age.dtype == float


def test_set_default():
    S = State(age=float, stage=int)
    S.set_default_values(age=0, stage=1, Z=5)
    assert S.default_values["alive"] == True
    assert S.default_values["age"] == 0
    assert S.default_values["stage"] == 1
    assert S.default_values["Z"] == 5.0


def test_set_default_err1():
    """Error if trying to set default for an undefined variable"""
    S = State(age=float)
    with pytest.raises(ValueError):
        S.set_default_values(length=4.3)


def test_set_default_err2():
    """Error if trying to set default for pid"""
    S = State(age=float)
    with pytest.raises(ValueError):
        S.set_default_values(pid=4)


def test_set_default_err2():
    """Error if trying to set an array as default value"""
    S = State(length=float)
    with pytest.raises(TypeError):
        S.set_default_values(length=[1.2, 4.3])
