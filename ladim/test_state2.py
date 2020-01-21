import numpy as np
import pytest

from state import State


def test_minimal():
    state = State()
    assert len(state) == 0
    assert state.npid == 0
    assert set(state.variables) == set(["pid", "alive", "X", "Y", "Z"])
    assert np.all(state.pid == [])
    assert np.all(state.alive == [])
    assert np.all(state.X == [])
    assert np.all(state.Y == [])
    assert np.all(state.Z == [])
    assert state.pid.dtype == int
    assert state.alive.dtype == bool
    assert state.X.dtype == float


def test_append_scalar():
    state = State()
    state.append(X=200, Z=5, Y=100)
    assert len(state) == 1
    assert state.npid == 1
    assert np.all(state.pid == [0])
    assert np.all(state.alive == [True])
    assert np.all(state.X == [200])


def test_append_array():
    state = State()
    state.append(X=np.array([200, 201]), Y=100, Z=[5, 10])
    assert len(state) == 2
    assert state.npid == 2
    assert np.all(state.pid == [0, 1])
    print("----", state.alive)
    assert np.all(state.alive == [True, True])
    assert np.all(state.X == [200.0, 201.0])
    assert np.all(state.Y == [100.0, 100.0])
    assert np.all(state.Z == [5.0, 10.0])


def test_extra_variables():
    state = State(age=float, stage="int")
    assert len(state) == 0
    assert state.age.dtype == float
    assert state.stage.dtype == int
    state.set_default_value("age", 1.0)
    state.append(X=1, Y=2, Z=3, stage=4)
    assert len(state.age) == 1
    assert np.all(state.age == [1])
    assert np.all(state.stage == [4])
