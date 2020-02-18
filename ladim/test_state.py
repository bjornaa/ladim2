import numpy as np  # type: ignore
import pytest  # type: ignore

from state import State  # type: ignore

# ------------
# __init__
# ------------


def test_minimal():
    """Init State with no arguments"""
    S = State()
    assert len(S) == 0
    assert S.npid == 0
    assert set(S.variables) == {"pid", "alive", "X", "Y", "Z"}
    assert S.pid.dtype == int
    assert S.alive.dtype == bool
    assert S.X.dtype == np.float64
    assert S.Y.dtype == float
    assert S.Z.dtype == "float"
    assert S.default_values["alive"] == True
    assert np.all(S.Y == [])


def test_init_args():
    """Init State with extra variables"""
    S = State(age=float, stage=int)
    assert S.age.dtype == float
    assert np.all(S.age == [])


def test_init_args_dict():
    """Init state with extra variables in a dictionary"""
    D = {"age": np.float64, "stage": int}
    S = State(**D)
    assert S.age.dtype == float
    assert np.all(S.age == [])


# -------------------------
# set_default_values
# -------------------------


def test_set_default():
    S = State(age=float, stage=int)
    S.set_default_values(age=0, stage=1, Z=5)
    assert S.default_values["alive"] == True
    assert S.default_values["age"] == 0
    assert S.default_values["stage"] == 1
    assert S.default_values["Z"] == 5.0


def test_set_default_err1():
    """Trying to set default for an undefined variable"""
    S = State(age=float)
    with pytest.raises(ValueError):
        S.set_default_values(length=4.3)


def test_set_default_err2():
    """Trying to set default for pid"""
    S = State(age=float)
    with pytest.raises(ValueError):
        S.set_default_values(pid=4)


def test_set_default_err3():
    """Trying to set an array as default value"""
    S = State(length=float)
    with pytest.raises(TypeError):
        S.set_default_values(length=[1.2, 4.3])


# --------------------
# append
# --------------------


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
    assert np.all(state.alive == [True, True])
    assert np.all(state.X == [200.0, 201.0])
    assert np.all(state.Y == [100.0, 100.0])
    assert np.all(state.Z == [5.0, 10.0])


def test_extra_variables():
    state = State(age=float, stage="int")
    assert len(state) == 0
    assert state.age.dtype == float
    assert state.stage.dtype == int
    state.set_default_values(age=1.0)
    state.append(X=1, Y=2, Z=3, stage=4)
    assert len(state.age) == 1
    assert np.all(state.age == [1])
    assert np.all(state.stage == [4])


def test_append_illegal_variable():
    """Append an undefined variable"""
    state = State(age=float)
    with pytest.raises(ValueError):
        state.append(X=1, Y=2, Z=3, age=0, length=20)


def test_missing_default():
    state = State(age=float, stage=int)
    # No default for stage
    state.set_default_values(age=0.0)
    with pytest.raises(TypeError):
        state.append(X=1, Y=2, Z=3)


# --------------
# Compactify
# --------------


def test_compactify():
    S = State()
    S.set_default_values(Z=5)
    S.append(X=[10, 11], Y=[1, 2])
    assert len(S) == 2
    S.append(X=[21, 22], Y=[3, 4])
    assert len(S) == 4
    # Remove second particle
    S.alive[1] = False
    S.compactify()
    assert len(S) == 3
    assert np.all(S.alive == True)
    assert np.all(S.pid == [0, 2, 3])
    assert np.all(S.X == [10, 21, 22])
    assert S.X.flags["C_CONTIGUOUS"]
