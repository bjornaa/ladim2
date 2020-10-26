import numpy as np  # type: ignore
import pytest  # type: ignore

from ladim2.state import State  # type: ignore

# ------------
# __init__
# ------------


def test_minimal():
    """Init State with no arguments"""
    S = State()
    assert len(S) == 0
    assert S.npid == 0
    assert set(S.variables) == {"pid", "X", "Y", "Z", "active", "alive"}
    assert S.instance_variables == {"pid", "X", "Y", "Z", "active", "alive"}
    assert S.particle_variables == set()
    assert S.pid.dtype == int
    assert all(S.pid == [])
    assert S.X.dtype == np.float64
    assert all(S.variables["X"] == [])
    assert all(S["X"] == [])
    assert all(S.X == [])
    assert S.alive.dtype == bool
    assert S.Y.dtype == float
    assert S.Z.dtype == "f8"
    assert S.default_values["alive"] == True


def test_init_args():
    """Init State with extra variables"""
    S = State(
        variables=dict(age=float, stage=int, release_time=np.datetime64),
        particle_variables=["release_time"],
    )
    assert "age" in S.instance_variables
    assert S.age.dtype == float
    assert S.stage.dtype == int
    assert all(S.age == [])
    assert S.particle_variables == {"release_time"}
    assert S.release_time.dtype == np.datetime64
    # cassert call(S.release_time == [])  # Does not work
    assert np.all(S.release_time == np.array([], np.datetime64))


# -------------------------
# set_default_values
# -------------------------


def test_set_default():
    S = State(variables=dict(age=float, stage=int))
    S.set_default_values(age=0, stage=1, Z=5)
    assert S.default_values["active"] == True
    assert S.default_values["alive"] == True
    assert S.default_values["age"] == 0
    assert S.default_values["stage"] == 1
    assert S.default_values["Z"] == 5.0


def test_set_default_err1():
    """Trying to set default for an undefined variable"""
    S = State(variables={"age": float})
    with pytest.raises(ValueError):
        S.set_default_values(length=4.3)


def test_set_default_err2():
    """Trying to set default for pid"""
    S = State(dict(age=float))
    with pytest.raises(ValueError):
        S.set_default_values(pid=4)


def test_set_default_err3():
    """Trying to set an array as default value"""
    S = State(dict(length=float))
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
    assert np.all(state.active == [True])
    assert np.all(state.alive == [True])
    assert np.all(state.X == [200])


def test_append_array():
    """Append an array to a non-empty state"""
    state = State()
    state.append(X=200, Z=5, Y=100)
    length = len(state)
    npid = state.npid
    X = state.X
    state.append(X=np.array([201, 202]), Y=110, Z=[5, 10])
    assert len(state) == length + 2
    assert state.npid == npid + 2
    assert np.all(state.pid == [0, 1, 2])
    assert np.all(state["pid"] == [0, 1, 2])
    assert np.all(state.variables["pid"] == [0, 1, 2])
    assert np.all(state.active == 3 * [True])
    assert np.all(state.alive == 3 * [True])
    assert np.all(state.X == [200, 201.0, 202.0])
    assert np.all(state["X"] == [200, 201.0, 202.0])
    assert np.all(state.variables["X"] == [200, 201.0, 202.0])
    assert np.all(state.Y == [100.0, 110.0, 110.0])
    assert np.all(state.Z == [5.0, 5.0, 10.0])


def test_extra_variables():
    state = State(dict(age=float, stage="int"))
    assert len(state) == 0
    assert state.age.dtype == float
    assert state.stage.dtype == int
    state.set_default_values(age=1.0)
    state.append(X=1, Y=2, Z=3, stage=4)
    assert len(state.age) == 1
    assert np.all(state.age == [1])
    assert np.all(state.stage == [4])


def test_append_nonvariable():
    """Append an undefined variable"""
    state = State(dict(age=float))
    with pytest.raises(ValueError):
        state.append(X=1, Y=2, Z=3, age=0, length=20)


def test_missing_default():
    state = State(dict(age=float, stage=int))
    # No default for stage
    state.set_default_values(age=0.0)
    with pytest.raises(TypeError):
        state.append(X=1, Y=2, Z=3)


def test_not_append_pid():
    """Can not append to pid"""
    S = State()
    with pytest.raises(ValueError):
        S.append(X=10, Y=20, Z=5, pid=101)


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
    # Kill second particle
    S.alive[1] = False
    S.compactify()
    assert len(S) == 3
    assert S.npid == 4
    assert np.all(S.active == True)
    assert np.all(S.alive == True)
    assert np.all(S.pid == [0, 2, 3])
    assert np.all(S.X == [10, 21, 22])
    # The arrays should be contiguous after removing an element
    assert S.X.flags["C_CONTIGUOUS"]


def test_not_compactify_particle_variables():
    S = State(variables=dict(age=float, X0=float), particle_variables=["X0"])
    S.set_default_values(Z=5, age=0)
    X0 = [10, 11, 12, 13]
    Y0 = [20, 21, 22, 23]
    S.append(X=X0, Y=Y0, X0=X0)
    S.alive[1] = False
    S.compactify()
    assert len(S) == 3
    assert all(S.pid == [0, 2, 3])
    assert all(S.X == [10, 12, 13])
    assert len(S.age) == 3
    # particle_variable X0 is not compactified
    assert all(S.X0 == X0)

def test_update_and_append_and_compactify():
    """Check that updating bug has been fixed"""
    S = State()

    # One particle
    S.append(X=100, Y=10, Z=5)
    assert all(S.pid == [0])
    assert all(S.X == [100])

    # Update position
    S["X"] += 1
    assert all(S.X == [101])

    # Update first particle and add two new particles
    S["X"] += 1
    S.append(X=np.array([200, 300]), Y=np.array([20, 30]), Z=5)
    assert all(S.X == [102, 200, 300])

    # Update particle positions and kill the first particle, pid=0
    S["X"] = S["X"] + 1.0
    S["alive"][0] = False
    S.compactify()
    assert all(S.X == [201, 301])

    # Update positions
    S["X"] = S["X"] + 1
    assert all(S.X == [202, 302])
