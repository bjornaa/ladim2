import numpy as np
import pytest
from .state import State

def test_ok():
    s = State()
    s.append(X=[100, 110], Y= [200, 210], Z=5)
    assert all(s.X == [100, 110])


def test_update_error_not_variable():
    s = State()
    s.append(X=[100, 110], Y= [200, 210], Z=5)
    with pytest.raises(KeyError):
        s["Lon"] = [4.5, 4.6]

def test_update_error_wrong_size():
    # Alternative broadcast the scalar, equivalent to s["X"] = [110, 100]
    s = State()
    s.append(X=[100, 110], Y= [200, 210], Z=5)
    with pytest.raises(KeyError):
        s["X"] = 110
    with pytest.raises(KeyError):
        s["X"] = [101, 111, 121]

def test_update_item():
    s = State()
    s.append(X=[100, 110], Y= [200, 210], Z=5)
    s["X"] += 1
    assert all(s.variables["X"] == [101, 111])

def test_update_attr():
    s = State()
    s.append(X=[100, 110], Y= [200, 210], Z=5)
    s.X += 1
    assert all(s.X == [101, 111])
    assert all(s.variables["X"] == [101, 111])

def test_missing_initial():
    """Need initial values"""
    s = State()
    with pytest.raises(TypeError):
        s.append(X=[100, 110], Z=5)

    s = State(dict(X0=float), particle_variables = ["X0"])
    with pytest.raises(TypeError):
        s.append(X=[100, 110], Y= [200, 210], Z=5)
