# Move particle in a circle (clockwise)

# from collections import namedtuple
from typing import Tuple, Union, Sequence, Callable

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from state import State

Vector = Union[float, np.ndarray]


def get_velocity1(
    state: State,
    sample_func: Callable[[Vector, Vector], Tuple[Vector, Vector]],
    dt: int,
) -> Tuple[Vector, Vector]:
    """Euler Forward"""
    x0, y0 = state.X, state.Y
    return sample_func(x0, y0)


def get_velocity2(
    state: State,
    sample_func: Callable[[Vector, Vector], Tuple[Vector, Vector]],
    dt: int,
) -> Tuple[Vector, Vector]:
    """2nd order Runge Kutta

        s = 1/2  gives midpoint
        s = 2/3  gives Ralston
        s = 1    gives Heun

    For circular motion, s is eliminated, all schemes gives same result
    also elliptical.

    """
    s = 1.0
    m = 1.0 / (2 * s)
    x0, y0 = state.X, state.Y
    u0, v0 = sample_func(x0, y0)
    x1, y1 = x0 + s * dt * u0, y0 + s * dt * v0
    u1, v1 = sample_func(x1, y1)
    return ((1 - m) * u0 + m * u1, (1 - m) * v0 + m * v1)


def get_velocity4(
    state: State,
    sample_func: Callable[[Vector, Vector], Tuple[Vector, Vector]],
    dt: int,
) -> Tuple[Vector, Vector]:
    """4th order Runge Kutta"""
    x0, y0 = state.X, state.Y
    u0, v0 = sample_func(x0, y0)
    x1, y1 = x0 + 0.5 * dt * u0, y0 + 0.5 * dt * v0
    u1, v1 = sample_func(x1, y1)
    x2, y2 = x0 + 0.5 * dt * u1, y0 + 0.5 * dt * v1
    u2, v2 = sample_func(x2, y2)
    x3, y3 = x0 + dt * u2, y0 + dt * v2
    u3, v3 = sample_func(x3, y3)
    return (u0 + 2 * u1 + 2 * u2 + u3) / 6, (v0 + 2 * v1 + 2 * v2 + v3) / 6
