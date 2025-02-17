"""Utility functions for sampling output from the ROMS

Horizontal sampling
-------------------

:func:`sample2D`
  Sample a 2D field given at rho-points
:func:`sample2DU`
  Sample a 2D field given at u-points
:func:`sample2DV`
  Sample a 2D field given at v-points

"""

# -----------------------------------
# Bjørn Ådlandsvik, bjorn@imr.no
# Institute of Marine Research
# Bergen, Norway
# 2010-09-30
# -----------------------------------

from __future__ import annotations

from typing import Optional

import numpy as np

# Type aliases
Field = np.ndarray[tuple[int, int], np.dtype[np.float64]]
ParticleArray = np.ndarray[tuple[int], np.dtype[np.float64]]


# ---------------------


def sample2D2(F: Field, X: ParticleArray, Y: ParticleArray) -> ParticleArray:
    """Bilinear sample of a 2D field

    *F* : 2D array

    *X*, *Y* : position in grid coordinates, scalars or compatible arrays

    Note reversed axes, for integers i and j we have
      ``sample2D(F, i, j) = F[j,i]``

    Using linear interpolation

    """

    I: np.ndarray[tuple[int], np.dtype[np.int64]] = X.astype("int")
    J: np.ndarray[tuple[int], np.dtype[np.int64]] = Y.astype("int")
    P = X - I
    Q = Y - J

    W00 = (1 - P) * (1 - Q)
    W01 = (1 - P) * Q
    W10 = P * (1 - Q)
    W11 = P * Q

    result: ParticleArray = (
        W00 * F[J, I] + W01 * F[J + 1, I] + W10 * F[J, I + 1] + W11 * F[J + 1, I + 1]
    )
    return result


# --------------------------------------------------


def sample2DUV(
    U: Field, V: Field, X: ParticleArray, Y: ParticleArray
) -> tuple[ParticleArray, ParticleArray]:
    """2D interpolation of velocity"""
    return sample2D(U, X + 0.5, Y), sample2D(V, X, Y + 0.5)


# -------------------------------------------------


def sample2D_masked(
    F: Field, M: Field, X: ParticleArray, Y: ParticleArray
) -> ParticleArray:
    """Bilinear sample of a 2D field

    F = 2D array, M = mask (=1 on sea, = 0 on land)
    X, Y = position in grid coordinates, scalars or compatible arrays

    Note reversed axes, for integers i and j we have
    sample2D(F, i, j) = F[j,i]

    Using linear interpolation

    """

    masked = True
    I: np.ndarray[tuple[int], np.dtype[np.int64]] = X.astype("int")
    J: np.ndarray[tuple[int], np.dtype[np.int64]] = Y.astype("int")
    P = X - I
    Q = Y - J

    W00 = M[J, I] * (1 - P) * (1 - Q)
    W01 = M[J + 1, I] * (1 - P) * Q
    W10 = M[J, I + 1] * P * (1 - Q)
    W11 = M[J + 1, I + 1] * P * Q
    if masked:
        W00 = M[J, I] * W00
        W01 = M[J + 1, I] * W01
        W10 = M[J, I + 1] * W10
        W11 = M[J + 1, I + 1] * W11
    SW = W00 + W01 + W10 + W11
    F_interp: ParticleArray = np.where(
        SW > 0,
        (W00 * F[J, I] + W01 * F[J + 1, I] + W10 * F[J, I + 1] + W11 * F[J + 1, I + 1])
        / SW,
        0.0,
    )
    # F_interp = np.ma.masked_where(SW == 0, F_interp)

    return F_interp


# ------------------


def sample2D(
    F: Field,
    X: ParticleArray,
    Y: ParticleArray,
    mask: Optional[Field] = None,
    undef_value: float = 0.0,
    outside_value: Optional[float] = None,
) -> ParticleArray:
    """Bilinear sample of a 2D field

    *F* : 2D array

    *X*, *Y* : position in grid coordinates, scalars or compatible arrays

    *mask* : if present must be a 2D matrix with 1 at valid
             and zero at non-valid points

    *undef_value* : value to put at undefined points

    *outside_value* : value to return outside the grid
                defaults to None,
                raising ValueError if any points are outside

    Note reversed axes, for integers i and j we have
      ``sample2D(F, i, j) = F[j,i]``

    If jmax, imax = F.shape
    then inside values requires 0 <= x < imax-1, 0 <= y < jmax-1

    Using bilinear interpolation

    """

    jmax, imax = F.shape

    # Find integer I, J such that
    # 0 <= I <= X < I+1 <= imax-1, 0 <= J <= Y < J+1 <= jmax-1
    # and local increments P and Q
    I: np.ndarray[tuple[int], np.dtype[np.int64]] = np.asarray(X).astype("int")
    J: np.ndarray[tuple[int], np.dtype[np.int64]] = np.asarray(Y).astype("int")
    P = X - I
    Q = Y - J
    outside = (X < 0) | (X >= imax - 1) | (Y < 0) | (Y >= jmax - 1)
    if np.any(outside):
        if outside_value is None:
            raise ValueError("point outside grid")
        I = np.where(outside, 0, I)
        J = np.where(outside, 0, J)

    # Weights for bilinear interpolation
    W00 = (1 - P) * (1 - Q)
    W01 = (1 - P) * Q
    W10 = P * (1 - Q)
    W11 = P * Q
    SW: float | ParticleArray = 1.0  # Sum of weights

    if mask is not None:
        W00 = mask[J, I] * W00
        W01 = mask[J + 1, I] * W01
        W10 = mask[J, I + 1] * W10
        W11 = mask[J + 1, I + 1] * W11
        SW = W00 + W01 + W10 + W11

    SW = np.where(SW == 0, -1.0, SW)  # atype: ignore
    result: ParticleArray = np.where(
        SW <= 0,
        undef_value,
        (W00 * F[J, I] + W01 * F[J + 1, I] + W10 * F[J, I + 1] + W11 * F[J + 1, I + 1])
        / SW,
    )

    # Set in outside_values
    if outside_value:
        result = np.where(outside, outside_value, result)

    return result


# -----------------------------------------


def bilin_inv(
    f: ParticleArray,
    g: ParticleArray,
    F: Field,
    G: Field,
    maxiter: int = 7,
    tol: float = 1.0e-7,
) -> tuple[ParticleArray, ParticleArray]:
    """Inverse bilinear interpolation

    f, g : scalars or 1D arrays of same length
    F, G : 2D arrays of the same shape

    returns x, y : shaped like f and g
    such that F and G linearly interpolated to x, y returns f and g

    """

    imax, jmax = F.shape
    if G.shape != (imax, jmax):
        msg = "Shape mismatch in 2D arrays"
        raise ValueError(msg)

    f = np.asarray(f)
    g = np.asarray(g)
    fshape = f.shape
    if g.shape != fshape:
        raise ValueError("Targets must have the same length")

    # initial guess = midpoint
    x: ParticleArray = np.zeros_like(f) + 0.5 * imax
    y: ParticleArray = np.zeros_like(f) + 0.5 * jmax

    for _t in range(maxiter):
        i: np.ndarray[tuple[int], np.dtype[np.int64]] = x.astype(int)
        j: np.ndarray[tuple[int], np.dtype[np.int64]] = y.astype(int)
        p, q = x - i, y - j

        # Bilinear estimate of F[x,y] and G[x,y]
        Fs = (
            (1 - p) * (1 - q) * F[i, j]
            + p * (1 - q) * F[i + 1, j]
            + (1 - p) * q * F[i, j + 1]
            + p * q * F[i + 1, j + 1]
        )
        Gs = (
            (1 - p) * (1 - q) * G[i, j]
            + p * (1 - q) * G[i + 1, j]
            + (1 - p) * q * G[i, j + 1]
            + p * q * G[i + 1, j + 1]
        )

        H = (Fs - f) ** 2 + (Gs - g) ** 2
        if np.all(H < tol):
            break

        # Estimate Jacobi matrix
        Fx = (1 - q) * (F[i + 1, j] - F[i, j]) + q * (F[i + 1, j + 1] - F[i, j + 1])
        Fy = (1 - p) * (F[i, j + 1] - F[i, j]) + p * (F[i + 1, j + 1] - F[i + 1, j])
        Gx = (1 - q) * (G[i + 1, j] - G[i, j]) + q * (G[i + 1, j + 1] - G[i, j + 1])
        Gy = (1 - p) * (G[i, j + 1] - G[i, j]) + p * (G[i + 1, j + 1] - G[i + 1, j])

        # Newton-Raphson step
        # Jinv = np.linalg.inv([[Fx, Fy], [Gx, Gy]])
        # incr = - np.dot(Jinv, [Fs-f, Gs-g])
        # x = x + incr[0], y = y + incr[1]
        det = Fx * Gy - Fy * Gx
        x -= (Gy * (Fs - f) - Fy * (Gs - g)) / det
        y -= (-Gx * (Fs - f) + Fx * (Gs - g)) / det

    return x, y


# ----------------------------
