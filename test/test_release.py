# import os
from io import StringIO

import numpy as np  # type: ignore

# import pandas as pd
import pytest
from ladim.release import ParticleReleaser
from ladim.timekeeper import TimeKeeper


class Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


timer = TimeKeeper(
    start=np.datetime64("2015-03-31 12"),
    stop=np.datetime64("2015-04-04"),
    dt=np.timedelta64(15, "m"),
)
datatypes0 = dict(mult=int, X=float, Y=float, Z=float)
state0 = Dummy(dtypes=datatypes0)
modules0 = dict(time=timer, state=state0, grid=None)


def test_read_release():
    f = StringIO(
        """
        release_time      X    Y    Z   super
        2015-04-01      100  200    5    1000
        2015-04-01T00   111  220    5    2000
        "2015-04-03 12" 200  300.0  5    3333
    """
    )
    datatypes = dict(datatypes0, super=int)

    A = ParticleReleaser.read_release_file(f, datatypes)
    assert len(A) == 3
    assert A.index[1] == np.datetime64("2015-04-01 00:00:00")
    assert all(A.X == [100.0, 111.0, 200.0])
    assert all(A.super == [1000, 2000, 3333.0])


def test_read_release_no_header():
    f = StringIO(
        """
        2    2015-04-01     100 200   5
        1    2015-04-01T00  111 220   5
        3   "2015-04-03 12" 200 300.0 5
    """
    )

    A = ParticleReleaser.read_release_file(
        f, datatypes0, names=["mult", "release_time", "X", "Y", "Z"]
    )
    assert len(A) == 3
    assert A.index[1] == np.datetime64("2015-04-01 00:00:00")
    assert all(A.X == [100.0, 111.0, 200.0])


def test_read_release_no_header_no_names():
    f = StringIO(
        """
        2    2015-04-01     100 200   5
        1    2015-04-01T00  111 220   5
        3   "2015-04-03 12" 200 300.0 5
    """
    )

    # Have more explicit error
    with pytest.raises(ValueError):
        ParticleReleaser.read_release_file(f, datatypes0)


def test_read_release_both_header_names():
    """Do not accept header in file and as argument"""
    # Chang to priority for file??

    f = StringIO(
        """
        mult release_time     X   Y   Z
        2    2015-04-01     100 200   5
        1    2015-04-01T00  111 220   5
        3   "2015-04-03 12" 200 300.0 5
    """
    )

    with pytest.raises(ValueError):
        ParticleReleaser.read_release_file(
            f, datatypes0, names=["mult", "release_time", "X", "Y", "Z"]
        )


def test_clean_add_mult():
    """If no mult column, clean_release_data should add mult=1"""

    f = StringIO(
        """
        release_time     X   Y   Z
        2015-04-01     100 200   5
        2015-04-01T00  111 220   5
        "2015-04-03 12" 200 300.0 5
    """
    )

    pr = ParticleReleaser(modules0, f)
    assert all(pr._df.mult == [1, 1, 1])


def test_clean_nopos():
    """Missing position info in release file"""

    f = StringIO(
        """
        release_time    Z
        2015-04-01      5
        2015-04-01T00   5
        "2015-04-03 12" 5
    """
    )

    with pytest.raises(SystemExit):
        ParticleReleaser(modules0, f)


def test_clean_convert_lonlat():
    """Test automatic conversion from geographical to grid coordinates"""

    # Use grid coords = lon, 2*lat (suitable for 60 deg north)
    class Grid:
        def ll2xy(self, lon, lat):
            return lon, 2 * lat

    # grid = Grid()

    f = StringIO(
        """
        release_time     lon   lat   Z
        2015-04-01         4    60   5
        2015-04-01T00    4.2  59.5   5
        "2015-04-03 12"  3.9  58.0   5
    """
    )

    modules = dict(modules0, grid=Grid())
    pr = ParticleReleaser(modules, f)
    df = pr._df
    assert all(df["Y"] == 2 * np.array([60, 59.5, 58.0]))


def test_clean_lonlat_no_convert():
    """lon/lat gives exception if no ll2xy in Grid"""

    f = StringIO(
        """
        release_time     lon   lat   Z
        2015-04-01         4    60   5
        2015-04-01T00    4.2  59.5   5
        "2015-04-03 12"  3.9  58.0   5
    """
    )

    with pytest.raises(SystemExit):
        ParticleReleaser(modules0, f)


def test_remove_late_release():
    """Remove releases after stop"""

    f = StringIO(
        """
        release_time       X     Y   Z
        2015-04-01         4    60   5
        2015-05-01T00    4.2  59.5   5
        "2015-05-03 12"  3.9  58.0   5
    """
    )
    pr = ParticleReleaser(modules0, f)
    df = pr._df
    assert len(pr.steps) == 1
    assert pr.steps[0] == 12 * 4  # 12 hours, 4 steps per hour
    assert all(df.index == np.datetime64("2015-04-01"))
    assert df.X[0] == 4


def test_remove_early_release():
    """Remove too early release entries"""

    # time[0] < time[1] == start < time[2] < stop
    f = StringIO(
        """
        release_time       X     Y   Z
        2015-03-01         4    60   5
        2015-03-31T12    4.2  59.5   5
        "2015-04-03 12"  3.9  58.0   5
    """
    )
    pr = ParticleReleaser(modules0, f)
    df = pr._df
    assert len(pr.steps) == 2
    assert pr.steps[0] == 0
    assert pr.steps[1] == 3 * 24 * 4  # 3 days, 4 steps per hour
    assert df.index[0] == np.datetime64("2015-03-31 12:00:00")
    assert df.index[1] == np.datetime64("2015-04-03 12", "s")
    assert all(df.X == [4.2, 3.9])


def test_remove_early_release2():
    """Remove too early release entries, no entry at start"""

    # time[0] < time[1] < start < time[2] < stop
    f = StringIO(
        """
        release_time       X     Y   Z
        2015-03-01         4    60   5
        2015-03-15T00:00 4.2  59.5   5
        "2015-04-03 12"  3.9  58.0   5
    """
    )
    pr = ParticleReleaser(modules0, f)
    df = pr._df
    assert len(pr.steps) == 1
    assert pr.steps[0] == 3 * 24 * 4  # 3 days, 4 steps per hour
    assert df.index[0] == np.datetime64("2015-04-03 12:00:00")
    assert all(df.X == [3.9])


def test_too_late_release():
    """All release after stop"""

    f = StringIO(
        """
        release_time       X     Y   Z
        2015-05-01         4    60   5
        2015-05-01T00    4.2  59.5   5
        "2015-05-03 12"  3.9  58.0   5
    """
    )
    with pytest.raises(SystemExit):
        ParticleReleaser(modules0, f)


def test_too_early_release():
    """All release before start"""

    f = StringIO(
        """
        release_time       X     Y   Z
        2015-03-01         4    60   5
        2015-03-10T00    4.2  59.5   5
        "2015-03-20 12"  3.9  58.0   5
    """
    )
    with pytest.raises(SystemExit):
        ParticleReleaser(modules0, f)


def test_continuous0():
    """Single constant continuous release"""

    f = StringIO(
        """
    release_time     X     Y    Z
    2015-03-01     100   400    5
    """
    )
    freq = np.timedelta64(12, "h")
    pr = ParticleReleaser(modules0, f, continuous=True, release_frequency=freq)

    df = pr._df
    assert len(df) == 7
    assert len(pr.steps) == 7
    assert pr.steps[0] == 0
    assert pr.steps[1] == 12 * 4
    assert df.index[0] == timer.start_time
    assert timer.stop_time - freq <= df.index[-1] < timer.stop_time  # type: ignore


def test_continuous1():
    """Non-constant continuous release"""

    f = StringIO(
        """
    release_time     X     Y    Z
    2015-03-01     100   400    5
    2015-04-02     101   401    5
    """
    )
    freq = np.timedelta64(12, "h")
    pr = ParticleReleaser(modules0, f, continuous=True, release_frequency=freq)

    df = pr._df
    assert df.index[3] == np.datetime64("2015-04-02", "s")
    assert all(df.Y == 3 * [400] + 4 * [401])


def test_continuous2():
    """Multiple non-constant continuous release"""

    f = StringIO(
        """
    release_time     X     Y    Z
    2015-03-01     100   400    5
    2015-03-01     110   410    5
    2015-04-02     101   401    5
    2015-04-02     111   411    5
    2015-04-02     121   421    5
    """
    )
    freq = np.timedelta64(12, "h")
    pr = ParticleReleaser(modules0, f, continuous=True, release_frequency=freq)

    df = pr._df
    assert df.index[0] == np.datetime64("2015-03-31 12", "s")
    assert df.index[2] == np.datetime64("2015-04-01", "s")
    assert df.index[17] == np.datetime64("2015-04-03 12", "s")
    assert all(df.Y == 3 * [400, 410] + 4 * [401, 411, 421])


def test_continuous_freq_mismatch():
    """
    File times and freq does not match

    If file time is not on a release time,
    the entry in the release file is silently ignored

    Might be better to issue a warning.

    In the test below, the 2nd and 3rd lines are ignored

    """

    f = StringIO(
        """
    release_time      X     Y    Z
    2015-03-01      100   400    0
    2015-04-01T06   200   500    1
    2015-04-01T18   300   600    2
    2015-04-02T12   400   700    3
    """
    )

    freq = np.timedelta64(12, "h")
    pr = ParticleReleaser(modules0, f, continuous=True, release_frequency=freq)
    df = pr._df
    assert len(pr.steps) == 7
    assert all(pr.steps == 48 * np.arange(7))
    assert df.index[3] == timer.start_time + 3 * freq
    assert all(df.X == 4 * [100] + 3 * [400])


def test_release_time_column():
    """With release time in datatypes include a release_time column"""

    f = StringIO(
        """
        release_time      X    Y    Z   super
        2015-04-01      100  200    5    1000
        2015-04-01T00   111  220    5    2000
        "2015-04-03 12" 200  300.0  5    3333
    """
    )
    datatypes = dict(datatypes0, super=int, release_time=np.dtype("M8[s]"))
    modules = modules0.copy()
    modules["state"] = Dummy(dtypes=datatypes)
    pr = ParticleReleaser(modules=modules, release_file=f, timer=timer)
    A = pr._df
    assert A.index[1] == np.datetime64("2015-04-01 00:00:00")
    assert A["release_time"][1] == np.datetime64("2015-04-01", "s")
    assert A["release_time"][2] == np.datetime64("2015-04-03 12:00:00")
    assert A["release_time"].dtype == np.dtype("M8[ns]")


def test_iterate1():
    f = StringIO(
        """
        release_time      X    Y    Z   super
        2015-04-01      100  200    5    1000
        2015-04-02T00   111  220    5    2000
        "2015-04-03 12" 200  300.0  5    3333
    """
    )

    pr = ParticleReleaser(modules0, f)
    A = next(pr)
    assert len(A) == 1
    assert A.X[0] == 100
    A = next(pr)
    assert len(A) == 1
    assert A.X[0] == 111
    A = next(pr)
    assert len(A) == 1
    assert A.X[0] == 200
    with pytest.raises(StopIteration):
        next(pr)


def test_iterate2():
    """Multiple non-constant continuous release"""

    f = StringIO(
        """
    release_time     X     Y    Z
    2015-03-01     100   400    5
    2015-03-01     110   410    5
    2015-04-02     101   401    5
    2015-04-02     111   411    5
    2015-04-02     121   421    5
    2015-04-03     102   402    5
    2015-04-04     103   403    5
    """
    )
    freq = np.timedelta64(12, "h")
    pr = ParticleReleaser(modules0, f, continuous=True, release_frequency=freq)

    for _k in range(3):
        A = next(pr)
        assert all(A.X == [100, 110])
    for _k in range(2):
        A = next(pr)
        assert all(A.X == [101, 111, 121])
    for _k in range(2):
        A = next(pr)
        assert all(A.X == [102])
    with pytest.raises(StopIteration):
        next(pr)
