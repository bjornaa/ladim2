import numpy as np
import datetime
import pytest

from ladim2.timer import Timer


def test_init():
    start_time = "2020-04-04 12"
    stop_time = np.datetime64("2020-04-05 12", "h")
    dt = 3600

    # Default reference_time
    t = Timer(start=start_time, stop=stop_time, dt=dt)
    assert str(t.start_time) == "2020-04-04T12:00:00"
    assert str(t.stop_time) == "2020-04-05T12:00:00"
    assert t.dt == dt
    assert t.reference_time == t.start_time
    assert t.Nsteps == 24

    # Explicit reference time
    t = Timer(start=start_time, stop=stop_time, dt=dt, reference="2020-01-01")
    assert str(t.reference_time) == "2020-01-01T00:00:00"

    # dt as timedelta
    t = Timer(start_time, stop_time, dt=np.timedelta64(1, "h"))
    assert t.dt == 3600

    # start - stop not divisible by dt
    t = Timer(start="2020-04-04", stop="2020-04-04 12:30", dt=3600)
    # Stop time is kept as given,
    # should perhaps be rounded down to the last time step
    # Alternatively: could raise an exception
    assert str(t.stop_time) == "2020-04-04T12:30:00"
    # Number of time steps is an integer
    assert t.Nsteps == 12


def test_time2step():
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:00:00"
    t = Timer(start=start_time, stop=stop_time, dt=3600)

    # Accept an iso-string
    assert t.time2step("2020-04-05") == 12

    # Accept a python datetime instance
    date = datetime.datetime(2020, 4, 4, 15)
    assert t.time2step(np.datetime64(date)) == 3

    # Accept a numpy datetime64 instance
    date = np.datetime64("2020-04-05")
    assert t.time2step(np.datetime64(date)) == 12

    # Accept time between time steps
    # This should perhaps raise exception
    assert t.time2step("2020-04-04 12:30:00") == 0


def test_step2isotime():
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:30:00"
    t = Timer(start=start_time, stop=stop_time, dt=3600)

    # Typical use
    assert t.step2isotime(10) == "2020-04-04T22:00:00"
    # Negative time steps are accepted
    assert t.step2isotime(-26) == "2020-04-03T10:00:00"
    # Fractional time steps are accepted
    # Should raise exception ?
    assert t.step2isotime(3.5) == "2020-04-04T15:30:00"
    assert t.step2isotime(np.pi) == "2020-04-04T15:08:29"


def test_step2nctime():
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:00:00"
    t = Timer(start=start_time, stop=stop_time, dt=3600)

    assert(t.cf_units() == "seconds since 2020-04-04T12:00:00")
    assert(t.cf_units('s') == "seconds since 2020-04-04T12:00:00")
    assert(t.cf_units('h') == "hours since 2020-04-04T12:00:00")
    assert(t.cf_units(unit='h') == "hours since 2020-04-04T12:00:00")

    assert(t.step2nctime(10) == 36000)
    assert(t.step2nctime(10, unit='s') == 36000)
    assert(t.step2nctime(10, unit="m") == 600)
    assert(t.step2nctime(10, unit="h") == 10)
