"""Unit tests for TimeKeeper"""

import datetime
import numpy as np

import pytest

from ladim2.timekeeper import TimeKeeper, parse_isoperiod


def test_init():
    """Check that initalization works as expected"""
    start_time = "2020-04-04 12"
    stop_time = np.datetime64("2020-04-05 12")
    dt = 3600

    # Default reference_time
    t = TimeKeeper(start=start_time, stop=stop_time, dt=dt)
    assert str(t.start_time) == "2020-04-04T12:00:00"
    assert str(t.stop_time) == "2020-04-05T12:00:00"
    assert t.dt == dt
    assert t.reference_time == t.start_time
    assert t.Nsteps == 24

    # Explicit reference time
    t = TimeKeeper(start=start_time, stop=stop_time, dt=dt, reference="2020-01-01")
    assert str(t.reference_time) == "2020-01-01T00:00:00"

    # dt as timedelta (both datetime and numpy)
    t = TimeKeeper(start_time, stop_time, dt=datetime.timedelta(seconds=3600))
    assert t.dt == 3600
    t = TimeKeeper(start_time, stop_time, dt=np.timedelta64(1, "h"))
    assert t.dt == 3600

    # dt as yaml-type timedeltas
    t = TimeKeeper(start_time, stop_time, dt=[1, "h"])
    assert t.dt == 3600

    # dt specified in ISO 8601 format
    t = TimeKeeper(start_time, stop_time, dt="PT1H")
    assert t.dt == 3600

    # start - stop not divisible by dt
    t0 = TimeKeeper(start="2020-04-04", stop="2020-04-04 12:00", dt=3600)
    t1 = TimeKeeper(start="2020-04-04", stop="2020-04-04 12:30", dt=3600)
    # Stop time is kept as given,
    # should perhaps be rounded down to the last time step
    # Alternatively: could raise an exception
    assert str(t1.stop_time) == "2020-04-04T12:30:00"
    # Number of time steps is an integer rounded down
    assert t1.Nsteps == t0.Nsteps


def test_time2step():
    """Test conversion from time to step number"""
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:00:00"
    t = TimeKeeper(start=start_time, stop=stop_time, dt=3600)

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
    """Conversion from step nr to iso-standard time"""
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:30:00"
    t = TimeKeeper(start=start_time, stop=stop_time, dt=3600)

    # Typical use
    assert t.step2isotime(10) == "2020-04-04T22:00:00"
    # Negative time steps are accepted
    assert t.step2isotime(-26) == "2020-04-03T10:00:00"
    # Fractional time steps are accepted
    # Should raise exception ?
    assert t.step2isotime(3.5) == "2020-04-04T15:30:00"
    assert t.step2isotime(np.pi) == "2020-04-04T15:08:29"


def test_step2nctime():
    """Conversion from step nr to time since reference time"""
    start_time = "2020-04-04 12"
    stop_time = "2020-04-05 12:00:00"
    t = TimeKeeper(start=start_time, stop=stop_time, dt=3600)

    assert t.cf_units() == "seconds since 2020-04-04T12:00:00"
    assert t.cf_units("s") == "seconds since 2020-04-04T12:00:00"
    assert t.cf_units("h") == "hours since 2020-04-04T12:00:00"
    assert t.cf_units(unit="h") == "hours since 2020-04-04T12:00:00"

    assert t.step2nctime(10) == 36000
    assert t.step2nctime(10, unit="s") == 36000
    assert t.step2nctime(10, unit="m") == 600
    assert t.step2nctime(10, unit="h") == 10

    # With explicit reference time
    reference_time = "2020-04-04"  # 12 h = 43200 s before start
    t = TimeKeeper(start=start_time, stop=stop_time, reference=reference_time, dt=3600)

    assert t.cf_units() == "seconds since 2020-04-04T00:00:00"
    assert t.step2nctime(0) == 43200
    assert t.step2nctime(0, unit="h") == 12
    assert t.step2nctime(10) == 43200 + 10 * 3600


def test_parse_isoperiod():
    """Test the parsing of iso 8601 time period"""

    assert parse_isoperiod("PT2H") == np.timedelta64(2, "h")
    assert parse_isoperiod("PT09M") == np.timedelta64(9, "m")
    assert parse_isoperiod("PT30S") == np.timedelta64(30, "s")
    assert parse_isoperiod("PT600S") == np.timedelta64(10, "m")
    assert parse_isoperiod("PT16M40S") == np.timedelta64(1000, "s")

    with pytest.raises(ValueError):  # Must have at least of the items
        parse_isoperiod("PT")
    with pytest.raises(ValueError):  # T is necessary
        parse_isoperiod("P30S")
    with pytest.raises(ValueError):  # P is necessary
        parse_isoperiod("T30S")
    with pytest.raises(ValueError):  # Must use upper case
        parse_isoperiod("PT30s")
    with pytest.raises(ValueError):  # Only one item per unit
        parse_isoperiod("PT100S30S")
    with pytest.raises(ValueError):  # Only units H, M, S are accepted
        parse_isoperiod("PT20A")
    with pytest.raises(ValueError):  # Must follow the logcal H,M,S order
        parse_isoperiod("PT40S16M")
    with pytest.raises(ValueError):  # The P must start the string
        parse_isoperiod("APT40S")
    with pytest.raises(ValueError):  # Nothing after the items
        parse_isoperiod("PT40SZ")
