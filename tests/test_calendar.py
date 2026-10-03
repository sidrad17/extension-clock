"""Tests for src/calendar.py: weekend month-end, Columbus Day closure, T-4 across a holiday."""
import numpy as np
import pandas as pd
import pytest

from src.calendar import BondCalendar

# Weekdays with a blank DGS10 in the FRED file between 2023-09-01 and 2024-06-30 (checked on the 2026-10-03
# snapshot): the real bond-market holidays, so this fixture is the real bond calendar for those months.
HOLIDAYS = ["2023-09-04", "2023-10-09", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15", "2024-02-19",
            "2024-03-29", "2024-05-27", "2024-06-19"]


@pytest.fixture
def cal():
    days = pd.bdate_range("2023-09-01", "2024-06-30").difference(pd.DatetimeIndex(HOLIDAYS))
    return BondCalendar(days)


def test_from_dgs10_drops_missing():
    s = pd.Series([4.1, np.nan, 4.2], index=pd.to_datetime(["2023-10-06", "2023-10-09", "2023-10-10"]))
    assert list(BondCalendar.from_dgs10(s).days) == list(pd.to_datetime(["2023-10-06", "2023-10-10"]))


def test_weekend_month_end(cal):
    assert cal.month_end("2023-09") == pd.Timestamp("2023-09-29")   # Sep 30, 2023 is a Saturday
    assert cal.month_end("2024-06") == pd.Timestamp("2024-06-28")   # Jun 30, 2024 is a Sunday


def test_columbus_day_closed(cal):
    assert not cal.is_bday("2023-10-09")
    assert cal.offset("2023-10-10", -1) == pd.Timestamp("2023-10-06")
    with pytest.raises(KeyError):
        cal.offset("2023-10-09", -1)


def test_t_minus_4_across_holiday(cal):
    T = cal.month_end("2024-05")
    assert T == pd.Timestamp("2024-05-31")
    assert cal.offset(T, -4) == pd.Timestamp("2024-05-24")          # skips Memorial Day, 2024-05-27
    T = cal.month_end("2024-03")
    assert T == pd.Timestamp("2024-03-28")                          # Good Friday 2024-03-29 closed, then weekend
    assert cal.offset(T, -4) == pd.Timestamp("2024-03-22")


def test_nth_bday_and_month_ends(cal):
    assert cal.nth_bday("2023-09", 2) == pd.Timestamp("2023-09-05")  # Labor Day 2023-09-04
    assert cal.nth_bday("2023-10", 3) == pd.Timestamp("2023-10-04")
    me = cal.month_ends()
    assert len(me) == 10 and me.index[0] == pd.Period("2023-09") and me.iloc[-1] == pd.Timestamp("2024-06-28")


def test_incomplete_month_has_no_month_end():
    cal = BondCalendar(pd.bdate_range("2024-05-01", "2024-06-14"))
    with pytest.raises(ValueError):
        cal.month_end("2024-06")
    assert list(cal.month_ends().index) == [pd.Period("2024-05")]
    with pytest.raises(IndexError):
        cal.offset("2024-05-01", -1)
