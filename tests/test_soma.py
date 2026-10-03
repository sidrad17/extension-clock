"""Tests for src/data/soma.py: the point-in-time SOMA date and the snapshot loader (no network)."""
import pandas as pd
import pytest

from src.calendar import BondCalendar
from src.data import soma

ASOF = pd.DatetimeIndex(["2024-11-13", "2024-11-20", "2024-11-27"])


def test_usable_asof_waits_a_business_day_for_the_release():
    # Thanksgiving 2024-11-28 (Thursday) closed: the 11-27 holdings are released after it.
    days = pd.bdate_range("2024-11-01", "2024-12-31").drop(pd.Timestamp("2024-11-28"))
    cal = BondCalendar(days)
    assert soma.usable_asof(ASOF, cal, "2024-11-20") == pd.Timestamp("2024-11-13")   # Wed: own date not released
    assert soma.usable_asof(ASOF, cal, "2024-11-21") == pd.Timestamp("2024-11-13")   # Thu: released that day, time unknown
    assert soma.usable_asof(ASOF, cal, "2024-11-22") == pd.Timestamp("2024-11-20")   # Fri: released Thursday
    assert soma.usable_asof(ASOF, cal, "2024-11-29") == pd.Timestamp("2024-11-20")   # Fri after the holiday
    assert soma.usable_asof(ASOF, cal, "2024-12-02") == pd.Timestamp("2024-11-27")
    assert soma.usable_asof(ASOF, cal, "2024-11-12") is None


def test_load_soma_parses_and_cuts_at_end(tmp_path):
    rows = [["2024-09-25", "912828AA1", "2030-01-15", "", "", "4.0", "1000000000", "", "0.1", "0", "0", "NotesBonds"],
            ["2024-10-02", "912828AA1", "2030-01-15", "", "", "4.0", "1500000000", "", "0.15", "500000000", "0",
             "NotesBonds"]]
    pd.DataFrame(rows, columns=soma.FIELDS).to_csv(tmp_path / soma.SNAPSHOT_NAME, index=False)
    pd.DataFrame({"asOfDate": ["2024-09-25", "2024-10-02"]}).to_csv(tmp_path / soma.ASOF_NAME, index=False)
    h = soma.load_soma(end="2024-09-30", snapshot_dir=tmp_path)
    assert list(h.asof) == [pd.Timestamp("2024-09-25")]
    assert h.on(pd.Timestamp("2024-09-25"))["912828AA1"] == 1e9
    with pytest.raises(KeyError):
        h.on(pd.Timestamp("2024-10-02"))
    imp = soma.outstanding_implied(snapshot_dir=tmp_path, end=None)
    assert imp["implied_outstanding"].tolist() == pytest.approx([10e9, 10e9])
