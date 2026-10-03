"""Bond-market business days, month-end T and T-k offsets (CLAUDE.md 7.2).

A bond business day is a date with a non-missing DGS10 (CLAUDE.md 7.2). FRED lists every weekday and leaves
bond-market holidays blank, so this captures SIFMA closures that equity calendars miss: Columbus Day
(2023-10-09) and Veterans Day closures, and Good Friday when the bond market shut (2024-03-29 is blank; 2021-04-02
has a value). T = the last bond business day of the month; offset(T, -k) = T-k.
"""
from __future__ import annotations

import pandas as pd

from config.settings import IS_END


def _period(m) -> pd.Period:
    return m if isinstance(m, pd.Period) else pd.Period(m, freq="M")


class BondCalendar:
    def __init__(self, days, covered_through=None):
        """`covered_through`: last date the source data covers (holiday rows included); defaults to the last day."""
        idx = pd.DatetimeIndex(days).normalize().unique().sort_values()
        if len(idx) == 0:
            raise ValueError("empty bond calendar")
        self.days = idx
        self.covered_through = pd.Timestamp(covered_through).normalize() if covered_through is not None else idx[-1]
        self._pos = pd.Series(range(len(idx)), index=idx)

    @classmethod
    def from_dgs10(cls, dgs10: pd.Series) -> "BondCalendar":
        """Bond business days = dates where DGS10 is not missing (CLAUDE.md 7.2). FRED has a row for every
        weekday, so the series' last row (even if blank) is how far the data reaches."""
        return cls(dgs10.dropna().index, covered_through=dgs10.index.max())

    def __contains__(self, d) -> bool:
        return pd.Timestamp(d).normalize() in self._pos.index

    def is_bday(self, d) -> bool:
        return d in self

    def _month_complete(self, p: pd.Period) -> bool:
        # The data must reach the month's last weekday (weekends never have rows), so a month cut short by the end
        # of the data never yields a false T.
        last_weekday = pd.bdate_range(p.start_time, p.end_time)[-1]
        return self.covered_through >= last_weekday

    def month_days(self, m) -> pd.DatetimeIndex:
        p = _period(m)
        return self.days[(self.days >= p.start_time) & (self.days <= p.end_time)]

    def month_end(self, m) -> pd.Timestamp:
        """T: last bond business day of month m (a Period, 'YYYY-MM' or any date in the month)."""
        p = _period(m)
        if not self._month_complete(p):
            raise ValueError(f"{p}: data covers through {self.covered_through.date()}, month not complete")
        days = self.month_days(p)
        if len(days) == 0:
            raise ValueError(f"{p}: no bond business days in calendar")
        return days[-1]

    def month_ends(self, start=None, end=None) -> pd.Series:
        """T for every complete month in [start, end], indexed by monthly Period."""
        first = _period(start) if start is not None else _period(self.days[0])
        last = _period(end) if end is not None else _period(self.days[-1])
        out = {}
        for p in pd.period_range(first, last, freq="M"):
            if self._month_complete(p) and len(self.month_days(p)):
                out[p] = self.month_days(p)[-1]
        return pd.Series(out, name="T", dtype="datetime64[ns]")

    def offset(self, d, k: int) -> pd.Timestamp:
        """Move k bond business days from bond day d (k < 0 = back): offset(T, -4) = T-4."""
        d = pd.Timestamp(d).normalize()
        if d not in self._pos.index:
            raise KeyError(f"{d.date()} is not a bond business day")
        i = int(self._pos[d]) + k
        if not 0 <= i < len(self.days):
            raise IndexError(f"offset {k} from {d.date()} leaves the calendar")
        return self.days[i]

    def nth_bday(self, m, n: int) -> pd.Timestamp:
        """n-th bond business day of month m (n = 1 is the first)."""
        days = self.month_days(m)
        if not 1 <= n <= len(days):
            raise IndexError(f"{_period(m)} has {len(days)} bond business days, asked for #{n}")
        return days[n - 1]


def load_calendar(end: str | None = IS_END) -> BondCalendar:
    """Bond calendar from the snapshot DGS10, dates <= end."""
    from src.data.fred import load_series
    return BondCalendar.from_dgs10(load_series("DGS10", end=end))
