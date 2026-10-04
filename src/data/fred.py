"""Keyless FRED CSV loader for Treasury constant-maturity yields and DTB3 (CLAUDE.md 7.1).

Live response inspected 2026-10-03 (rule 8): `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` returns
content-type application/csv, header `observation_date,DGS10`, one row per weekday (holidays included, value
blank), values in percent. Missing values were "" in that download; older vintages used "." so both are read as
missing. Coverage in that download: DGS1/3/5/10 and DGS20 from 1962-01-02, DGS2 1976-06-01, DGS7 1969-07-01,
DGS30 1977-02-15, DTB3 1954-01-04; all through 2026-10-01.

Known gaps, handled explicitly (never forward-filled, CLAUDE.md 7.1):
* DGS20 1987-01-01 to 1993-09-30: blank in the published file (1,761 weekdays).
* DGS30 2002-02-19 to 2006-02-08: Treasury did not issue 30-year bonds then. The 2026-10-03 vintage HAS values in
  this window, but they splice badly at both ends (2002-02-19: DGS30 +17bp while DGS20 +3bp; 2006-02-09: DGS30
  -16bp while DGS20 -3bp), consistent with a 30-year rate extrapolated from the 20-year rather than a market
  yield. The snapshot keeps the published values; `load_series(mask_known_gaps=True)` (the default) blanks the
  window so the curve is flat beyond 20 years there and the 30-year return series has a gap. Flagged for review.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.settings import CURVE_KNOTS, IS_END, RF_SERIES
from src.data.snapshot import active_dir

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = list(CURVE_KNOTS) + [RF_SERIES]
KNOWN_GAPS = {"DGS20": ("1987-01-01", "1993-09-30"), "DGS30": ("2002-02-19", "2006-02-08")}
MAX_FFILL_DAYS = 5  # CLAUDE.md 7.1: never forward-fill across a gap longer than 5 days


def snapshot_name(series_id: str) -> str:
    return f"fred_{series_id}.csv"


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a fredgraph.csv body into a float Series indexed by date (NaN where blank or ".")."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if list(df.columns) != ["observation_date", series_id]:
        raise ValueError(f"FRED {series_id}: unexpected header {list(df.columns)}; re-inspect the CSV (rule 8)")
    values = pd.to_numeric(df[series_id].str.strip().replace({"": np.nan, ".": np.nan}), errors="raise")
    idx = pd.DatetimeIndex(pd.to_datetime(df["observation_date"], format="%Y-%m-%d"), name="date")
    return pd.Series(values.to_numpy(dtype=float), index=idx, name=series_id)


def fetch_series(series_id: str, session: requests.Session | None = None) -> tuple[str, pd.Series]:
    """Download one series; returns (raw CSV text as published, parsed Series)."""
    s = session or requests.Session()
    r = s.get(FRED_CSV_URL, params={"id": series_id}, timeout=120)
    r.raise_for_status()
    return r.text, parse_fred_csv(r.text, series_id)


def mask_gaps(s: pd.Series) -> pd.Series:
    """Blank the known-gap window for this series (see module docstring)."""
    if s.name not in KNOWN_GAPS:
        return s
    lo, hi = KNOWN_GAPS[s.name]
    out = s.copy()
    out.loc[lo:hi] = np.nan
    return out


def load_series(series_id: str, end: str | None = IS_END, mask_known_gaps: bool = True,
                snapshot_dir: Path | None = None) -> pd.Series:
    """One FRED series from the snapshot (src/data/snapshot.py::active_dir unless given), dates <= end, missing
    values left as NaN (no filling here)."""
    text = ((snapshot_dir or active_dir()) / snapshot_name(series_id)).read_text(encoding="utf-8")
    s = parse_fred_csv(text, series_id)
    if mask_known_gaps:
        s = mask_gaps(s)
    if end is not None:
        s = s.loc[: pd.Timestamp(end)]
    return s


def ffill_short_gaps(s: pd.Series, max_gap: int = MAX_FFILL_DAYS) -> pd.Series:
    """Forward-fill runs of at most `max_gap` consecutive NaN rows; longer runs stay entirely NaN.

    Rows are whatever index `s` carries (callers pass series on the bond-business-day calendar), so the limit is
    in business days. Leading NaNs are never filled.
    """
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    fillable = isna & (run_len <= max_gap)
    filled = s.ffill()
    return s.where(~fillable, filled)


def load_frame(series: list[str] | None = None, end: str | None = IS_END, index: pd.DatetimeIndex | None = None,
               fill: bool = True, snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Several series side by side; optionally reindexed to `index` (e.g. bond business days) with short-gap fill."""
    series = SERIES if series is None else series
    df = pd.concat([load_series(sid, end=end, snapshot_dir=snapshot_dir) for sid in series], axis=1)
    if index is not None:
        df = df.reindex(index)
    if fill:
        df = df.apply(ffill_short_gaps)
    return df
