"""Fed SOMA Treasury holdings by CUSIP from the NY Fed Markets Data API (keyless), for the index's Fed deduction.

Why (index methodology): the Bloomberg US Treasury Index deducts Fed SOMA holdings, "both purchases at issuance
and net secondary market transactions", from amounts outstanding. The auctions feed only carries the Fed's purchases
AT AUCTION (soma_accepted, 2008-04 on), so QE-era public amounts were too large. This module supplies the holdings.

Live responses inspected 2026-10-03 (rule 8):
* `/api/soma/asofdates/list.json` -> {"soma": {"asOfDates": [...]}}: 1,213 dates, newest first, 2003-07-09 to
  2026-09-30. Weekly: 1,199 Wednesdays, 12 Tuesdays and 2 Thursdays (holiday weeks); gaps of 6, 7 or 8 days.
  So holdings by CUSIP start 2003-07-09; there is no earlier history in this API.
* `/api/soma/tsy/get/notesbonds/asof/{YYYY-MM-DD}.json` -> {"soma": {"holdings": [...]}}, one row per CUSIP held,
  12 keys, every value a string: asOfDate, cusip, maturityDate, issuer, spread, coupon, parValue,
  inflationCompensation, percentOutstanding, changeFromPriorWeek, changeFromPriorYear, securityType.
  securityType is always "NotesBonds" (TIPS, FRNs and bills have their own types and are not returned);
  issuer, spread and inflationCompensation are "" for notes and bonds. parValue is in dollars
  (2003-07-09, 9128277A4: "4996200000"); percentOutstanding is a fraction of the amount outstanding
  ("0.3121986943918337" = 31%). Rows: 130 (2003-07-09), 219 (2010-12-29), 316 (2020-12-30), 326 (2026-09-30).
* `/api/soma/tsy/get/release_log.json` -> {"soma": {"dates": [{"releaseDate", "asOfDate"}, ...]}}: last 13 weeks
  only; every release is the day after its as-of Wednesday (2026-10-01 for 2026-09-30, ..., 2026-07-09 for
  2026-07-08). Earlier release dates are not in the API.

Point in time (CLAUDE.md rule 3; our choice): the time of day of the release is not in the API, so a SOMA as-of
date d counts as known at the close of X only if d falls before the bond business day preceding X
(d < offset(X, -1)): the release the next business day must be over before X begins. On a Thursday X the latest
usable date is the previous week's Wednesday; on a Friday or later, that week's Wednesday.

The snapshot (`data/snapshot/soma_notesbonds.csv`) keeps every published column for the as-of dates the rebuild
can use: for each month from SOMA_FIRST_MONTH through the last complete month, the usable date at T-k for
k = 0..SOMA_MAX_OFFSET (covers the NOW universe at T(m-1), E = T-4 and the pre-registered entry grid T-2..T-6).
The full list of as-of dates goes to `soma_asof_dates.csv`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.settings import IS_END
from src.data.snapshot import active_dir

API = "https://markets.newyorkfed.org/api/soma"
ASOF_LIST_URL = f"{API}/asofdates/list.json"
HOLDINGS_URL = API + "/tsy/get/notesbonds/asof/{date}.json"
RELEASE_LOG_URL = f"{API}/tsy/get/release_log.json"
SNAPSHOT_NAME = "soma_notesbonds.csv"
ASOF_NAME = "soma_asof_dates.csv"
FIELDS = ["asOfDate", "cusip", "maturityDate", "issuer", "spread", "coupon", "parValue", "inflationCompensation",
          "percentOutstanding", "changeFromPriorWeek", "changeFromPriorYear", "securityType"]
SOMA_FIRST_MONTH = "2003-07"   # first month-end after the first as-of date (2003-07-09)
SOMA_MAX_OFFSET = 6            # T-0 .. T-6: NOW at T(m-1), E = T-4, entry grid T-2..T-6 (CLAUDE.md 7.12)


# ------------------------------------------------------------------------------------------------- download

def fetch_asof_dates(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(ASOF_LIST_URL, timeout=60)
    r.raise_for_status()
    dates = r.json()["soma"]["asOfDates"]
    return pd.DataFrame({"asOfDate": sorted(dates)})


def fetch_holdings(date: str, session: requests.Session | None = None, retries: int = 3) -> pd.DataFrame:
    """Notes and bonds held on one as-of date, every field as the published string."""
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            r = s.get(HOLDINGS_URL.format(date=date), timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    rows = r.json()["soma"]["holdings"]
    df = pd.DataFrame(rows, columns=FIELDS) if rows else pd.DataFrame(columns=FIELDS)
    extra = set().union(*(r.keys() for r in rows)) - set(FIELDS) if rows else set()
    if extra:
        raise RuntimeError(f"SOMA {date}: unexpected fields {sorted(extra)}; re-inspect the API (rule 8)")
    if len(df) and not (df["asOfDate"] == date).all():
        raise RuntimeError(f"SOMA {date}: rows carry another asOfDate")
    if len(df) and set(df["securityType"]) != {"NotesBonds"}:
        raise RuntimeError(f"SOMA {date}: securityType {sorted(set(df['securityType']))}, expected NotesBonds")
    return df


def needed_asof_dates(asof: pd.DatetimeIndex, cal, first_month: str = SOMA_FIRST_MONTH,
                      last_month=None, max_offset: int = SOMA_MAX_OFFSET) -> pd.DatetimeIndex:
    """Every as-of date that usable_asof() can return at T-k (k = 0..max_offset) for the months covered."""
    out = set()
    for m, T in cal.month_ends(first_month, last_month).items():
        for k in range(max_offset + 1):
            d = usable_asof(asof, cal, cal.offset(T, -k))
            if d is not None:
                out.add(d)
    return pd.DatetimeIndex(sorted(out))


def download(session: requests.Session | None = None, cal=None, pause: float = 0.1) -> tuple[pd.DataFrame,
                                                                                               pd.DataFrame]:
    """(as-of date list, holdings for the needed dates). `cal`: bond calendar over the whole snapshot."""
    s = session or requests.Session()
    asof = fetch_asof_dates(s)
    dates = needed_asof_dates(pd.DatetimeIndex(pd.to_datetime(asof["asOfDate"])), cal)
    frames = []
    for i, d in enumerate(dates):
        frames.append(fetch_holdings(d.date().isoformat(), s))
        if (i + 1) % 50 == 0:
            print(f"  SOMA holdings: {i + 1}/{len(dates)} dates")
        time.sleep(pause)
    h = pd.concat(frames, ignore_index=True).sort_values(["asOfDate", "cusip"], kind="mergesort")
    return asof, h.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------- point in time

def usable_asof(asof: pd.DatetimeIndex, cal, X) -> pd.Timestamp | None:
    """Latest SOMA as-of date known at the close of bond day X: d < offset(X, -1) (module docstring)."""
    cutoff = cal.offset(pd.Timestamp(X), -1)
    i = asof.searchsorted(cutoff, side="left")
    return asof[i - 1] if i > 0 else None


# ------------------------------------------------------------------------------------------------------ load

class SomaHoldings:
    """Par held by SOMA per (as-of date, CUSIP), in dollars."""

    def __init__(self, asof: pd.DatetimeIndex, par: pd.Series):
        self.asof = pd.DatetimeIndex(asof).sort_values()
        self.par = par                                  # MultiIndex (asOfDate, cusip) -> dollars
        self._stored = set(par.index.get_level_values(0).unique())

    def usable(self, cal, X) -> pd.Timestamp | None:
        return usable_asof(self.asof, cal, X)

    def on(self, d: pd.Timestamp) -> pd.Series:
        """Holdings by CUSIP on as-of date d (must be in the snapshot)."""
        if d not in self._stored:
            raise KeyError(f"SOMA holdings for {d.date()} are not in the snapshot; refresh with "
                           "`python scripts/download_all.py --only soma`")
        return self.par.xs(d, level=0)


def load_soma(end: str | None = IS_END, snapshot_dir: Path | None = None) -> SomaHoldings:
    """Snapshot holdings and as-of dates with asOfDate <= end (snapshot_dir: active_dir() unless given)."""
    snapshot_dir = snapshot_dir or active_dir()
    raw = pd.read_csv(snapshot_dir / SNAPSHOT_NAME, dtype=str, keep_default_na=False)
    asof = pd.DatetimeIndex(pd.to_datetime(
        pd.read_csv(snapshot_dir / ASOF_NAME, dtype=str)["asOfDate"], format="%Y-%m-%d"))
    d = pd.to_datetime(raw["asOfDate"], format="%Y-%m-%d")
    par = pd.Series(pd.to_numeric(raw["parValue"], errors="raise").to_numpy(float),
                    index=pd.MultiIndex.from_arrays([d, raw["cusip"]], names=["asOfDate", "cusip"]), name="par")
    if par.index.duplicated().any():
        raise ValueError("SOMA snapshot: duplicate (asOfDate, cusip)")
    if end is not None:
        cut = pd.Timestamp(end)
        par = par[par.index.get_level_values(0) <= cut]
        asof = asof[asof <= cut]
    return SomaHoldings(asof, par.sort_index())


def outstanding_implied(snapshot_dir: Path | None = None, end: str | None = IS_END) -> pd.DataFrame:
    """parValue / percentOutstanding per (date, CUSIP): the amount outstanding the Fed's file implies (validation)."""
    raw = pd.read_csv((snapshot_dir or active_dir()) / SNAPSHOT_NAME, dtype=str, keep_default_na=False)
    raw["asOfDate"] = pd.to_datetime(raw["asOfDate"], format="%Y-%m-%d")
    if end is not None:
        raw = raw[raw["asOfDate"] <= pd.Timestamp(end)]
    par = pd.to_numeric(raw["parValue"])
    pct = pd.to_numeric(raw["percentOutstanding"].replace("", np.nan))
    return raw.assign(par=par, pct=pct, implied_outstanding=par / pct)[
        ["asOfDate", "cusip", "par", "pct", "implied_outstanding"]]
