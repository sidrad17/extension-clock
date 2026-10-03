"""Primary dealer transactions in Treasury coupons (NY Fed Markets Data API, keyless): the volume input of the
cash capacity estimate (CLAUDE.md 7.13). The futures version uses CME volume and needs Databento.

Live responses inspected 2026-10-03 (rule 8):
* `/api/pd/list/seriesbreaks.json` -> {"pd": {"seriesbreaks": [{"label", "seriesbreak", "startdate", "enddate"}]}}:
  SBP2001 (1998-01-28 to 2001-06-30), SBP2013 (2001-07-01 to 2013-03-31), SBN2013, SBN2015, SBN2022, SBN2024
  (2024-07-03 on). The FR 2004 report changed its maturity buckets at each break.
* `/api/pd/get/{keyid}.json` -> {"pd": {"timeseries": [{"asofdate", "keyid", "value"}]}}: one row per week, every
  as-of date a Wednesday, no missing weeks, value a string in $ millions.
* `/api/pd/list/timeseries.json` describes only the current keys. PDTRGSC-G7L11 = "U.S. TREASURY SECURITIES
  (EXCLUDING TIPS) COUPONS DUE IN MORE THAN 7 YEARS BUT LESS THAN OR EQUAL TO 11 YEARS - DEALER TRANSACTIONS WITH
  INTER-DEALER BROKERS + ... WITH OTHER". It runs 2013-04-03 to 2026-09-23 (704 weeks).
* The API has no description for keys before April 2013. PDSUSGCS611OT (2001-07-04 to 2013-03-27, 613 weeks) is
  dealer outright transactions in coupons due in more than 6 and at most 11 years. Check: the PDSUSG*OT maturity
  buckets (bills, TIPS, coupons < 3, 3-6, 6-11, > 11 years) sum to the transactions by counterparty,
  PDCUSGIDBOT + PDCUSGOOT ($310,974M vs $310,972M on 2001-07-04; $611,373M vs $611,373M on 2007-06-27; $530,445M
  vs $530,446M on 2013-03-27).
* Values are daily averages over the week: total Treasury dealer transactions are $311bn on 2001-07-04 and $611bn
  on 2007-06-27. Weekly totals would imply about $60bn a day in 2001.

The 10-year note sits in the 6-11 year bucket before April 2013 and in the 7-11 year bucket after, so SERIES keeps
those two keys. The earlier SBP2001 key (coupons over 5 years) also holds the 30-year bond and is not used, so the
capacity estimate starts in 2001-07.

Point in time (our choice): dealers report each week ending Wednesday and the NY Fed publishes it on the Thursday
of the following week, so week w counts at the close of day X only if w + RELEASE_LAG_DAYS < X (the release is
before X). ADV at X = the mean of the last ADV_WEEKS weekly values known at X (4 weeks, about the 20 business days
of ADV_LOOKBACK).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from config.settings import IS_END
from src.data.snapshot import SNAPSHOT_DIR

API = "https://markets.newyorkfed.org/api/pd"
SERIES_URL = API + "/get/{keyid}.json"
SNAPSHOT_NAME = "pd_treasury_volume.csv"
FIELDS = ["asofdate", "keyid", "value"]
SERIES = {"PDSUSGCS611OT": "coupons due in more than 6 and at most 11 years (2001-07 to 2013-03)",
          "PDTRGSC-G7L11": "coupons due in more than 7 and at most 11 years (2013-04 on)"}
RELEASE_LAG_DAYS = 8
ADV_WEEKS = 4


def fetch_series(keyid: str, session: requests.Session | None = None, retries: int = 3) -> pd.DataFrame:
    """Every week of one key, fields as published (strings)."""
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            r = s.get(SERIES_URL.format(keyid=keyid), timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    rows = r.json()["pd"]["timeseries"]
    extra = set().union(*(x.keys() for x in rows)) - set(FIELDS) if rows else set()
    if extra:
        raise ValueError(f"{keyid}: unexpected fields {sorted(extra)}")
    return pd.DataFrame(rows, columns=FIELDS)


def download(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    df = pd.concat([fetch_series(k, s) for k in SERIES], ignore_index=True)
    return df.sort_values(["asofdate", "keyid"], kind="mergesort").reset_index(drop=True)


def parse(df: pd.DataFrame) -> pd.Series:
    """Weekly 10-year-bucket dealer volume in $ (daily average), indexed by as-of date (one key per week)."""
    d = pd.DataFrame({"asofdate": pd.to_datetime(df["asofdate"], format="%Y-%m-%d"), "keyid": df["keyid"],
                      "value": pd.to_numeric(df["value"], errors="raise") * 1e6})
    d = d[d["keyid"].isin(list(SERIES))]
    if d["asofdate"].duplicated().any():
        raise ValueError("two volume keys on the same week")
    return pd.Series(d["value"].to_numpy(), index=pd.DatetimeIndex(d["asofdate"]), name="adv_10y_bucket").sort_index()


def load_volume(end: str | None = IS_END, path=None) -> pd.Series:
    """Weekly volume from the snapshot, as-of dates <= end."""
    s = parse(pd.read_csv(path or SNAPSHOT_DIR / SNAPSHOT_NAME, dtype=str))
    return s if end is None else s.loc[: pd.Timestamp(end)]


def adv_known_at(dates, weekly: pd.Series, n_weeks: int = ADV_WEEKS, lag_days: int = RELEASE_LAG_DAYS) -> pd.Series:
    """ADV ($) known at the close of each date: mean of the last n_weeks weekly values released before it; NaN
    while fewer than n_weeks are known."""
    dates = pd.DatetimeIndex(dates)
    released = weekly.index + pd.Timedelta(days=lag_days)
    vals = weekly.to_numpy(float)
    out = []
    for d in dates:
        k = int(released.searchsorted(d, side="left"))         # weeks released strictly before d
        out.append(float(vals[k - n_weeks:k].mean()) if k >= n_weeks else np.nan)
    return pd.Series(out, index=dates, name="adv")
