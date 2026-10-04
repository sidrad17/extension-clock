"""Treasury auction records from the Fiscal Data auctions_query API (CLAUDE.md 7.1).

Live response inspected 2026-10-03 (rule 8). Top level: `data`, `meta`, `links`. `meta` has `count`, `total-count`,
`total-pages`, `labels`, `dataTypes`, `dataFormats`; `links` has `self`, `first`, `prev`, `next`, `last` (`next` is
None on the last page). Each row has 114 keys and every value is a string; missing values are the literal "null".
Keys used here: cusip, security_type, security_term, auction_date, issue_date, maturity_date, announcemt_date,
offering_amt, total_accepted, int_rate, high_yield, reopening, original_issue_date, soma_accepted, soma_holdings,
soma_included, comp_accepted, noncomp_accepted, fima_noncomp_accepted, inflation_index_security, floating_rate,
callable, currently_outstanding, first_int_payment_date, int_payment_frequency, closing_time_comp.
With `filter=security_type:in:(Note,Bond)` and `page[size]=10000` the whole set is one page: 2,791 rows,
auction_date 1979-10-31 to 2026-10-08 (2,337 Notes, 454 Bonds; 713 reopenings; 269 TIPS; 155 FRNs; 19 callable).
Rows with a future auction_date are announced auctions: total_accepted, soma_accepted and high_yield are "null";
int_rate is "null" for a new CUSIP (the coupon is set at auction) and the original coupon for a reopening.
Checked on that download: (cusip, auction_date) is unique; each CUSIP has one maturity_date and one int_rate;
issue_date >= auction_date >= announcemt_date on every row.

The snapshot (`data/snapshot/auctions.csv`) stores every column exactly as published (strings, "null" kept);
`load_auctions()` parses it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config.settings import EXCLUDE, IS_END
from src.data.snapshot import active_dir

AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
SNAPSHOT_NAME = "auctions.csv"

DATE_COLS = ["record_date", "auction_date", "issue_date", "maturity_date", "announcemt_date",
             "original_issue_date", "first_int_payment_date", "dated_date", "call_date", "called_date"]
NUM_COLS = ["offering_amt", "total_accepted", "total_tendered", "int_rate", "high_yield", "soma_accepted",
            "soma_holdings", "soma_tendered", "comp_accepted", "noncomp_accepted", "fima_noncomp_accepted",
            "currently_outstanding", "std_int_payment_per1000"]
FLAG_COLS = ["reopening", "inflation_index_security", "floating_rate", "callable", "soma_included"]
REQUIRED = ["cusip", "security_type", "auction_date", "issue_date", "maturity_date", "announcemt_date",
            "offering_amt", "total_accepted", "int_rate", "reopening", "soma_accepted",
            "inflation_index_security", "floating_rate", "callable"]


def fetch_auctions(session: requests.Session | None = None, page_size: int = 10_000) -> pd.DataFrame:
    """Download every Note/Bond auction (all pages), values left as published strings."""
    s = session or requests.Session()
    rows, page = [], 1
    while True:
        r = s.get(AUCTIONS_URL, params={"filter": "security_type:in:(Note,Bond)", "page[size]": page_size,
                                        "page[number]": page, "sort": "auction_date"}, timeout=120)
        r.raise_for_status()
        j = r.json()
        rows += j["data"]
        if not j["links"].get("next"):
            break
        page += 1
    df = pd.DataFrame(rows)
    if len(df) != j["meta"]["total-count"]:
        raise RuntimeError(f"auctions: got {len(df)} rows, API reports total-count {j['meta']['total-count']}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"auctions: API response lacks expected fields {missing}; re-inspect the API (rule 8)")
    return df.sort_values(["auction_date", "cusip"], kind="mergesort").reset_index(drop=True)


def clean_auctions(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the published strings: "null" -> missing, dates -> Timestamp, amounts -> float, Yes/No -> bool."""
    df = raw.copy().replace("null", np.nan)
    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%d")
    for c in NUM_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    for c in FLAG_COLS:
        if c in df:
            df[c] = df[c].map({"Yes": True, "No": False}).astype("boolean")
    return df


def public_amount(df: pd.DataFrame, deduct_soma: bool = True) -> pd.Series:
    """Par sold to the public at an auction already held (NaN for announced auctions not yet held).

    SOMA check (CLAUDE.md 7.1), run on the live data 2026-10-03:
    * 1,121 auctions report soma_accepted > 0 (2008-04-22 to 2026-09-24). On every one,
      total_accepted / (offering_amt + soma_accepted) lies in [1.000000, 1.000027] (mean 1.000002), while
      total_accepted / offering_amt averages 1.11 (max 1.70). Five of them:
        2008-04-22  5-Year   912828HW3  offering  8.000bn  SOMA  0.734bn  total_accepted  8.734bn
        2011-03-30  7-Year   912828QB9  offering 29.000bn  SOMA  1.301bn  total_accepted 30.301bn
        2017-03-27  2-Year   912828W97  offering 26.000bn  SOMA  3.148bn  total_accepted 29.148bn
        2021-05-25  2-Year   91282CCD1  offering 60.000bn  SOMA 11.646bn  total_accepted 71.646bn
        2026-09-24  7-Year   91282CRM5  offering 44.000bn  SOMA  6.624bn  total_accepted 50.624bn
      On the 634 auctions with soma_accepted == 0, total_accepted / offering_amt is in [1.000000, 1.000036].
      comp_accepted + noncomp_accepted + fima_noncomp_accepted ~= offering_amt, so FIMA sits inside the offering.
      => total_accepted INCLUDES the Fed's SOMA add-on. Public amount = total_accepted - soma_accepted.
    * Before 2008-04-10 every breakdown field (soma_*, comp_accepted, ...) is "null", yet total_accepted still
      exceeds offering_amt (1,033 auctions 1979-10-31 to 2008-03-27: mean ratio 1.14, median 1.12; only 7% within
      0.01% of the offering), the same pattern as the Fed add-ons above. For those auctions the public amount is
      offering_amt (capped at total_accepted). The excess may also contain foreign-official add-ons, which the
      feed cannot separate before 2008. Our choice, flagged for review at STOP 2.
    * soma_accepted covers only the Fed's purchases AT AUCTION. Secondary-market SOMA purchases (QE) are not in
      this feed. From 2003-08 the rebuild deducts NY Fed SOMA holdings by CUSIP instead (src/data/soma.py,
      src/index_rebuild.py "Fed holdings", PREREG_ADDENDUM.md); this auction-only amount is used before then.
    With deduct_soma=False (sensitivity, settings.DEDUCT_SOMA) the public amount is total_accepted.
    """
    total = df["total_accepted"]
    if not deduct_soma:
        return total.rename("public_amount")
    known = df["soma_accepted"].notna()
    amt = np.where(known, total - df["soma_accepted"].fillna(0.0), np.fmin(df["offering_amt"], total))
    return pd.Series(amt, index=df.index, name="public_amount").where(total.notna())


def drop_excluded(df: pd.DataFrame, exclude: dict = EXCLUDE) -> pd.DataFrame:
    """Drop TIPS, FRNs and callable bonds per settings.EXCLUDE (Bloomberg US Treasury Index rules; callable is our
    choice, pre-registered). Bills and CMBs never enter: the API query is restricted to Notes and Bonds."""
    keep = pd.Series(True, index=df.index)
    if exclude.get("TIPS"):
        keep &= ~df["inflation_index_security"].fillna(False).astype(bool)
    if exclude.get("FRN"):
        keep &= ~df["floating_rate"].fillna(False).astype(bool)
    if exclude.get("callable"):
        keep &= ~df["callable"].fillna(False).astype(bool)
    return df[keep]


def load_auctions(end: str | None = IS_END, deduct_soma: bool = True, exclude: dict | None = EXCLUDE,
                  path: Path | None = None) -> pd.DataFrame:
    """Parsed auctions from the snapshot with `auction_date <= end` (end=None keeps announced future auctions).

    Adds `public_amount` (see public_amount()). It is a RESULT of the auction: point-in-time code must use
    offering_amt for any tranche auctioned after the entry day (CLAUDE.md rule 3 and section 11).
    """
    path = path or active_dir() / SNAPSHOT_NAME
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = clean_auctions(raw)
    if end is not None:
        df = df[df["auction_date"] <= pd.Timestamp(end)]
    if exclude:
        df = drop_excluded(df, exclude)
    df = df.assign(public_amount=public_amount(df, deduct_soma))
    return df.reset_index(drop=True)
