"""Primary dealer net positions in Treasury coupons excluding TIPS, all maturities (NY Fed Markets Data API, keyless):
the input of H8's dealer signal zD (PREREG_DEALERS.md, tag `prereg-dealers`).

Live responses inspected 2026-10-03, after the prereg-dealers tag was pushed (rule 8):
* `/api/pd/list/seriesbreaks.json` -> {"pd": {"seriesbreaks": [{"label", "seriesbreak", "startdate", "enddate"}]}}:
  the six breaks in SERIES_BREAKS (the FR 2004 changed at each).
* `/api/pd/list/timeseries.json` describes only the SBN2024 keys (1,539). `/api/pd/latest/{seriesbreak}.json` lists
  every key of a break with its last week's values; that is how the older keys below were found.
* `/api/pd/get/{keyid}.json` -> {"pd": {"timeseries": [{"asofdate", "keyid", "value"}]}}: weekly, every as-of date a
  Wednesday, no missing week and no non-numeric value in any key used here, value a string in $ millions. Position
  keys are net (long minus short).
* No break publishes a total for coupons excluding TIPS, so the series is the sum of that break's coupon buckets
  (COUPON_KEYS):
  - SBP2001 (179 weeks, 1998-01-28 to 2001-06-27): PDPUSGCS5LNOP + PDPUSGCS5MNOP, net outright positions in coupons
    due in at most / more than 5 years. TIPS (PDPUSGTIISNOP) and bills (PDPUSGTBNOP) have their own keys; the NFP and
    NPP keys are other position types, not outright, and are not used. This break has no total to check against;
    its keys follow the SBP2013 naming, which the next check confirms.
  - SBP2013 (613 weeks, 2001-07-04 to 2013-03-27): PDPUSGCS3LNOP + PDPUSGCS36NOP + PDPUSGCS611NOP + PDPUSGCSM11NOP.
    Check: these 4 + bills (PDPUSGTBNOP) + TIPS (PDPUSGTIISNOP) = the published total PDPUSGTNOP in every week.
  - SBN2013, SBN2015 (2013-04-03 to 2021-12-29): PDPOSGSC-L2, -G2L3, -G3L6, -G6L7, -G7L11, -G11.
  - SBN2022, SBN2024 (2022-01-05 on): PDPOSGSC-L2, -G2L3, -G3L6, -G6L7, -G7L11, -G11L21, -G21.
  Check (704 weeks, 2013-04-03 on): coupon buckets + bills (PDPOSGS-B) + FRNs (PDPOSGS-BFRN, from 2015-01-07) + TIPS
  (PDPOSTIPS-L2, -G2, -G6L11, -G11) = PDPOSGST-TOT in every week, although that total's description says
  "excluding TIPS". FRNs (issued from 2014-01) have no key before 2015-01-07, and in 2014 the total equals coupons +
  bills + TIPS exactly, so any 2014 FRN positions sit inside one of those lines and cannot be separated.
* The snapshot keeps the coupon keys and the check keys, every field as published; parse() reruns both checks.

Gate (PREREG_DEALERS.md): download() and load_positions() call assert_dealers_prereg() first (active with GQH_DEV=1).
The point-in-time rule (publication date) is in src/dealers.py.
"""
from __future__ import annotations

import pandas as pd
import requests

from config.settings import IS_END
from src.data.pd_volume import SERIES_URL, fetch_series
from src.data.snapshot import active_dir
from src.trial_log import assert_dealers_prereg

SNAPSHOT_NAME = "pd_treasury_positions.csv"
SERIES_BREAKS = [("SBP2001", "1998-01-28", "2001-06-30"), ("SBP2013", "2001-07-01", "2013-03-31"),
                 ("SBN2013", "2013-04-01", "2014-12-31"), ("SBN2015", "2015-01-01", "2022-01-04"),
                 ("SBN2022", "2022-01-05", "2024-07-02"), ("SBN2024", "2024-07-03", "9999-12-31")]
_SBN_PRE = ["PDPOSGSC-L2", "PDPOSGSC-G2L3", "PDPOSGSC-G3L6", "PDPOSGSC-G6L7", "PDPOSGSC-G7L11"]
COUPON_KEYS = {"SBP2001": ["PDPUSGCS5LNOP", "PDPUSGCS5MNOP"],
               "SBP2013": ["PDPUSGCS3LNOP", "PDPUSGCS36NOP", "PDPUSGCS611NOP", "PDPUSGCSM11NOP"],
               "SBN2013": _SBN_PRE + ["PDPOSGSC-G11"], "SBN2015": _SBN_PRE + ["PDPOSGSC-G11"],
               "SBN2022": _SBN_PRE + ["PDPOSGSC-G11L21", "PDPOSGSC-G21"],
               "SBN2024": _SBN_PRE + ["PDPOSGSC-G11L21", "PDPOSGSC-G21"]}
SBP_CHECK = {"total": "PDPUSGTNOP", "other": ["PDPUSGTBNOP", "PDPUSGTIISNOP"]}
SBN_CHECK = {"total": "PDPOSGST-TOT",
             "other": ["PDPOSGS-B", "PDPOSGS-BFRN", "PDPOSTIPS-L2", "PDPOSTIPS-G2", "PDPOSTIPS-G6L11", "PDPOSTIPS-G11"]}
CHECK_TOL = 0.5                                       # $ millions; the published values are integers


def all_keys() -> list[str]:
    keys = {k for v in COUPON_KEYS.values() for k in v}
    keys |= {SBP_CHECK["total"], *SBP_CHECK["other"], SBN_CHECK["total"], *SBN_CHECK["other"]}
    return sorted(keys)


def download(session: requests.Session | None = None) -> pd.DataFrame:
    """Every week of every key used here, fields as published (strings)."""
    assert_dealers_prereg()
    s = session or requests.Session()
    df = pd.concat([fetch_series(k, s) for k in all_keys()], ignore_index=True)
    return df.sort_values(["asofdate", "keyid"], kind="mergesort").reset_index(drop=True)


def segment_of(dates: pd.DatetimeIndex) -> pd.Series:
    """The NY Fed series break of each as-of date."""
    out = pd.Series(pd.NA, index=dates, dtype=object)
    for sb, lo, hi in SERIES_BREAKS:
        out[(dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))] = sb
    if out.isna().any():
        raise ValueError(f"as-of dates outside every series break: {list(out.index[out.isna()][:3])}")
    return out


def parse(df: pd.DataFrame, end: str | None = IS_END) -> pd.DataFrame:
    """Weekly net position in Treasury coupons excluding TIPS ($ millions) and its series break, indexed by as-of
    date (as-of dates <= end). Raises if a week misses a bucket or a check fails (module docstring)."""
    d = df.assign(asofdate=pd.to_datetime(df["asofdate"], format="%Y-%m-%d"),
                  value=pd.to_numeric(df["value"], errors="raise"))
    if end is not None:
        d = d[d["asofdate"] <= pd.Timestamp(end)]
    w = d.pivot(index="asofdate", columns="keyid", values="value").sort_index()
    seg = segment_of(w.index)
    pos = pd.Series(float("nan"), index=w.index)
    for sb, keys in COUPON_KEYS.items():
        rows = seg == sb
        if not rows.any():
            continue
        block = w.loc[rows].reindex(columns=keys)
        if block.isna().any().any():
            raise ValueError(f"{sb}: a week misses a coupon bucket")
        pos[rows] = block.sum(axis=1)
    checks = {}
    for name, sbs, chk in (("SBP2013", ["SBP2013"], SBP_CHECK), ("SBN", ["SBN2013", "SBN2015", "SBN2022", "SBN2024"],
                                                               SBN_CHECK)):
        rows = seg.isin(sbs)
        if not rows.any():
            continue
        other = w.loc[rows].reindex(columns=chk["other"]).fillna(0.0).sum(axis=1)
        diff = (pos[rows] + other - w.loc[rows, chk["total"]]).abs()
        checks[name] = {"n_weeks": int(rows.sum()), "max_abs_diff_musd": float(diff.max())}
        if diff.max() > CHECK_TOL:
            raise ValueError(f"{name}: coupon buckets + other lines differ from {chk['total']} by {diff.max()}")
    out = pd.DataFrame({"position_musd": pos, "segment": seg})
    out.attrs["checks"] = checks
    return out


def load_positions(end: str | None = IS_END, path=None) -> pd.DataFrame:
    """Weekly coupon positions from the snapshot (parse()), as-of dates <= end."""
    assert_dealers_prereg()
    return parse(pd.read_csv(path or active_dir() / SNAPSHOT_NAME, dtype=str), end=end)


__all__ = ["SERIES_URL", "SNAPSHOT_NAME", "COUPON_KEYS", "SERIES_BREAKS", "all_keys", "download", "parse",
           "load_positions", "segment_of"]
