"""Auction event table for the Flow Clock: events, windows, size signal zS_e, month-end supply SA_m / zA_m.

PREREG_FLOWCLOCK.md "Definitions" (tag `prereg-flowclock`). Nothing here reads a return; returns, tests and the
book are in src/flowclock.py.

* Events: every nominal fixed-rate coupon auction in data/snapshot/auctions.csv, new issues and reopenings. TIPS
  (`inflation_index_security` = Yes) and FRNs (`floating_rate` = Yes) are dropped. Bills are not in the file (the
  query is Notes and Bonds, src/data/auctions.py). The 19 callable bonds (1979-1984) are nominal fixed-rate, so they
  stay. The auction date A must be a bond business day; the rest are dropped and counted (none through IS_END).
* Tenor: the ORIGINAL term (`original_security_term`, e.g. "10-Year" for a reopening whose `security_term` is
  "9-Year 10-Month"), read as years + months / 12 and mapped to the nearest CMT series in FC_TENORS. Ties go to the
  longer ("4-Year" -> DGS5).
* Windows: pre = close of A-5 to close of A; post = close of A to close of A+5, in bond business days
  (src/calendar.py). An event is skipped if its tenor's yield is missing on any bond day A-5..A+5. The yields are
  short-gap filled exactly as src/returns.py fills them, so a skipped event is one whose window return is undefined.
* Size signal zS_e = (amount - mean of the 6 prior amounts) / their std (ddof = 1, as src/signals.py):
  - priors are same-tenor auctions held before A, counted from the start of the tenor's current issuance run (a gap
    of more than ZS_BREAK_DAYS between consecutive auctions of the tenor starts a new run);
  - fewer than 6 priors: no zS_e (NaN, flag "none"); the event leaves H6c and gets w_e = 1;
  - 6 equal priors: zS_e = 0 if the amount is unchanged, else +/-ZS_CLIP (flag "sd0"); every zS_e is clipped to
    [-ZS_CLIP, ZS_CLIP] (flag "clipped" when the clip binds).
  Two versions, by what is known when each leg is entered:
  - zS_post (known at the close of A, the post leg's entry): the event's own offering_amt (always announced on or
    before A) against priors held before A.
  - zS_pre (known at the close of A-5, the pre leg's and the long-short's entry; used by H6c): the event's
    offering_amt only if `announcemt_date` <= A-5, else the previous same-tenor auction's; priors must also be
    announced by A-5. Our reading, fixed before any auction-window return: "known at A-5" applies to every amount
    used, so a prior auction announced after A-5 (only the off-cycle 10-year reopenings of 2008-10-08/09 in-sample)
    is not a prior for zS_pre. "The previous same-tenor auction" is the latest such prior.
  w_e = clip(1 + zS_e, WS_CLIP) with the version of the leg (w_pre, w_post); 1 where zS_e is missing.
* Samples: an event belongs to a sample when its whole window A-5..A+5 lies inside it (PREREG_FLOWCLOCK.md).
* Month-end supply: SA_m = sum over events with A in [T-8, T-4] of offering_amt x the modified duration of a par
  bond of the tenor's maturity at the curve yield on A (src/bonds.py), in $bn x years. Our choices, fixed before any
  return: every event counts, skipped or not (supply is supply; the skip rule concerns window returns); the curve
  yield equals the tenor's CMT yield whenever it is published. A <= T-4, so SA_m is known at the entry T-4.
  zA_m = SA_m standardized on past months only (src/signals.py past_zscore, ZSCORE_MIN_MONTHS = 36), with SA_m from
  SA_START = 1990-01 like the index rebuild, so zA_m starts in 1993-01 exactly as z_m does.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from config.flowclock import (FC_TENORS, POST_DAYS, PRE_DAYS, SA_FROM_T, WS_CLIP, ZS_BREAK_DAYS, ZS_CLIP,
                              ZS_PRIOR)
from src.bonds import KNOT_YEARS, mod_duration

TERM_RE = re.compile(r"^(\d+)-Year(?: (\d+)-Month)?$")


def term_years(term: str) -> float:
    """'5-Year 2-Month' -> 5.1667. Every original_security_term in the snapshot has this form."""
    m = TERM_RE.match(str(term).strip())
    if not m:
        raise ValueError(f"unrecognised security term {term!r}")
    return int(m.group(1)) + int(m.group(2) or 0) / 12.0


def nearest_tenor(years: float, tenors: list[str] = FC_TENORS) -> str:
    """Nearest CMT series by maturity; ties go to the longer one."""
    return min(tenors, key=lambda s: (abs(KNOT_YEARS[s] - years), -KNOT_YEARS[s]))


def size_z(amount: float, prior: np.ndarray, clip: float = ZS_CLIP) -> tuple[float, str]:
    """(zS, flag) from the prior amounts (module docstring)."""
    prior = np.asarray(prior, float)
    mu = prior.mean()
    if np.ptp(prior) == 0:
        return (0.0 if amount == mu else float(np.sign(amount - mu) * clip)), "sd0"
    z = (amount - mu) / prior.std(ddof=1)
    return float(np.clip(z, -clip, clip)), ("clipped" if abs(z) > clip else "ok")


def run_starts(dates: pd.Series, break_days: int = ZS_BREAK_DAYS) -> pd.Series:
    """For each auction date of one tenor, the first date of its issuance run (a gap > break_days starts a run)."""
    u = pd.DatetimeIndex(sorted(dates.unique()))
    start, out = u[0], {}
    for i, d in enumerate(u):
        if i and (d - u[i - 1]).days > break_days:
            start = d
        out[d] = start
    return dates.map(out)


def _size_signal(g: pd.DataFrame, cal) -> pd.DataFrame:
    """zS_post and zS_pre for every auction of ONE tenor (rows sorted by auction_date, cusip)."""
    g = g.sort_values(["auction_date", "cusip"], kind="mergesort")
    rs = run_starts(g["auction_date"])
    out = []
    for i, r in g.iterrows():
        A = r["auction_date"]
        held = g[(g["auction_date"] < A) & (g["auction_date"] >= rs.loc[i])]
        row = {"n_prior": len(held), "zS_post": np.nan, "zS_post_flag": "none", "zS_pre": np.nan,
               "zS_pre_flag": "none", "S_pre": np.nan, "S_pre_source": "", "prior_mean": np.nan, "prior_sd": np.nan}
        if len(held) >= ZS_PRIOR:
            p6 = held["offering_amt"].to_numpy()[-ZS_PRIOR:]
            row["prior_mean"], row["prior_sd"] = p6.mean(), p6.std(ddof=1)
            row["zS_post"], row["zS_post_flag"] = size_z(r["offering_amt"], p6)
        a5 = _offset_or_nat(cal, A, -PRE_DAYS)
        if pd.notna(a5):
            known = held[held["announcemt_date"] <= a5]
            if len(known) >= ZS_PRIOR:
                own = r["announcemt_date"] <= a5
                s = r["offering_amt"] if own else known["offering_amt"].iloc[-1]
                row["S_pre"], row["S_pre_source"] = s, ("own" if own else "previous")
                row["zS_pre"], row["zS_pre_flag"] = size_z(s, known["offering_amt"].to_numpy()[-ZS_PRIOR:])
        out.append(pd.Series(row, name=i))
    return pd.DataFrame(out)


def _offset_or_nat(cal, d, k):
    try:
        return cal.offset(d, k)
    except (KeyError, IndexError):
        return pd.NaT


def build_events(auctions: pd.DataFrame, cal, yields: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per nominal coupon auction on a bond business day, with tenor, windows, zS and weights.

    auctions: parsed auction records, nothing excluded (src/data/auctions.py::load_auctions(end, exclude=None)).
    yields: the FC_TENORS yields on bond business days, short-gap filled (for the skip flag); None leaves it NaN.
    df.attrs records the counts the note reports (TIPS, FRNs, off-calendar auctions dropped).
    """
    a = auctions.copy()
    tips = a["inflation_index_security"].fillna(False).astype(bool)
    frn = a["floating_rate"].fillna(False).astype(bool)
    n_tips, n_frn = int(tips.sum()), int((frn & ~tips).sum())
    a = a[~tips & ~frn]
    on_bday = a["auction_date"].map(lambda d: d in cal)
    n_off = int((~on_bday).sum())
    a = a[on_bday].copy()
    a["orig_years"] = a["original_security_term"].map(term_years)
    a["tenor"] = a["orig_years"].map(nearest_tenor)
    a["tenor_years"] = a["tenor"].map(KNOT_YEARS)
    sig = pd.concat([_size_signal(g, cal) for _, g in a.groupby("tenor", sort=True)])
    a = a.join(sig)
    a = a.sort_values(["auction_date", "tenor", "cusip"], kind="mergesort")
    ev = pd.DataFrame({
        "event_id": a["auction_date"].dt.strftime("%Y-%m-%d") + "_" + a["cusip"],
        "A": a["auction_date"], "announced": a["announcemt_date"], "cusip": a["cusip"],
        "security_term": a["security_term"], "original_term": a["original_security_term"],
        "reopening": a["reopening"].astype(bool), "tenor": a["tenor"], "tenor_years": a["tenor_years"],
        "offering_amt": a["offering_amt"],
    })
    ev["pre_entry"] = [_offset_or_nat(cal, d, -PRE_DAYS) for d in ev["A"]]
    ev["post_exit"] = [_offset_or_nat(cal, d, POST_DAYS) for d in ev["A"]]
    for c in ["n_prior", "prior_mean", "prior_sd", "zS_post", "zS_post_flag", "S_pre", "S_pre_source", "zS_pre",
              "zS_pre_flag"]:
        ev[c] = a[c].to_numpy()
    ev["n_prior"] = ev["n_prior"].astype(int)
    for c in ["prior_mean", "prior_sd", "zS_post", "S_pre", "zS_pre"]:
        ev[c] = ev[c].astype(float)
    ev["w_pre"] = (1.0 + ev["zS_pre"]).clip(*WS_CLIP).fillna(1.0)
    ev["w_post"] = (1.0 + ev["zS_post"]).clip(*WS_CLIP).fillna(1.0)
    ev["week"] = ev["A"].dt.to_period("W-SUN").astype(str)       # Monday-Sunday calendar week of A (H6 clusters)
    ev["window_complete"] = ev["pre_entry"].notna() & ev["post_exit"].notna()
    ev["skipped"] = np.nan if yields is None else skip_flags(ev, yields)
    ev = ev.reset_index(drop=True)
    ev.attrs.update({"n_tips_dropped": n_tips, "n_frn_dropped": n_frn, "n_not_bond_day": n_off})
    return ev


def skip_flags(ev: pd.DataFrame, yields: pd.DataFrame) -> pd.Series:
    """True if the tenor's yield is missing on any bond day A-5..A+5 (or the window leaves the calendar)."""
    out = []
    for t, e, x in zip(ev["tenor"], ev["pre_entry"], ev["post_exit"]):
        if pd.isna(e) or pd.isna(x):
            out.append(True)
            continue
        out.append(bool(yields[t].loc[e:x].isna().any()))
    return pd.Series(out, index=ev.index, dtype=bool)


def in_sample_mask(ev: pd.DataFrame, start, end) -> pd.Series:
    """Events whose whole window A-5..A+5 lies in [start, end] (PREREG_FLOWCLOCK.md "Samples")."""
    return ev["window_complete"] & (ev["pre_entry"] >= pd.Timestamp(start)) & (ev["post_exit"] <= pd.Timestamp(end))


# ------------------------------------------------------------------------------------------- month-end supply

def month_end_supply(ev: pd.DataFrame, cal, curve, months) -> pd.DataFrame:
    """SA_m ($bn x years) and the count of events with A in [T-8, T-4], per month (module docstring)."""
    lo_k, hi_k = SA_FROM_T
    rows = {}
    for m in months:
        T = cal.month_end(m)
        lo, hi = cal.offset(T, -lo_k), cal.offset(T, -hi_k)
        sel = ev[(ev["A"] >= lo) & (ev["A"] <= hi)]
        sa = 0.0
        for A, yrs, amt in zip(sel["A"], sel["tenor_years"], sel["offering_amt"]):
            y = float(curve.yields(A, yrs))
            sa += amt * float(mod_duration(y, y, yrs))
        rows[m] = {"T": T, "from": lo, "to": hi, "n_events": len(sel), "SA_bn_years": sa / 1e9,
                   "tenors": ",".join(sorted(sel["tenor"].unique()))}
    return pd.DataFrame.from_dict(rows, orient="index")
