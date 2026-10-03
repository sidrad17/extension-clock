"""Index rebuild validation checks shown at STOP 2 (CLAUDE.md 7.6). No returns are used anywhere here.

1. par_vs_mspd: rebuilt gross par of fixed-coupon notes and bonds outstanding (before any SOMA deduction, no
   maturity or size screen, callable bonds kept until their called_date, i.e. the MSPD definition) vs MSPD
   marketable Notes + Bonds (total_mil_amt) at each month-end. MSPD on Fiscal Data starts 2001-01.
   A security counts until it is PAID: maturity rolled to the next bond business day. Checked on the data: MSPD
   still counts notes maturing on a weekend month-end (e.g. 2022-12-31, a Saturday) because they are paid on the
   next business day; with the plain maturity date the December gap jumps by $60-140B in exactly those years, and
   the month-to-month change in the gap has a std of $56B, against $1.9B with the paid-date rule.
2. duration_table: rebuilt index duration 2015-2024, for humans to set beside published Treasury-index ETF
   durations (they look up and cite the published values).
3. extension_stats / outliers: distribution of Ext, c_m, FDD; months with |z| > 4 traced to their auctions. The z
   here is a full-sample DIAGNOSTIC over the in-sample months; no signal uses it (signals use past months only).
4. count_table: adds, removes, reopenings, estimated (offering_amt) and skipped-for-lookahead tranches.
5. SOMA by CUSIP (PREREG_ADDENDUM.md): soma_coverage puts the Fed's notes-and-bonds holdings beside the par table
   (MSPD totals include Fed holdings, so the par gap itself does not change); soma_outstanding_check compares the
   amount outstanding the Fed's file implies (parValue / percentOutstanding) with our gross par, CUSIP by CUSIP;
   duration_compare and era_compare set the SOMA-by-CUSIP rebuild beside the auction-only one.
Everything is restricted to dates <= IS_END (CLAUDE.md rule 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import IS_END, IS_START
from src.index_rebuild import BUCKETS

DIAG_Z = 4.0
ERAS = {"pre-2009 (1993-01..2008-12)": ("1993-01", "2008-12"),
        "QE years (2009-01..2022-05)": ("2009-01", "2022-05"),
        "post-2022 (2022-06..2024-09)": ("2022-06", "2024-09")}


def in_sample(monthly: pd.DataFrame) -> pd.DataFrame:
    lo, hi = pd.Period(IS_START, "M"), pd.Period(IS_END, "M")
    p = pd.PeriodIndex(monthly["month"], freq="M")
    return monthly[(p >= lo) & (p <= hi)]


# ------------------------------------------------------------------------------------------- 1. par vs MSPD

def paid_date(stop: pd.Series, bdays: pd.DatetimeIndex) -> pd.Series:
    """First bond business day on or after `stop` (dates past the calendar are left unchanged)."""
    pos = bdays.searchsorted(stop.to_numpy())
    inside = pos < len(bdays)
    rolled = np.where(inside, bdays.to_numpy()[np.minimum(pos, len(bdays) - 1)], stop.to_numpy())
    return pd.Series(pd.DatetimeIndex(rolled), index=stop.index)


def rebuilt_gross_par(auctions_all: pd.DataFrame, dates: pd.DatetimeIndex, bdays: pd.DatetimeIndex) -> pd.DataFrame:
    """Gross par outstanding ($ millions) at each date: issue_date <= d < paid date of maturity (or call).

    `auctions_all`: load_auctions(exclude={"TIPS": True, "FRN": True, "callable": False}), so callable bonds are in.
    Gross = total_accepted (includes the Fed's add-ons; before 2008 those are not reported separately).
    """
    a = auctions_all
    called = a.groupby("cusip")["called_date"].transform("max")
    stop = paid_date(a["maturity_date"].where(called.isna(), np.minimum(a["maturity_date"], called)), bdays)
    out = []
    for d in dates:
        live = (a["issue_date"] <= d) & (stop > d)
        out.append({"date": d, "rebuilt_mil": a.loc[live, "total_accepted"].sum() / 1e6,
                    "rebuilt_callable_mil": a.loc[live & a["callable"].fillna(False).astype(bool),
                                                  "total_accepted"].sum() / 1e6})
    return pd.DataFrame(out).set_index("date")


def par_vs_mspd(auctions_all: pd.DataFrame, mspd: pd.DataFrame, bdays: pd.DatetimeIndex) -> pd.DataFrame:
    m = mspd.loc[: pd.Timestamp(IS_END)]
    r = rebuilt_gross_par(auctions_all, m.index, bdays)
    out = r.join(m[["notes_bonds_total"]].rename(columns={"notes_bonds_total": "mspd_mil"}))
    out["gap_mil"] = out["rebuilt_mil"] - out["mspd_mil"]
    out["gap_pct"] = 100.0 * out["gap_mil"] / out["mspd_mil"]
    return out


def par_gap_summary(par: pd.DataFrame) -> pd.DataFrame:
    """December (or last available month) of each year, $ billions."""
    last = par.groupby(par.index.year).tail(1)
    return pd.DataFrame({"date": last.index.date, "rebuilt_bn": last["rebuilt_mil"] / 1e3,
                         "mspd_bn": last["mspd_mil"] / 1e3, "gap_bn": last["gap_mil"] / 1e3,
                         "gap_pct": last["gap_pct"], "callable_in_rebuilt_bn": last["rebuilt_callable_mil"] / 1e3})


def live_gross_by_cusip(auctions_all: pd.DataFrame, d: pd.Timestamp, bdays: pd.DatetimeIndex) -> pd.Series:
    """Gross par ($) by CUSIP outstanding at d (issued by d, not yet paid), the MSPD definition used above."""
    a = auctions_all
    called = a.groupby("cusip")["called_date"].transform("max")
    stop = paid_date(a["maturity_date"].where(called.isna(), np.minimum(a["maturity_date"], called)), bdays)
    live = (a["issue_date"] <= d) & (stop > d)
    return a.loc[live].groupby("cusip")["total_accepted"].sum()


def soma_coverage(auctions_all: pd.DataFrame, soma, implied: pd.DataFrame, cal, par: pd.DataFrame,
                  dates: pd.DatetimeIndex) -> pd.DataFrame:
    """At each date: the Fed's notes-and-bonds holdings on the SOMA date known then, how much of it sits in CUSIPs
    our auction records have outstanding ($ billions), and the par gap check: rebuilt gross par minus the amount
    outstanding the Fed's file implies (parValue / percentOutstanding), both on the SOMA date and summed over the
    CUSIPs the Fed holds, beside
    the MSPD par gap of section 1. Auction records never show Treasury buybacks (2000-2002, and again from 2024)."""
    rows = []
    for d in dates:
        X = cal.days[cal.days <= d][-1]
        s = soma.usable(cal, X)
        if s is None:
            continue
        h = soma.on(s)
        live = live_gross_by_cusip(auctions_all, d, cal.days)
        matched = h[h.index.isin(live.index)]
        imp = implied[implied["asOfDate"] == s].set_index("cusip")["implied_outstanding"].dropna()
        live_s = live_gross_by_cusip(auctions_all, s, cal.days)          # same day as the Fed's file
        both = imp.index.intersection(live_s.index)
        rows.append({"date": d.date(), "soma_date": s.date(), "soma_notes_bonds_bn": h.sum() / 1e9,
                     "soma_matched_bn": matched.sum() / 1e9, "matched_share_pct": 100 * matched.sum() / h.sum(),
                     "rebuilt_gross_bn": live.sum() / 1e9, "rebuilt_ex_fed_bn": (live.sum() - matched.sum()) / 1e9,
                     "fed_share_pct": 100 * matched.sum() / live.sum(),
                     "rebuilt_minus_fed_implied_bn": (live_s[both] - imp[both]).sum() / 1e9,
                     "mspd_par_gap_bn": par.loc[: d, "gap_mil"].iloc[-1] / 1e3})
    return pd.DataFrame(rows).set_index("date")


def soma_outstanding_check(auctions_all: pd.DataFrame, implied: pd.DataFrame, bdays: pd.DatetimeIndex) -> dict:
    """parValue / percentOutstanding (the Fed's view of the amount outstanding) vs our gross par per CUSIP, on
    every stored SOMA date: share of CUSIP-dates within 0.5% and 2%, and the worst ones."""
    out = []
    for d, g in implied.dropna(subset=["implied_outstanding"]).groupby("asOfDate"):
        live = live_gross_by_cusip(auctions_all, d, bdays)
        m = g.set_index("cusip").join(live.rename("gross"), how="inner")
        out.append(m.assign(asOfDate=d))
    m = pd.concat(out)
    m["ratio"] = m["implied_outstanding"] / m["gross"]
    dev = (m["ratio"] - 1).abs()
    worst = m.assign(dev=dev).nlargest(5, "dev")
    return {"cusip_dates": int(len(m)), "within_0.5pct": round(float((dev <= 0.005).mean()), 4),
            "within_2pct": round(float((dev <= 0.02).mean()), 4), "median_ratio": round(float(m["ratio"].median()), 6),
            "worst": [{"asOfDate": str(r.asOfDate.date()), "cusip": c, "implied_bn": round(r.implied_outstanding / 1e9, 3),
                       "gross_bn": round(r.gross / 1e9, 3)} for c, r in worst.iterrows()]}


def par_gap_by_decade(par: pd.DataFrame) -> dict:
    dec = (par.index.year // 10) * 10
    g = par.groupby(dec)["gap_pct"]
    return {f"{k}s": {"mean_gap_pct": round(float(v.mean()), 3), "min": round(float(v.min()), 3),
                      "max": round(float(v.max()), 3), "months": int(v.size)} for k, v in g}


# ------------------------------------------------------------------------------------------ 2. duration

def duration_table(monthly: pd.DataFrame, years=range(2015, 2025)) -> pd.DataFrame:
    p = pd.PeriodIndex(monthly["month"], freq="M")
    rows = []
    for y in years:
        sel = monthly[p.year == y]
        if sel.empty:
            continue
        last = sel.iloc[-1]
        rows.append({"year": y, "last_month": last["month"], "D_next_last": last["D_next"],
                     "D_now_mean": sel["D_now"].mean(), "published_etf_duration": "", "source": ""})
    return pd.DataFrame(rows)


def duration_compare(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Duration table for the auction-only rebuild (before) and the SOMA-by-CUSIP rebuild (after)."""
    b = duration_table(before).set_index("year")
    a = duration_table(after).set_index("year")
    return pd.DataFrame({"last_month": a["last_month"],
                         "D_next_last_before": b["D_next_last"], "D_next_last_after": a["D_next_last"],
                         "D_now_mean_before": b["D_now_mean"], "D_now_mean_after": a["D_now_mean"],
                         "change_D_now_mean": a["D_now_mean"] - b["D_now_mean"],
                         "published_etf_duration": "", "source": ""})


def _era(monthly: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    p = pd.PeriodIndex(monthly["month"], freq="M")
    return monthly[(p >= pd.Period(lo, "M")) & (p <= pd.Period(hi, "M"))]


def era_compare(before: pd.DataFrame, after: pd.DataFrame, entry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean Ext and FDD by era, auction-only (before) vs SOMA by CUSIP (after); optional NOW-at-entry diagnostic."""
    rows = []
    for name, (lo, hi) in ERAS.items():
        b, a = _era(before, lo, hi), _era(after, lo, hi)
        r = {"era": name, "months": len(a), "months_cusip_rule": int((a["soma_rule"] == "cusip").sum())}
        for col in ["Ext", "FDD"]:
            r[f"{col}_before"] = b[col].mean()
            r[f"{col}_after"] = a[col].mean()
            r[f"{col}_change"] = a[col].mean() - b[col].mean()
        if entry is not None:
            e = _era(entry, lo, hi)
            r["Ext_now_at_entry"] = e["Ext"].mean()
            r["FDD_now_at_entry"] = e["FDD"].mean()
        r["corr_FDD_before_after"] = float(np.corrcoef(b["FDD"], a["FDD"])[0, 1])
        r["D_now_mean_before"] = b["D_now"].mean()
        r["D_now_mean_after"] = a["D_now"].mean()
        r["fed_bn_mean_now"] = a["soma_now_bn"].mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("era")


# ---------------------------------------------------------------------------- 3. distribution and outliers

def extension_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    s = in_sample(monthly)
    cash = s["c_m"] * s["D_next"]
    df = pd.DataFrame({"Ext": s["Ext"], "c_m": s["c_m"], "cash_term": cash, "FDD": s["FDD"]})
    out = df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    out["share_negative"] = (df < 0).mean()
    return out


def seasonality(monthly: pd.DataFrame) -> pd.DataFrame:
    s = in_sample(monthly)
    cal_month = pd.PeriodIndex(s["month"], freq="M").month
    return s.groupby(cal_month)[["Ext", "c_m", "FDD", "n_adds", "n_removes"]].mean().rename_axis("calendar_month")


def outliers(monthly: pd.DataFrame, changes: pd.DataFrame, z: float = DIAG_Z) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Months with |z| > z on Ext, c_m or FDD (full in-sample diagnostic z), and their traced CUSIP changes."""
    s = in_sample(monthly)
    flags = []
    for col in ["Ext", "c_m", "FDD"]:
        zz = (s[col] - s[col].mean()) / s[col].std(ddof=1)
        for i in s.index[zz.abs() > z]:
            flags.append({"month": s.at[i, "month"], "series": col, "value": s.at[i, col], "z": zz[i]})
    flagged = pd.DataFrame(flags, columns=["month", "series", "value", "z"])
    traced = changes[changes["month"].isin(flagged["month"])].sort_values(["month", "contribution"],
                                                                          ascending=[True, False])
    return flagged, traced


def top_months(monthly: pd.DataFrame, changes: pd.DataFrame, col: str = "FDD", n: int = 3) -> pd.DataFrame:
    """The n largest in-sample months of `col` with their CUSIP changes (largest contributions first)."""
    s = in_sample(monthly).nlargest(n, col)
    return changes[changes["month"].isin(s["month"])].sort_values(["month", "contribution"], ascending=[True, False])


# ----------------------------------------------------------------------------------------------- 4. counts

def count_table(monthly: pd.DataFrame) -> pd.DataFrame:
    p = pd.PeriodIndex(monthly["month"], freq="M")
    cols = ["n_adds", "n_removes", "n_reopens", "n_estimated", "n_skipped"]
    out = monthly.groupby(p.year)[cols].sum()
    out["months"] = monthly.groupby(p.year).size()
    out["n_next_dec"] = monthly.groupby(p.year)["n_next"].last()
    return out.rename_axis("year")


def bucket_check(monthly: pd.DataFrame) -> dict:
    """Max |sum_b dC_b - Ext| and |sum_b fdd_b - FDD| (both must be ~0, CLAUDE.md 7.5)."""
    ext = (monthly[[f"ext_{b}" for b in BUCKETS]].sum(axis=1) - monthly["Ext"]).abs().max()
    fdd = (monthly[[f"fdd_{b}" for b in BUCKETS]].sum(axis=1) - monthly["FDD"]).abs().max()
    return {"max_abs_ext_bucket_error": float(ext), "max_abs_fdd_bucket_error": float(fdd)}


# --------------------------------------------------------------------------------------------- formatting

def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", index: bool = True) -> str:
    """Markdown table without extra dependencies."""
    d = df.reset_index() if index else df
    cells = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
              ("" if isinstance(v, (float, np.floating)) else str(v)) for v in row] for row in d.to_numpy()]
    head = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join(lines)
