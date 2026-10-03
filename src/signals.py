"""Point-in-time signals: z_fdd, component z's, w_m, surprise, pension pressure, dummies (CLAUDE.md 7.7).

Every signal for month m uses only data known at the close of its entry day E = T - 4 (CLAUDE.md rule 3):
* FDD_m, Ext_m and c_m come from the index rebuild, which is point in time at E (src/index_rebuild.py).
* z_m = (x_m - mean(x_<m)) / std(x_<m) over PRIOR months only (expanding window, sample std, ddof = 1), defined
  once at least ZSCORE_MIN_MONTHS (36) prior values exist; otherwise NaN, and then w_m = 1 (CLAUDE.md 7.7). The
  rebuild starts 1990-01, so z is defined from 1993-01 = IS_START (PREREG_ADDENDUM.md section 4).
* Components (component test, CLAUDE.md 7.11): Ext_m alone and the cash term c_m x D_next alone, each z-scored the
  same way. FDD = Ext + cash term when REINVEST_COUPONS (the pre-registered default).
* w_m = clip(1 + z_m, W_CLIP) = clip(1 + z_m, 0, 2) (HYPOTHESIS.md, settings.W_CLIP).
* surprise_m = Ext_m - mean(Ext in the same calendar month of the 3 prior years), all three required
  (CLAUDE.md 7.7, PREREG_ADDENDUM.md (c)); z_surprise uses the same past-only rule, so it starts 1996-01.
* pension_m = equity total return (Ken French Mkt-RF + RF) minus the 10-year cash-bond total return, both
  cumulative from the first bond business day of month m through E - 1 (= T - 5 for the pre-registered E = T - 4,
  CLAUDE.md 7.7), z-scored on past months. Ending at E - 1 keeps it known at E for any entry offset.
* Dummies (CLAUDE.md 7.7): quarter-end (Mar, Jun, Sep, Dec), year-end (Dec), refunding month (Feb, May, Aug, Nov;
  REF_m in PREREG_ADDENDUM.md), and fomc = 1 if a SCHEDULED FOMC decision date falls in (E, T] (known at E;
  unscheduled actions never enter a signal, CLAUDE.md 7.1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import W_CLIP, ZSCORE_MIN_MONTHS

REFUNDING_MONTHS = (2, 5, 8, 11)
QUARTER_END_MONTHS = (3, 6, 9, 12)


def past_zscore(x: pd.Series, min_months: int = ZSCORE_MIN_MONTHS) -> pd.Series:
    """z_m from prior values only: (x_m - mean(x_<m)) / std(x_<m), NaN until `min_months` prior values exist.

    `x` must be in month order. NaN values of x are skipped in the past mean and std and do not count toward
    `min_months`.
    """
    past = x.shift(1).expanding(min_periods=min_months)
    return (x - past.mean()) / past.std(ddof=1)


def weight(z: pd.Series, clip: tuple[float, float] = W_CLIP) -> pd.Series:
    """w_m = clip(1 + z_m, 0, 2); w_m = 1 where z_m is undefined (CLAUDE.md 7.7)."""
    return (1.0 + z).clip(*clip).fillna(1.0)


def surprise(ext: pd.Series, years: int = 3) -> pd.Series:
    """Ext_m minus the mean of Ext in the same calendar month of the `years` prior years (all required).

    `ext` is indexed by monthly Period.
    """
    lags = [ext.reindex(ext.index - 12 * k).to_numpy(float) for k in range(1, years + 1)]
    base = np.mean(np.vstack(lags), axis=0)          # NaN if any prior year is missing
    return pd.Series(ext.to_numpy(float) - base, index=ext.index, name="surprise")


def window_has(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if any date in `dates` falls in (start, end]."""
    i = dates.searchsorted(start, side="right")
    return bool(i < len(dates) and dates[i] <= end)


def pension_pressure(months: pd.DataFrame, cal, equity_pct: pd.Series, bond_total: pd.Series) -> pd.Series:
    """Equity minus 10-year cash-bond cumulative total return, first bond day of month m through E - 1.

    months: indexed by Period with an `E` column. equity_pct: daily equity total return in percent (Ken French
    Mkt-RF + RF). bond_total: daily 10-year par-bond total return (decimal) on bond business days.
    """
    out = {}
    for m, row in months.iterrows():
        first = cal.month_days(m)[0]
        last = cal.offset(row["E"], -1)
        eq = equity_pct.loc[first:last]
        bd = bond_total.loc[first:last]
        if len(eq) == 0 or len(bd) == 0 or eq.isna().any() or bd.isna().any():
            out[m] = np.nan
            continue
        out[m] = float(np.prod(1.0 + eq.to_numpy() / 100.0) - np.prod(1.0 + bd.to_numpy()))
    return pd.Series(out, name="pension", dtype=float)


def build_signals(monthly: pd.DataFrame, fomc_scheduled: pd.DatetimeIndex, pension: pd.Series | None = None,
                  min_months: int = ZSCORE_MIN_MONTHS) -> pd.DataFrame:
    """Monthly signal table indexed by Period, from the rebuild output (src/index_rebuild.py).

    `monthly` must start at the rebuild start (1990-01) so the past-only z has its warm-up; `fomc_scheduled` are
    scheduled FOMC decision dates only (src/data/fomc.py, scheduled_only=True).
    """
    m = monthly.copy()
    idx = pd.PeriodIndex(m["month"], freq="M")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("monthly rebuild must be one row per month in order")
    s = pd.DataFrame(index=idx)
    s["T"] = pd.to_datetime(m["T"].to_numpy())
    s["E"] = pd.to_datetime(m["E"].to_numpy())
    s["FDD"] = m["FDD"].to_numpy(float)
    s["Ext"] = m["Ext"].to_numpy(float)
    s["cash"] = (m["c_m"] * m["D_next"]).to_numpy(float)
    s["z"] = past_zscore(s["FDD"], min_months)
    s["z_ext"] = past_zscore(s["Ext"], min_months)
    s["z_cash"] = past_zscore(s["cash"], min_months)
    s["w"] = weight(s["z"])
    s["surprise"] = surprise(s["Ext"])
    s["z_surprise"] = past_zscore(s["surprise"], min_months)
    if pension is not None:
        s["pension"] = pension.reindex(idx)
        s["z_pension"] = past_zscore(s["pension"], min_months)
    s["ref"] = idx.month.isin(REFUNDING_MONTHS).astype(int)
    s["quarter_end"] = idx.month.isin(QUARTER_END_MONTHS).astype(int)
    s["year_end"] = (idx.month == 12).astype(int)
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()
    s["fomc"] = [int(window_has(fomc, e, t)) for e, t in zip(s["E"], s["T"])]
    return s
