"""Daily constant-maturity par-bond returns per tenor and excess over T-bill (CLAUDE.md 7.4).

On each bond business day t (previous bond day t-1, dt_days calendar days apart):
  buy at t-1 a par bond with coupon = y(t-1) and maturity = tenor;
  value it at t at yield y(t) with maturity tenor - dt_days/365.25 (bonds.py year convention);
  return(t) = clean_price/100 - 1 + coupon x dt_days/365          (CLAUDE.md 7.4 formula)
  excess(t) = return(t) - DTB3(t-1) x dt_days/365.
The price in the formula is the CLEAN price: carry enters once, through coupon x dt. (The full price already
contains the accrued coupon, so using it would count carry twice.) DTB3 is quoted on a discount basis and used as
is (CLAUDE.md 7.4). Returns are decimals (0.001 = 0.1%). Yields are short-gap filled (<= 5 bond days, fred.py);
inside a known gap (DGS20 1987-01..1993-09, DGS30 2002-02..2006-02) the return is NaN.

Every call goes through the gate guards: Gate 1 for any return computation and Gate 2 for dates after IS_END
(CLAUDE.md rules 1-2, 7.14; active with GQH_DEV=1).
"""
from __future__ import annotations

import pandas as pd

from config.settings import IS_END, RF_SERIES, TENORS
from src.bonds import KNOT_YEARS, price
from src.trial_log import assert_gate1, guard_end


def par_bond_returns(y: pd.Series, tenor_years: float, rf: pd.Series | None = None) -> pd.DataFrame:
    """Daily total and excess returns of the constant-maturity par bond; y and rf in percent on one calendar."""
    y = y.astype(float)
    y_prev = y.shift(1)
    dt_days = pd.Series(y.index, index=y.index).diff().dt.days.astype(float)
    p = price(y_prev.to_numpy(), y.to_numpy(), tenor_years - dt_days.to_numpy() / 365.25, clean=True)
    ret = pd.Series(p, index=y.index) / 100.0 - 1.0 + y_prev / 100.0 * dt_days / 365.0
    out = pd.DataFrame({"ret": ret})
    if rf is not None:
        rf_prev = rf.reindex(y.index).astype(float).shift(1)
        out["excess"] = ret - rf_prev / 100.0 * dt_days / 365.0
    return out


def tenor_returns(end: str | None = IS_END, tenors: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(total, excess) daily returns on bond business days, one column per FRED tenor (default settings.TENORS)."""
    assert_gate1()
    guard_end(end)
    from src.calendar import load_calendar
    from src.data.fred import load_frame
    series = list((tenors or TENORS).values())
    cal = load_calendar(end=end)
    y = load_frame(series + [RF_SERIES], end=end, index=cal.days, fill=True)
    total, excess = {}, {}
    for sid in series:
        r = par_bond_returns(y[sid], KNOT_YEARS[sid], y[RF_SERIES])
        total[sid], excess[sid] = r["ret"], r["excess"]
    return pd.DataFrame(total), pd.DataFrame(excess)
