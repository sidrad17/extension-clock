"""Coupon bond price, modified duration, convexity and curve interpolation (CLAUDE.md 7.3).

Conventions (CLAUDE.md 7.3; units follow the data, so coupon and yield are in PERCENT, price per 100 face):
* semiannual coupons, semiannual compounding at the yield;
* years to maturity = calendar days / 365.25; number of remaining coupons n = ceil(2 x years);
* coupons fall on half-year steps counted back from maturity, so the first period is fractional
  (t_1 = 2 x years - (n - 1) half-years, in (0, 1]) and pays a full coupon. This is a stated approximation:
  real coupon dates and day counts (actual/actual) are not modelled.
* full price discounts every cash flow; clean price = full price - accrued, accrued = c/2 x (1 - t_1).
  At whole half-year maturities t_1 = 1, accrued = 0 and a bond with coupon = yield prices at exactly 100.
All functions accept scalars or numpy arrays (broadcast together).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import CURVE_KNOTS, IS_END

KNOT_YEARS = {"DGS1": 1.0, "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0, "DGS10": 10.0, "DGS20": 20.0,
              "DGS30": 30.0}
_EPS = 1e-9  # guards ceil(2 x years) against floating error at whole half-years


def _cashflows(coupon, ytm, years):
    """Times (half-years), cash flows and discount factors on a common (bonds x max_periods) grid."""
    c, y, T = np.broadcast_arrays(np.asarray(coupon, float), np.asarray(ytm, float), np.asarray(years, float))
    shape = c.shape
    c, y, T = c.ravel(), y.ravel(), T.ravel()
    valid = np.isfinite(c) & np.isfinite(y) & np.isfinite(T) & (T > 0)
    n = np.where(valid, np.ceil(2.0 * np.where(valid, T, 0.0) - _EPS), 0).astype(int)
    n = np.maximum(n, np.where(valid, 1, 0))
    k = np.arange(max(int(n.max()) if n.size else 1, 1))
    t = 2.0 * T[:, None] - k[None, :]                         # half-years to each flow, k = 0 is maturity
    live = k[None, :] < n[:, None]
    cf = np.where(live, c[:, None] / 2.0, 0.0)
    cf[:, 0] += np.where(valid, 100.0, 0.0)
    disc = np.where(live, (1.0 + y[:, None] / 200.0) ** (-np.where(live, t, 0.0)), 0.0)
    t_first = 2.0 * T - (n - 1)                                # fraction of the first coupon period
    return t, cf, disc, c, y, t_first, valid, shape


def _out(x, valid, shape):
    x = np.where(valid, x, np.nan).reshape(shape)
    return x.item() if x.ndim == 0 else x


def accrued(coupon, ytm, years):
    """Accrued interest per 100 face under the fractional-first-period convention."""
    _, _, _, c, _, t_first, valid, shape = _cashflows(coupon, ytm, years)
    return _out(c / 2.0 * (1.0 - t_first), valid, shape)


def price(coupon, ytm, years, clean: bool = False):
    """Full (default) or clean price per 100 face; coupon and ytm in percent."""
    t, cf, disc, c, _, t_first, valid, shape = _cashflows(coupon, ytm, years)
    p = (cf * disc).sum(axis=1)
    if clean:
        p = p - c / 2.0 * (1.0 - t_first)
    return _out(p, valid, shape)


def mod_duration(coupon, ytm, years):
    """Modified duration in years: Macaulay (PV-weighted time, years) / (1 + y/2), on the full price."""
    t, cf, disc, _, y, _, valid, shape = _cashflows(coupon, ytm, years)
    pv = cf * disc
    p = pv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mac = (pv * t).sum(axis=1) / 2.0 / p
        d = mac / (1.0 + y / 200.0)
    return _out(d, valid, shape)


def convexity(coupon, ytm, years):
    """Convexity in years^2: (1/P) d2P/dy2 with y in decimal, semiannual compounding."""
    t, cf, disc, _, y, _, valid, shape = _cashflows(coupon, ytm, years)
    pv = cf * disc
    p = pv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (pv * t * (t + 1.0)).sum(axis=1) / (4.0 * p * (1.0 + y / 200.0) ** 2)
    return _out(cx, valid, shape)


# ------------------------------------------------------------------------------------------------- the curve

class Curve:
    """Daily par-yield curve from FRED CMT knots (percent), CLAUDE.md 7.3.

    `frame`: rows = dates, columns = knot series ids (subset of KNOT_YEARS). On each date the yield at maturity x
    interpolates linearly in maturity across the knots available that day (NaN knots skipped) and is flat beyond
    the shortest and longest available knot.
    """

    def __init__(self, frame: pd.DataFrame):
        unknown = set(frame.columns) - set(KNOT_YEARS)
        if unknown:
            raise ValueError(f"unknown curve knots {unknown}")
        cols = sorted(frame.columns, key=KNOT_YEARS.get)
        self.frame = frame[cols]
        self.knots = np.array([KNOT_YEARS[c] for c in cols])

    def yields(self, date, years):
        row = self.frame.loc[pd.Timestamp(date)].to_numpy(dtype=float)
        ok = np.isfinite(row)
        if not ok.any():
            raise ValueError(f"no curve knots on {pd.Timestamp(date).date()}")
        out = np.interp(np.asarray(years, float), self.knots[ok], row[ok])
        return out.item() if np.ndim(out) == 0 else out


def load_curve(end: str | None = IS_END) -> Curve:
    """Curve on bond business days (DGS10 present), knots short-gap filled (<= 5 days), from the snapshot."""
    from src.calendar import load_calendar
    from src.data.fred import load_frame
    cal = load_calendar(end=end)
    return Curve(load_frame(list(CURVE_KNOTS), end=end, index=cal.days, fill=True))


_DEFAULT_CURVE: Curve | None = None


def curve_yield(date, years, curve: Curve | None = None):
    """Yield (percent) at `years` on `date`; uses the in-sample snapshot curve unless one is passed."""
    global _DEFAULT_CURVE
    if curve is None:
        if _DEFAULT_CURVE is None:
            _DEFAULT_CURVE = load_curve()
        curve = _DEFAULT_CURVE
    return curve.yields(date, years)
