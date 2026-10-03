"""Capacity of the cash month-end trade: net Sharpe against capital with square-root impact (CLAUDE.md 7.13).

CLAUDE.md 7.13 states the model for futures (contract ADV); without Databento the cash version uses the cash
market's volume (our choices, fixed before any capacity result):
* Volume: NY Fed primary dealer transactions in the coupon bucket that holds the 10-year note (6-11 years before
  April 2013, 7-11 years after; daily averages, src/data/pd_volume.py). ADV at entry E = the mean of the last 4
  weekly values released before E (ADV_LOOKBACK = 20 business days). Dealer volume misses trades between
  non-dealers, so it understates the market and the estimate is conservative.
* Sample: months whose E has 4 released weeks, 2001-08 to 2024-09; the base Sharpe is reported on the same months.
* Scaling: sizing is linear in capital (the 3x notional cap is a multiple of capital), so at capital K every window
  trades Q = notional x K / CAPITAL of par per side, before the participation cap.
* Participation cap (ADV_CAP = 5%): Q is cut to 5% of ADV; the window's P&L scales by the same factor.
* Impact per side = sigma_price x sqrt(Q / ADV) x Q, with sigma_price = sigma_bp x D x 1e-4, the daily price
  volatility of the traded par bond implied by the 60-day 10-year yield volatility the sizing rule uses (known at
  E). It is charged at entry and at exit, on top of CASH_COST_BP. Rough, as CLAUDE.md says.
* Net daily return at K = the base run's daily return x the participation factor, minus impact / K on the entry and
  exit days. The drawdown rule's decisions are kept from the base run (an approximation: impact would deepen
  drawdowns slightly at large K).
* Grid: capital 10^7 ... 10^10 in steps of 10^0.25 ($10M to $10B). Halving capital: where the net Sharpe falls to
  half its value at $10M (log-linear interpolation between grid points); "above the grid" if it never does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import ADV_CAP
from src.backtest import StrategyResult
from src.stats import sharpe

CAPITALS = [float(10 ** (7 + i / 4)) for i in range(13)]


def capacity_curve(res: StrategyResult, adv: pd.Series, capitals: list[float] = CAPITALS,
                   adv_cap: float = ADV_CAP) -> dict:
    """Module docstring. adv: ADV ($) known at each trade's entry, indexed by the entry dates."""
    t = res.trades.copy()
    t["adv"] = adv.reindex(pd.DatetimeIndex(t["entry"])).to_numpy()
    t = t[np.isfinite(t["adv"])]
    if t.empty:
        raise ValueError("capacity: no window has a known ADV")
    first = t["entry"].min()
    d = res.daily.loc[first:, "excess"]
    pos = pd.Series(np.arange(len(d)), index=d.index)
    base = d.to_numpy(float)
    cap0 = res.cfg.capital
    if not {"sigma_bp", "duration"} <= set(t.columns):
        raise ValueError("capacity: trades need sigma_bp and duration")
    sig_px = (t["sigma_bp"] * t["duration"] * 1e-4).to_numpy(float)
    rows = []
    for K in capitals:
        q = t["notional"].to_numpy(float) * K / cap0
        q_cap = np.minimum(q, adv_cap * t["adv"].to_numpy(float))
        f = np.divide(q_cap, q, out=np.ones_like(q), where=q > 0)                 # w = 0 months trade nothing
        impact = sig_px * np.sqrt(q_cap / t["adv"].to_numpy(float)) * q_cap        # $ per side
        r = np.zeros_like(base)
        for j, (e, x) in enumerate(zip(t["entry"], t["exit"])):
            i0, i1 = int(pos[e]), int(pos[x])
            r[i0:i1 + 1] += base[i0:i1 + 1] * f[j]
            r[i0] -= impact[j] / K
            r[i1] -= impact[j] / K
        years = ((d.index[-1] - d.index[0]).days + 1) / 365.25
        rows.append({"capital": K, "sharpe": sharpe(r),
                     "ann_excess_return_pct": (float(np.prod(1.0 + r)) ** (1.0 / years) - 1.0) * 100.0,
                     "mean_participation": float(np.mean(q_cap / t["adv"].to_numpy(float))),
                     "max_participation": float(np.max(q_cap / t["adv"].to_numpy(float))),
                     "share_windows_capped": float(np.mean(q > q_cap)),
                     "impact_pct_per_year": float(2.0 * (impact / K).sum() / years * 100.0)})
    grid = pd.DataFrame(rows)
    s0 = float(grid["sharpe"].iloc[0])
    half = s0 / 2.0
    k_half = None
    below = np.nonzero(grid["sharpe"].to_numpy() <= half)[0]
    if s0 > 0 and len(below):
        i = int(below[0])
        if i == 0:
            k_half = float(grid["capital"].iloc[0])
        else:
            s_a, s_b = grid["sharpe"].iloc[i - 1], grid["sharpe"].iloc[i]
            l_a, l_b = np.log10(grid["capital"].iloc[i - 1]), np.log10(grid["capital"].iloc[i])
            k_half = float(10 ** (l_a + (s_a - half) / (s_a - s_b) * (l_b - l_a)))
    return {"sample": [str(t["entry"].min().date()), str(d.index[-1].date())], "n_windows": int(len(t)),
            "sharpe_no_impact_same_months": sharpe(base), "sharpe_at_10m": s0,
            "capital_where_sharpe_halves": k_half, "halves_within_grid": k_half is not None,
            "median_adv_bn": float(t["adv"].median() / 1e9),
            "median_notional_x_capital": float((t["notional"] / cap0).median()),
            "grid": grid.to_dict(orient="list")}
