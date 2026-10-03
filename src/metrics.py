"""Required performance metrics (CLAUDE.md 7.13). Betas, crowding monitor and tails arrive in Phase 5.

Definitions (CLAUDE.md 7.13; conventions are our choice, fixed before any return was computed):
* annual return (geometric): NAV_end^(1/years) - 1, years = calendar days of the period / 365.25, where NAV
  compounds the daily returns. Reported for the total-return NAV (strategy P&L plus the T-bill on capital) and for
  the excess NAV (strategy P&L only).
* volatility: std (ddof = 1) of daily excess returns x sqrt(252).
* Sharpe: mean / std of daily excess returns (over DTB3) x sqrt(252), net of costs (and gross, for reference).
* max drawdown: largest peak-to-trough fall of the NAV, total-return and excess.
* turnover: sum |notional traded| / CAPITAL / years; a window trades its notional twice (in at entry, out at exit).
  Capital is fixed (src/backtest.py), so average capital = CAPITAL.
* hit rate: share of windows with net P&L > 0; worst window: lowest net window P&L, % of capital.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import StrategyResult
from src.stats import DAYS_PER_YEAR, sharpe


def max_drawdown(nav: pd.Series) -> float:
    """Largest fall from a running peak, as a positive fraction (NAV starts at 1 before the first day)."""
    v = np.concatenate([[1.0], nav.to_numpy(float)])
    peak = np.maximum.accumulate(v)
    return float((1.0 - v / peak).max())


def period_years(days: pd.DatetimeIndex) -> float:
    """Calendar length of the period, first day to last day inclusive, in years."""
    return float(((days[-1] - days[0]).days + 1) / 365.25)


def required_metrics(res: StrategyResult) -> dict:
    d, t = res.daily, res.trades
    years = period_years(d.index)
    nav_total = (1.0 + d["total"]).cumprod()
    nav_excess = (1.0 + d["excess"]).cumprod()
    gross_excess = d["excess"].copy()
    cost_per_day = pd.Series(0.0, index=d.index)
    for _, tr in t.iterrows():
        cost_per_day.loc[tr["entry"]] += tr["cost"] / 2.0
        cost_per_day.loc[tr["exit"]] += tr["cost"] / 2.0
    gross_excess += cost_per_day / res.cfg.capital
    worst = t["net_pnl"].idxmin()
    return {
        "period": [d.index[0].date().isoformat(), d.index[-1].date().isoformat()],
        "years": years,
        "ann_return_pct": (float(nav_total.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0,
        "ann_excess_return_pct": (float(nav_excess.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0,
        "vol_pct": float(d["excess"].std(ddof=1) * np.sqrt(DAYS_PER_YEAR) * 100.0),
        "sharpe": sharpe(d["excess"]),
        "sharpe_gross": sharpe(gross_excess),
        "max_drawdown_pct": max_drawdown(nav_total) * 100.0,
        "max_drawdown_excess_pct": max_drawdown(nav_excess) * 100.0,
        "turnover_x_per_year": float(2.0 * t["notional"].sum() / res.cfg.capital / years),
        "hit_rate": float((t["net_pnl"] > 0).mean()),
        "worst_window": {"month": str(worst), "net_pnl_pct": float(t.at[worst, "net_pnl"] / res.cfg.capital * 100)},
        "n_windows": int(len(t)),
        "mean_window_net_pnl_pct": float(t["net_pnl"].mean() / res.cfg.capital * 100.0),
        "cost_pct_per_year": float(t["cost"].sum() / res.cfg.capital / years * 100.0),
        "mean_notional_x_capital": float(t["notional"].mean() / res.cfg.capital),
        "n_notional_capped": int(t["capped"].sum()),
        "n_fomc_halved": int((t["fomc_mult"] < 1).sum()),
        "n_drawdown_halved": int((t["dd_mult"] < 1).sum()),
    }


def equity_curves(results: dict[str, StrategyResult]) -> pd.DataFrame:
    """Daily total-return and excess NAV per strategy (the equity curve, CLAUDE.md 7.13)."""
    cols = {}
    for name, res in results.items():
        cols[f"nav_total_{name}"] = (1.0 + res.daily["total"]).cumprod()
        cols[f"nav_excess_{name}"] = (1.0 + res.daily["excess"]).cumprod()
    return pd.DataFrame(cols).rename_axis("date")
