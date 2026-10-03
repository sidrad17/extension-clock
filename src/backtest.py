"""Window positions and daily NAV for the cash strategies (CLAUDE.md 7.10). Futures arrive in Phase 4.

Cash version (CLAUDE.md 7.10):
* Each month m holds one window: long the tenor's constant-maturity par bond from the close of the entry day to
  the close of the exit day (headline: E = T - 4 to T, 10-year).
* Notional = DV01 / (D x 1e-4), D = modified duration at entry of the par bond (coupon = yield at entry, maturity
  = tenor, src/bonds.py); DV01 from src/risk.py (forecast-sized: x w_m; calendar-only: x 1). The notional is held
  constant through the window: daily P&L = notional x daily excess return of the par bond (src/returns.py), so the
  position is financed at the T-bill rate.
* Cost = cost_bp x cost_mult x DV01 per round trip (CASH_COST_BP = 0.5 bp of yield), half charged on the entry day
  and half on the exit day.
* Capital is fixed at CAPITAL (sizing never compounds). Daily excess return = P&L / CAPITAL; total return adds the
  T-bill on capital (DTB3, as in src/returns.py). NAV series compound these daily returns.
Strategies: calendar_only and forecast_sized (headline). curve_allocated (the H2 trade) arrives with H2 (Phase 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import ENTRY_OFFSET, EXIT_OFFSET, PLACEBO_BDAYS
from src.bonds import mod_duration
from src.risk import DrawdownRule, RiskConfig, sigma_bp, size_trade
from src.signals import window_has


# --------------------------------------------------------------------------------------------- windows

def month_end_windows(cal, months, entry_offset: int = ENTRY_OFFSET, exit_offset: int = EXIT_OFFSET) -> pd.DataFrame:
    """entry = T - entry_offset, exit = T + exit_offset, indexed by month (CLAUDE.md 7.10; headline T-4 -> T)."""
    rows = {}
    for m in months:
        T = cal.month_end(m)
        rows[m] = {"T": T, "entry": cal.offset(T, -entry_offset), "exit": cal.offset(T, exit_offset)}
    return pd.DataFrame.from_dict(rows, orient="index")


def placebo_windows(cal, months, bdays: tuple[int, int] = PLACEBO_BDAYS) -> pd.DataFrame:
    """Placebo for signal month m: enter at the close of business day bdays[0] - 1 of month m + 1, exit at the close
    of business day bdays[1] (CLAUDE.md 7.11: business days 4-7, clear of the T+1..T+3 reversal and the 15th).

    Indexed by the SIGNAL month m: the latest forced-demand signal known at the placebo entry is z_m (rule 3), so
    the placebo tests pair window m + 1 with z_m (our choice, fixed before any return was computed).
    """
    rows = {}
    for m in months:
        nxt = m + 1
        rows[m] = {"window_month": nxt, "entry": cal.nth_bday(nxt, bdays[0] - 1), "exit": cal.nth_bday(nxt, bdays[1])}
    return pd.DataFrame.from_dict(rows, orient="index")


def hold_days(cal, entry: pd.Timestamp, exit_: pd.Timestamp) -> int:
    """Bond business days held: return days in (entry, exit]."""
    return int(cal.days.get_loc(exit_) - cal.days.get_loc(entry))


def window_return(daily: pd.Series, entry: pd.Timestamp, exit_: pd.Timestamp) -> float:
    """Compounded return over the days in (entry, exit]; NaN if any day is missing."""
    r = daily.loc[entry:exit_].iloc[1:]
    if len(r) == 0 or r.isna().any():
        return np.nan
    return float(np.prod(1.0 + r.to_numpy()) - 1.0)


def window_returns(daily: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """Window excess returns (decimal) for each row of `windows` (entry, exit columns)."""
    return pd.Series([window_return(daily, e, x) for e, x in zip(windows["entry"], windows["exit"])],
                     index=windows.index, dtype=float)


def window_yield_change_bp(y: pd.Series, windows: pd.DataFrame) -> pd.Series:
    """y(exit) - y(entry) in bp."""
    return pd.Series((y.reindex(windows["exit"]).to_numpy() - y.reindex(windows["entry"]).to_numpy()) * 100.0,
                     index=windows.index, dtype=float)


# --------------------------------------------------------------------------------------------- strategy

@dataclass
class StrategyResult:
    name: str
    trades: pd.DataFrame      # one row per window: sizing, P&L in $
    daily: pd.DataFrame       # per bond day: pnl ($), excess, total (fractions of capital), notional ($)
    cfg: RiskConfig


def run_strategy(name: str, windows: pd.DataFrame, w: pd.Series, daily_excess: pd.Series, rf: pd.Series,
                 y_tenor: pd.Series, y10: pd.Series, tenor_years: float, fomc_scheduled: pd.DatetimeIndex,
                 cal, days: pd.DatetimeIndex, cfg: RiskConfig | None = None) -> StrategyResult:
    """Run one cash strategy over `windows` (indexed by month, in time order, with entry and exit columns).

    w: size multiplier per month (forecast-sized: w_m; calendar-only: 1). daily_excess / rf: the tenor's daily par
    bond excess return and the T-bill return per bond day (fractions). y_tenor / y10: yields in percent (y10 sizes
    every trade, y_tenor gives the duration). days: the bond days of the NAV (sample period).
    """
    cfg = cfg or RiskConfig()
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()
    pnl = pd.Series(0.0, index=days)
    notional_d = pd.Series(0.0, index=days)
    dd = DrawdownRule(cfg)
    rows = []
    prev_exit = pd.Timestamp.min
    for m, win in windows.iterrows():
        entry, exit_ = win["entry"], win["exit"]
        if entry < days[0] or exit_ > days[-1]:
            raise ValueError(f"{m}: window {entry.date()}..{exit_.date()} outside the NAV days")
        if entry <= prev_exit:
            raise ValueError(f"{m}: window overlaps the previous one")
        prev_exit = exit_
        sig = sigma_bp(y10, entry, cfg.vol_lookback)
        fomc_in = window_has(fomc, entry, exit_)
        dd_mult = dd.decide()
        y_e = float(y_tenor.loc[entry])
        dur = float(mod_duration(y_e, y_e, tenor_years))
        hd = hold_days(cal, entry, exit_)
        size = size_trade(sig, float(w.loc[m]), fomc_in, dd_mult, dur, cfg, hold_days=hd)
        r = daily_excess.loc[entry:exit_].iloc[1:]
        if r.isna().any():
            raise ValueError(f"{m}: missing returns in the window")
        gross = size["notional"] * r
        cost = cfg.cost_bp * cfg.cost_mult * size["dv01"]
        pnl.loc[gross.index] += gross.to_numpy()
        pnl.loc[entry] -= cost / 2.0
        pnl.loc[exit_] -= cost / 2.0
        notional_d.loc[gross.index] = size["notional"]
        dd.update(pnl.loc[entry:exit_].to_numpy() / cfg.capital)
        rows.append({"month": m, "entry": entry, "exit": exit_, "hold_days": hd, "sigma_bp": sig,
                     "w": float(w.loc[m]), "fomc": int(fomc_in), "fomc_mult": size["fomc_mult"],
                     "dd_mult": dd_mult, "duration": dur, "dv01": size["dv01"], "notional": size["notional"],
                     "capped": int(size["capped"]), "gross_pnl": float(gross.sum()), "cost": cost,
                     "net_pnl": float(gross.sum()) - cost})
    trades = pd.DataFrame(rows).set_index("month")
    excess = pnl / cfg.capital
    daily = pd.DataFrame({"pnl": pnl, "excess": excess, "total": rf.reindex(days).to_numpy() + excess,
                          "notional": notional_d})
    if daily["total"].isna().any():
        raise ValueError(f"{name}: missing T-bill returns in the NAV period")
    return StrategyResult(name=name, trades=trades, daily=daily, cfg=cfg)
