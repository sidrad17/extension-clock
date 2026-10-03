"""Position sizing, notional cap, FOMC haircut and drawdown rule, each a toggle (CLAUDE.md 7.8).

Sizing (CLAUDE.md 7.8, settings.py, pre-registered):
* sigma_bp = std (ddof = 1) of the VOL_LOOKBACK (60) daily 10-year yield changes in bp ending the bond day before
  entry (E - 1 = T - 5 for the pre-registered E = T - 4). The 10-year yield sizes every trade, whatever the tenor.
* DV01_target = RISK_PER_TRADE x CAPITAL / (sigma_bp x sqrt(hold_days)) in $ per bp, so a 1-sigma move over the
  holding period loses RISK_PER_TRADE (1%) of capital; hold_days = 4 for T - 4 -> T and for the placebo window.
  Then x w_m (forecast-sized) or x 1 (calendar-only).
* Rule 2 (FOMC_HALF): x 0.5 if a SCHEDULED FOMC decision date is in (entry, exit]. Scheduled dates are public about
  a year ahead, so the rule is known at entry; unscheduled actions are never used (CLAUDE.md 7.1, 7.8).
* Rule 5 (DD_RULE): x 0.5 while the strategy is in drawdown mode. Our reading of "drawdown > 2x expected yearly vol
  -> half size until new high", fixed before any return was computed: drawdown is measured on the strategy's own
  excess-return NAV (its P&L, without the T-bill on idle capital) from past days only, i.e. through the previous
  window's exit (the strategy is flat between windows); expected yearly vol is the vol the sizing rule targets,
  RISK_PER_TRADE x sqrt(12) (one 1%-of-capital 1-sigma window per month), so the trigger is a drawdown above
  DD_MULT x 1% x sqrt(12) = 6.93% of capital. Once triggered, size stays halved until the NAV makes a new high
  (exceeds its peak at the trigger); the trigger is then checked again.
* Notional cap: notional <= NOTIONAL_CAP x CAPITAL (3x), applied after the multipliers above.
Each rule has an on/off switch in RiskConfig; the pre-registered setting is all on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import (CAPITAL, CASH_COST_BP, DD_MULT, DD_RULE, FOMC_HALF, NOTIONAL_CAP, RISK_PER_TRADE,
                             VOL_LOOKBACK)

WINDOWS_PER_YEAR = 12


@dataclass(frozen=True)
class RiskConfig:
    capital: float = CAPITAL
    risk_per_trade: float = RISK_PER_TRADE
    vol_lookback: int = VOL_LOOKBACK
    fomc_half: bool = FOMC_HALF
    dd_rule: bool = DD_RULE
    dd_mult: float = DD_MULT
    notional_cap_on: bool = True
    notional_cap: float = NOTIONAL_CAP
    cost_bp: float = CASH_COST_BP        # yield bp per round trip, x DV01 (cash version, CLAUDE.md 7.10)
    cost_mult: float = 1.0               # COST_STRESS = 2.0 in the sensitivity grid

    @property
    def expected_yearly_vol(self) -> float:
        """Vol the sizing rule targets, as a fraction of capital: 1% per window x sqrt(12 windows)."""
        return self.risk_per_trade * np.sqrt(WINDOWS_PER_YEAR)

    @property
    def dd_threshold(self) -> float:
        return self.dd_mult * self.expected_yearly_vol


def sigma_bp(y10: pd.Series, entry: pd.Timestamp, lookback: int = VOL_LOOKBACK) -> float:
    """Std of the `lookback` daily 10-year yield changes (bp) ending the bond day before `entry`.

    y10: 10-year yield in percent on bond business days (the calendar index).
    """
    pos = y10.index.get_loc(entry)
    if pos - 1 - lookback < 0:
        raise ValueError(f"not enough yield history before {entry.date()} for sigma")
    window = y10.iloc[pos - 1 - lookback: pos].to_numpy(float)      # lookback + 1 levels -> lookback changes
    d = np.diff(window) * 100.0
    if not np.isfinite(d).all():
        raise ValueError(f"missing 10-year yields in the sigma window before {entry.date()}")
    return float(d.std(ddof=1))


def base_dv01(sigma: float, cfg: RiskConfig, hold_days: int = 4) -> float:
    """$ per bp so that a 1-sigma move over `hold_days` loses risk_per_trade x capital."""
    return cfg.risk_per_trade * cfg.capital / (sigma * np.sqrt(hold_days))


class DrawdownRule:
    """Rule 5 state machine on the strategy's excess-return NAV (module docstring)."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.nav = 1.0
        self.peak = 1.0
        self.half = False
        self.peak_at_trigger = np.nan

    def update(self, daily_excess: np.ndarray) -> None:
        """Feed realised daily excess returns (fractions of capital) in time order."""
        for r in np.asarray(daily_excess, float):
            self.nav *= 1.0 + r
            self.peak = max(self.peak, self.nav)

    @property
    def drawdown(self) -> float:
        return 1.0 - self.nav / self.peak

    def decide(self) -> float:
        """Size multiplier for the next trade, from past NAV only (1.0 or 0.5)."""
        if not self.cfg.dd_rule:
            return 1.0
        if self.half and self.peak > self.peak_at_trigger:
            self.half = False
        if not self.half and self.drawdown > self.cfg.dd_threshold:
            self.half = True
            self.peak_at_trigger = self.peak
        return 0.5 if self.half else 1.0


def size_trade(sigma: float, w: float, fomc_in_window: bool, dd_mult: float, duration: float, cfg: RiskConfig,
               hold_days: int = 4) -> dict:
    """DV01 ($/bp) and notional ($) of one window trade in a par bond of modified duration `duration` (years)."""
    fomc_mult = 0.5 if (cfg.fomc_half and fomc_in_window) else 1.0
    dv01 = base_dv01(sigma, cfg, hold_days) * w * fomc_mult * dd_mult
    notional = dv01 / (duration * 1e-4)
    capped = False
    if cfg.notional_cap_on and notional > cfg.notional_cap * cfg.capital:
        notional = cfg.notional_cap * cfg.capital
        dv01 = notional * duration * 1e-4
        capped = True
    return {"dv01": dv01, "notional": notional, "capped": capped, "fomc_mult": fomc_mult, "dd_mult": dd_mult}
