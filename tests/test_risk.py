"""Tests for src/risk.py: sigma window, DV01 target, FOMC haircut, drawdown rule, notional cap."""
import numpy as np
import pandas as pd
import pytest

from src.risk import DrawdownRule, RiskConfig, base_dv01, sigma_bp, size_trade


def test_sigma_window_ends_day_before_entry():
    days = pd.bdate_range("2020-01-01", periods=100)
    y = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 0.05, 100)) + 2.0, index=days)
    entry = days[80]
    expected = (np.diff(y.iloc[19:80].to_numpy()) * 100).std(ddof=1)    # 60 changes ending day 79
    assert sigma_bp(y, entry, 60) == pytest.approx(expected)
    y2 = y.copy()
    y2.iloc[80:] += 5.0                          # entry day and later never enter sigma
    assert sigma_bp(y2, entry, 60) == pytest.approx(expected)


def test_dv01_target_formula():
    cfg = RiskConfig()
    # 1% of $10M / (6bp x sqrt(4)) = $8,333.33 per bp
    assert base_dv01(6.0, cfg, 4) == pytest.approx(0.01 * 10_000_000 / 12.0)


def test_fomc_and_drawdown_halve_and_cap_binds():
    cfg = RiskConfig()
    plain = size_trade(6.0, 1.0, False, 1.0, 8.0, cfg)
    halved = size_trade(6.0, 1.0, True, 0.5, 8.0, cfg)
    assert halved["dv01"] == pytest.approx(plain["dv01"] / 4)
    assert plain["notional"] == pytest.approx(plain["dv01"] / (8.0 * 1e-4))
    off = size_trade(6.0, 1.0, True, 1.0, 8.0, RiskConfig(fomc_half=False))
    assert off["dv01"] == pytest.approx(plain["dv01"])
    big = size_trade(1.0, 2.0, False, 1.0, 8.0, cfg)          # very low vol -> cap at 3x capital
    assert big["capped"] and big["notional"] == pytest.approx(3 * cfg.capital)
    assert big["dv01"] == pytest.approx(3 * cfg.capital * 8.0 * 1e-4)


def test_drawdown_rule_hysteresis():
    cfg = RiskConfig()
    thr = cfg.dd_threshold
    assert thr == pytest.approx(2 * 0.01 * np.sqrt(12))
    dd = DrawdownRule(cfg)
    assert dd.decide() == 1.0
    dd.update([0.02])                      # new peak 1.02
    dd.update([-(thr + 0.01)])             # drawdown above threshold
    assert dd.decide() == 0.5
    dd.update([0.03])                      # partial recovery, still below the peak
    assert dd.drawdown < thr and dd.decide() == 0.5     # stays half until a new high
    dd.update([0.10])                      # new high
    assert dd.decide() == 1.0
    assert DrawdownRule(RiskConfig(dd_rule=False)).decide() == 1.0
