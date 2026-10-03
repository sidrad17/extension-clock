"""Tests for src/returns.py: carry, duration response, weekend accrual, gate guard."""
import numpy as np
import pandas as pd
import pytest

import src.trial_log as tl
from src.bonds import convexity, mod_duration
from src.returns import par_bond_returns, tenor_returns


def _flat(level, days):
    return pd.Series(level, index=days, dtype=float)


def test_constant_yield_earns_carry_once():
    days = pd.bdate_range("2024-01-01", "2024-01-31")
    r = par_bond_returns(_flat(5.0, days), 10.0, _flat(2.0, days))
    dt = pd.Series(days, index=days).diff().dt.days
    assert np.isnan(r["ret"].iloc[0])
    # At an unchanged yield the bond earns its yield once (not twice): clean-price change + coupon x dt equals the
    # full-price growth (1 + y/2)^(2 dt) - 1 (semiannual compounding, 1.2% below simple y x dt). The residual
    # (< 3e-7 over a weekend) is the CLAUDE.md 7.4 day counts: carry uses dt/365, maturity uses dt/365.25.
    expected = 1.025 ** (2 * dt / 365.25) - 1
    assert np.allclose(r["ret"].iloc[1:], expected.iloc[1:], rtol=0, atol=3e-7)
    assert np.allclose(r["excess"].iloc[1:], (r["ret"] - 0.02 * dt / 365).iloc[1:], atol=1e-15)
    mon = r.loc["2024-01-08", "ret"]
    tue = r.loc["2024-01-09", "ret"]
    assert mon == pytest.approx(3 * tue, rel=0.03)                    # Friday -> Monday accrues 3 days


def test_yield_rise_loses_duration():
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    r = par_bond_returns(pd.Series([5.0, 5.10], index=days), 10.0)["ret"].iloc[1]
    d, c, dy = mod_duration(5.0, 5.0, 10.0), convexity(5.0, 5.0, 10.0), 0.001
    assert r == pytest.approx(-d * dy + 0.5 * c * dy ** 2 + 0.05 / 365, abs=2e-5)


def test_returns_refuse_test_window_without_gate2(monkeypatch):
    monkeypatch.setattr(tl, "dev_mode", lambda: True)
    monkeypatch.setattr(tl, "_tags", lambda: [tl.GATE1_TAG])
    monkeypatch.setattr(tl, "_prereg_changed", lambda: False)
    monkeypatch.setattr(tl, "_head_tags", lambda: [])
    with pytest.raises(tl.GateError):
        tenor_returns(end="2025-01-31")
    with pytest.raises(tl.GateError):
        tenor_returns(end=None)
