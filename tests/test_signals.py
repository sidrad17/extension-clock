"""Tests for src/signals.py: past-only z-scores and w clipping."""
import numpy as np
import pandas as pd
import pytest

from src.signals import build_signals, past_zscore, surprise, weight, window_has


def _monthly(n=60, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("1990-01", periods=n, freq="M")
    T = [m.end_time.normalize() - pd.offsets.BDay(0) for m in months]
    ext = rng.normal(0.06, 0.03, n)
    c = np.abs(rng.normal(0.003, 0.002, n))
    dn = np.full(n, 5.0)
    return pd.DataFrame({"month": months.astype(str), "T": T, "E": [t - pd.offsets.BDay(4) for t in T],
                         "Ext": ext, "c_m": c, "D_next": dn, "FDD": ext + c * dn})


def test_z_uses_only_past_months():
    x = pd.Series(np.random.default_rng(1).normal(size=80))
    z = past_zscore(x, 36)
    x2 = x.copy()
    x2.iloc[60:] += 100.0                       # change the future
    z2 = past_zscore(x2, 36)
    pd.testing.assert_series_equal(z.iloc[:60], z2.iloc[:60])
    # value check: z at 50 uses rows 0..49 only
    assert z.iloc[50] == pytest.approx((x.iloc[50] - x.iloc[:50].mean()) / x.iloc[:50].std(ddof=1))


def test_z_needs_min_months():
    x = pd.Series(np.arange(40, dtype=float))
    z = past_zscore(x, 36)
    assert z.iloc[:36].isna().all() and z.iloc[36:].notna().all()


def test_w_clipping_and_default():
    z = pd.Series([np.nan, -3.0, -1.0, 0.0, 0.5, 1.0, 4.0])
    assert weight(z).tolist() == [1.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.0]


def test_surprise_needs_three_prior_years():
    idx = pd.period_range("1990-01", periods=48, freq="M")
    ext = pd.Series(np.arange(48, dtype=float), index=idx)
    s = surprise(ext)
    assert s.iloc[:36].isna().all()
    # month 36 (1993-01): mean of months 24, 12, 0 = 12 -> 36 - 12 = 24
    assert s.iloc[36] == pytest.approx(24.0)


def test_window_has_is_half_open():
    d = pd.DatetimeIndex(["2024-01-31"])
    assert window_has(d, pd.Timestamp("2024-01-25"), pd.Timestamp("2024-01-31"))
    assert not window_has(d, pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-06"))


def test_build_signals_components_and_dummies():
    m = _monthly()
    fomc = pd.DatetimeIndex([m["T"].iloc[40], m["E"].iloc[41]])      # in (E, T] for month 40; at E for 41 -> no
    s = build_signals(m, fomc)
    assert np.allclose(s["cash"] + s["Ext"], s["FDD"])
    assert s["fomc"].iloc[40] == 1 and s["fomc"].iloc[41] == 0
    feb = s.index.month == 2
    assert (s.loc[feb, "ref"] == 1).all() and s.loc[s.index.month == 3, "ref"].eq(0).all()
    assert (s.loc[s.index.month == 12, ["quarter_end", "year_end"]] == 1).all().all()
    assert s["z"].iloc[:36].isna().all() and (s["w"].iloc[:36] == 1.0).all()
    assert s["z"].iloc[36:].notna().all()
