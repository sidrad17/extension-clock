"""Tests for src/stats.py: Newey-West vs statsmodels, deterministic bootstraps, Sharpe."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.stats import (bootstrap_mean_ci, newey_west_ols, nw_mean, nw_regression, paired_sharpe_diff_bootstrap,
                       sharpe)


def _ar_data(n=300, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal()
    return x, 0.2 + 0.4 * x + e


def test_newey_west_matches_statsmodels():
    x, y = _ar_data()
    X = sm.add_constant(x)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ours = newey_west_ols(y, X, lags=3, names=["const", "x"])
    assert ours["params"]["x"]["b"] == pytest.approx(ref.params[1], rel=1e-10)
    assert ours["params"]["x"]["se"] == pytest.approx(ref.bse[1], rel=1e-10)
    assert ours["params"]["const"]["se"] == pytest.approx(ref.bse[0], rel=1e-10)
    lo, hi = ref.conf_int()[1]
    assert ours["params"]["x"]["ci"] == pytest.approx([lo, hi], rel=1e-9)


def test_nw_regression_drops_nan_and_aligns():
    x, y = _ar_data(100)
    ys, xs = pd.Series(y), pd.Series(x)
    ys.iloc[5] = np.nan
    r = nw_regression(ys, {"x": xs})
    assert r["n"] == 99
    m = nw_mean(pd.Series(y))
    assert m["b"] == pytest.approx(y.mean())


def test_bootstraps_are_deterministic():
    x = np.random.default_rng(0).normal(size=200)
    assert bootstrap_mean_ci(x) == bootstrap_mean_ci(x)
    days = pd.bdate_range("2000-01-03", periods=600)
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.0004, 0.01, 600), index=days)
    b = pd.Series(rng.normal(0.0002, 0.01, 600), index=days)
    r1 = paired_sharpe_diff_bootstrap(a, b, n_boot=2000)
    r2 = paired_sharpe_diff_bootstrap(a, b, n_boot=2000)
    assert r1 == r2
    assert r1["diff"] == pytest.approx(sharpe(a) - sharpe(b))
    assert r1["ci"][0] < r1["diff"] < r1["ci"][1]


def test_paired_bootstrap_identical_series_has_zero_diff():
    days = pd.bdate_range("2000-01-03", periods=300)
    a = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300), index=days)
    r = paired_sharpe_diff_bootstrap(a, a.copy(), n_boot=500)
    assert r["diff"] == 0 and r["ci"] == [0.0, 0.0]


def test_sharpe_annualisation():
    r = np.array([0.01, -0.005, 0.002, 0.004])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
