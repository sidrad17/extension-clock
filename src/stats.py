"""Newey-West OLS, bootstrap confidence intervals and the paired Sharpe-difference test (CLAUDE.md 7.11).

Conventions (our choice, fixed before any return was computed):
* Newey-West: Bartlett kernel with HAC_LAGS = 3 lags, no small-sample correction, normal-based 95% intervals and
  t = b / se. This is statsmodels' default for `OLS.fit(cov_type="HAC", cov_kwds={"maxlags": 3})`, which
  tests/test_stats.py checks against.
* Bootstraps are seeded with SEED = 7 (numpy default_rng) and use BOOT_N = 10,000 draws; intervals are the 2.5% and
  97.5% percentiles of the bootstrap distribution.
* Sharpe ratio = mean / std (ddof = 1) of DAILY excess returns x sqrt(252).
The Deflated Sharpe (Bailey & Lopez de Prado) arrives with Phase 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BOOT_N, HAC_LAGS, SEED

Z95 = 1.959963984540054   # standard normal 97.5% quantile
DAYS_PER_YEAR = 252


def newey_west_ols(y, X, lags: int = HAC_LAGS, names: list[str] | None = None) -> dict:
    """OLS of y on X (X must already contain the constant) with Newey-West (Bartlett) standard errors.

    Rows with a NaN in y or X are dropped. Observations are assumed to be in time order (lags run over rows).
    Returns {"n", "params": {name: {"b", "se", "t", "ci"}}}.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Newey-West OLS needs more rows than regressors (n={n}, k={k})")
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = xu[L:].T @ xu[:-L]
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    names = names or [f"x{i}" for i in range(k)]
    params = {nm: {"b": float(b[i]), "se": float(se[i]), "t": float(b[i] / se[i]),
                   "ci": [float(b[i] - Z95 * se[i]), float(b[i] + Z95 * se[i])]} for i, nm in enumerate(names)}
    return {"n": int(n), "params": params}


def nw_regression(y: pd.Series, regressors: dict[str, pd.Series], lags: int = HAC_LAGS) -> dict:
    """Convenience wrapper: y on a constant plus named regressors, aligned on y's index."""
    df = pd.concat([y.rename("_y")] + [s.rename(k) for k, s in regressors.items()], axis=1)
    X = np.column_stack([np.ones(len(df))] + [df[k].to_numpy(float) for k in regressors])
    return newey_west_ols(df["_y"].to_numpy(float), X, lags=lags, names=["const"] + list(regressors))


def nw_mean(x: pd.Series, lags: int = HAC_LAGS) -> dict:
    """Mean of x with a Newey-West interval (regression on a constant)."""
    r = newey_west_ols(np.asarray(x, float), np.ones((len(x), 1)), lags=lags, names=["mean"])
    return {"n": r["n"], **r["params"]["mean"]}


def bootstrap_mean_ci(x, n_boot: int = BOOT_N, seed: int = SEED, level: float = 0.95) -> dict:
    """Mean of x with an i.i.d. percentile bootstrap interval (seeded, deterministic)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return {"mean": float(x.mean()), "ci": [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))],
            "n": int(len(x))}


def sharpe(daily_excess) -> float:
    """Annualized Sharpe of daily excess returns: mean / std (ddof=1) x sqrt(252)."""
    r = np.asarray(daily_excess, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else float("nan")


def _sharpe_from_sums(s: np.ndarray, q: np.ndarray, n: np.ndarray) -> np.ndarray:
    mean = s / n
    var = (q - n * mean ** 2) / (n - 1)
    return mean / np.sqrt(var) * np.sqrt(DAYS_PER_YEAR)


def paired_sharpe_diff_bootstrap(daily_a: pd.Series, daily_b: pd.Series, n_boot: int = BOOT_N,
                                 seed: int = SEED, level: float = 0.95) -> dict:
    """Sharpe(a) - Sharpe(b) with a paired bootstrap by calendar month (CLAUDE.md 7.11, H4).

    Both series are daily excess returns on the same dates. Each draw resamples calendar months with replacement
    (the same months for both strategies, so the pairing is kept) and recomputes both daily Sharpe ratios from the
    resampled months' days. Returns the point estimates, the interval and p_le_0 = share of draws with diff <= 0.
    """
    df = pd.concat([daily_a.rename("a"), daily_b.rename("b")], axis=1)
    if df.isna().any().any():
        raise ValueError("paired Sharpe bootstrap: NaN in daily returns")
    month = df.index.to_period("M")
    g = df.groupby(month)
    n = g.size().to_numpy(float)
    sa, sb = g["a"].sum().to_numpy(), g["b"].sum().to_numpy()
    qa, qb = (df["a"] ** 2).groupby(month).sum().to_numpy(), (df["b"] ** 2).groupby(month).sum().to_numpy()
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(n), np.full(len(n), 1.0 / len(n)), size=n_boot).astype(float)
    shp_a = _sharpe_from_sums(counts @ sa, counts @ qa, counts @ n)
    shp_b = _sharpe_from_sums(counts @ sb, counts @ qb, counts @ n)
    diff = shp_a - shp_b
    a = (1.0 - level) / 2.0
    point_a, point_b = sharpe(df["a"]), sharpe(df["b"])
    return {"sharpe_a": point_a, "sharpe_b": point_b, "diff": point_a - point_b,
            "ci": [float(np.quantile(diff, a)), float(np.quantile(diff, 1 - a))],
            "p_le_0": float((diff <= 0).mean()), "n_months": int(len(n)), "n_boot": int(n_boot), "seed": int(seed)}
