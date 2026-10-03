"""Newey-West OLS, bootstrap confidence intervals and the paired Sharpe-difference test (CLAUDE.md 7.11).

Conventions (our choice, fixed before any return was computed):
* Newey-West: Bartlett kernel with HAC_LAGS = 3 lags, no small-sample correction, normal-based 95% intervals and
  t = b / se. This is statsmodels' default for `OLS.fit(cov_type="HAC", cov_kwds={"maxlags": 3})`, which
  tests/test_stats.py checks against.
* Bootstraps are seeded with SEED = 7 (numpy default_rng) and use BOOT_N = 10,000 draws; intervals are the 2.5% and
  97.5% percentiles of the bootstrap distribution.
* Sharpe ratio = mean / std (ddof = 1) of DAILY excess returns x sqrt(252).
* Deflated Sharpe (Bailey & Lopez de Prado 2014, CLAUDE.md 7.11): N = rows in runs/trials.csv; V = sample variance
  (ddof = 1) of every Sharpe ratio in the log (both columns, sharpe_fc and sharpe_cal, every row that has one),
  converted to daily units (/ sqrt(252)); skewness and kurtosis (non-excess, plain moment estimators) from the
  strategy's own daily excess returns, T = its number of days. SR0 = sqrt(V) x ((1 - g) PHI^-1(1 - 1/N)
  + g PHI^-1(1 - 1/(N e))), g = Euler-Mascheroni; DSR = PHI((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR
  + (kurt - 1)/4 SR^2)), SR the daily Sharpe. Our choices, fixed before any Phase 5 result: N counts every row (reruns,
  placebo and grid cells included), which makes SR0 larger, not smaller. Phase 4 (team, CLAUDE.md section 15, before
  any futures result): the headline uses N = distinct variants (distinct config_hash) and V over their Sharpes
  (src/trial_log.py::trial_counts); the Phase 5 version (N = every row, V over every row) is reported beside it.
  This function takes N and the trial Sharpes as given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from config.settings import BOOT_N, HAC_LAGS, SEED

Z95 = 1.959963984540054   # standard normal 97.5% quantile
DAYS_PER_YEAR = 252
EULER_GAMMA = 0.5772156649015329


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


# --------------------------------------------------------------------------------------------- Deflated Sharpe

def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """SR0: the expected maximum Sharpe of n_trials independent trials with zero true Sharpe and variance var_sr
    across trials (same units as sqrt(var_sr))."""
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    return float(np.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * sps.norm.ppf(1.0 - 1.0 / n)
                                    + EULER_GAMMA * sps.norm.ppf(1.0 - 1.0 / (n * np.e))))


def probabilistic_sharpe(sr: float, sr0: float, n_obs: int, skew: float, kurt: float) -> tuple[float, float]:
    """(PSR, z): probability that the true per-period Sharpe exceeds sr0, given the observed per-period sr over
    n_obs returns with the given skewness and (non-excess) kurtosis."""
    z = (sr - sr0) * np.sqrt(n_obs - 1.0) / np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    return float(sps.norm.cdf(z)), float(z)


def deflated_sharpe(daily_excess, trial_sharpes_ann, n_trials: int, periods: int = DAYS_PER_YEAR) -> dict:
    """Deflated Sharpe of one strategy (module docstring). trial_sharpes_ann: annualized Sharpe ratios from the
    trial log; n_trials: rows in the log."""
    r = np.asarray(daily_excess, float)
    r = r[np.isfinite(r)]
    t = np.asarray(trial_sharpes_ann, float)
    t = t[np.isfinite(t)]
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    var_ann = float(t.var(ddof=1))
    sr0 = expected_max_sharpe(var_ann / periods, n_trials)
    dsr, z = probabilistic_sharpe(sr, sr0, len(r), skew, kurt)
    psr0, _ = probabilistic_sharpe(sr, 0.0, len(r), skew, kurt)
    return {"sharpe_ann": sr * np.sqrt(periods), "n_days": int(len(r)), "skew": skew, "kurtosis": kurt,
            "n_trials": int(n_trials), "n_trial_sharpes": int(len(t)), "var_trial_sharpe_ann": var_ann,
            "sr0_ann": sr0 * np.sqrt(periods), "dsr": dsr, "z": z, "psr_vs_zero": psr0}
