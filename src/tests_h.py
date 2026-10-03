"""Hypothesis tests: H1 (+ components, addendum (a)-(b)), H4, placebo and luck test, event path (CLAUDE.md 7.11).

Units: R_m is the 10-year par-bond window excess return in PERCENT (T - 4 -> T); z_m in standard deviations (past
months only, src/signals.py); yield terms are -dy in bp over the same window. Newey-West with HAC_LAGS = 3 over
monthly observations in calendar order (src/stats.py). H2, H3, H5 and the addendum's (c) arrive in Phase 5.

Our choices, fixed before any return was computed:
* Terciles of z_m use the in-sample cut points of z_m (a description of the test sample, not a trading rule); each
  tercile mean has an i.i.d. bootstrap 95% interval (BOOT_N, SEED).
* Placebo (CLAUDE.md 7.11): business days 4-7 of month m + 1 paired with z_m, the latest signal known at the
  placebo entry (src/backtest.py::placebo_windows). Same tests as the month-end window: mean, slope on z,
  component-free H1 form, and both strategies (H4 form) on the placebo window with the same risk rules.
* Luck test (CLAUDE.md 7.11, "random windows"): per month, the candidate windows are 4 consecutive bond business days
  of RETURN (entry at the close of the day before the first one) that lie inside month m and avoid T(m-1)+1..+3
  (the first three bond days of month m), T(m)-4..T(m), and calendar days 14-16 (the 15th +/- 1). Each of
  N_RANDOM_PLACEBO = 1,000 seeded draws picks one candidate per month uniformly and averages the window excess
  returns over months; the month-end mean over the same months is set against that distribution.
* Event path (figure 1): cumulative excess return from the close of T - 10 through T + 5, months whose T + 5 is in
  the sample; 95% band = mean +/- 1.96 x standard error across months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BOOT_N, N_RANDOM_PLACEBO, SEED
from src.stats import bootstrap_mean_ci, nw_mean, nw_regression, paired_sharpe_diff_bootstrap

TERCILES = ["low", "mid", "high"]
EXCLUDE_DAYS_OF_MONTH = (14, 15, 16)


def _coef(reg: dict, name: str) -> dict:
    p = reg["params"][name]
    return {"b": p["b"], "se": p["se"], "t": p["t"], "ci": p["ci"]}


def h1(R_pct: pd.Series, dy_neg_bp: pd.Series, z: pd.Series) -> dict:
    """R_m = a + b z_m + e_m (Newey-West), the same in yield terms, plus tercile means with bootstrap CIs."""
    df = pd.concat([R_pct.rename("R"), dy_neg_bp.rename("Y"), z.rename("z")], axis=1).dropna()
    reg = nw_regression(df["R"], {"z": df["z"]})
    reg_y = nw_regression(df["Y"], {"z": df["z"]})
    terc = pd.qcut(df["z"], 3, labels=TERCILES)
    terciles, terciles_y = {}, {}
    for lab in TERCILES:
        sel = df[terc == lab]
        terciles[lab] = {**bootstrap_mean_ci(sel["R"]), "z_min": float(sel["z"].min()), "z_max": float(sel["z"].max())}
        terciles_y[lab] = bootstrap_mean_ci(sel["Y"])
    return {**_coef(reg, "z"), "n": reg["n"], "a": reg["params"]["const"]["b"],
            "yield_bp": {**_coef(reg_y, "z"), "n": reg_y["n"]},
            "mean_R": nw_mean(df["R"]), "mean_neg_dy_bp": nw_mean(df["Y"]),
            "terciles": terciles, "terciles_yield_bp": terciles_y,
            "high_minus_low": terciles["high"]["mean"] - terciles["low"]["mean"],
            "sample": [str(df.index[0]), str(df.index[-1])],
            "units": "R_m: 10-year window excess return, %; z: sd of FDD (past months only); yield_bp: -dy, bp"}


def tercile_labels(z: pd.Series) -> pd.Series:
    zz = z.dropna()
    return pd.qcut(zz, 3, labels=TERCILES).astype(str).reindex(z.index)


def h1_components(R_pct: pd.Series, dy_neg_bp: pd.Series, z_ext: pd.Series, z_cash: pd.Series) -> dict:
    """R_m on z(Ext) and z(cash term) together (CLAUDE.md 7.11); both predicted > 0."""
    df = pd.concat([R_pct.rename("R"), dy_neg_bp.rename("Y"), z_ext.rename("ext"), z_cash.rename("cash")],
                   axis=1).dropna()
    reg = nw_regression(df["R"], {"ext": df["ext"], "cash": df["cash"]})
    reg_y = nw_regression(df["Y"], {"ext": df["ext"], "cash": df["cash"]})
    return {"ext": _coef(reg, "ext"), "cash": _coef(reg, "cash"), "n": reg["n"],
            "yield_bp": {"ext": _coef(reg_y, "ext"), "cash": _coef(reg_y, "cash")},
            "corr_z_ext_z_cash": float(df["ext"].corr(df["cash"]))}


def h1_addendum(R_pct: pd.Series, dy_neg_bp: pd.Series, z: pd.Series, ref: pd.Series) -> dict:
    """PREREG_ADDENDUM.md (a) and (b). (c), the surprise extension, arrives in Phase 5."""
    df = pd.concat([R_pct.rename("R"), dy_neg_bp.rename("Y"), z.rename("z"), ref.rename("ref")], axis=1).dropna()
    out = {}
    for key, col in [("pct", "R"), ("yield_bp", "Y")]:
        a = nw_regression(df[col], {"z": df["z"], "ref": df["ref"]})
        b = nw_regression(df[col], {"ref": df["ref"], "z_ref": df["z"] * df["ref"],
                                    "z_oth": df["z"] * (1 - df["ref"])})
        out[key] = {"refunding_dummy": {"b": _coef(a, "z"), "c": _coef(a, "ref"), "n": a["n"]},
                    "within_refunding": {**_coef(b, "z_ref"), "n": int(df["ref"].sum())},
                    "within_other": {**_coef(b, "z_oth"), "n": int((1 - df["ref"]).sum())}}
    res = out["pct"]
    res["yield_bp"] = out["yield_bp"]
    res["surprise"] = {"status": "Phase 5 (PREREG_ADDENDUM.md (c))"}
    return res


def h4(daily_fc: pd.Series, daily_cal: pd.Series, n_boot: int = BOOT_N, seed: int = SEED) -> dict:
    """Net Sharpe forecast-sized minus calendar-only, paired bootstrap by month (CLAUDE.md 7.11)."""
    r = paired_sharpe_diff_bootstrap(daily_fc, daily_cal, n_boot=n_boot, seed=seed)
    return {"sharpe_fc": r["sharpe_a"], "sharpe_cal": r["sharpe_b"], "diff": r["diff"], "ci": r["ci"],
            "p_le_0": r["p_le_0"], "n_months": r["n_months"], "n_boot": r["n_boot"], "seed": r["seed"]}


# ------------------------------------------------------------------------------------------------ luck test

def luck_candidates(cal, months, daily: pd.Series, n_days: int = 4) -> dict:
    """Window excess returns (decimal) of every candidate random window per month (module docstring)."""
    out = {}
    for m in months:
        days = cal.month_days(m)
        T = cal.month_end(m)
        excluded = set(days[:3]) | {cal.offset(T, -k) for k in range(0, 5)}
        allowed = np.array([(d not in excluded) and (d.day not in EXCLUDE_DAYS_OF_MONTH) for d in days])
        rets = []
        for i in range(len(days) - n_days + 1):
            if allowed[i:i + n_days].all():
                entry = cal.offset(days[i], -1)
                r = daily.loc[entry:days[i + n_days - 1]].iloc[1:]
                if len(r) == n_days and not r.isna().any():
                    rets.append(float(np.prod(1.0 + r.to_numpy()) - 1.0))
        out[m] = np.array(rets)
    return out


def luck_test(candidates: dict, month_end_R: pd.Series, n_draws: int = N_RANDOM_PLACEBO, seed: int = SEED) -> dict:
    """Distribution of the mean random-window return vs the month-end mean (both in %, same months)."""
    months = [m for m in month_end_R.dropna().index if len(candidates.get(m, [])) > 0]
    rng = np.random.default_rng(seed)
    lens = np.array([len(candidates[m]) for m in months])
    u = rng.random((n_draws, len(months)))
    pick = np.floor(u * lens[None, :]).astype(int)
    vals = np.column_stack([candidates[m][pick[:, j]] for j, m in enumerate(months)])
    means = vals.mean(axis=1) * 100.0
    me = float(month_end_R.loc[months].mean())
    return {"label": "luck test", "n_draws": int(n_draws), "seed": int(seed), "n_months": len(months),
            "months_without_candidate": int(month_end_R.dropna().size - len(months)),
            "mean_candidates_per_month": float(lens.mean()),
            "month_end_mean_pct": me, "random_mean_pct": float(means.mean()), "random_sd_pct": float(means.std(ddof=1)),
            "random_p2_5_pct": float(np.quantile(means, 0.025)), "random_p50_pct": float(np.quantile(means, 0.5)),
            "random_p97_5_pct": float(np.quantile(means, 0.975)),
            "share_random_ge_month_end": float((means >= me).mean()), "draw_means_pct": means}


# ------------------------------------------------------------------------------------------------ event path

def event_paths(cal, months, daily: pd.Series, lo: int = -10, hi: int = 5) -> pd.DataFrame:
    """Cumulative excess return (%) from the close of T + lo to the close of T + k, k = lo..hi; one row per month.
    Months whose T + hi lies beyond the return series are dropped."""
    rows = {}
    last = daily.index[-1]
    for m in months:
        T = cal.month_end(m)
        try:
            start, end = cal.offset(T, lo), cal.offset(T, hi)
        except IndexError:
            continue
        if end > last:
            continue
        r = daily.loc[start:end].iloc[1:].to_numpy()
        if np.isnan(r).any():
            continue
        rows[m] = np.concatenate([[0.0], (np.cumprod(1.0 + r) - 1.0)]) * 100.0
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(range(lo, hi + 1)))


def event_path_summary(paths: pd.DataFrame, terciles: pd.Series) -> dict:
    """Mean path and 95% band by tercile (mean +/- 1.96 x s.e. across months)."""
    out = {}
    lab = terciles.reindex(paths.index)
    for t in TERCILES:
        p = paths[lab == t]
        mean = p.mean()
        se = p.std(ddof=1) / np.sqrt(len(p))
        out[t] = {"n": int(len(p)), "k": [int(k) for k in p.columns], "mean_pct": mean.tolist(),
                  "lo_pct": (mean - 1.959963984540054 * se).tolist(), "hi_pct": (mean + 1.959963984540054 * se).tolist()}
    return out
