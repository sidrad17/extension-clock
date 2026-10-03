"""Required performance metrics, betas, crowding monitor, tails and period tables (CLAUDE.md 7.13).

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

Phase 5 (our choices, fixed before any Phase 5 result):
* Betas: daily strategy excess return on the buy-and-hold 10-year par bond's daily excess return and the equity
  market's daily excess return, OLS with Newey-West (HAC_LAGS) errors over every NAV day. Equity excess on bond day
  t = the Ken French market total return compounded over the equity days in (t-1, t] (so bond holidays and equity
  holidays line up) minus the same T-bill accrual the strategy's excess return uses. Alpha in % per year (x 252).
* Crowding monitor (descriptive): per month, C(k) = cumulative 10-year excess return from the close of T-10 to the
  close of T+k. Share of the T-10..T gain earned before entry = mean C(-4) / mean C(0) over the months of a group
  (a ratio of means: per-month ratios explode when C(0) is near 0). Reported by year and by z tercile, with both
  means, so a group whose T-10..T gain is near zero or negative can be read correctly.
* Tails: the 5 worst windows by net P&L, with the 10-year yield change over the window, that change in sigmas
  (dy / (sigma_bp x sqrt(hold days))), and flags: scheduled and unscheduled FOMC decisions in (E, T] (unscheduled
  ones are descriptive only, CLAUDE.md 7.1), refunding month, quarter-end, drawdown rule on.
* Period tables (descriptive): metrics of a slice of a full-sample run's daily returns (no rerun), and a table by
  calendar year: excess return (compounded), volatility, Sharpe, max drawdown within the year (excess NAV starting
  the year at 1), max drawdown from the all-time peak reached during the year, worst calendar month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import StrategyResult
from src.stats import DAYS_PER_YEAR, nw_regression, sharpe


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


# ------------------------------------------------------------------------------------------------ Phase 5

def equity_excess_on_bond_days(equity_pct: pd.Series, bond_days: pd.DatetimeIndex, rf: pd.Series) -> pd.Series:
    """Equity market excess return per bond day (module docstring, betas). rf: T-bill accrual per bond day."""
    eq = equity_pct.dropna() / 100.0
    pos = bond_days.searchsorted(eq.index, side="left")           # bond day closing each equity day's interval
    inside = pos < len(bond_days)
    grp = pd.Series(np.log1p(eq.to_numpy()[inside]), index=bond_days[pos[inside]])
    total = np.expm1(grp.groupby(level=0).sum()).reindex(bond_days, fill_value=0.0)   # no equity day: 0
    total[(bond_days <= eq.index[0]) | (bond_days > eq.index[-1])] = np.nan              # outside the equity data
    total.iloc[0] = np.nan                                         # interval before the first bond day is unknown
    return (total - rf.reindex(bond_days)).rename("equity_excess")


def betas(daily_excess: pd.Series, bond_excess: pd.Series, equity_excess: pd.Series) -> dict:
    idx = daily_excess.index
    df = pd.DataFrame({"s": daily_excess.to_numpy(float), "bond": bond_excess.reindex(idx).to_numpy(float),
                       "equity": equity_excess.reindex(idx).to_numpy(float)}, index=idx).dropna()
    reg = nw_regression(df["s"], {"bond": df["bond"], "equity": df["equity"]})
    p = reg["params"]
    fitted = p["const"]["b"] + p["bond"]["b"] * df["bond"] + p["equity"]["b"] * df["equity"]
    r2 = 1.0 - float(((df["s"] - fitted) ** 2).sum() / ((df["s"] - df["s"].mean()) ** 2).sum())
    return {"alpha_pct_per_year": p["const"]["b"] * DAYS_PER_YEAR * 100.0, "alpha_t": p["const"]["t"],
            "beta_bond_10y": p["bond"]["b"], "beta_bond_t": p["bond"]["t"], "beta_bond_ci": p["bond"]["ci"],
            "beta_equity": p["equity"]["b"], "beta_equity_t": p["equity"]["t"], "beta_equity_ci": p["equity"]["ci"],
            "r2": r2, "n_days": reg["n"],
            "corr_bond": float(df["s"].corr(df["bond"])), "corr_equity": float(df["s"].corr(df["equity"]))}


def crowding(paths: pd.DataFrame, terciles: pd.Series, entry_k: int = -4) -> dict:
    """paths: months x k (cumulative %, k = -10..0, from src/tests_h.event_paths). Module docstring."""
    pre, full = paths[entry_k], paths[0]

    def block(sel) -> dict:
        a, b = float(pre[sel].mean()), float(full[sel].mean())
        return {"n": int(sel.sum()), "mean_T10_to_E_pct": a, "mean_T10_to_T_pct": b,
                "share_before_entry": a / b if b != 0 else None}

    years = paths.index.year
    out = {"all": block(pd.Series(True, index=paths.index)),
           "by_year": {str(y): block(pd.Series(years == y, index=paths.index)) for y in sorted(set(years))}}
    lab = terciles.reindex(paths.index)
    out["by_tercile"] = {t: block(lab == t) for t in ("low", "mid", "high")}
    out["note"] = ("share = mean C(-4) / mean C(0), C(k) = cumulative 10-year excess return from the close of T-10; "
                   "descriptive (crowding monitor, CLAUDE.md 7.13)")
    return out


def tails(res: StrategyResult, y10: pd.Series, sig: pd.DataFrame, fomc_all: pd.DatetimeIndex,
          fomc_scheduled: pd.DatetimeIndex, n: int = 5) -> list[dict]:
    """The n worst windows (module docstring). sig: the monthly signal table (z, w, ref, quarter_end)."""
    t = res.trades.sort_values("net_pnl").head(n)
    out = []
    for m, tr in t.iterrows():
        e, x = tr["entry"], tr["exit"]
        dy = float((y10.loc[x] - y10.loc[e]) * 100.0)
        move = dy / (tr["sigma_bp"] * np.sqrt(tr["hold_days"]))
        sched = [str(d.date()) for d in fomc_scheduled if e < d <= x]
        unsched = [str(d.date()) for d in fomc_all if e < d <= x and d not in set(fomc_scheduled)]
        causes = [f"10-year yield {dy:+.1f}bp ({move:+.1f} sd of a {int(tr['hold_days'])}-day move)"]
        if sched:
            causes.append("scheduled FOMC decision in window")
        if unsched:
            causes.append("unscheduled FOMC action in window")
        if tr["dd_mult"] < 1:
            causes.append("drawdown rule on (half size)")
        out.append({"month": str(m), "entry": str(e.date()), "exit": str(x.date()),
                    "net_pnl_pct": float(tr["net_pnl"] / res.cfg.capital * 100.0), "dy10_bp": dy,
                    "move_in_sd": float(move), "sigma_bp": float(tr["sigma_bp"]), "w": float(tr["w"]),
                    "z": float(sig.at[m, "z"]) if m in sig.index else None,
                    "fomc_scheduled": sched, "fomc_unscheduled": unsched, "refunding_month": int(sig.at[m, "ref"]),
                    "quarter_end": int(sig.at[m, "quarter_end"]), "dd_mult": float(tr["dd_mult"]),
                    "causes": "; ".join(causes)})
    return out


def slice_metrics(daily: pd.DataFrame, start, end) -> dict:
    """Return, volatility, Sharpe and drawdown of daily['excess'] (and 'total') between start and end (inclusive);
    a slice of a full-sample run, not a rerun (module docstring)."""
    d = daily.loc[pd.Timestamp(start):pd.Timestamp(end)]
    years = period_years(d.index)
    nav_x = (1.0 + d["excess"]).cumprod()
    out = {"period": [d.index[0].date().isoformat(), d.index[-1].date().isoformat()], "years": years,
           "ann_excess_return_pct": (float(nav_x.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0,
           "vol_pct": float(d["excess"].std(ddof=1) * np.sqrt(DAYS_PER_YEAR) * 100.0),
           "sharpe": sharpe(d["excess"]), "max_drawdown_excess_pct": max_drawdown(nav_x) * 100.0}
    if "total" in d:
        nav_t = (1.0 + d["total"]).cumprod()
        out["ann_return_pct"] = (float(nav_t.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0
        out["max_drawdown_pct"] = max_drawdown(nav_t) * 100.0
    return out


def yearly_table(daily_excess: pd.Series) -> pd.DataFrame:
    """Drawdown table by calendar year (module docstring); percent units."""
    nav = (1.0 + daily_excess).cumprod()
    peak_all = np.maximum.accumulate(np.concatenate([[1.0], nav.to_numpy()]))[1:]
    dd_all = pd.Series(1.0 - nav.to_numpy() / peak_all, index=nav.index)
    rows = {}
    for y, r in daily_excess.groupby(daily_excess.index.year):
        monthly = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
        rows[int(y)] = {"excess_return_pct": float(np.prod(1.0 + r.to_numpy()) - 1.0) * 100.0,
                        "vol_pct": float(r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR) * 100.0),
                        "sharpe": sharpe(r), "max_drawdown_in_year_pct": max_drawdown((1.0 + r).cumprod()) * 100.0,
                        "max_drawdown_from_peak_pct": float(dd_all.loc[r.index].max()) * 100.0,
                        "worst_month": str(monthly.idxmin()), "worst_month_pct": float(monthly.min()) * 100.0,
                        "n_days": int(len(r))}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")
