"""Flow Clock: auction-window returns, H6a-c, H7, the supply leg and the netted book (PREREG_FLOWCLOCK.md).

Events, windows and the size signal come from src/auction_events.py. Returns are the constant-maturity par-bond
excess returns of src/returns.py, in %, compounded over the window's return days exactly as src/backtest.py does.

Tests (PREREG_FLOWCLOCK.md "Tests"):
* H6a: mean pre-window return < 0. H6b: mean post-window return > 0. H6c: LS_e = R_post - R_pre on zS_e (the size
  known at A-5), beta > 0. Standard errors clustered by the calendar week of A (Monday-Sunday). We use statsmodels'
  cluster-robust covariance with its default small-sample factor G/(G-1) x (N-1)/(N-K), and normal intervals, the
  same convention as the Newey-West intervals in src/stats.py (our choice, fixed before any auction-window return).
  Yield terms (bp) are reported beside them, for comparison with Lou, Yan & Zhang: dy_pre = y(A) - y(A-5),
  dy_post = y(A+5) - y(A), and the long-short in yield terms dy_pre - dy_post.
* H7: R_m (the H1 10-year T-4..T excess return, %) on zA_m, Newey-West with HAC_LAGS = 3; c > 0 and a > 0.

Strategy (PREREG_FLOWCLOCK.md "Strategy"). Every leg is a constant-notional position in its tenor's par bond, sized
at entry, as in src/backtest.py:
* Supply legs: short the tenor from the close of A-5 to the close of A, long it from the close of A to the close of
  A+5. DV01 = LEG_RISK x CAPITAL / (sigma_bp x sqrt(5)), sigma_bp = std (ddof = 1) of the tenor's 60 daily yield
  changes ending the bond day before entry (src/risk.py::sigma_bp on the tenor's yield). Calendar version: w = 1;
  size-weighted: x w_pre (pre leg) or w_post (post leg) from src/auction_events.py.
* Demand leg: the month-end calendar-only trade exactly as already tested (src/backtest.py): long the 10-year from
  T-4 to T, DV01 = RISK_PER_TRADE x CAPITAL / (sigma_10y x sqrt(4)).
* Book: legs are summed in DV01 by tenor each day. P&L per leg = notional x the tenor's daily excess return.
  Costs = CASH_COST_BP / 2 (half a round trip) x cost_mult x |change in net DV01 of each tenor| at each close, so
  offsetting trades in the same tenor on the same day net out. Turnover = sum |change in net notional| / CAPITAL /
  years. Capital is fixed at CAPITAL; daily excess return = (P&L - cost) / CAPITAL; total adds the T-bill.
* Risk rules, the existing rules extended to many legs (our reading, fixed before any auction-window return):
  - FOMC: a leg is halved if a SCHEDULED FOMC decision date is in (entry, exit] (src/risk.py rule 2, per leg).
  - Drawdown (src/risk.py rule 5): one state machine on the strategy's own excess NAV through the bond day before
    each entry; legs entered while it is on are halved. The trigger is DD_MULT x the yearly vol the sizing targets.
    For the month-end trade that is 1% x sqrt(12); here it is sqrt(12 x RISK_PER_TRADE^2 [demand leg, if any] +
    n x LEG_RISK^2), n = scheduled supply-leg entries in the 252 bond days ending the day before (a calendar count,
    so it is known in advance). With the demand leg alone this is exactly the month-end rule.
  - Notional cap: gross notional of open legs <= NOTIONAL_CAP x CAPITAL. Legs entering at a close (after that
    close's exits) are scaled by one common factor to fit. With one leg at a time this is the month-end rule.
  The book run with the demand leg alone reproduces src/backtest.py's calendar_only daily returns exactly
  (checked in run_all.py and tests/test_flowclock.py).
* Metrics: as src/metrics.py, with hit rate and worst unit per event (supply legs: pre + post net P&L, each leg
  charged CASH_COST_BP x cost_mult x its DV01) or per calendar month (the book).

Phase 5, descriptive (our choices, fixed before any Phase 5 result):
* Risk rules on and off: each rule switched off alone (FOMC half size, drawdown rule, notional cap), then all three,
  for the book, the calendar supply leg and the demand leg alone (calendar_only with the same switches), in-sample.
* By tenor: from the in-sample calendar supply leg's legs (no rerun): events, net P&L in % of capital per year
  (attributed costs: CASH_COST_BP x cost_mult x the leg's DV01), pre and post legs separately, hit rate per event.
* By decade (1993-1999, 2000-2009, 2010-2019, 2020-2024-09): H6 on the events whose A falls in the decade; Sharpe,
  return and drawdown of the full in-sample runs' daily returns sliced to the decade (src/metrics.py
  slice_metrics, not a rerun).
* Drawdown table by year: src/metrics.py yearly_table on the in-sample book, calendar supply leg and demand leg.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config.flowclock import LEG_HOLD_DAYS, LEG_RISK
from config.settings import HEADLINE_TENOR, RISK_PER_TRADE
from src.backtest import hold_days, window_return
from src.bonds import KNOT_YEARS, mod_duration
from src.metrics import max_drawdown, period_years
from src.risk import DrawdownRule, RiskConfig, sigma_bp
from src.signals import window_has
from src.stats import DAYS_PER_YEAR, Z95, nw_regression, paired_sharpe_diff_bootstrap, sharpe
from src.tests_h import TERCILES, _coef

TRAILING_BDAYS = 252


# ------------------------------------------------------------------------------------------------- returns

def event_returns(ev: pd.DataFrame, excess: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """Per event: R_pre, R_post, LS = R_post - R_pre (%), dy_pre_bp, dy_post_bp, LS_bp = dy_pre - dy_post."""
    rows = []
    for t, e, A, x in zip(ev["tenor"], ev["pre_entry"], ev["A"], ev["post_exit"]):
        if pd.isna(e) or pd.isna(x):
            rows.append([np.nan] * 4)
            continue
        y = yields[t]
        rows.append([window_return(excess[t], e, A) * 100.0, window_return(excess[t], A, x) * 100.0,
                     (y.loc[A] - y.loc[e]) * 100.0, (y.loc[x] - y.loc[A]) * 100.0])
    out = pd.DataFrame(rows, index=ev.index, columns=["R_pre", "R_post", "dy_pre_bp", "dy_post_bp"])
    out["LS"] = out["R_post"] - out["R_pre"]
    out["LS_bp"] = out["dy_pre_bp"] - out["dy_post_bp"]
    return out


# ------------------------------------------------------------------------------------------------- tests

def cluster_ols(y, X: np.ndarray, groups, names: list[str]) -> dict:
    """OLS with standard errors clustered by `groups` (statsmodels defaults, normal intervals)."""
    codes = pd.factorize(pd.Series(groups))[0]
    fit = sm.OLS(np.asarray(y, float), X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    params = {}
    for i, nm in enumerate(names):
        b, se = float(fit.params[i]), float(fit.bse[i])
        params[nm] = {"b": b, "se": se, "t": b / se, "ci": [b - Z95 * se, b + Z95 * se]}
    return {"n": int(len(codes)), "n_clusters": int(codes.max() + 1), "params": params}


def cluster_mean(x: pd.Series, groups: pd.Series) -> dict:
    """Mean with week-clustered errors; with fewer than 2 clusters (possible in Phase 5's by-decade cells) the mean
    is reported and its error is undefined (NaN)."""
    if pd.Series(groups).nunique() < 2:
        b = float(np.mean(x)) if len(x) else np.nan
        return {"n": int(len(x)), "n_clusters": int(pd.Series(groups).nunique()), "b": b, "se": np.nan, "t": np.nan,
                "ci": [np.nan, np.nan]}
    r = cluster_ols(x.to_numpy(float), np.ones((len(x), 1)), groups.to_numpy(), ["mean"])
    return {"n": r["n"], "n_clusters": r["n_clusters"], **r["params"]["mean"]}


def h6(ev: pd.DataFrame) -> dict:
    """H6a-c on one sample of events (rows = the sample's events, with event_returns() columns joined)."""
    ok = ~ev["skipped"].astype(bool) & ev["R_pre"].notna() & ev["R_post"].notna()
    e = ev[ok]
    c = e[e["zS_pre"].notna()]
    X = np.column_stack([np.ones(len(c)), c["zS_pre"].to_numpy(float)])
    reg = cluster_ols(c["LS"], X, c["week"], ["alpha", "zS"])
    reg_bp = cluster_ols(c["LS_bp"], X, c["week"], ["alpha", "zS"])
    by_tenor = {}
    for t, g in e.groupby("tenor"):
        by_tenor[t] = {"n": int(len(g)),
                       "R_pre": cluster_mean(g["R_pre"], g["week"]), "R_post": cluster_mean(g["R_post"], g["week"]),
                       "dy_pre_bp": float(g["dy_pre_bp"].mean()), "dy_post_bp": float(g["dy_post_bp"].mean())}
    return {
        "n_events": int(len(ev)), "n_used": int(len(e)), "n_skipped": int((~ok).sum()), "n_H6c": int(len(c)),
        "n_no_zS": int(e["zS_pre"].isna().sum()), "sample": [str(e["A"].min().date()), str(e["A"].max().date())],
        "H6a": {**cluster_mean(e["R_pre"], e["week"]), "prediction": "mean < 0"},
        "H6b": {**cluster_mean(e["R_post"], e["week"]), "prediction": "mean > 0"},
        "H6c": {**_coef(reg, "zS"), "alpha": reg["params"]["alpha"]["b"], "n": reg["n"],
                "n_clusters": reg["n_clusters"], "prediction": "beta > 0"},
        "long_short_mean": cluster_mean(e["LS"], e["week"]),
        "yield_bp": {"dy_pre": cluster_mean(e["dy_pre_bp"], e["week"]),
                     "dy_post": cluster_mean(e["dy_post_bp"], e["week"]),
                     "H6c": {**_coef(reg_bp, "zS"), "n": reg_bp["n"]}},
        "by_tenor": by_tenor,
        "units": "R: tenor's constant-maturity excess return over the window, %; dy: yield change, bp; zS: size "
                 "known at A-5, sd of the 6 prior same-tenor amounts; SEs clustered by calendar week of A",
    }


def h7(R_pct: pd.Series, dy_neg_bp: pd.Series, zA: pd.Series) -> dict:
    """R_m = a + c zA_m + e_m, Newey-West (HAC_LAGS); the same in yield terms (-dy, bp)."""
    df = pd.concat([R_pct.rename("R"), dy_neg_bp.rename("Y"), zA.rename("zA")], axis=1).dropna()
    reg = nw_regression(df["R"], {"zA": df["zA"]})
    reg_y = nw_regression(df["Y"], {"zA": df["zA"]})
    return {"c": _coef(reg, "zA"), "a": _coef(reg, "const"), "n": reg["n"],
            "yield_bp": {"c": _coef(reg_y, "zA"), "a": _coef(reg_y, "const")},
            "sample": [str(df.index[0]), str(df.index[-1])], "prediction": "c > 0 and a > 0",
            "units": "R_m: 10-year T-4..T excess return, %; zA: sd of SA_m (past months only)"}


# ------------------------------------------------------------------------------------------------- legs

LEG_COLS = ["leg_id", "unit_id", "kind", "tenor", "tenor_years", "entry", "exit", "sign", "base_risk", "hold_days",
            "w"]


def supply_legs(ev: pd.DataFrame, weighted: bool) -> pd.DataFrame:
    """Two legs per event: short pre (A-5 -> A), long post (A -> A+5); w = 1 or w_pre / w_post."""
    pre = pd.DataFrame({"unit_id": ev["event_id"], "kind": "pre", "tenor": ev["tenor"], "entry": ev["pre_entry"],
                        "exit": ev["A"], "sign": -1.0, "w": ev["w_pre"] if weighted else 1.0})
    post = pd.DataFrame({"unit_id": ev["event_id"], "kind": "post", "tenor": ev["tenor"], "entry": ev["A"],
                         "exit": ev["post_exit"], "sign": 1.0, "w": ev["w_post"] if weighted else 1.0})
    legs = pd.concat([pre, post], ignore_index=True)
    legs["leg_id"] = legs["unit_id"] + "_" + legs["kind"]
    legs["tenor_years"] = legs["tenor"].map(KNOT_YEARS)
    legs["base_risk"] = LEG_RISK
    legs["hold_days"] = LEG_HOLD_DAYS
    return legs[LEG_COLS]


def demand_legs(windows: pd.DataFrame, cal) -> pd.DataFrame:
    """The month-end calendar-only trade: long the 10-year from T-4 to T, w = 1 (src/backtest.py)."""
    legs = pd.DataFrame({"unit_id": [str(m) for m in windows.index], "kind": "demand", "tenor": HEADLINE_TENOR,
                         "entry": windows["entry"].to_numpy(), "exit": windows["exit"].to_numpy(), "sign": 1.0,
                         "w": 1.0})
    legs["leg_id"] = legs["unit_id"] + "_demand"
    legs["tenor_years"] = KNOT_YEARS[HEADLINE_TENOR]
    legs["base_risk"] = RISK_PER_TRADE
    legs["hold_days"] = [hold_days(cal, e, x) for e, x in zip(legs["entry"], legs["exit"])]
    return legs[LEG_COLS]


# ------------------------------------------------------------------------------------------------- the book

@dataclass
class BookResult:
    name: str
    legs: pd.DataFrame        # one row per leg: sizing, gross P&L, attributed cost, net P&L ($)
    daily: pd.DataFrame       # per bond day: gross_pnl, cost, pnl ($), excess, total, gross_notional, traded_notional
    cfg: RiskConfig
    unit: str                 # hit-rate unit: "event" or "month"


class _BookDrawdown(DrawdownRule):
    """src/risk.py rule 5 with the trigger passed at each decision (module docstring)."""

    def decide_at(self, threshold: float) -> float:
        if not self.cfg.dd_rule:
            return 1.0
        if self.half and self.peak > self.peak_at_trigger:
            self.half = False
        if not self.half and self.drawdown > threshold:
            self.half = True
            self.peak_at_trigger = self.peak
        return 0.5 if self.half else 1.0


def run_book(name: str, legs: pd.DataFrame, excess: pd.DataFrame, yields: pd.DataFrame, rf: pd.Series,
             fomc_scheduled: pd.DatetimeIndex, cal, days: pd.DatetimeIndex, cfg: RiskConfig | None = None,
             demand_per_year: int = 0, supply_entries: pd.DatetimeIndex | None = None,
             unit: str = "event") -> BookResult:
    """Daily NAV of a set of legs traded as one book (module docstring).

    excess / yields: daily excess returns (fractions) and yields (percent) per tenor on the bond calendar.
    demand_per_year: month-end legs per year in the drawdown trigger (12 for the book, 0 for the supply leg).
    supply_entries: entry dates of every scheduled supply leg (the full event calendar) for the trigger's count.
    """
    cfg = cfg or RiskConfig()
    legs = legs.sort_values(["entry", "kind", "leg_id"], kind="mergesort").reset_index(drop=True)
    if legs["entry"].min() < days[0] or legs["exit"].max() > days[-1]:
        raise ValueError(f"{name}: legs outside the NAV days {days[0].date()}..{days[-1].date()}")
    if (legs["exit"] <= legs["entry"]).any():
        raise ValueError(f"{name}: a leg exits on or before its entry")
    tenors = sorted(legs["tenor"].unique(), key=KNOT_YEARS.get)
    tcol = legs["tenor"].map({t: i for i, t in enumerate(tenors)}).to_numpy()
    x = excess.reindex(days)[tenors].to_numpy(float)
    pos = pd.Series(np.arange(len(days)), index=days)
    e_i = pos.reindex(legs["entry"]).to_numpy()
    x_i = pos.reindex(legs["exit"]).to_numpy()
    if np.isnan(e_i.astype(float)).any() or np.isnan(x_i.astype(float)).any():
        raise ValueError(f"{name}: a leg's entry or exit is not a NAV day")
    e_i, x_i = e_i.astype(int), x_i.astype(int)
    entries = pd.Series(np.arange(len(legs))).groupby(e_i).apply(list).to_dict()
    exits = set(x_i.tolist())
    sign = legs["sign"].to_numpy(float)
    sup = pd.DatetimeIndex(sorted(supply_entries)) if supply_entries is not None else pd.DatetimeIndex([])
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()

    L, nd, nT = len(legs), len(days), len(tenors)
    sig = np.full(L, np.nan)
    dur, dv01, notional = sig.copy(), sig.copy(), sig.copy()
    fomc_m, dd_m, cap_m = np.ones(L), np.ones(L), np.ones(L)
    gross = np.zeros(L)
    pnl, cost, gnot, traded = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
    net_dv01, net_not = np.zeros(nT), np.zeros(nT)
    dd = _BookDrawdown(cfg)
    open_: list[int] = []
    cap_total = cfg.notional_cap * cfg.capital
    for i in range(nd):
        for j in open_:                                     # P&L of positions held from the previous close
            r = x[i, tcol[j]]
            if not np.isfinite(r):
                raise ValueError(f"{name}: missing {tenors[tcol[j]]} return on {days[i].date()} (leg {legs.at[j, 'leg_id']})")
            g = sign[j] * notional[j] * r
            gross[j] += g
            pnl[i] += g
        if i in exits:                                      # exits at this close, then entries
            open_ = [j for j in open_ if x_i[j] != i]
        new = entries.get(i, [])
        if new:
            d = days[i]
            n_sup = 0
            if len(sup):
                p = int(cal.days.get_loc(d))
                lo = cal.days[max(p - TRAILING_BDAYS, 0)]
                n_sup = int(sup.searchsorted(cal.days[p - 1], side="right") - sup.searchsorted(lo, side="left"))
            exp_vol = np.sqrt(demand_per_year * cfg.risk_per_trade ** 2 + n_sup * LEG_RISK ** 2)
            ddm = dd.decide_at(cfg.dd_mult * exp_vol)
            for j in new:
                t = legs.at[j, "tenor"]
                sig[j] = sigma_bp(yields[t], d, cfg.vol_lookback)
                y_e = float(yields.at[d, t])
                dur[j] = float(mod_duration(y_e, y_e, legs.at[j, "tenor_years"]))
                fomc_m[j] = 0.5 if (cfg.fomc_half and window_has(fomc, d, legs.at[j, "exit"])) else 1.0
                dd_m[j] = ddm
                base = legs.at[j, "base_risk"] * cfg.capital / (sig[j] * np.sqrt(legs.at[j, "hold_days"]))
                dv01[j] = base * legs.at[j, "w"] * fomc_m[j] * ddm
                notional[j] = dv01[j] / (dur[j] * 1e-4)
            open_gross = float(sum(notional[k] for k in open_))
            new_gross = float(sum(notional[j] for j in new))
            if cfg.notional_cap_on and new_gross > 0 and open_gross + new_gross > cap_total:
                f = max(cap_total - open_gross, 0.0) / new_gross
                for j in new:
                    cap_m[j] = f
                    notional[j] *= f
                    dv01[j] *= f
            open_ += new
        nd01, nno = np.zeros(nT), np.zeros(nT)
        for j in open_:
            nd01[tcol[j]] += sign[j] * dv01[j]
            nno[tcol[j]] += sign[j] * notional[j]
        cost[i] = np.abs(nd01 - net_dv01).sum() * cfg.cost_bp / 2.0 * cfg.cost_mult
        traded[i] = np.abs(nno - net_not).sum()
        net_dv01, net_not = nd01, nno
        gnot[i] = float(sum(notional[j] for j in open_))
        dd.update(np.array([(pnl[i] - cost[i]) / cfg.capital]))
    if open_:
        raise ValueError(f"{name}: legs still open after the last NAV day")

    leg_cost = cfg.cost_bp * cfg.cost_mult * dv01
    out_legs = legs.assign(sigma_bp=sig, duration=dur, fomc_mult=fomc_m, dd_mult=dd_m, cap_mult=cap_m, dv01=dv01,
                           notional=notional, gross_pnl=gross, cost=leg_cost, net_pnl=gross - leg_cost)
    exc = (pnl - cost) / cfg.capital
    daily = pd.DataFrame({"gross_pnl": pnl, "cost": cost, "pnl": pnl - cost, "excess": exc,
                          "total": rf.reindex(days).to_numpy() + exc, "gross_notional": gnot,
                          "traded_notional": traded}, index=days)
    if daily["total"].isna().any():
        raise ValueError(f"{name}: missing T-bill returns in the NAV period")
    return BookResult(name=name, legs=out_legs, daily=daily, cfg=cfg, unit=unit)


def book_metrics(res: BookResult) -> dict:
    """Required metrics (src/metrics.py definitions; hit rate per event or per calendar month)."""
    d, L, cap = res.daily, res.legs, res.cfg.capital
    years = period_years(d.index)
    nav_total, nav_excess = (1.0 + d["total"]).cumprod(), (1.0 + d["excess"]).cumprod()
    if res.unit == "event":
        u = L.groupby("unit_id")["net_pnl"].sum() / cap * 100.0
    else:
        held = d["gross_notional"].groupby(d.index.to_period("M")).sum() > 0
        u = (d["pnl"].groupby(d.index.to_period("M")).sum() / cap * 100.0)[held]
    inv = d["gross_notional"] > 0
    return {
        "period": [d.index[0].date().isoformat(), d.index[-1].date().isoformat()], "years": years,
        "ann_return_pct": (float(nav_total.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0,
        "ann_excess_return_pct": (float(nav_excess.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0,
        "vol_pct": float(d["excess"].std(ddof=1) * np.sqrt(DAYS_PER_YEAR) * 100.0),
        "sharpe": sharpe(d["excess"]), "sharpe_gross": sharpe(d["excess"] + d["cost"] / cap),
        "max_drawdown_pct": max_drawdown(nav_total) * 100.0,
        "max_drawdown_excess_pct": max_drawdown(nav_excess) * 100.0,
        "turnover_x_per_year": float(d["traded_notional"].sum() / cap / years),
        "hit_rate": float((u > 0).mean()), "hit_unit": res.unit, "n_units": int(len(u)),
        "worst_unit": {"id": str(u.idxmin()), "net_pnl_pct": float(u.min())},
        "mean_unit_net_pnl_pct": float(u.mean()),
        "cost_pct_per_year": float(d["cost"].sum() / cap / years * 100.0),
        "n_legs": int(len(L)), "legs_by_kind": {k: int(v) for k, v in L["kind"].value_counts().sort_index().items()},
        "share_days_invested": float(inv.mean()),
        "mean_gross_notional_x_capital": float(d.loc[inv, "gross_notional"].mean() / cap),
        "max_gross_notional_x_capital": float(d["gross_notional"].max() / cap),
        "n_legs_capped": int((L["cap_mult"] < 1).sum()), "n_fomc_halved": int((L["fomc_mult"] < 1).sum()),
        "n_drawdown_halved": int((L["dd_mult"] < 1).sum()),
    }


def compare_sharpe(daily_a: pd.Series, daily_b: pd.Series, a: str, b: str) -> dict:
    """Net Sharpe of a minus b, paired bootstrap by calendar month (src/stats.py, as H4)."""
    r = paired_sharpe_diff_bootstrap(daily_a, daily_b)
    return {f"sharpe_{a}": r["sharpe_a"], f"sharpe_{b}": r["sharpe_b"], "diff": r["diff"], "ci": r["ci"],
            "p_le_0": r["p_le_0"], "n_months": r["n_months"], "n_boot": r["n_boot"], "seed": r["seed"]}


# ------------------------------------------------------------------------------------------------- event path

def tercile_by_rank(z: pd.Series) -> pd.Series:
    """Terciles of z by average rank, so equal values share a tercile (a quarter of zS are exactly 0, and a plain
    quantile cut would put all of them in the low tercile). Our choice, fixed before any auction-window return."""
    p = z.dropna().rank(method="average", pct=True)
    lab = pd.Series(np.where(p <= 1 / 3, "low", np.where(p <= 2 / 3, "mid", "high")), index=p.index)
    return lab.reindex(z.index)


def auction_event_paths(ev: pd.DataFrame, excess: pd.DataFrame, yields: pd.DataFrame, cal, lo: int, hi: int,
                        start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative excess return (%) and yield change (bp) from the close of A+lo to the close of A+k, k = lo..hi,
    for events whose whole span lies in [start, end] with no missing data. Index: event_id."""
    ret, dy = {}, {}
    for eid, t, A in zip(ev["event_id"], ev["tenor"], ev["A"]):
        try:
            s, f = cal.offset(A, lo), cal.offset(A, hi)
        except (KeyError, IndexError):
            continue
        if s < pd.Timestamp(start) or f > pd.Timestamp(end):
            continue
        r = excess[t].loc[s:f].iloc[1:].to_numpy()
        y = yields[t].loc[s:f].to_numpy()
        if np.isnan(r).any() or np.isnan(y).any():
            continue
        ret[eid] = np.concatenate([[0.0], np.cumprod(1.0 + r) - 1.0]) * 100.0
        dy[eid] = (y - y[0]) * 100.0
    cols = list(range(lo, hi + 1))
    return (pd.DataFrame.from_dict(ret, orient="index", columns=cols),
            pd.DataFrame.from_dict(dy, orient="index", columns=cols))


def path_summary(paths: pd.DataFrame, labels: pd.Series, weeks: pd.Series) -> dict:
    """Mean path and 95% band by tercile; the band uses standard errors clustered by the week of A (as H6)."""
    out = {}
    lab, wk = labels.reindex(paths.index), weeks.reindex(paths.index)
    for t in TERCILES:
        p = paths[lab == t]
        g = wk[lab == t]
        n, G = len(p), g.nunique()
        mean = p.mean()
        s = (p - mean).groupby(g.to_numpy()).sum()
        se = np.sqrt((s ** 2).sum() * G / (G - 1)) / n
        out[t] = {"n": int(n), "n_weeks": int(G), "k": [int(k) for k in p.columns], "mean": mean.tolist(),
                  "lo": (mean - Z95 * se).tolist(), "hi": (mean + Z95 * se).tolist()}
    return out


# ------------------------------------------------------------------------------------------------- Phase 5

DECADES = {"1993-1999": ("1993-01-01", "1999-12-31"), "2000-2009": ("2000-01-01", "2009-12-31"),
           "2010-2019": ("2010-01-01", "2019-12-31"), "2020-2024": ("2020-01-01", "2024-12-31")}


def by_tenor(res: BookResult, years: float) -> dict:
    """Net P&L of the supply legs by tenor (module docstring); % of capital."""
    L = res.legs[res.legs["kind"].isin(["pre", "post"])]
    cap = res.cfg.capital
    out = {}
    for t, g in L.groupby("tenor", sort=False):
        ev = g.groupby("unit_id")["net_pnl"].sum()
        out[t] = {"n_events": int(ev.size),
                  "net_pnl_pct_per_year": float(g["net_pnl"].sum() / cap / years * 100.0),
                  "pre_net_pnl_pct_per_year": float(g.loc[g["kind"] == "pre", "net_pnl"].sum() / cap / years * 100),
                  "post_net_pnl_pct_per_year": float(g.loc[g["kind"] == "post", "net_pnl"].sum() / cap / years * 100),
                  "mean_event_net_pnl_bp_of_capital": float(ev.mean() / cap * 1e4),
                  "hit_rate": float((ev > 0).mean()),
                  "cost_pct_per_year": float(g["cost"].sum() / cap / years * 100.0)}
    order = sorted(out, key=KNOT_YEARS.get)
    return {t: out[t] for t in order}
