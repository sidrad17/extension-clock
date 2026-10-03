"""CMT switch diagnostic (Phase 4d, descriptive; CLAUDE.md section 15). Nothing here changes a rule, signal, sizing or
cost, and nothing here is a strategy: it decomposes the existing Flow Clock supply leg.

Question (team, Oct 3, 2026): is part of the cash auction effect an artifact of how Treasury builds constant-maturity
yields? The CMT curve is fit to ~3:30 PM bid prices of the most recently auctioned securities, so on or right after an
auction its input bond switches from the old issue to the new one, while the supply leg flips short -> long at the
close of A. Futures have no such switch. If the switch drives part of the cash effect, the cash - futures gap should
sit on the switch days (A, the settlement day S) and be absent at reopenings, where the input bond does not change.

Every choice below was fixed before any Phase 4d number was computed (our choices, approved by the team with the
request; "team" = stated in the request).

Cash vs futures by day (team: 2010-07-01 to 2024-09-30, the same events, bp of capital at pre-registered sizing):
* Events: the futures supply leg's events (Flow Clock events with A-5..A+5 inside [FUT_START, IS_END], not skipped).
  An event is used when both its legs can be priced in futures: the leg's roll-rule contract (with the TN / UB
  fallback) has settlements at entry and exit and a valid DV01 (src/futures.py prepare_legs, status "ok"). Other
  events are counted by the first failing leg's status. Contract rounding, the notional cap and the drawdown rule do
  not drop an event (sizing below).
* Sizing, the same DV01 for both instruments: each leg's pre-registered DV01 = LEG_RISK x CAPITAL / (sigma_bp x
  sqrt(5)) x the FOMC half size (PREREG_FLOWCLOCK.md; sigma_bp from the event tenor's CMT yield, as both books size
  it). The drawdown rule and the notional cap are left out: each book runs them on its own NAV and notional, so they
  size the same leg differently in cash and futures and would put a sizing gap on every day. Futures contracts =
  DV01 / DV01 per contract, not rounded.
* Daily P&L before costs, $ and bp of capital: cash leg = sign x DV01 / (D x 1e-4) x the tenor's daily constant-
  maturity excess return (src/flowclock.run_book; D = par-bond modified duration at the entry yield); futures leg =
  sign x contracts x (settlement - previous settlement) x $ per point, a bond day without a settlement carrying the
  previous one (src/futures.run_futures_book). Both are excess returns. Each engine is checked against these
  functions on every run (cash_check, futures_check).
* Days: day k = close A+k-1 -> close A+k, k = -4..5; k <= 0 is the short pre leg, k >= 1 the long post leg; day A-5
  is the entry close and carries no P&L. Day A = k = 0. Day S = the auction's issue (settlement) date
  (`issue_date`); its P&L is that day's when S is in A+1..A+5; events whose S falls later have no position on S and
  are counted. Total = the sum over the 10 days.
* Statistics: per day, means across events of cash, futures and the gap (cash - futures) with standard errors
  clustered by the Monday-Sunday week of A (src/flowclock.cluster_mean, as H6). Share of the gap on a day = the mean
  gap on that day / the mean total gap (the S share counts 0 for events without S in the window, so A and S shares
  add up). Also per tenor, and for DGS10, DGS20, DGS30 by new issue vs reopening.
* Units: bp of capital; for days A and S also the gap per $1 of DV01 (bp of yield, signed as the leg's P&L: on day
  A, the short pre leg, + means the CMT yield rose more than the futures imply).
* Licensed data: only these aggregates (means, errors, counts) leave the --futures build; no per-event or per-day
  futures P&L is written (CLAUDE.md section 15, Phase 4b).

Reopening control (team: cash, full in-sample, DGS10, DGS20, DGS30):
* Reopening = the event's CUSIP was auctioned before (auction records, data/snapshot/auctions.csv); the count of
  disagreements with Fiscal Data's `reopening` flag is reported. New issue = otherwise.
* The pre-registered event returns (src/flowclock.event_returns, %, before costs): R_pre (the short pre leg earns
  -R_pre), R_post, and the total LS = R_post - R_pre, on the in-sample events of H6 (not skipped, both returns
  defined). Per tenor and pooled: counts, first and last A, week-clustered means; reopening minus new issue by OLS
  on a reopening dummy with week clusters (pooled: plus tenor dummies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.flowclock import POST_DAYS, PRE_DAYS
from src.bonds import KNOT_YEARS, mod_duration
from src.flowclock import cluster_mean, cluster_ols
from src.risk import RiskConfig, sigma_bp
from src.signals import window_has

DAY_K = list(range(1 - PRE_DAYS, POST_DAYS + 1))           # -4..5
ISSUE_TENORS = ["DGS10", "DGS20", "DGS30"]
POOLED = "DGS10+DGS20+DGS30"
RETURN_COLS = {"R_pre": "pre leg: tenor return A-5 -> A, % (the short leg earns -R_pre)",
               "R_post": "post leg: tenor return A -> A+5, %", "LS": "total: R_post - R_pre, %"}


def day_label(k: int) -> str:
    return "A" if k == 0 else f"A{k:+d}"


def weeks_of(dates: pd.Series) -> pd.Series:
    """Monday-Sunday calendar week of A (as src/auction_events.py)."""
    return pd.to_datetime(dates).dt.to_period("W-SUN").astype(str)


# ------------------------------------------------------------------------------------------------- sizing and P&L

def common_dv01(legs: pd.DataFrame, yields: pd.DataFrame, fomc_scheduled, cfg: RiskConfig | None = None) -> pd.Series:
    """Pre-registered DV01 per leg: base_risk x capital / (sigma_bp x sqrt(hold days)) x w x FOMC half size; no
    drawdown rule and no notional cap (module docstring). legs: src/flowclock.py LEG_COLS."""
    cfg = cfg or RiskConfig()
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()
    out = []
    for t, e, x, br, hd, w in zip(legs["tenor"], legs["entry"], legs["exit"], legs["base_risk"], legs["hold_days"],
                                  legs["w"]):
        f = 0.5 if (cfg.fomc_half and window_has(fomc, e, x)) else 1.0
        out.append(br * cfg.capital / (sigma_bp(yields[t], e, cfg.vol_lookback) * np.sqrt(hd)) * w * f)
    return pd.Series(out, index=legs.index, dtype=float)


def cash_notional(legs: pd.DataFrame, dv01: pd.Series, yields: pd.DataFrame) -> pd.Series:
    """Par-bond notional delivering `dv01` at entry: DV01 / (D x 1e-4), D at the tenor's entry yield (run_book)."""
    d = [float(mod_duration(y, y, n)) for y, n in
         zip([float(yields.at[e, t]) for e, t in zip(legs["entry"], legs["tenor"])], legs["tenor_years"])]
    return dv01 / (np.asarray(d) * 1e-4)


def cash_leg_daily(legs: pd.DataFrame, notional: pd.Series, excess: pd.DataFrame, cal) -> pd.DataFrame:
    """$ P&L of each leg on each bond day it is held (entry, exit]: sign x notional x the tenor's daily excess return
    (src/flowclock.run_book). Long form: leg_id, date, pnl."""
    x = excess.reindex(cal.days)
    a, b = cal.days.get_indexer(legs["entry"]), cal.days.get_indexer(legs["exit"])
    n = b - a
    if (a < 0).any() or (b < 0).any() or (n < 1).any():
        raise ValueError("cash legs: entry or exit off the bond calendar, or exit not after entry")
    day = np.repeat(a, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1
    col = legs["tenor"].map({t: i for i, t in enumerate(x.columns)}).to_numpy()
    r = x.to_numpy(float)[day, np.repeat(col, n)]
    if not np.isfinite(r).all():
        raise ValueError("cash legs: a missing tenor return inside a holding period")
    pnl = np.repeat(legs["sign"].to_numpy(float) * np.asarray(notional, float), n) * r
    return pd.DataFrame({"leg_id": np.repeat(legs["leg_id"].to_numpy(), n), "date": cal.days[day], "pnl": pnl})


def futures_leg_daily(legs: pd.DataFrame, contracts: pd.Series, m, cal) -> pd.DataFrame:
    """$ P&L of each leg on each bond day it is held: sign x contracts x (settlement - previous settlement) x $ per
    point; a day without a settlement carries the previous one (src/futures.run_futures_book). legs: leg_id, sign,
    entry, exit, contract, root, settle_entry. Long form: leg_id, date, pnl."""
    settle = m.settle.reindex(cal.days)
    rows = []
    for lid, e, xt, s, c, r, s0, n in zip(legs["leg_id"], legs["entry"], legs["exit"], legs["sign"], legs["contract"],
                                         legs["root"], legs["settle_entry"], contracts):
        a, b = cal.days.get_loc(e), cal.days.get_loc(xt)
        px = settle[c].to_numpy(float)[a + 1:b + 1]
        last, g = float(s0), np.zeros(len(px))
        for i, p in enumerate(px):
            if np.isfinite(p):
                g[i] = s * n * (p - last) * m.point_value[r]
                last = p
        rows.append(pd.DataFrame({"leg_id": lid, "date": cal.days[a + 1:b + 1], "pnl": g}))
    return pd.concat(rows, ignore_index=True)


def event_matrix(daily: pd.DataFrame, legs: pd.DataFrame, A: pd.Series, cal, scale: pd.Series | float) -> pd.DataFrame:
    """Units x day k (DAY_K): the summed P&L of the unit's legs on day A+k, each leg's P&L divided by `scale` (a
    number, or a Series by leg_id). legs: leg_id, unit_id; A: Series unit_id -> auction date."""
    d = daily.merge(legs[["leg_id", "unit_id"]], on="leg_id", how="left")
    sc = d["leg_id"].map(scale) if isinstance(scale, pd.Series) else scale
    d["v"] = d["pnl"] / sc
    pos = pd.Series(np.arange(len(cal.days)), index=cal.days)
    d["k"] = pos.reindex(d["date"]).to_numpy() - pos.reindex(A.reindex(d["unit_id"])).to_numpy()
    out = d.pivot_table(index="unit_id", columns="k", values="v", aggfunc="sum")
    return out.reindex(index=A.index, columns=DAY_K)


def s_offsets(A: pd.Series, S: pd.Series, cal) -> pd.Series:
    """Bond-day offset of the settlement date S from A per unit (NaN if S is not a bond day)."""
    pos = pd.Series(np.arange(len(cal.days)), index=cal.days)
    return pd.Series(pos.reindex(S).to_numpy() - pos.reindex(A).to_numpy(), index=A.index)


def on_s(mat: pd.DataFrame, off: pd.Series) -> pd.Series:
    """Each unit's value on its S day (0 where S is not in A+1..A+5)."""
    out = pd.Series(0.0, index=mat.index)
    for k in range(1, POST_DAYS + 1):
        sel = off == k
        out[sel] = mat.loc[sel, k]
    return out


# ------------------------------------------------------------------------------------------------- summaries

def _cm(x: pd.Series, wk: pd.Series) -> dict:
    r = cluster_mean(x, wk)
    return {k: r[k] for k in ("b", "se", "t", "n", "n_clusters")}


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def gap_summary(cash: pd.DataFrame, fut: pd.DataFrame, cash_u: pd.DataFrame, fut_u: pd.DataFrame, wk: pd.Series,
                off: pd.Series) -> dict:
    """Day path, legs, day A, day S and shares for one set of units (module docstring). cash / fut: bp of capital;
    cash_u / fut_u: per $1 of DV01 (bp of yield)."""
    gap = cash - fut
    tot = gap.sum(axis=1)
    mt = float(tot.mean())
    path = {}
    for k in DAY_K:
        path[day_label(k)] = {"cash": _cm(cash[k], wk), "futures": _cm(fut[k], wk), "gap": _cm(gap[k], wk),
                              "share_of_total_gap": _ratio(gap[k].mean(), mt)}
    pre, post = [k for k in DAY_K if k <= 0], [k for k in DAY_K if k >= 1]
    legs = {name: {"cash": _cm(cash[ks].sum(axis=1), wk), "futures": _cm(fut[ks].sum(axis=1), wk),
                   "gap": _cm(gap[ks].sum(axis=1), wk)}
            for name, ks in (("pre_leg_A-4_to_A", pre), ("post_leg_A+1_to_A+5", post), ("total", DAY_K))}
    in_w = off.between(1, POST_DAYS)
    gs, cs, fs = on_s(gap, off), on_s(cash, off), on_s(fut, off)
    gu_s = on_s(cash_u - fut_u, off)
    day_s = {"n_units_S_in_window": int(in_w.sum()),
             "S_offset_counts": {day_label(int(k)): int(v) for k, v in off.value_counts().sort_index().items()},
             "n_units_S_after_window": int((off > POST_DAYS).sum()), "n_units_S_not_bond_day": int(off.isna().sum())}
    if in_w.any():
        day_s.update({"cash": _cm(cs[in_w], wk[in_w]), "futures": _cm(fs[in_w], wk[in_w]),
                      "gap": _cm(gs[in_w], wk[in_w]), "gap_per_unit_dv01_bp": _cm(gu_s[in_w], wk[in_w])})
    day_s["share_of_total_gap"] = _ratio(gs.mean(), mt)
    a_s = gap[0] + gs
    return {
        "n_units": int(len(gap)), "n_weeks": int(wk.nunique()),
        "day_path": path, "legs": legs,
        "day_A": {**path["A"], "gap_per_unit_dv01_bp": _cm(cash_u[0] - fut_u[0], wk)},
        "day_S": day_s,
        "gap_on_A_and_S": {**_cm(a_s, wk), "share_of_total_gap": _ratio(a_s.mean(), mt)},
        "gap_other_days": {**_cm(tot - a_s, wk), "share_of_total_gap": _ratio((tot - a_s).mean(), mt)},
    }


def cash_vs_futures(ev: pd.DataFrame, issue_date: pd.Series, reopen: pd.Series, su_legs: pd.DataFrame, su_prep: pd.DataFrame, m,
                    excess: pd.DataFrame, yields: pd.DataFrame, fomc_scheduled, cal, capital: float,
                    check=None) -> dict:
    """The cash vs futures day-by-day diagnostic (module docstring), aggregates only.

    ev: the futures supply leg's events (event_id, A, tenor); issue_date and reopen (reopening_flags): aligned with ev.
    su_legs / su_prep: src/futures.py supply legs and prepare_legs output (same index).
    check: the base futures supply book run with record_leg_daily=True (src/futures.run_futures_book), whose traded
    legs' daily P&L futures_leg_daily must reproduce; None skips the check (reported).
    """
    ok = su_prep["status"] == "ok"
    st = su_legs.assign(status=su_prep["status"].to_numpy()).sort_values(["unit_id", "kind"], ascending=[True, False])
    bad = st[st["status"] != "ok"].groupby("unit_id")["status"].first()
    units = ev.loc[~ev["event_id"].isin(bad.index), "event_id"]
    legs = su_legs[ok & su_legs["unit_id"].isin(units)]
    prep = su_prep.loc[legs.index]
    dv01 = common_dv01(legs, yields, fomc_scheduled)
    c_long = cash_leg_daily(legs, cash_notional(legs, dv01, yields), excess, cal)
    fl = legs[["leg_id", "sign", "entry", "exit"]].assign(contract=prep["contract"], root=prep["root"],
                                                          settle_entry=prep["settle_entry"])
    f_long = futures_leg_daily(fl, dv01 / prep["dv01_contract"], m, cal)
    e = ev.assign(reopen=reopen.astype(bool).to_numpy()).set_index("event_id").loc[units]
    A = e["A"]
    by_leg = pd.Series(dv01.to_numpy(), index=legs["leg_id"])
    cash, fut = (event_matrix(x, legs, A, cal, capital / 1e4) for x in (c_long, f_long))
    cash_u, fut_u = (event_matrix(x, legs, A, cal, by_leg) for x in (c_long, f_long))
    if cash.isna().any().any() or fut.isna().any().any():
        raise ValueError("cmt switch: a unit is missing a day of P&L")
    wk = weeks_of(A)
    off = s_offsets(A, pd.to_datetime(issue_date.set_axis(ev["event_id"]).loc[units]), cal)
    out = {"sample": [str(A.min().date()), str(A.max().date())],
           "n_events": int(len(ev)), "n_events_used": int(len(units)),
           "n_events_dropped_by_leg_status": {k: int(v) for k, v in bad.value_counts().sort_index().items()},
           "all": gap_summary(cash, fut, cash_u, fut_u, wk, off)}
    out["by_tenor"] = {}
    roots = prep.groupby(legs["tenor"].to_numpy())["root"].value_counts()
    for t, g in e.groupby("tenor", sort=False):
        u = g.index
        out["by_tenor"][t] = {"futures_legs_by_root": {r: int(n) for r, n in roots.loc[t].items()},
                              **gap_summary(cash.loc[u], fut.loc[u], cash_u.loc[u], fut_u.loc[u], wk.loc[u],
                                            off.loc[u])}
    out["by_tenor"] = {t: out["by_tenor"][t] for t in sorted(out["by_tenor"], key=KNOT_YEARS.get)}
    out["by_issue_type"] = {}
    for t in ISSUE_TENORS + [POOLED]:
        sel = e["tenor"].isin(ISSUE_TENORS) if t == POOLED else e["tenor"] == t
        if not sel.any():
            continue
        out["by_issue_type"][t] = {}
        for name, flag in (("new_issue", False), ("reopening", True)):
            u = e.index[sel & (e["reopen"] == flag)]
            if len(u) == 0:
                continue
            s = gap_summary(cash.loc[u], fut.loc[u], cash_u.loc[u], fut_u.loc[u], wk.loc[u], off.loc[u])
            out["by_issue_type"][t][name] = {k: s[k] for k in ("n_units", "n_weeks", "day_A", "day_S", "legs")}
    out["futures_check"] = futures_check(check, legs, prep, m, cal) if check is not None else {
        "status": "skipped: no base run passed"}
    return out


def futures_check(base, legs: pd.DataFrame, prep: pd.DataFrame, m, cal) -> dict:
    """futures_leg_daily at the traded contract counts against the base futures supply book's own per-leg daily
    P&L (src/futures.run_futures_book, record_leg_daily=True): the day alignment the diagnostic relies on."""
    bl = base.legs
    tr = bl[(bl["contracts"] > 0) & bl["leg_id"].isin(legs["leg_id"])]
    pi = prep.set_axis(legs["leg_id"].to_numpy())
    fl = tr[["leg_id", "sign", "entry", "exit", "contract", "root"]].assign(
        settle_entry=pi.loc[tr["leg_id"], "settle_entry"].to_numpy())
    mine = futures_leg_daily(fl, tr["contracts"], m, cal).set_index(["leg_id", "date"])["pnl"]
    ld = base.leg_daily
    eng = pd.Series(ld["gross_pnl"].to_numpy(), index=pd.MultiIndex.from_arrays(
        [bl["leg_id"].to_numpy()[ld["leg"].to_numpy()], base.daily.index[ld["day"].to_numpy()]],
        names=["leg_id", "date"]))
    eng = eng[eng.index.get_level_values(0).isin(tr["leg_id"])]
    j = mine.to_frame("mine").join(eng.rename("engine"), how="outer").fillna(0.0)
    diff = float((j["mine"] - j["engine"]).abs().max()) if len(j) else 0.0
    if diff > 1e-6:
        raise RuntimeError(f"cmt switch: futures daily P&L differs from the futures engine by ${diff}")
    return {"n_traded_legs_checked": int(len(tr)), "max_abs_diff_usd": diff}


def cash_check(book_legs: pd.DataFrame, excess: pd.DataFrame, cal) -> dict:
    """cash_leg_daily at the book's own notionals against the cash supply book's per-leg gross P&L (run_book)."""
    d = cash_leg_daily(book_legs, book_legs["notional"], excess, cal)
    s = d.groupby("leg_id")["pnl"].sum().reindex(book_legs["leg_id"]).to_numpy()
    diff = float(np.abs(s - book_legs["gross_pnl"].to_numpy()).max())
    if diff > 1e-6:
        raise RuntimeError(f"cmt switch: cash daily P&L differs from the cash book by ${diff}")
    return {"n_legs_checked": int(len(book_legs)), "max_abs_diff_usd": diff}


# ------------------------------------------------------------------------------------------------- reopenings

def reopening_flags(ev: pd.DataFrame, auctions: pd.DataFrame) -> pd.Series:
    """True if the event's CUSIP was auctioned before its A (any auction record)."""
    first = auctions.groupby("cusip")["auction_date"].min()
    return pd.Series(ev["A"].to_numpy() > ev["cusip"].map(first).to_numpy(), index=ev.index)


def reopening_control(ev: pd.DataFrame, reopen: pd.Series) -> dict:
    """Reopening control (module docstring). ev: in-sample events with event_returns() columns, A, tenor, week."""
    ok = ~ev["skipped"].astype(bool) & ev["R_pre"].notna() & ev["R_post"].notna() & ev["tenor"].isin(ISSUE_TENORS)
    e = ev[ok].assign(reopen=reopen[ok].astype(bool))
    out = {}
    for t in ISSUE_TENORS + [POOLED]:
        g = e if t == POOLED else e[e["tenor"] == t]
        cell = {}
        for name, flag in (("new_issue", False), ("reopening", True)):
            h = g[g["reopen"] == flag]
            cell[name] = {"n": int(len(h))}
            if len(h):
                cell[name].update({"first_A": str(h["A"].min().date()), "last_A": str(h["A"].max().date()),
                                   **{c: _cm(h[c], h["week"]) for c in RETURN_COLS}})
        if g["reopen"].nunique() == 2:
            cols = [np.ones(len(g)), g["reopen"].to_numpy(float)]
            names = ["const", "reopening"]
            if t == POOLED:
                for d in [x for x in ISSUE_TENORS if x in set(g["tenor"])][1:]:     # tenor dummies (first = base)
                    cols.append((g["tenor"] == d).to_numpy(float))
                    names.append(d)
            X = np.column_stack(cols)
            cell["reopening_minus_new"] = {}
            for c in RETURN_COLS:
                r = cluster_ols(g[c], X, g["week"], names)
                cell["reopening_minus_new"][c] = {**{k: r["params"]["reopening"][k] for k in ("b", "se", "t", "ci")},
                                                  "n": r["n"], "n_clusters": r["n_clusters"]}
        out[t] = cell
    return out
