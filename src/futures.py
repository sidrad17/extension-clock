"""Futures layer: contract selection (roll rule), DV01 from data, futures P&L and costs (CLAUDE.md 7.9, section 15).

Every rule below was fixed before any futures return was computed (CLAUDE.md section 15; "team" = team decision of
Oct 3, 2026; the rest our choices, approved by the team at the same time).

* Contracts (team): the month-end calendar-only leg trades ZN (settings.HEADLINE_FUTURE, gate1-prereg). Flow Clock
  supply legs trade config.futures.SUPPLY_CONTRACT by auction tenor (2y, 3y -> ZT; 5y -> ZF; 7y -> ZN; 10y -> TN;
  20y -> ZB; 30y -> UB), with TN -> ZN and UB -> ZB before the product exists (below).
* Roll rule (CLAUDE.md 7.9): the first intention day (FID) of a quarterly contract is FID_BDAYS_BEFORE (2) business
  days before the first business day of its delivery month. Business days = the bond calendar (src/calendar.py),
  extended with plain weekdays after its last day, so no data after IS_END is read (the extension only matters for
  contracts delivering after IS_END, whose FID is months after any in-sample exit). At a leg's entry E, among the
  product's quarterly outrights with FID > exit + ROLL_BUFFER_BDAYS (5) business days, take the one with the highest
  volume on the bond day before E (the UTC-day ohlcv-1d bar dated E-1, complete before the close of E); ties go to
  the nearer delivery. If the product has no volume at all that day, step back one bond day (at most
  VOLUME_MAX_BACK). No eligible contract with volume -> the leg is not traded ("no_contract"). The contract is held
  from entry to exit; no leg rolls.
* DV01 per contract (CLAUDE.md 7.9): OLS slope, with intercept, of the contract's daily settlement change x $ per
  point on -dy (bp) over the DV01_LOOKBACK (60) bond-day changes ending E-1 (the sigma rule's window,
  src/risk.py); pairs with a missing settlement or yield are dropped; fewer than DV01_MIN_OBS (50) valid pairs, or a
  slope <= 0, -> not sized ("no_dv01"). The month-end leg regresses on DGS10 (settings.FUT_YIELD_MAP); a supply leg
  on its event tenor's own CMT yield, the yield that sized it, so the contract count delivers the leg's DV01 target
  in that yield (ZT on DGS3 for a 3-year auction).
* Fallback (team, TN and UB): "before the product exists" = before the first entry date, over the legs mapped to
  the product, at which its roll-rule contract can be sized (contract with volume, settlements at entry and exit,
  valid DV01). Legs entering earlier use the fallback product; later, an unsizable leg is not traded.
* Sizing: DV01 target as in the cash engines (src/risk.py, src/flowclock.py): base_risk x CAPITAL / (sigma_bp x
  sqrt(hold days)) x w x FOMC half size (scheduled decision in (entry, exit]) x drawdown multiplier. sigma_bp is the
  std of the leg tenor's 60 daily yield changes ending E-1 (the 10-year for the month-end leg). Contracts = target /
  DV01 per contract, rounded half up; 0 -> not traded ("zero_contracts"). Notional = contracts x settlement at
  entry x $ per point. Notional cap: gross open notional <= NOTIONAL_CAP x CAPITAL; legs entering at a close (after
  that close's exits) are scaled by one common factor and rounded down ("capped_to_zero" if that leaves 0).
* Drawdown rule: src/flowclock.py's state machine on the strategy's own excess NAV through the day before entry,
  trigger DD_MULT x sqrt(demand_per_year x RISK_PER_TRADE^2 + n x LEG_RISK^2), n = scheduled supply-leg entries in
  the trailing 252 bond days. For the month-end leg alone (12 per year, no supply) this is src/risk.py's rule.
* P&L (CLAUDE.md 7.9): each day a leg is open, sign x contracts x (settlement - previous settlement) x $ per point;
  a bond day without a settlement carries the previous one (the move counts the next day). Settlements must exist
  at entry and exit ("no_settle" otherwise). Futures P&L is an excess return: excess = P&L / CAPITAL; total adds
  the T-bill (DTB3) on capital, as the cash engines.
* Costs (team): 1 tick + FUT_COMMISSION_RT ($2) per contract round trip, half at entry and half at exit, x
  cost_mult (COST_STRESS = 2 for the stress test). The tick value is the contract's on that day, from the exchange
  definitions (src/data/databento_futures.py). At each close the cost is charged on |change in net contracts| of
  each contract, so offsetting legs in the same contract net out (the futures form of the Flow Clock book's cost
  netting). Per-leg attributed cost (hit rate, P&L tables) = contracts x (tick at entry + tick at exit + 2 x $2) / 2.
* Turnover: sum over closes of |change in net contracts| x settlement x $ per point / CAPITAL / years.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from config.futures import (DV01_MIN_OBS, FALLBACK, FID_BDAYS_BEFORE, SUPPLY_CONTRACT, VOLUME_LAG_BDAYS,
                            VOLUME_MAX_BACK)
from config.settings import DV01_LOOKBACK, FUT_COMMISSION_RT, HEADLINE_FUTURE, ROLL_BUFFER_BDAYS
from src.backtest import StrategyResult
from src.flowclock import TRAILING_BDAYS, BookResult, _BookDrawdown
from src.risk import RiskConfig, sigma_bp
from src.signals import window_has

STATUS_OK = "ok"


# ------------------------------------------------------------------------------------------------ roll rule

class RollCalendar:
    """Business days for the roll rule: the bond calendar, then plain weekdays after its last covered day."""

    def __init__(self, cal, years_ahead: int = 2):
        ext = pd.bdate_range(cal.covered_through + pd.Timedelta(days=1),
                             cal.covered_through + pd.DateOffset(years=years_ahead))
        self.days = cal.days.append(ext)
        self._pos = pd.Series(np.arange(len(self.days)), index=self.days)

    def offset(self, d, k: int) -> pd.Timestamp:
        return self.days[int(self._pos[pd.Timestamp(d)]) + k]

    def first_on_or_after(self, d) -> pd.Timestamp:
        return self.days[int(self.days.searchsorted(pd.Timestamp(d), side="left"))]


def first_intention_day(delivery: pd.Period, rc: RollCalendar) -> pd.Timestamp:
    """FID_BDAYS_BEFORE business days before the first business day of the delivery month."""
    return rc.offset(rc.first_on_or_after(delivery.start_time), -FID_BDAYS_BEFORE)


# ------------------------------------------------------------------------------------------------ market data

@dataclass
class Market:
    """Settlements and volumes on bond days, contract attributes and tick values (src/data/databento_futures.py)."""
    days: pd.DatetimeIndex            # bond days of the data (settle / volume rows)
    settle: pd.DataFrame              # days x contract, price points; NaN = no settlement that trade date
    volume: pd.DataFrame              # days x contract; 0 = no bar
    root: pd.Series                   # contract -> root
    delivery: pd.Series               # contract -> monthly Period
    fid: pd.Series                    # contract -> first intention day
    point_value: dict                 # root -> $ per price point
    ticks: dict                       # contract -> (snapshot dates, tick values $)
    rc: RollCalendar

    def tick_value(self, contract: str, date) -> float:
        """The contract's tick value on `date`: its latest snapshot on or before the date, else its first one."""
        snaps, vals = self.ticks[contract]
        k = int(snaps.searchsorted(np.datetime64(pd.Timestamp(date)), side="right")) - 1
        return float(vals[max(k, 0)])

    def contracts_of(self, root: str) -> list[str]:
        if not hasattr(self, "_by_root"):
            self._by_root = {r: list(g.index) for r, g in self.root.groupby(self.root)}
        return self._by_root.get(root, [])


def build_market(settle_long: pd.DataFrame, volume_long: pd.DataFrame, defs: pd.DataFrame, cal,
                 point_value: dict) -> Market:
    """Market from the loaders' long tables (settlements, volume, definition snapshots) on the bond calendar."""
    first = min(settle_long["trade_date"].min(), volume_long["date"].min())
    days = cal.days[cal.days >= first]
    settle = settle_long.pivot(index="trade_date", columns="contract", values="settle").reindex(days)
    contracts = sorted(set(settle.columns) | set(volume_long["contract"]))
    settle = settle.reindex(columns=contracts)
    volume = volume_long.pivot(index="date", columns="contract", values="volume").reindex(index=days,
                                                                                        columns=contracts)
    volume = volume.fillna(0.0).astype(float)
    attrs = pd.concat([settle_long[["contract", "root", "delivery"]], volume_long[["contract", "root", "delivery"]]])
    attrs = attrs.drop_duplicates("contract").set_index("contract").reindex(contracts)
    rc = RollCalendar(cal)
    fid = pd.Series({c: first_intention_day(attrs.at[c, "delivery"], rc) for c in contracts})
    ticks = {}
    for c, g in defs.groupby("contract"):
        ticks[c] = (g["snapshot"].to_numpy(dtype="datetime64[ns]"), g["tick_value"].to_numpy(float))
    return Market(days=days, settle=settle, volume=volume, root=attrs["root"], delivery=attrs["delivery"], fid=fid,
                  point_value=dict(point_value), ticks=ticks, rc=rc)


def select_contract(m: Market, root: str, entry: pd.Timestamp, exit_: pd.Timestamp) -> dict | None:
    """The roll rule (module docstring): contract, its FID, the volume day and volume; None if none qualifies."""
    cutoff = m.rc.offset(exit_, ROLL_BUFFER_BDAYS)
    allc = m.contracts_of(root)
    cands = [c for c in allc if m.fid[c] > cutoff]
    pos = int(m.days.get_loc(entry))
    for back in range(VOLUME_LAG_BDAYS, VOLUME_LAG_BDAYS + VOLUME_MAX_BACK):
        if pos - back < 0:
            return None
        d = m.days[pos - back]
        if not (m.volume.loc[d, allc] > 0).any():            # product closed that day: step back
            continue
        if not cands:
            return None
        v = m.volume.loc[d, cands]
        if not (v.max() > 0):
            return None
        best = sorted(v.index[v == v.max()], key=lambda c: m.delivery[c])[0]
        return {"contract": best, "fid": m.fid[best], "volume_day": d, "volume": float(v[best])}
    return None


def dv01_regression(m: Market, contract: str, y: pd.Series, entry: pd.Timestamp,
                    lookback: int = DV01_LOOKBACK, min_obs: int = DV01_MIN_OBS) -> dict:
    """Slope of settlement change ($ per contract) on -dy (bp) over the `lookback` changes ending E-1."""
    pos = int(m.days.get_loc(entry))
    idx = m.days[max(pos - 1 - lookback, 0): pos]                 # levels E-61 .. E-1 -> 60 changes
    s = m.settle[contract].reindex(idx).to_numpy(float)
    yy = y.reindex(idx).to_numpy(float)
    ds = np.diff(s) * m.point_value[m.root[contract]]
    x = -np.diff(yy) * 100.0
    ok = np.isfinite(ds) & np.isfinite(x)
    n = int(ok.sum())
    if n < min_obs:
        return {"dv01": np.nan, "r2": np.nan, "n": n}
    x, ds = x[ok], ds[ok]
    xc = x - x.mean()
    slope = float((xc * (ds - ds.mean())).sum() / (xc ** 2).sum())
    r = np.corrcoef(x, ds)[0, 1]
    return {"dv01": slope, "r2": float(r ** 2), "n": n}


def size_info(m: Market, root: str, tenor: str, entry: pd.Timestamp, exit_: pd.Timestamp,
              yields: pd.DataFrame) -> dict:
    """Path-independent sizing inputs of one leg in one product (module docstring)."""
    out = {"root": root, "status": STATUS_OK, "contract": "", "fid": pd.NaT, "volume_day": pd.NaT, "volume": np.nan,
           "dv01_contract": np.nan, "dv01_r2": np.nan, "dv01_n": 0, "settle_entry": np.nan, "settle_exit": np.nan}
    c = select_contract(m, root, entry, exit_)
    if c is None:
        return {**out, "status": "no_contract"}
    out.update(c)
    k = c["contract"]
    out["settle_entry"], out["settle_exit"] = m.settle.at[entry, k], m.settle.at[exit_, k]
    if not (np.isfinite(out["settle_entry"]) and np.isfinite(out["settle_exit"])):
        return {**out, "status": "no_settle"}
    reg = dv01_regression(m, k, yields[tenor], entry)
    out.update({"dv01_contract": reg["dv01"], "dv01_r2": reg["r2"], "dv01_n": reg["n"]})
    if not (np.isfinite(reg["dv01"]) and reg["dv01"] > 0):
        return {**out, "status": "no_dv01"}
    return out


def prepare_legs(legs: pd.DataFrame, m: Market, yields: pd.DataFrame) -> pd.DataFrame:
    """Contract, DV01 and status per leg, with the TN / UB fallback rule applied (module docstring).
    legs: src/flowclock.py LEG_COLS plus `product` (the mapped root)."""
    prim = [size_info(m, p, t, e, x, yields) for p, t, e, x in
            zip(legs["product"], legs["tenor"], legs["entry"], legs["exit"])]
    prim = pd.DataFrame(prim, index=legs.index)
    first_ok = {}
    for p in legs["product"].unique():
        ok = (legs["product"] == p) & (prim["status"] == STATUS_OK)
        first_ok[p] = legs.loc[ok, "entry"].min() if ok.any() else pd.Timestamp.max
    rows = []
    for i, leg in legs.iterrows():
        p = leg["product"]
        if p in FALLBACK and leg["entry"] < first_ok[p]:
            info = size_info(m, FALLBACK[p], leg["tenor"], leg["entry"], leg["exit"], yields)
            rows.append({**info, "fallback_used": True})
        else:
            rows.append({**prim.loc[i].to_dict(), "fallback_used": False})
    out = pd.DataFrame(rows, index=legs.index)
    out.attrs["first_sizable_entry"] = {p: (None if d == pd.Timestamp.max else str(pd.Timestamp(d).date()))
                                        for p, d in first_ok.items()}
    return out


# ------------------------------------------------------------------------------------------------ legs

def month_end_legs(demand: pd.DataFrame) -> pd.DataFrame:
    """The month-end calendar-only legs (src/flowclock.demand_legs) on settings.HEADLINE_FUTURE (ZN)."""
    return demand.assign(product=HEADLINE_FUTURE)


def supply_legs_futures(supply: pd.DataFrame) -> pd.DataFrame:
    """Flow Clock supply legs (src/flowclock.supply_legs) on config.futures.SUPPLY_CONTRACT by auction tenor."""
    return supply.assign(product=supply["tenor"].map(SUPPLY_CONTRACT))


# ------------------------------------------------------------------------------------------------ the engine

def run_futures_book(name: str, legs: pd.DataFrame, prep: pd.DataFrame, m: Market, yields: pd.DataFrame,
                     rf: pd.Series, fomc_scheduled: pd.DatetimeIndex, cal, days: pd.DatetimeIndex,
                     cfg: RiskConfig | None = None, demand_per_year: int = 0,
                     supply_entries: pd.DatetimeIndex | None = None, unit: str = "event",
                     commission_rt: float = FUT_COMMISSION_RT) -> BookResult:
    """Daily NAV of futures legs traded as one book (module docstring). legs + prep share an index; days = the NAV
    bond days. Returns src/flowclock.py's BookResult, so src/flowclock.book_metrics applies."""
    cfg = cfg or RiskConfig()
    from config.flowclock import LEG_RISK
    order = legs.sort_values(["entry", "kind", "leg_id"], kind="mergesort").index
    legs, prep = legs.loc[order].reset_index(drop=True), prep.loc[order].reset_index(drop=True)
    if legs["entry"].min() < days[0] or legs["exit"].max() > days[-1]:
        raise ValueError(f"{name}: legs outside the NAV days {days[0].date()}..{days[-1].date()}")
    pos = pd.Series(np.arange(len(days)), index=days)
    e_i = pos.reindex(legs["entry"]).to_numpy()
    x_i = pos.reindex(legs["exit"]).to_numpy()
    if np.isnan(e_i.astype(float)).any() or np.isnan(x_i.astype(float)).any():
        raise ValueError(f"{name}: a leg's entry or exit is not a NAV day")
    e_i, x_i = e_i.astype(int), x_i.astype(int)
    entries = pd.Series(np.arange(len(legs))).groupby(e_i).apply(list).to_dict()
    sup = pd.DatetimeIndex(sorted(supply_entries)) if supply_entries is not None else pd.DatetimeIndex([])
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()
    settle = m.settle.reindex(days)

    L, nd = len(legs), len(days)
    sign = legs["sign"].to_numpy(float)
    status = prep["status"].to_numpy(object).copy()
    contract = prep["contract"].to_numpy(object)
    pv = np.array([m.point_value.get(r, np.nan) for r in prep["root"]])
    sig, dv01_t, raw, cap_m = np.full(L, np.nan), np.full(L, np.nan), np.full(L, np.nan), np.ones(L)
    fomc_m, dd_m = np.ones(L), np.ones(L)
    n_c = np.zeros(L)
    notional, gross, last = np.zeros(L), np.zeros(L), np.full(L, np.nan)
    tick_e, tick_x = np.full(L, np.nan), np.full(L, np.nan)
    pnl, cost, gnot, traded = np.zeros(nd), np.zeros(nd), np.zeros(nd), np.zeros(nd)
    net_prev: dict[str, float] = {}
    dd = _BookDrawdown(cfg)
    open_: list[int] = []
    cap_total = cfg.notional_cap * cfg.capital
    for i in range(nd):
        day = days[i]
        for j in open_:                                        # P&L of positions held from the previous close
            s = settle.at[day, contract[j]]
            if np.isfinite(s):
                g = sign[j] * n_c[j] * (s - last[j]) * pv[j]
                gross[j] += g
                pnl[i] += g
                last[j] = s
        open_ = [j for j in open_ if x_i[j] != i]               # exits at this close, then entries
        new = entries.get(i, [])
        if new:
            n_sup = 0
            if len(sup):
                p = int(cal.days.get_loc(day))
                lo = cal.days[max(p - TRAILING_BDAYS, 0)]
                n_sup = int(sup.searchsorted(cal.days[p - 1], side="right") - sup.searchsorted(lo, side="left"))
            ddm = dd.decide_at(cfg.dd_mult * np.sqrt(demand_per_year * cfg.risk_per_trade ** 2 + n_sup * LEG_RISK ** 2))
            sized = []
            for j in new:
                t = legs.at[j, "tenor"]
                sig[j] = sigma_bp(yields[t], day, cfg.vol_lookback)
                fomc_m[j] = 0.5 if (cfg.fomc_half and window_has(fomc, day, legs.at[j, "exit"])) else 1.0
                dd_m[j] = ddm
                base = legs.at[j, "base_risk"] * cfg.capital / (sig[j] * np.sqrt(legs.at[j, "hold_days"]))
                dv01_t[j] = base * legs.at[j, "w"] * fomc_m[j] * ddm
                if status[j] != STATUS_OK:
                    continue
                raw[j] = dv01_t[j] / prep.at[j, "dv01_contract"]
                n_c[j] = np.floor(raw[j] + 0.5)
                if n_c[j] == 0:
                    status[j] = "zero_contracts"
                    continue
                notional[j] = n_c[j] * prep.at[j, "settle_entry"] * pv[j]
                sized.append(j)
            open_gross = float(notional[open_].sum()) if open_ else 0.0
            new_gross = float(notional[sized].sum()) if sized else 0.0
            if cfg.notional_cap_on and new_gross > 0 and open_gross + new_gross > cap_total:
                f = max(cap_total - open_gross, 0.0) / new_gross
                for j in sized:
                    cap_m[j] = f
                    n_c[j] = np.floor(n_c[j] * f)
                    notional[j] = n_c[j] * prep.at[j, "settle_entry"] * pv[j]
                    if n_c[j] == 0:
                        status[j] = "capped_to_zero"
                sized = [j for j in sized if n_c[j] > 0]
            for j in sized:
                last[j] = prep.at[j, "settle_entry"]
                tick_e[j] = m.tick_value(contract[j], day)
                tick_x[j] = m.tick_value(contract[j], legs.at[j, "exit"])
            open_ += sized
        net: dict[str, float] = {}
        for j in open_:
            net[contract[j]] = net.get(contract[j], 0.0) + sign[j] * n_c[j]
        for c in set(net) | set(net_prev):
            dn = abs(net.get(c, 0.0) - net_prev.get(c, 0.0))
            if dn:
                cost[i] += dn * (m.tick_value(c, day) + commission_rt) / 2.0 * cfg.cost_mult
                traded[i] += dn * settle.at[day, c] * m.point_value[m.root[c]]
        net_prev = {c: v for c, v in net.items() if v}
        gnot[i] = float(notional[open_].sum()) if open_ else 0.0
        dd.update(np.array([(pnl[i] - cost[i]) / cfg.capital]))
    if open_:
        raise ValueError(f"{name}: legs still open after the last NAV day")

    traded_leg = n_c > 0
    leg_cost = np.where(traded_leg, n_c * ((tick_e + commission_rt) / 2 + (tick_x + commission_rt) / 2)
                        * cfg.cost_mult, 0.0)
    keep = ["leg_id", "unit_id", "kind", "tenor", "product", "entry", "exit", "sign", "base_risk", "hold_days", "w"]
    out_legs = legs[keep].assign(
        root=prep["root"].to_numpy(), fallback_used=prep["fallback_used"].to_numpy(bool), contract=contract,
        fid=prep["fid"].to_numpy(), volume_day=prep["volume_day"].to_numpy(), volume=prep["volume"].to_numpy(),
        status=status, sigma_bp=sig, fomc_mult=fomc_m, dd_mult=dd_m, cap_mult=cap_m, dv01_target=dv01_t,
        dv01_contract=prep["dv01_contract"].to_numpy(float), dv01_r2=prep["dv01_r2"].to_numpy(float),
        dv01_n=prep["dv01_n"].to_numpy(int), contracts_raw=raw, contracts=n_c,
        dv01=n_c * prep["dv01_contract"].to_numpy(float), notional=notional, point_value=pv, tick_value_entry=tick_e,
        tick_value_exit=tick_x, gross_pnl=gross, cost=leg_cost, net_pnl=gross - leg_cost)
    exc = (pnl - cost) / cfg.capital
    daily = pd.DataFrame({"gross_pnl": pnl, "cost": cost, "pnl": pnl - cost, "excess": exc,
                          "total": rf.reindex(days).to_numpy() + exc, "gross_notional": gnot,
                          "traded_notional": traded}, index=days)
    if daily["total"].isna().any():
        raise ValueError(f"{name}: missing T-bill returns in the NAV period")
    return BookResult(name=name, legs=out_legs, daily=daily, cfg=cfg, unit=unit)


# ------------------------------------------------------------------------------------------------ results

def traded(res: BookResult) -> BookResult:
    """The same result with only the legs that traded (for src/flowclock.book_metrics' per-unit hit rate)."""
    return BookResult(name=res.name, legs=res.legs[res.legs["contracts"] > 0].reset_index(drop=True),
                      daily=res.daily, cfg=res.cfg, unit=res.unit)


def as_strategy(res: BookResult) -> StrategyResult:
    """A month-end futures book (one leg per month) as src/backtest.py's StrategyResult, so
    src/metrics.required_metrics applies (traded windows only; capital fixed)."""
    t = res.legs[res.legs["contracts"] > 0].copy()
    t["month"] = pd.PeriodIndex(t["unit_id"], freq="M")
    t["capped"] = (t["cap_mult"] < 1).astype(int)
    t["fomc"] = (t["fomc_mult"] < 1).astype(int)
    trades = t.set_index("month")[["entry", "exit", "hold_days", "sigma_bp", "w", "fomc", "fomc_mult", "dd_mult",
                                   "dv01", "notional", "capped", "gross_pnl", "cost", "net_pnl", "contract",
                                   "contracts"]]
    d = res.daily
    daily = pd.DataFrame({"pnl": d["pnl"], "excess": d["excess"], "total": d["total"],
                          "notional": d["gross_notional"]}, index=d.index)
    return StrategyResult(name=res.name, trades=trades, daily=daily, cfg=res.cfg)


def leg_summary(res: BookResult) -> dict:
    """Counts by status, product and fallback; DV01 regression and sizing diagnostics of the traded legs."""
    L = res.legs
    t = L[L["contracts"] > 0]
    by_root = {}
    for r, g in t.groupby("root"):
        by_root[r] = {"n_legs": int(len(g)), "n_fallback": int(g["fallback_used"].sum()),
                      "contracts_median": float(g["contracts"].median()),
                      "dv01_contract_median": float(g["dv01_contract"].median()),
                      "dv01_r2_median": float(g["dv01_r2"].median()), "dv01_r2_min": float(g["dv01_r2"].min()),
                      "dv01_n_min": int(g["dv01_n"].min()),
                      "tick_value_entry": sorted(float(v) for v in g["tick_value_entry"].unique())}
    ratio = t["dv01"] / t["dv01_target"]
    return {"n_legs": int(len(L)), "n_traded": int(len(t)),
            "status": {k: int(v) for k, v in L["status"].value_counts().sort_index().items()},
            "by_product": {k: by_root[k] for k in sorted(by_root)},
            "by_tenor_product": {f"{a}->{b}": int(v) for (a, b), v in t.groupby(["tenor", "root"]).size().items()},
            "n_contracts_capped": int((t["cap_mult"] < 1).sum()),
            "realised_over_target_dv01": {"median": float(ratio.median()), "p5": float(ratio.quantile(0.05)),
                                          "p95": float(ratio.quantile(0.95))},
            "n_distinct_contracts": int(t["contract"].nunique())}


# ------------------------------------------------------------------------------------------------ derived tables

LEG_TABLE_COLS = ["strategy", "cost_mult", "leg_id", "unit_id", "kind", "tenor", "product", "root", "fallback_used",
                  "contract", "entry", "exit", "fid", "volume_day", "volume", "status", "sign", "base_risk",
                  "hold_days", "w", "sigma_bp", "fomc_mult", "dd_mult", "cap_mult", "dv01_target", "dv01_contract",
                  "dv01_r2", "dv01_n", "contracts_raw", "contracts", "dv01", "notional", "point_value",
                  "tick_value_entry", "tick_value_exit", "gross_pnl", "cost", "net_pnl"]
DAILY_TABLE_COLS = ["gross_pnl", "cost", "gross_notional", "traded_notional"]
DATE_COLS = ["entry", "exit", "fid", "volume_day"]


def to_tables(results: dict[str, BookResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(legs, daily) derived tables of several futures books, keyed by strategy name. No settlement levels."""
    legs, daily = [], {}
    for k, r in results.items():
        legs.append(r.legs.assign(strategy=k, cost_mult=r.cfg.cost_mult)[LEG_TABLE_COLS])
        for c in DAILY_TABLE_COLS:
            daily[f"{k}__{c}"] = r.daily[c]
    leg_t = pd.concat(legs, ignore_index=True)
    for c in DATE_COLS:
        leg_t[c] = pd.to_datetime(leg_t[c]).dt.strftime("%Y-%m-%d")
    day_t = pd.DataFrame(daily)
    day_t.index = day_t.index.strftime("%Y-%m-%d")
    return leg_t, day_t.rename_axis("date")


def from_tables(leg_t: pd.DataFrame, day_t: pd.DataFrame, rf: pd.Series, units: dict[str, str],
                base_cfg: RiskConfig | None = None) -> dict[str, BookResult]:
    """Rebuild the BookResults from the derived tables (the keyless reproduction path; module docstring)."""
    base_cfg = base_cfg or RiskConfig()
    day_t = day_t.copy()
    day_t.index = pd.DatetimeIndex(pd.to_datetime(day_t.index))
    out = {}
    for k, g in leg_t.groupby("strategy", sort=False):
        g = g.reset_index(drop=True).copy()
        for c in DATE_COLS:
            g[c] = pd.to_datetime(g[c])
        g["fallback_used"] = g["fallback_used"].astype(str).str.lower().eq("true")
        cm = float(g["cost_mult"].iloc[0])
        cfg = replace(base_cfg, cost_mult=cm)
        d = pd.DataFrame({c: day_t[f"{k}__{c}"].to_numpy(float) for c in DAILY_TABLE_COLS}, index=day_t.index)
        d = d.dropna(how="all")
        d["pnl"] = d["gross_pnl"] - d["cost"]
        d["excess"] = d["pnl"] / cfg.capital
        d["total"] = rf.reindex(d.index).to_numpy() + d["excess"]
        out[k] = BookResult(name=k, legs=g.drop(columns=["strategy", "cost_mult"]), daily=d, cfg=cfg, unit=units[k])
    return out


# ------------------------------------------------------------------------------------------------ data checks

def front_series(m: Market, root: str) -> pd.DataFrame:
    """Per bond day t: the settlement change from t-1 to t of the contract with the highest volume on t-1."""
    allc = m.contracts_of(root)
    rows = []
    for i in range(1, len(m.days)):
        d0, d1 = m.days[i - 1], m.days[i]
        v = m.volume.loc[d0, allc]
        if not (v.max() > 0):
            continue
        c = v.idxmax()
        rows.append((d1, c, m.settle.at[d1, c] - m.settle.at[d0, c]))
    return pd.DataFrame(rows, columns=["date", "contract", "dsettle"]).set_index("date")


def alignment_checks(m: Market, y10: pd.Series, dates: list[str], root: str = HEADLINE_FUTURE) -> dict:
    """Trade-date alignment of the settlements (CLAUDE.md 7.1): correlation of the front contract's settlement
    change with -dy10 at lags -1, 0, +1, and the changes around three known large-move dates."""
    f = front_series(m, root)
    dy = (y10.reindex(m.days).diff() * 100.0).reindex(f.index)
    s = f["dsettle"] * m.point_value[root]
    lag = {}
    for k in (-1, 0, 1):
        ny = -(y10.reindex(m.days).diff() * 100.0).shift(-k).reindex(f.index)
        ok = s.notna() & ny.notna()
        lag[str(k)] = float(np.corrcoef(s[ok], ny[ok])[0, 1])
    known = {}
    for d in dates:
        d = pd.Timestamp(d)
        p = int(m.days.get_loc(d))
        known[str(d.date())] = {str(m.days[p + k].date()): {"contract": str(f.at[m.days[p + k], "contract"]),
                                                            "dsettle_usd_per_contract": float(s.at[m.days[p + k]]),
                                                            "dy10_bp": float(dy.at[m.days[p + k]])}
                                for k in (-1, 0, 1)}
    ok = s.notna() & dy.notna()
    return {"root": root, "n_days": int(ok.sum()), "corr_dsettle_neg_dy10_by_lag": lag,
            "note": "lag k: corr(settlement change on t, -dy10 on t + k); k = 0 should dominate if ts_ref is the "
                    "trade date", "known_dates": known}


def settlement_gaps(m: Market, days: pd.DatetimeIndex) -> dict:
    """Bond days in `days` on which no contract of a root has a settlement (gaps in the statistics feed), by root."""
    out = {}
    for r in sorted(set(m.root)):
        cols = m.contracts_of(r)
        s = m.settle.reindex(days)[cols]
        listed = s.notna().any(axis=0)
        first = s.loc[:, listed].apply(pd.Series.first_valid_index).min() if listed.any() else None
        if first is None:
            continue
        none = s.loc[first:].notna().sum(axis=1) == 0
        out[r] = [str(d.date()) for d in none.index[none]]
    return out
