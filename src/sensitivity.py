"""Pre-registered sensitivity grid (CLAUDE.md 7.12): every cell reported, the headline cell marked.

Grid: entry T-2 ... T-6 x exit {T, T+1} x tenor {5y, 10y, 30y} x INCLUSION_RULE {auctioned, settled} x
REINVEST_COUPONS {True, False} x DEDUCT_SOMA {True, False} x cost {1x, 2x} = 480 cells. Each cell: the H1 slope
(window excess return of the cell's tenor, %, on the cell's z, Newey-West) with its 95% interval and n, and the net
Sharpe of forecast_sized and calendar_only, both cash, in-sample (1993-01 to 2024-09).

How each dimension enters (our reading, fixed before any Phase 5 result):
* entry T-k changes the index rebuild itself, which is point in time at E = T-k (src/index_rebuild.py: which
  tranches use offering_amt, which SOMA date is known), and the trade's entry. INCLUSION_RULE and DEDUCT_SOMA change
  the rebuild. REINVEST_COUPONS = False uses FDD = Ext (the same rebuild). z, w are rebuilt from each cell's series
  exactly as src/signals.py does, from 1990-01.
* exit T+1 holds one more day (DV01 sized for hold_days = k + 1, src/risk.py); a month whose exit lies after IS_END
  (2024-09 for T+1) is left out of that cell.
* tenor: the cell's constant-maturity par bond; sizing stays on the 10-year yield's sigma (src/risk.py), duration
  from the tenor's yield. A month whose tenor yield or return is missing in the window (DGS30 2002-02 to 2006-02,
  masked in src/data/fred.py) is not traded and leaves H1; such months are counted per cell.
* cost: cost_mult = 1 or COST_STRESS (2) on CASH_COST_BP; H1 does not depend on it.
Rules on (the pre-registered setting). The 20 rebuilds (5 entries x 2 rules x 2 SOMA settings) run in parallel
processes (GQH_WORKERS, default min(8, CPUs)); each is deterministic, so results do not depend on the worker count.

Speed: the cell strategies run on fast_daily(), a numpy version of src/backtest.py::run_strategy with the same
arithmetic in the same order. run_all.py checks it against run_strategy on the headline cell and one other cell on
every run and stops on any difference above 1e-12.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from multiprocessing import get_context

import numpy as np
import pandas as pd

from config.settings import COST_STRESS, ENTRY_OFFSET, EXIT_OFFSET, HEADLINE_TENOR, INCLUSION_RULE
from src.backtest import hold_days
from src.bonds import KNOT_YEARS, mod_duration
from src.risk import DrawdownRule, RiskConfig, sigma_bp, size_trade
from src.signals import past_zscore, weight, window_has
from src.stats import nw_regression, sharpe

GRID = {"entry": [2, 3, 4, 5, 6], "exit": [0, 1], "tenor": ["DGS5", "DGS10", "DGS30"],
        "inclusion_rule": ["auctioned_by_rebalance", "settled_by_month_end"], "reinvest_coupons": [True, False],
        "deduct_soma": [True, False], "cost_mult": [1.0, COST_STRESS]}
HEADLINE = {"entry": ENTRY_OFFSET, "exit": EXIT_OFFSET, "tenor": HEADLINE_TENOR, "inclusion_rule": INCLUSION_RULE,
            "reinvest_coupons": True, "deduct_soma": True, "cost_mult": 1.0}
DIMS = list(GRID)
REBUILD_COLS = ["month", "T", "E", "Ext", "FDD"]


# ------------------------------------------------------------------------------------------------ rebuilds

def _rebuild_job(key: tuple) -> tuple[tuple, pd.DataFrame]:
    from src.index_rebuild import RebuildConfig, run_default
    entry, rule, soma = key
    monthly, _ = run_default(cfg=RebuildConfig(inclusion_rule=rule, deduct_soma=soma, entry_offset=entry))
    return key, monthly[REBUILD_COLS].copy()


def n_workers() -> int:
    env = os.environ.get("GQH_WORKERS", "").strip()
    return max(int(env), 1) if env else max(min(8, os.cpu_count() or 1), 1)


def grid_rebuilds(workers: int | None = None) -> dict[tuple, pd.DataFrame]:
    """{(entry, inclusion_rule, deduct_soma): monthly rebuild columns} for every grid rebuild."""
    keys = list(product(GRID["entry"], GRID["inclusion_rule"], GRID["deduct_soma"]))
    workers = workers or n_workers()
    if workers <= 1:
        return dict(_rebuild_job(k) for k in keys)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
        return dict(ex.map(_rebuild_job, keys))


def cell_signal(monthly: pd.DataFrame, reinvest: bool) -> pd.DataFrame:
    """z and w of one rebuild, from its first month (src/signals.py rules), indexed by Period."""
    x = pd.Series((monthly["FDD"] if reinvest else monthly["Ext"]).to_numpy(float),
                  index=pd.PeriodIndex(monthly["month"], freq="M"))
    z = past_zscore(x)
    return pd.DataFrame({"z": z, "w": weight(z), "E": pd.to_datetime(monthly["E"].to_numpy())}, index=x.index)


# ------------------------------------------------------------------------------------------------ fast path

@dataclass
class Prepared:
    months: pd.PeriodIndex
    entry: pd.DatetimeIndex
    i_entry: np.ndarray
    i_exit: np.ndarray
    sigma: np.ndarray
    fomc: np.ndarray
    dur: np.ndarray
    hold: np.ndarray
    rets: list
    valid: np.ndarray
    R_pct: pd.Series


def prepare(cal, months, entry_k: int, exit_k: int, tenor: str, excess: pd.Series, y_tenor: pd.Series,
            y10: pd.Series, fomc_scheduled: pd.DatetimeIndex, days: pd.DatetimeIndex,
            cfg: RiskConfig) -> Prepared:
    """Per-window inputs that do not depend on w or costs (the work run_strategy repeats for every run)."""
    fomc = pd.DatetimeIndex(fomc_scheduled).sort_values()
    pos = pd.Series(np.arange(len(days)), index=days)
    keep, ent, ie, ix, sig, fo, du, ho, rets, ok, R = [], [], [], [], [], [], [], [], [], [], []
    for m in months:
        T = cal.month_end(m)
        try:
            e, x = cal.offset(T, -entry_k), cal.offset(T, exit_k)
        except IndexError:
            continue
        if e < days[0] or x > days[-1]:
            continue
        keep.append(m)
        ent.append(e)
        ie.append(int(pos[e]))
        ix.append(int(pos[x]))
        sig.append(sigma_bp(y10, e, cfg.vol_lookback))
        fo.append(window_has(fomc, e, x))
        y_e = float(y_tenor.loc[e])
        du.append(float(mod_duration(y_e, y_e, KNOT_YEARS[tenor])) if np.isfinite(y_e) else np.nan)
        ho.append(hold_days(cal, e, x))
        r = excess.loc[e:x].iloc[1:].to_numpy(float)
        good = bool(np.isfinite(y_e) and len(r) == ho[-1] and np.isfinite(r).all())
        ok.append(good)
        rets.append(r)
        R.append(float(np.prod(1.0 + r) - 1.0) * 100.0 if good else np.nan)
    idx = pd.PeriodIndex(keep, freq="M")
    return Prepared(months=idx, entry=pd.DatetimeIndex(ent), i_entry=np.array(ie), i_exit=np.array(ix),
                    sigma=np.array(sig), fomc=np.array(fo, bool), dur=np.array(du), hold=np.array(ho), rets=rets,
                    valid=np.array(ok, bool), R_pct=pd.Series(R, index=idx, dtype=float))


def fast_daily(prep: Prepared, w: np.ndarray, n_days: int, cfg: RiskConfig) -> np.ndarray:
    """Daily excess returns (fractions of capital) of run_strategy on the prepared windows; months marked invalid
    are not traded. w: size multiplier per prepared month."""
    pnl = np.zeros(n_days)
    dd = DrawdownRule(cfg)
    for j in range(len(prep.months)):
        if not prep.valid[j]:
            continue
        e, x = prep.i_entry[j], prep.i_exit[j]
        dd_mult = dd.decide()
        size = size_trade(prep.sigma[j], float(w[j]), bool(prep.fomc[j]), dd_mult, prep.dur[j], cfg,
                          hold_days=int(prep.hold[j]))
        gross = size["notional"] * prep.rets[j]
        cost = cfg.cost_bp * cfg.cost_mult * size["dv01"]
        pnl[e + 1:x + 1] += gross
        pnl[e] -= cost / 2.0
        pnl[x] -= cost / 2.0
        dd.update(pnl[e:x + 1] / cfg.capital)
    return pnl / cfg.capital


# ------------------------------------------------------------------------------------------------ the grid

def run_grid(rebuilds: dict, cal, months, excess: pd.DataFrame, yields: pd.DataFrame, fomc_scheduled,
             days: pd.DatetimeIndex, base_cfg: RiskConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Every cell (module docstring). excess / yields: daily excess returns and yields per FRED tenor (yields must
    include DGS10, which sizes every trade). Returns (cells, prepared) where prepared[(entry, exit, tenor)] is kept
    for the checks against run_strategy in run_all.py."""
    base_cfg = base_cfg or RiskConfig()
    cfgs = {c: RiskConfig(**{**asdict(base_cfg), "cost_mult": c}) for c in GRID["cost_mult"]}
    sigs = {(e, r, s, rv): cell_signal(rebuilds[(e, r, s)], rv)
            for (e, r, s) in rebuilds for rv in GRID["reinvest_coupons"]}
    rows, prepared = [], {}
    for entry_k, exit_k, tenor in product(GRID["entry"], GRID["exit"], GRID["tenor"]):
        prep = prepare(cal, months, entry_k, exit_k, tenor, excess[tenor], yields[tenor], yields[HEADLINE_TENOR],
                       fomc_scheduled, days, base_cfg)
        prepared[(entry_k, exit_k, tenor)] = prep
        cal_sharpe = {c: sharpe(fast_daily(prep, np.ones(len(prep.months)), len(days), cfgs[c]))
                      for c in GRID["cost_mult"]}
        for rule, soma, reinvest in product(GRID["inclusion_rule"], GRID["deduct_soma"], GRID["reinvest_coupons"]):
            sig = sigs[(entry_k, rule, soma, reinvest)]
            if not (sig.loc[prep.months, "E"].to_numpy() == prep.entry.to_numpy()).all():
                raise RuntimeError(f"grid: rebuild E differs from the window entry (T-{entry_k})")
            z = sig.loc[prep.months, "z"]
            if z.isna().any():
                raise RuntimeError("grid: z undefined inside the in-sample period")
            df = pd.concat([prep.R_pct.rename("R"), z.rename("z")], axis=1).dropna()
            reg = nw_regression(df["R"], {"z": df["z"]})["params"]["z"]
            w = sig.loc[prep.months, "w"].to_numpy(float)
            for c in GRID["cost_mult"]:
                s_fc = sharpe(fast_daily(prep, w, len(days), cfgs[c]))
                rows.append({"entry": entry_k, "exit": exit_k, "tenor": tenor, "inclusion_rule": rule,
                             "reinvest_coupons": reinvest, "deduct_soma": soma, "cost_mult": c, "n": int(len(df)),
                             "n_months": int(len(prep.months)), "n_not_traded": int((~prep.valid).sum()),
                             "H1_b": reg["b"], "H1_lo": reg["ci"][0], "H1_hi": reg["ci"][1], "H1_t": reg["t"],
                             "sharpe_fc": s_fc, "sharpe_cal": cal_sharpe[c], "sharpe_diff": s_fc - cal_sharpe[c]})
    cells = pd.DataFrame(rows)
    cells["headline"] = np.logical_and.reduce([cells[k] == v for k, v in HEADLINE.items()])
    if cells["headline"].sum() != 1:
        raise RuntimeError("grid: the headline cell is not exactly one cell")
    return cells.sort_values(DIMS, kind="mergesort").reset_index(drop=True), prepared


def summary(cells: pd.DataFrame) -> dict:
    """Shares and medians over all cells, by dimension, and the headline cell (descriptive; every cell is in the
    table)."""
    def agg(df: pd.DataFrame) -> dict:
        return {"n_cells": int(len(df)), "share_H1_b_gt_0": float((df["H1_b"] > 0).mean()),
                "share_H1_lo_gt_0": float((df["H1_lo"] > 0).mean()), "median_H1_b": float(df["H1_b"].median()),
                "share_fc_beats_cal": float((df["sharpe_diff"] > 0).mean()),
                "median_sharpe_fc": float(df["sharpe_fc"].median()),
                "median_sharpe_cal": float(df["sharpe_cal"].median()),
                "share_sharpe_cal_gt_0": float((df["sharpe_cal"] > 0).mean()),
                "median_sharpe_diff": float(df["sharpe_diff"].median())}
    out = {"all": agg(cells), "by_dimension": {}}
    for d in DIMS:
        out["by_dimension"][d] = {str(v): agg(g) for v, g in cells.groupby(d, sort=True)}
    out["headline_cell"] = {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in cells.loc[cells["headline"]].iloc[0].to_dict().items()}
    return out


def to_markdown(cells: pd.DataFrame) -> str:
    """Every cell, the headline marked (CLAUDE.md 7.12: reported in full)."""
    lines = ["# Sensitivity grid (CLAUDE.md 7.12), in-sample 1993-01 to 2024-09, cash, rules on", "",
             "H1: window excess return of the cell's tenor (%) on the cell's z, Newey-West 95% interval. Sharpe: net "
             "of costs, daily, annualized. **>>** marks the pre-registered headline cell.", "",
             "| | entry | exit | tenor | inclusion | reinvest | SOMA | cost | n | H1 b | 95% CI | Sharpe fc | "
             "Sharpe cal | fc - cal |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cells.iterrows():
        lines.append(f"| {'**>>**' if r['headline'] else ''} | T-{r['entry']} | T+{r['exit']} | {r['tenor']} | "
                     f"{'auctioned' if r['inclusion_rule'] == 'auctioned_by_rebalance' else 'settled'} | "
                     f"{r['reinvest_coupons']} | {r['deduct_soma']} | {r['cost_mult']:g}x | {r['n']} | "
                     f"{r['H1_b']:.4f} | [{r['H1_lo']:.4f}, {r['H1_hi']:.4f}] | {r['sharpe_fc']:.3f} | "
                     f"{r['sharpe_cal']:.3f} | {r['sharpe_diff']:+.3f} |")
    return "\n".join(lines) + "\n"
