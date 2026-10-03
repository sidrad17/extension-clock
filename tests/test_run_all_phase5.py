"""Smoke test of run_all.month_end_phase5 on synthetic data: every loader is replaced, so no snapshot return is
computed. It checks that the orchestration runs end to end, logs its trials, passes its engine checks and fills
every results block."""
import numpy as np
import pandas as pd
import pytest

import run_all
from src import report
from src.backtest import month_end_windows, run_strategy, window_returns, window_yield_change_bp
from src.calendar import BondCalendar
from src.index_rebuild import BUCKETS
from src.risk import RiskConfig
from src.sensitivity import GRID
from src.signals import build_signals

DAYS = pd.bdate_range("1995-01-02", "2004-06-30")
CAL = BondCalendar(DAYS)
RNG = np.random.default_rng(4)
SER = ["DGS2", "DGS5", "DGS10", "DGS20", "DGS30"]
YLD = pd.DataFrame({t: lvl + np.cumsum(RNG.normal(0, 0.04, len(DAYS))) for t, lvl in
                    zip(SER, [4.0, 4.5, 5.0, 5.3, 5.5])}, index=DAYS)
YLD.loc["2001-03-01":"2001-05-31", "DGS30"] = np.nan                   # a gap, like DGS30 2002-2006
EXC = pd.DataFrame({t: RNG.normal(0.0001, 0.003, len(DAYS)) for t in SER}, index=DAYS)
EXC.loc["2001-03-01":"2001-06-01", "DGS30"] = np.nan
RF = pd.Series(0.0001, index=DAYS)
MONTHS = pd.period_range("1995-01", "2004-05", freq="M")
IS_START, IS_END = "1998-01-01", "2004-05-31"


def monthly(entry: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(entry)
    rows = []
    for m in MONTHS:
        T = CAL.month_end(m)
        r = {"month": str(m), "T": T.date().isoformat(), "E": CAL.offset(T, -entry).date().isoformat()}
        ext = {b: rng.normal(0.01, 0.01) for b in BUCKETS}
        cash = {b: abs(rng.normal(0.004, 0.002)) for b in BUCKETS}
        r.update({f"ext_{b}": ext[b] for b in BUCKETS})
        r.update({f"cash_{b}": cash[b] for b in BUCKETS})
        r.update({f"fdd_{b}": ext[b] + cash[b] for b in BUCKETS})
        r["Ext"] = sum(ext.values())
        r["FDD"] = r["Ext"] + sum(cash.values())
        r["c_m"], r["D_next"] = abs(rng.normal(0.004, 0.001)), 5.0
        rows.append(r)
    return pd.DataFrame(rows)


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "OUTPUTS", tmp_path)
    monkeypatch.setattr(run_all, "FIG_DIR", tmp_path / "figures")
    (tmp_path / "figures").mkdir()
    (tmp_path / "tables").mkdir()
    monkeypatch.setattr(run_all, "IS_START", IS_START)
    monkeypatch.setattr(run_all, "IS_END", IS_END)
    monkeypatch.setattr(run_all, "POSTPUB_START", "2002-01-01")
    monkeypatch.setattr(run_all, "load_frame", lambda cols, **kw: YLD.reindex(columns=cols))
    weeks = pd.date_range("1995-01-04", "2004-06-30", freq="W-WED")
    monkeypatch.setattr(run_all, "load_volume", lambda **kw: pd.Series(1e11, index=weeks))
    fomc = pd.DatetimeIndex(["1998-03-31", "1999-06-30", "2001-01-31", "2003-12-10"])
    monkeypatch.setattr(run_all, "load_fomc_dates", lambda **kw: fomc)
    keys = [(e, r, s) for e in GRID["entry"] for r in GRID["inclusion_rule"] for s in GRID["deduct_soma"]]
    base = {e: monthly(e) for e in GRID["entry"]}
    monkeypatch.setattr(run_all, "grid_rebuilds", lambda: {k: base[k[0]] for k in keys})
    logged = []
    monkeypatch.setattr(run_all, "log_trials", lambda rows: logged.extend(rows) or len(rows))
    return fomc, logged


def test_month_end_phase5_runs_end_to_end(patched):
    fomc, logged = patched
    mon = monthly(4)
    pension = pd.Series(RNG.normal(size=len(MONTHS)), index=MONTHS)
    sig = build_signals(mon, fomc, pension)
    is_months = sig.index[(sig.index >= pd.Period(IS_START, "M")) & (sig.index <= pd.Period(IS_END, "M"))]
    s_is = sig.loc[is_months]
    win = month_end_windows(CAL, is_months)
    x10, y10 = EXC["DGS10"], YLD["DGS10"]
    days = DAYS[(DAYS >= IS_START) & (DAYS <= IS_END)]
    common = dict(daily_excess=x10, rf=RF, y_tenor=y10, y10=y10, tenor_years=10.0, fomc_scheduled=fomc, cal=CAL,
                  days=days, cfg=RiskConfig())
    strat = {"forecast_sized": run_strategy("fs", win, s_is["w"], **common),
             "calendar_only": run_strategy("co", win, pd.Series(1.0, index=is_months), **common)}
    R = window_returns(x10, win) * 100.0
    Y = -window_yield_change_bp(y10, win)
    res_h1 = {"b": float(np.polyfit(s_is["z"], R, 1)[0]), "ci": [0.0, 0.0], "n": len(R)}
    eq_ex = pd.Series(RNG.normal(0, 0.01, len(DAYS)), index=DAYS)
    out = run_all.month_end_phase5(cal=CAL, monthly=mon, sig=sig, s_is=s_is, is_months=is_months, win=win, R=R,
                                   Y=Y, x10=x10, rf=RF, y10=y10, fomc=fomc, days=days, common=common, strat=strat,
                                   res_h1=res_h1, total=EXC + RF.to_numpy()[:, None], excess=EXC, eq_ex=eq_ex,
                                   base_cfg={"strategy": "cash"}, git={"commit": "x", "dirty": False}, t0=0.0)
    for k in ("H2", "H3", "H5", "curve_allocated", "cost_stress", "risk_rules_on_off", "sensitivity", "betas",
              "crowding", "tails", "capacity", "post_publication"):
        assert out[k], k
    assert len(out["sensitivity"]["cells"]) == 480
    assert max(out["sensitivity"]["engine_check_max_abs_diff"].values()) <= 1e-12
    assert out["curve_allocated"]["n_months_bucket_excluded"] >= 1                  # the DGS30 gap
    assert set(out["risk_rules_on_off"]) == {"all_on", "fomc_off", "drawdown_off", "cap_off", "all_off"}
    assert set(out["figures"]) == {"capacity", "extension_series", "curve_map"}
    windows = [r["window"] for r in logged]
    assert windows.count("in_sample_grid") == 480 and len(logged) == 480 + 8
    assert {"in_sample_cost2x", "post_publication", "in_sample_curve_allocated"} <= set(windows)
    rc = report.clean({"x": out["sensitivity"], "y": out["capacity"], "z": out["tails"]})
    assert rc["x"]["summary"]["all"]["n_cells"] == 480
