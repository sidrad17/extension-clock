"""Phase 5 tests on synthetic data: trial log rule, Deflated Sharpe, addendum (c), H2, H3, H5, the curve-allocated
trade, the sensitivity fast path, capacity, dealer volume, crowding, betas, period tables and Flow Clock by tenor."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import src.trial_log as tl
from src.backtest import month_end_windows, reversal_windows, run_curve_allocated, run_strategy
from src.calendar import BondCalendar
from src.capacity import capacity_curve
from src.data.pd_volume import adv_known_at, parse
from src.flowclock import by_tenor, run_book, supply_legs
from src.metrics import (betas, crowding, equity_excess_on_bond_days, slice_metrics, tails, yearly_table)
from src.risk import RiskConfig
from src.sensitivity import HEADLINE, cell_signal, fast_daily, prepare, summary, to_markdown
from src.stats import deflated_sharpe, expected_max_sharpe, probabilistic_sharpe, sharpe
from src.tests_h import h1_addendum, h2, h2_panel, h3, h5

DAYS = pd.bdate_range("2017-01-02", "2020-12-31")
CAL = BondCalendar(DAYS)
RNG = np.random.default_rng(21)
TEN = {"1-3y": "DGS2", "3-7y": "DGS5", "7-10y": "DGS10", "10-20y": "DGS20", "20y+": "DGS30"}
Y = pd.DataFrame({t: lvl + np.cumsum(RNG.normal(0, 0.05, len(DAYS))) for t, lvl in
                  [("DGS2", 1.5), ("DGS5", 2.0), ("DGS10", 2.5), ("DGS20", 2.8), ("DGS30", 3.0)]}, index=DAYS)
X = pd.DataFrame({t: RNG.normal(0.0001, 0.003, len(DAYS)) for t in Y.columns}, index=DAYS)
RF = pd.Series(0.00005, index=DAYS)
MONTHS = pd.period_range("2017-06", "2020-11", freq="M")
NAV = DAYS[(DAYS >= "2017-06-01") & (DAYS <= "2020-11-30")]
FOMC = pd.DatetimeIndex(["2017-06-14", "2017-12-13", "2018-03-21", "2018-09-26", "2019-07-31", "2020-01-29"])


# ------------------------------------------------------------------------------------------------ trial log

def test_trial_log_appends_only_in_dev_mode(monkeypatch, tmp_path):
    path = tmp_path / "trials.csv"
    git = {"commit": "abc", "dirty": False}
    monkeypatch.setattr(tl, "dev_mode", lambda: False)
    row = tl.log_trial({"a": 1}, "in_sample", {"strategy": "cash", "sharpe_fc": 0.5}, git=git, path=path)
    assert row["config_hash"] and not path.exists() and tl.trial_count(path) == 0
    monkeypatch.setattr(tl, "dev_mode", lambda: True)
    tl.log_trial({"a": 1}, "in_sample", {"strategy": "cash", "sharpe_fc": 0.5}, git=git, path=path)
    rows = [tl.trial_row({"b": i}, "grid", {"sharpe_cal": 0.1 * i}, git=git) for i in range(3)]
    assert tl.log_trials(rows, path=path) == 3
    assert tl.trial_count(path) == 4
    df = tl.read_trials(path)
    assert df["sharpe_fc"].notna().sum() == 1 and df["sharpe_cal"].notna().sum() == 3
    assert list(df.columns) == tl.TRIAL_COLUMNS


# ------------------------------------------------------------------------------------------------ Deflated Sharpe

def test_deflated_sharpe_matches_bailey_lopez_de_prado_example():
    # their numerical example: SR 2.5 annualized over 1,250 days, N = 100, V = 1/2 (annualized), skew -3, kurt 10
    sr0 = expected_max_sharpe(0.5 / 250, 100)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    dsr, _ = probabilistic_sharpe(2.5 / np.sqrt(250), sr0, 1250, -3.0, 10.0)
    assert dsr == pytest.approx(0.9004, abs=1e-4)


def test_deflated_sharpe_wrapper():
    r = np.random.default_rng(3).normal(0.0005, 0.01, 2000)
    one = deflated_sharpe(r, [0.4, 0.6, 0.5], n_trials=1)
    assert one["sr0_ann"] == 0.0 and one["dsr"] == pytest.approx(one["psr_vs_zero"])
    many = deflated_sharpe(r, np.linspace(-1, 1, 50), n_trials=500)
    assert many["sr0_ann"] > 0 and many["dsr"] < one["dsr"]
    assert many["sharpe_ann"] == pytest.approx(sharpe(r))


# ------------------------------------------------------------------------------------------------ tests

def _monthly(n=240, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    return idx, rng


def test_surprise_and_h3_recover_slopes():
    idx, rng = _monthly(3000)
    z = pd.Series(rng.normal(size=len(idx)), index=idx)
    zs = pd.Series(rng.normal(size=len(idx)), index=idx)
    R = 0.2 + 0.05 * zs + pd.Series(rng.normal(0, 0.3, len(idx)), index=idx)
    ref = pd.Series(idx.month.isin([2, 5, 8, 11]).astype(int), index=idx)
    a = h1_addendum(R, -R * 10, z, ref, zs)
    assert a["surprise"]["ci"][0] < 0.05 < a["surprise"]["ci"][1] and a["surprise"]["n"] == len(idx)
    assert h1_addendum(R, -R * 10, z, ref)["surprise"] == {}
    R3 = -0.05 - 0.04 * z + pd.Series(rng.normal(0, 0.2, len(idx)), index=idx)
    r = h3(R3, -R3 * 10, z, R)
    assert r["ci"][0] < -0.04 < r["ci"][1] and r["mean_R3"]["b"] < 0
    assert r["reversal_share_of_run_up"] == pytest.approx(R3.mean() / R.mean())
    assert r["terciles"]["high"]["mean"] < r["terciles"]["low"]["mean"]


def test_reversal_windows_first_three_days_and_drop_at_end():
    w = reversal_windows(CAL, pd.period_range("2020-10", "2020-12", freq="M"), 3)
    assert w.loc[pd.Period("2020-10", "M"), "entry"] == pd.Timestamp("2020-10-30")
    assert w.loc[pd.Period("2020-10", "M"), "exit"] == pd.Timestamp("2020-11-04")
    assert pd.Period("2020-12", "M") not in w.index                  # T + 3 beyond the calendar


def test_h5_matches_ols_point_estimates():
    idx, rng = _monthly(400, 2)
    sig = pd.DataFrame({"z": rng.normal(size=400), "z_pension": rng.normal(size=400),
                        "quarter_end": idx.month.isin([3, 6, 9, 12]).astype(int), "year_end": (idx.month == 12) * 1,
                        "ref": idx.month.isin([2, 5, 8, 11]).astype(int), "fomc": rng.integers(0, 2, 400)}, index=idx)
    R = 0.1 + 0.03 * sig["z_pension"] + pd.Series(rng.normal(0, 0.3, 400), index=idx)
    r = h5(R, -R, sig)
    ref = sm.OLS(R.to_numpy(), sm.add_constant(sig.to_numpy(float))).fit()
    assert [r["coefs"][k]["b"] for k in ["const", "z", "z_pension", "quarter_end", "year_end", "ref", "fomc"]] == \
        pytest.approx(list(ref.params))
    assert r["n"] == 400 and r["r2"] == pytest.approx(ref.rsquared)


def test_h2_absorbs_month_effects_and_matches_dummy_regression():
    idx, rng = _monthly(300, 7)
    b = list(TEN)
    x = pd.DataFrame(rng.normal(size=(300, 5)), index=idx, columns=b)
    month_fx = rng.normal(0, 5, 300)[:, None]                         # common shock: absorbed by month effects
    y = pd.DataFrame(month_fx + 0.7 * x.to_numpy() + rng.normal(0, 1, (300, 5)), index=idx, columns=b)
    y.iloc[:10, 4] = np.nan                                           # a missing bucket leaves those months' panel
    r = h2(y, x, x * 2.0)
    assert r["ci"][0] < 0.7 < r["ci"][1]
    assert r["n_obs"] == 1490 and r["n_months"] == 300 and r["n_by_bucket"]["20y+"] == 290
    assert r["ext_only"]["coef"] == pytest.approx(r["coef"] / 2.0)
    long = h2_panel(y, x)
    dummies = pd.get_dummies(long["month"], dtype=float)
    ref = sm.OLS(long["y"].to_numpy(), np.column_stack([long["x"].to_numpy(), dummies.to_numpy()])).fit()
    assert r["coef"] == pytest.approx(ref.params[0])
    assert np.allclose(long.groupby("month")["yd"].mean(), 0.0)


# ------------------------------------------------------------------------------------------------ curve trade

def _curve(demand, cfg=None, yields=Y, excess=X):
    win = month_end_windows(CAL, MONTHS)
    return run_curve_allocated("c", win, demand, TEN, excess, yields, RF, Y["DGS10"], FOMC, CAL, NAV,
                               cfg or RiskConfig())


def test_curve_allocated_with_one_bucket_is_calendar_only():
    win = month_end_windows(CAL, MONTHS)
    d = pd.DataFrame(0.0, index=MONTHS, columns=list(TEN))
    d["7-10y"] = 0.3
    X10 = X.copy()
    X10["DGS10"] = X["DGS10"] * 4.0 - 0.003                          # losses so the drawdown rule fires
    res = _curve(d, excess=X10)
    ref = run_strategy("cal", win, pd.Series(1.0, index=MONTHS), X10["DGS10"], RF, Y["DGS10"], Y["DGS10"], 10.0,
                       FOMC, CAL, NAV, RiskConfig())
    assert np.array_equal(res.daily["excess"].to_numpy(), ref.daily["excess"].to_numpy())
    assert (ref.trades["dd_mult"] < 1).any()
    assert res.trades["dd_mult"].tolist() == ref.trades["dd_mult"].tolist()


def test_curve_allocated_weights_flat_months_and_missing_buckets():
    d = pd.DataFrame(RNG.normal(0.01, 0.02, (len(MONTHS), 5)), index=MONTHS, columns=list(TEN))
    d.iloc[3] = -0.01                                                  # every bucket <= 0: no position
    y = Y.copy()
    m = MONTHS[6]
    T = CAL.month_end(m)
    y.loc[CAL.offset(T, -4), "DGS30"] = np.nan                        # 20y+ yield missing at entry that month
    res = _curve(d, cfg=RiskConfig(fomc_half=False, dd_rule=False), yields=y)
    assert res.flat_months == [MONTHS[3]] and MONTHS[3] not in res.trades.index
    assert [e["month"] for e in res.excluded] == [m] and res.excluded[0]["buckets"] == ["20y+"]
    tr = res.trades.loc[m]
    pos = d.loc[m].drop("20y+").clip(lower=0)
    for b in ["1-3y", "3-7y", "7-10y", "10-20y"]:
        assert tr[f"a_{b}"] == pytest.approx(pos[b] / pos.sum())
    assert tr["a_20y+"] == 0.0
    other = res.trades.drop(index=m)
    assert np.allclose(other[[f"a_{b}" for b in TEN]].sum(axis=1), 1.0)


def test_curve_allocated_notional_cap_scales_all_buckets():
    d = pd.DataFrame(0.0, index=MONTHS, columns=list(TEN))
    d["1-3y"] = 1.0                                                    # all risk in the 2-year: large notional
    res = _curve(d, cfg=RiskConfig(fomc_half=False, dd_rule=False, notional_cap=0.5))
    assert (res.trades["capped"] == 1).all()
    assert np.allclose(res.trades["notional"], 0.5 * 1e7)


# ------------------------------------------------------------------------------------------------ sensitivity

def test_fast_daily_equals_run_strategy_and_skips_missing_months():
    cfg = RiskConfig(cost_mult=2.0)
    X10 = X.copy()
    X10["DGS10"] = X["DGS10"] * 4.0 - 0.003
    for k, x in ((4, 0), (2, 1)):
        prep = prepare(CAL, MONTHS, k, x, "DGS10", X10["DGS10"], Y["DGS10"], Y["DGS10"], FOMC, NAV, cfg)
        w = np.clip(1 + RNG.normal(size=len(prep.months)), 0, 2)
        win = month_end_windows(CAL, prep.months, k, x)
        ref = run_strategy("r", win, pd.Series(w, index=prep.months), X10["DGS10"], RF, Y["DGS10"], Y["DGS10"],
                           10.0, FOMC, CAL, NAV, cfg)
        assert np.array_equal(fast_daily(prep, w, len(NAV), cfg), ref.daily["excess"].to_numpy())
        assert prep.R_pct.notna().all()
    x30 = X["DGS30"].copy()
    m = MONTHS[5]
    x30.loc[CAL.month_end(m)] = np.nan
    prep = prepare(CAL, MONTHS, 4, 0, "DGS30", x30, Y["DGS30"], Y["DGS10"], FOMC, NAV, cfg)
    j = list(prep.months).index(m)
    assert not prep.valid[j] and prep.valid.sum() == len(MONTHS) - 1 and np.isnan(prep.R_pct.loc[m])
    out = fast_daily(prep, np.ones(len(prep.months)), len(NAV), cfg)
    e, t = prep.i_entry[j], prep.i_exit[j]
    assert (out[e:t + 1] == 0).all()


def test_cell_signal_summary_and_markdown():
    mon = pd.DataFrame({"month": [str(p) for p in pd.period_range("1990-01", periods=60, freq="M")],
                        "E": pd.bdate_range("1990-01-25", periods=60, freq="BME").strftime("%Y-%m-%d"),
                        "Ext": RNG.normal(0.05, 0.02, 60), "FDD": RNG.normal(0.07, 0.02, 60)})
    s_true, s_false = cell_signal(mon, True), cell_signal(mon, False)
    assert s_true["z"].iloc[:36].isna().all() and s_true["z"].iloc[36:].notna().all()
    assert (s_true["w"].iloc[:36] == 1).all() and not np.allclose(s_true["z"].iloc[36:], s_false["z"].iloc[36:])
    rows = []
    for e in (2, 4):
        for c in (1.0, 2.0):
            rows.append({**HEADLINE, "entry": e, "cost_mult": c, "n": 10, "H1_b": 0.1 * e - 0.25, "H1_lo": -0.5,
                         "H1_hi": 0.5, "sharpe_fc": 0.5, "sharpe_cal": 0.6, "sharpe_diff": -0.1})
    cells = pd.DataFrame(rows)
    cells["headline"] = (cells["entry"] == 4) & (cells["cost_mult"] == 1.0)
    sm_ = summary(cells)
    assert sm_["all"]["n_cells"] == 4 and sm_["all"]["share_H1_b_gt_0"] == 0.5
    assert sm_["by_dimension"]["entry"]["4"]["share_H1_b_gt_0"] == 1.0
    assert sm_["headline_cell"]["entry"] == 4
    md = to_markdown(cells)
    assert md.count("**>>**") == 2 and md.count("\n| ") == 1 + 4     # header + 4 rows (one marked)


# ------------------------------------------------------------------------------------------------ capacity

def test_adv_known_at_uses_released_weeks_only():
    weeks = pd.date_range("2020-01-01", periods=8, freq="W-WED")
    weekly = pd.Series(np.arange(1, 9) * 1e9, index=weeks)
    adv = adv_known_at(pd.DatetimeIndex(["2020-01-30", "2020-02-06", "2020-02-07"]), weekly)
    # 2020-01-30: weeks released before it are 01-01, 01-08, 01-15 (+8 days < 01-30), only 3 -> NaN
    assert np.isnan(adv.iloc[0])
    assert adv.iloc[1] == pytest.approx(np.mean([1, 2, 3, 4]) * 1e9)    # 01-22 week released 01-30 < 02-06
    assert adv.iloc[2] == pytest.approx(np.mean([2, 3, 4, 5]) * 1e9)    # 01-29 week released 02-06 < 02-07
    raw = pd.DataFrame({"asofdate": ["2013-03-27", "2013-04-03"], "keyid": ["PDSUSGCS611OT", "PDTRGSC-G7L11"],
                        "value": ["121731", "75101"]})
    s = parse(raw)
    assert s.tolist() == [121731e6, 75101e6]


def test_capacity_small_capital_matches_base_and_cap_binds():
    win = month_end_windows(CAL, MONTHS)
    res = run_strategy("c", win, pd.Series(1.0, index=MONTHS), X["DGS10"] + 0.0004, RF, Y["DGS10"], Y["DGS10"],
                       10.0, FOMC, CAL, NAV, RiskConfig())
    adv = pd.Series(1e11, index=pd.DatetimeIndex(res.trades["entry"]))
    c = capacity_curve(res, adv, capitals=[1e3, 1e7, 1e9, 1e10, 1e11])
    g = pd.DataFrame(c["grid"])
    assert g["sharpe"].iloc[0] == pytest.approx(c["sharpe_no_impact_same_months"], abs=1e-3)
    assert (np.diff(g["sharpe"]) <= 1e-12).all()                         # flat once every window is capped
    assert g["share_windows_capped"].iloc[-1] == 1.0 and g["max_participation"].iloc[-1] == pytest.approx(0.05)
    tr = res.trades.iloc[0]
    K = 1e9
    q = tr["notional"] * K / 1e7
    imp = tr["sigma_bp"] * tr["duration"] * 1e-4 * np.sqrt(q / 1e11) * q
    assert g["impact_pct_per_year"].iloc[2] > 0
    years = ((NAV[-1] - res.trades["entry"].min()).days + 1) / 365.25
    assert imp / K * 2 / years * 100 < g["impact_pct_per_year"].iloc[2]
    assert c["capital_where_sharpe_halves"] is None or c["capital_where_sharpe_halves"] > 1e3


# ------------------------------------------------------------------------------------------------ metrics

def test_crowding_ratio_of_means():
    idx = pd.PeriodIndex(["2019-01", "2019-02", "2020-01"], freq="M")
    paths = pd.DataFrame({-10: [0.0, 0.0, 0.0], -4: [0.1, 0.3, -0.1], 0: [0.4, 0.4, 0.2]}, index=idx)
    terc = pd.Series(["low", "high", "high"], index=idx)
    c = crowding(paths, terc)
    assert c["by_year"]["2019"]["share_before_entry"] == pytest.approx(0.2 / 0.4)
    assert c["by_tercile"]["high"]["share_before_entry"] == pytest.approx(0.1 / 0.3)
    assert c["all"]["n"] == 3


def test_equity_excess_compounds_across_bond_holidays():
    bond = pd.DatetimeIndex(["2020-10-08", "2020-10-09", "2020-10-13"])    # 10-12 Columbus Day: bond closed
    eq = pd.Series([1.0, 2.0, 1.0, -1.0], index=pd.DatetimeIndex(["2020-10-08", "2020-10-09", "2020-10-12",
                                                                     "2020-10-13"]))
    rf = pd.Series(0.0001, index=bond)
    ex = equity_excess_on_bond_days(eq, bond, rf)
    assert np.isnan(ex.iloc[0])
    assert ex.iloc[1] == pytest.approx(0.02 - 0.0001)
    assert ex.iloc[2] == pytest.approx(1.01 * 0.99 - 1 - 0.0001)


def test_betas_recover_known_loadings():
    rng = np.random.default_rng(8)
    d = pd.bdate_range("2000-01-03", periods=3000)
    bx, ex = pd.Series(rng.normal(0, 0.004, 3000), index=d), pd.Series(rng.normal(0, 0.01, 3000), index=d)
    s = 0.0001 + 0.5 * bx - 0.1 * ex + pd.Series(rng.normal(0, 0.001, 3000), index=d)
    b = betas(s, bx, ex)
    assert b["beta_bond_10y"] == pytest.approx(0.5, abs=0.02) and b["beta_equity"] == pytest.approx(-0.1, abs=0.01)
    assert b["n_days"] == 3000 and b["alpha_pct_per_year"] == pytest.approx(0.0001 * 252 * 100, rel=0.5)


def test_yearly_table_and_slice_metrics():
    d = pd.bdate_range("2019-01-01", "2020-12-31")
    r = pd.Series(0.0, index=d)
    r.loc["2019-03-01"], r.loc["2019-03-04"], r.loc["2020-06-01"] = 0.10, -0.20, 0.05
    t = yearly_table(r)
    assert t.loc[2019, "excess_return_pct"] == pytest.approx((1.1 * 0.8 - 1) * 100)
    assert t.loc[2019, "max_drawdown_in_year_pct"] == pytest.approx(20.0)
    assert t.loc[2020, "max_drawdown_in_year_pct"] == pytest.approx(0.0)
    assert t.loc[2020, "max_drawdown_from_peak_pct"] == pytest.approx((1 - 0.88 / 1.1) * 100)
    assert t.loc[2019, "worst_month"] == "2019-03"
    sm_ = slice_metrics(pd.DataFrame({"excess": r, "total": r}), "2020-01-01", "2020-12-31")
    assert sm_["max_drawdown_excess_pct"] == 0.0 and sm_["period"][0] == "2020-01-01"


def test_tails_lists_worst_windows_with_causes():
    win = month_end_windows(CAL, MONTHS)
    res = run_strategy("c", win, pd.Series(1.0, index=MONTHS), X["DGS10"], RF, Y["DGS10"], Y["DGS10"], 10.0, FOMC,
                       CAL, NAV, RiskConfig())
    sig = pd.DataFrame({"z": 0.0, "w": 1.0, "ref": 0, "quarter_end": 0}, index=MONTHS)
    t = tails(res, Y["DGS10"], sig, FOMC.union(pd.DatetimeIndex(["2020-03-03"])), FOMC, n=3)
    assert len(t) == 3 and t[0]["net_pnl_pct"] <= t[1]["net_pnl_pct"] <= t[2]["net_pnl_pct"]
    assert t[0]["net_pnl_pct"] == pytest.approx(res.trades["net_pnl"].min() / 1e7 * 100)
    assert "10-year yield" in t[0]["causes"]


def test_flowclock_by_tenor_adds_up():
    rows = []
    for i, (t, A) in enumerate([("DGS2", "2019-09-18"), ("DGS5", "2019-10-16"), ("DGS2", "2019-11-13")]):
        A = pd.Timestamp(A)
        rows.append({"event_id": f"e{i}", "tenor": t, "A": A, "pre_entry": CAL.offset(A, -5),
                     "post_exit": CAL.offset(A, 5), "w_pre": 1.0, "w_post": 1.0})
    res = run_book("s", supply_legs(pd.DataFrame(rows), False), X, Y, RF, pd.DatetimeIndex([]), CAL, NAV,
                   RiskConfig(fomc_half=False, dd_rule=False))
    bt = by_tenor(res, 2.0)
    assert list(bt) == ["DGS2", "DGS5"] and bt["DGS2"]["n_events"] == 2
    total = sum(v["net_pnl_pct_per_year"] for v in bt.values()) * 2.0 / 100 * 1e7
    assert total == pytest.approx(res.legs["net_pnl"].sum())
    assert bt["DGS2"]["pre_net_pnl_pct_per_year"] + bt["DGS2"]["post_net_pnl_pct_per_year"] == pytest.approx(
        bt["DGS2"]["net_pnl_pct_per_year"])
