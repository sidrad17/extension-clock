"""Tests for src/flowclock.py on synthetic data: window returns, leg sizing known at entry, cost netting by tenor,
risk rules, the demand leg reproducing src/backtest.py, no lookahead, H6 and the event-path helpers."""
import numpy as np
import pandas as pd
import pytest

from config.flowclock import LEG_RISK
from src.backtest import month_end_windows, run_strategy, window_return
from src.bonds import mod_duration
from src.calendar import BondCalendar
from src.flowclock import (auction_event_paths, book_metrics, demand_legs, event_returns, h6, path_summary, run_book,
                           supply_legs, tercile_by_rank)
from src.risk import RiskConfig, sigma_bp

DAYS = pd.bdate_range("2019-01-02", "2020-12-31")
CAL = BondCalendar(DAYS)
RNG = np.random.default_rng(11)
Y = pd.DataFrame({t: lvl + np.cumsum(RNG.normal(0, 0.05, len(DAYS))) for t, lvl in
                  [("DGS2", 1.5), ("DGS5", 2.0), ("DGS10", 2.5)]}, index=DAYS)
X = pd.DataFrame({t: RNG.normal(0.0001, 0.003, len(DAYS)) for t in Y.columns}, index=DAYS)
RF = pd.Series(0.00005, index=DAYS)
NAV = DAYS[(DAYS >= "2019-06-03") & (DAYS <= "2020-11-30")]
OFF = RiskConfig(fomc_half=False, dd_rule=False)


def events(spec):
    """spec: list of (event_id, tenor, A, w_pre, w_post)."""
    rows = []
    for eid, t, A, wp, wq in spec:
        A = pd.Timestamp(A)
        rows.append({"event_id": eid, "tenor": t, "A": A, "pre_entry": CAL.offset(A, -5),
                     "post_exit": CAL.offset(A, 5), "w_pre": wp, "w_post": wq, "zS_pre": wp - 1.0,
                     "week": A.to_period("W-SUN").strftime("%Y-%m-%d"), "skipped": False})
    return pd.DataFrame(rows)


def book(legs, cfg=OFF, **kw):
    return run_book("t", legs, X, Y, RF, kw.pop("fomc", pd.DatetimeIndex([])), CAL, NAV, cfg, **kw)


def test_event_returns_windows():
    ev = events([("e1", "DGS5", "2019-09-18", 1, 1)])
    r = event_returns(ev, X, Y).iloc[0]
    A, e, x = pd.Timestamp("2019-09-18"), CAL.offset(pd.Timestamp("2019-09-18"), -5), CAL.offset(pd.Timestamp("2019-09-18"), 5)
    assert r["R_pre"] == pytest.approx(window_return(X["DGS5"], e, A) * 100)
    assert r["R_post"] == pytest.approx((np.prod(1 + X["DGS5"].loc[A:x].iloc[1:]) - 1) * 100)
    assert r["LS"] == pytest.approx(r["R_post"] - r["R_pre"])
    assert r["dy_pre_bp"] == pytest.approx((Y.at[A, "DGS5"] - Y.at[e, "DGS5"]) * 100)
    assert r["LS_bp"] == pytest.approx(r["dy_pre_bp"] - r["dy_post_bp"])


def test_supply_legs_sizing_sign_and_pnl():
    ev = events([("e1", "DGS2", "2019-09-18", 1.4, 0.6)])
    for weighted, (wp, wq) in ((False, (1.0, 1.0)), (True, (1.4, 0.6))):
        res = book(supply_legs(ev, weighted))
        pre, post = res.legs.set_index("kind").loc["pre"], res.legs.set_index("kind").loc["post"]
        A = pd.Timestamp("2019-09-18")
        e = CAL.offset(A, -5)
        assert pre["sign"] == -1 and post["sign"] == 1 and pre["exit"] == post["entry"] == A
        s_pre = sigma_bp(Y["DGS2"], e)                       # tenor's own yield, 60 changes ending e - 1
        assert pre["dv01"] == pytest.approx(LEG_RISK * 1e7 / (s_pre * np.sqrt(5)) * wp)
        assert post["dv01"] == pytest.approx(LEG_RISK * 1e7 / (sigma_bp(Y["DGS2"], A) * np.sqrt(5)) * wq)
        assert pre["notional"] == pytest.approx(pre["dv01"] / (float(mod_duration(Y.at[e, "DGS2"], Y.at[e, "DGS2"], 2.0)) * 1e-4))
        assert pre["gross_pnl"] == pytest.approx(-pre["notional"] * X["DGS2"].loc[e:A].iloc[1:].sum())
        # the flip at A trades both DV01s; nothing nets (same direction)
        assert res.daily.at[A, "cost"] == pytest.approx((pre["dv01"] + post["dv01"]) * 0.25)
        assert res.daily["cost"].sum() == pytest.approx(res.legs["cost"].sum())
        assert res.daily["pnl"].sum() == pytest.approx(res.legs["net_pnl"].sum())


def test_sizing_known_at_entry():
    ev = events([("e1", "DGS2", "2019-09-18", 1, 1)])
    base = book(supply_legs(ev, False)).legs.set_index("kind")
    Y2 = Y.copy()
    Y2.loc[pd.Timestamp("2019-09-11"):, "DGS2"] += 1.0      # from the pre entry (A-5) on: sigma for pre unchanged
    s1, s2 = sigma_bp(Y["DGS2"], pd.Timestamp("2019-09-11")), sigma_bp(Y2["DGS2"], pd.Timestamp("2019-09-11"))
    assert CAL.offset(pd.Timestamp("2019-09-18"), -5) == pd.Timestamp("2019-09-11")
    assert s1 == pytest.approx(s2)
    X2 = X.copy()
    X2.loc[pd.Timestamp("2019-09-19"):] *= -3.0              # returns after A cannot change either leg's size
    alt = run_book("t", supply_legs(ev, False), X2, Y, RF, pd.DatetimeIndex([]), CAL, NAV, RiskConfig()).legs
    ref = run_book("t", supply_legs(ev, False), X, Y, RF, pd.DatetimeIndex([]), CAL, NAV, RiskConfig()).legs
    assert np.allclose(alt["dv01"], ref["dv01"])
    assert base.loc["pre", "sigma_bp"] == pytest.approx(s1)


def test_opposite_legs_in_one_tenor_net_to_zero_cost():
    ev = events([("e1", "DGS5", "2019-09-18", 1, 1)])
    legs = supply_legs(ev, False)
    mirror = legs.assign(sign=-legs["sign"], leg_id=legs["leg_id"] + "_m", unit_id="e2")
    res = book(pd.concat([legs, mirror], ignore_index=True))
    assert res.daily["cost"].sum() == pytest.approx(0.0, abs=1e-9)
    assert res.daily["pnl"].abs().sum() == pytest.approx(0.0, abs=1e-6)
    assert res.daily["traded_notional"].sum() == pytest.approx(0.0, abs=1e-6)
    assert res.legs["cost"].sum() > 0                        # gross attribution per leg still charges each leg


def test_demand_leg_alone_reproduces_backtest_calendar_only():
    months = pd.period_range("2019-07", "2020-10", freq="M")
    win = month_end_windows(CAL, months)
    fomc = pd.DatetimeIndex(["2019-07-31", "2019-10-30", "2020-01-29", "2020-04-29", "2020-07-29"])
    X10 = X.copy()
    X10["DGS10"] = X["DGS10"] * 4.0 - 0.003                 # steady losses so the drawdown rule fires
    cfg = RiskConfig()
    ref = run_strategy("calendar_only", win, pd.Series(1.0, index=months), X10["DGS10"], RF, Y["DGS10"], Y["DGS10"],
                       10.0, fomc, CAL, NAV, cfg)
    res = run_book("d", demand_legs(win, CAL), X10, Y, RF, fomc, CAL, NAV, cfg, demand_per_year=12, unit="month")
    assert np.allclose(res.daily["excess"], ref.daily["excess"], atol=1e-15, rtol=0)
    assert (res.legs["dd_mult"] < 1).any() and (res.legs["fomc_mult"] < 1).any()
    assert res.legs["dd_mult"].tolist() == ref.trades["dd_mult"].tolist()


def test_fomc_halves_a_leg_only_inside_entry_exit():
    A = pd.Timestamp("2019-09-18")
    ev = events([("e1", "DGS2", A, 1, 1)])
    res = book(supply_legs(ev, False), cfg=RiskConfig(dd_rule=False), fomc=pd.DatetimeIndex([A]))
    k = res.legs.set_index("kind")["fomc_mult"]
    assert k["pre"] == 0.5 and k["post"] == 1.0            # A is in (A-5, A] but not in (A, A+5]


def test_notional_cap_on_gross_open_notional():
    spec = [(f"e{i}", "DGS2", d, 2.0, 2.0) for i, d in enumerate(["2019-09-18", "2019-09-19", "2019-09-20"])]
    calm = Y.copy()
    calm["DGS2"] = 1.5 + np.cumsum(RNG.normal(0, 0.002, len(DAYS)))         # tiny vol -> huge notional
    res = run_book("t", supply_legs(events(spec), True), X, calm, RF, pd.DatetimeIndex([]), CAL, NAV,
                   RiskConfig(dd_rule=False))
    assert res.daily["gross_notional"].max() <= 3.0 * 1e7 * (1 + 1e-12)
    assert (res.legs["cap_mult"] < 1).any()
    assert book_metrics(res)["n_legs_capped"] == int((res.legs["cap_mult"] < 1).sum())


def test_drawdown_decisions_use_past_days_only():
    dates = pd.bdate_range("2019-07-01", "2020-10-30", freq="7B")
    spec = [(f"e{i}", "DGS2", d, 1, 1) for i, d in enumerate(dates)]
    legs = supply_legs(events(spec), False)
    Xb = X.copy()
    Xb["DGS2"] = X["DGS2"] * 6.0
    cfg = RiskConfig(fomc_half=False)
    ref = run_book("t", legs, Xb, Y, RF, pd.DatetimeIndex([]), CAL, NAV, cfg).legs.set_index("leg_id")
    cut = pd.Timestamp("2020-03-02")
    Xc = Xb.copy()
    Xc.loc[cut:, "DGS2"] *= -1.0
    alt = run_book("t", legs, Xc, Y, RF, pd.DatetimeIndex([]), CAL, NAV, cfg).legs.set_index("leg_id")
    known = ref["entry"] < cut                               # entered before any changed return is realised
    assert known.sum() > 5
    assert np.allclose(ref.loc[known, "dv01"], alt.loc[known, "dv01"])
    assert (ref["dd_mult"] < 1).any() or (alt["dd_mult"] < 1).any()


def test_legs_outside_nav_refused():
    ev = events([("e1", "DGS2", "2020-11-27", 1, 1)])       # A+5 after the last NAV day
    with pytest.raises(ValueError):
        book(supply_legs(ev, False))


def test_h6_means_and_clusters():
    spec = [(f"e{i}", "DGS2", d, 1 + 0.1 * (i % 5), 1) for i, d in enumerate(pd.bdate_range("2019-08-01", periods=40, freq="3B"))]
    ev = events(spec)
    ev = ev.join(event_returns(ev, X, Y))
    r = h6(ev)
    assert r["H6a"]["b"] == pytest.approx(ev["R_pre"].mean())
    assert r["H6b"]["b"] == pytest.approx(ev["R_post"].mean())
    assert r["H6a"]["n_clusters"] == ev["week"].nunique() < len(ev)
    assert r["n_H6c"] == len(ev) and np.isfinite(r["H6c"]["se"])


def test_tercile_by_rank_keeps_ties_together():
    z = pd.Series([-2.0, -1.5, -1.0, -0.5] * 3 + [0.0] * 10 + [0.5, 1.0, 1.5, 2.0, 2.5] * 3 + [np.nan])
    lab = tercile_by_rank(z)
    assert lab[z == 0].nunique() == 1 and lab[z == 0].iloc[0] == "mid"
    assert lab.iloc[-1] is np.nan or pd.isna(lab.iloc[-1])
    assert set(lab.dropna()) == {"low", "mid", "high"}


def test_event_paths_and_bands():
    spec = [(f"e{i}", "DGS5", d, 1 + (i % 3) - 1, 1) for i, d in enumerate(pd.bdate_range("2019-08-01", periods=30, freq="4B"))]
    ev = events(spec)
    rp, dp = auction_event_paths(ev, X, Y, CAL, -10, 10, "2019-06-03", "2020-11-30")
    assert list(rp.columns) == list(range(-10, 11)) and (rp[-10] == 0).all() and (dp[-10] == 0).all()
    lab = tercile_by_rank(ev.set_index("event_id")["zS_pre"])
    s = path_summary(rp, lab, ev.set_index("event_id")["week"])
    for t in ("low", "mid", "high"):
        assert s[t]["n"] > 0 and all(lo <= m <= hi for lo, m, hi in zip(s[t]["lo"], s[t]["mean"], s[t]["hi"]))
