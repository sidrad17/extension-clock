"""Tests for src/cmt_switch.py (Phase 4d, descriptive) on synthetic data: no snapshot return is computed. The daily P&L
functions must reproduce the cash and futures book engines leg by leg and day by day, days must land on the right
offset from A, shares must add up, and the reopening split must recover known group means."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import run_all
from src import cmt_switch as cs
from src import futures as fut
from src.flowclock import run_book, supply_legs
from src.risk import RiskConfig
from tests.test_futures import CAL, DAYS, MKT, RF, YLD

RNG = np.random.default_rng(11)
EXC = pd.DataFrame({t: RNG.normal(0.0001, 0.003, len(DAYS)) for t in YLD.columns}, index=DAYS)
FOMC = pd.DatetimeIndex(["2015-06-17", "2015-09-17", "2016-03-16"])
A = pd.to_datetime(["2015-02-10", "2015-03-24", "2015-05-12", "2015-06-15", "2015-08-11", "2015-11-09",
                    "2016-02-09", "2016-05-10", "2016-06-14", "2016-08-09"])
TEN = ["DGS3", "DGS10", "DGS30", "DGS10", "DGS2", "DGS5", "DGS7", "DGS20", "DGS30", "DGS10"]


def events() -> pd.DataFrame:
    return pd.DataFrame({"event_id": [f"{a.date()}_{t}" for a, t in zip(A, TEN)], "tenor": TEN, "A": A,
                         "pre_entry": [CAL.offset(a, -5) for a in A], "post_exit": [CAL.offset(a, 5) for a in A],
                         "w_pre": 1.0, "w_post": 1.0, "skipped": False, "window_complete": True})


def nav_days():
    return DAYS[(DAYS >= "2014-12-01") & (DAYS <= "2016-09-30")]


def test_common_dv01_is_the_book_dv01_without_drawdown_rule_and_cap():
    legs = supply_legs(events(), False)
    cfg = RiskConfig(dd_rule=False, notional_cap_on=False)
    book = run_book("s", legs, EXC, YLD, RF, FOMC, CAL, nav_days(), cfg=cfg)
    d = cs.common_dv01(legs, YLD, FOMC)
    mine = pd.Series(d.to_numpy(), index=legs["leg_id"])
    assert np.allclose(mine.reindex(book.legs["leg_id"]).to_numpy(), book.legs["dv01"].to_numpy(), rtol=1e-12)
    assert (book.legs["fomc_mult"] < 1).any()                                   # the FOMC half size is in both
    n = cs.cash_notional(legs, d, YLD)
    assert np.allclose(pd.Series(n.to_numpy(), index=legs["leg_id"]).reindex(book.legs["leg_id"]).to_numpy(),
                       book.legs["notional"].to_numpy(), rtol=1e-12)


def test_cash_daily_pnl_reproduces_the_cash_book():
    book = run_book("s", supply_legs(events(), False), EXC, YLD, RF, FOMC, CAL, nav_days(), cfg=RiskConfig())
    chk = cs.cash_check(book.legs, EXC, CAL)
    assert chk["n_legs_checked"] == 20 and chk["max_abs_diff_usd"] < 1e-6
    bad = book.legs.assign(gross_pnl=book.legs["gross_pnl"] + 1.0)
    with pytest.raises(RuntimeError):
        cs.cash_check(bad, EXC, CAL)


def futures_setup(m=MKT):
    ev = events()
    legs = fut.supply_legs_futures(supply_legs(ev, False))
    prep = fut.prepare_legs(legs, m, YLD)
    base = fut.run_futures_book("s", legs, prep, m, YLD, RF, FOMC, CAL, nav_days(), cfg=RiskConfig(),
                                supply_entries=pd.DatetimeIndex(list(ev["pre_entry"]) + list(ev["A"])),
                                record_leg_daily=True)
    return ev, legs, prep, base


def test_futures_daily_pnl_reproduces_the_engine_with_a_missing_settlement():
    settle = MKT.settle.copy()
    gap_day = CAL.offset(A[4], 2)                                    # inside the 2-year event's post leg
    settle.loc[gap_day] = np.nan                                     # no settlement for any contract that day
    m = replace(MKT, settle=settle)
    ev, legs, prep, base = futures_setup(m)
    ok = prep["status"] == "ok"
    chk = cs.futures_check(base, legs[ok], prep[ok], m, CAL)
    assert chk["n_traded_legs_checked"] >= 15 and chk["max_abs_diff_usd"] < 1e-6
    lid = f"{A[4].date()}_DGS2_post"
    one = legs[legs["leg_id"] == lid]
    p1 = prep.loc[one.index]
    fl = one[["leg_id", "sign", "entry", "exit"]].assign(contract=p1["contract"], root=p1["root"],
                                                         settle_entry=p1["settle_entry"])
    d = cs.futures_leg_daily(fl, pd.Series(1.0, index=one.index), m, CAL).set_index("date")["pnl"]
    assert d.loc[gap_day] == 0.0 and d.loc[CAL.offset(gap_day, 1)] != 0.0       # the move counts the next day


def test_event_matrix_puts_each_day_at_its_offset_from_A():
    ev = events()
    legs = supply_legs(ev, False)
    pos = pd.DataFrame({t: np.arange(len(DAYS), dtype=float) for t in YLD.columns}, index=DAYS)  # r(d) = position
    daily = cs.cash_leg_daily(legs, pd.Series(1.0, index=legs.index), pos, CAL)
    Au = ev.set_index("event_id")["A"]
    mat = cs.event_matrix(daily, legs, Au, CAL, 1.0)
    pA = np.array([DAYS.get_loc(a) for a in Au])
    for k in cs.DAY_K:
        sign = -1.0 if k <= 0 else 1.0
        assert np.allclose(mat[k].to_numpy(), sign * (pA + k)), k
    S = pd.Series([CAL.offset(a, 3) for a in Au], index=Au.index)
    S.iloc[0] = CAL.offset(Au.iloc[0], 7)                            # settles after the window
    off = cs.s_offsets(Au, S, CAL)
    s = cs.on_s(mat, off)
    assert s.iloc[0] == 0.0 and np.allclose(s.iloc[1:].to_numpy(), pA[1:] + 3)


def test_gap_summary_shares_add_up():
    idx = [f"e{i}" for i in range(60)]
    c = pd.DataFrame(RNG.normal(0.5, 2, (60, 10)), index=idx, columns=cs.DAY_K)
    f = pd.DataFrame(RNG.normal(0.2, 2, (60, 10)), index=idx, columns=cs.DAY_K)
    wk = pd.Series([f"w{i // 3}" for i in range(60)], index=idx)
    off = pd.Series(RNG.integers(1, 9, 60), index=idx).astype(float)
    g = cs.gap_summary(c, f, c / 2, f / 2, wk, off)
    shares = [g["day_path"][cs.day_label(k)]["share_of_total_gap"] for k in cs.DAY_K]
    assert sum(shares) == pytest.approx(1.0)
    assert g["gap_on_A_and_S"]["share_of_total_gap"] == pytest.approx(
        g["day_A"]["share_of_total_gap"] + g["day_S"]["share_of_total_gap"])
    assert g["gap_on_A_and_S"]["share_of_total_gap"] + g["gap_other_days"]["share_of_total_gap"] == pytest.approx(1)
    assert g["legs"]["total"]["gap"]["b"] == pytest.approx(sum(g["day_path"][cs.day_label(k)]["gap"]["b"]
                                                               for k in cs.DAY_K))
    assert g["day_S"]["n_units_S_in_window"] == int(off.between(1, 5).sum())
    assert g["day_A"]["gap_per_unit_dv01_bp"]["b"] == pytest.approx(g["day_A"]["gap"]["b"] / 2)


def reopen_data():
    rng = np.random.default_rng(3)
    rows, auc = [], []
    d = pd.Timestamp("2000-01-12")
    for i in range(240):
        t = cs.ISSUE_TENORS[i % 3]
        cusip = f"C{t}{i // 6}"                                       # each CUSIP: 1 new issue + reopenings
        reopen = i % 6 >= 3
        r_pre = (-0.10 if not reopen else 0.0) + rng.normal(0, 0.02)
        r_post = (0.08 if not reopen else 0.01) + rng.normal(0, 0.02)
        rows.append({"event_id": f"e{i}", "A": d, "cusip": cusip, "tenor": t, "week": f"w{i // 2}",
                     "R_pre": r_pre, "R_post": r_post, "LS": r_post - r_pre, "skipped": False,
                     "reopening": reopen})
        auc.append({"cusip": cusip, "auction_date": d})
        d += pd.Timedelta(days=7)
    return pd.DataFrame(rows), pd.DataFrame(auc)


def test_reopening_flags_and_control_recover_group_means():
    ev, auc = reopen_data()
    flags = cs.reopening_flags(ev, auc)
    assert (flags == ev["reopening"]).all()
    rc = cs.reopening_control(ev, flags)
    for t in cs.ISSUE_TENORS:
        g = ev[ev["tenor"] == t]
        for name, flag in (("new_issue", False), ("reopening", True)):
            h = g[g["reopening"] == flag]
            assert rc[t][name]["n"] == len(h)
            assert rc[t][name]["LS"]["b"] == pytest.approx(h["LS"].mean())
        diff = g.loc[g["reopening"], "LS"].mean() - g.loc[~g["reopening"], "LS"].mean()
        assert rc[t]["reopening_minus_new"]["LS"]["b"] == pytest.approx(diff)
    pooled = rc[cs.POOLED]
    assert pooled["new_issue"]["n"] + pooled["reopening"]["n"] == 240
    assert pooled["new_issue"]["R_pre"]["t"] < -5 and abs(pooled["reopening"]["R_pre"]["b"]) < 0.01


def test_run_all_cmt_cash():
    ev, auc = reopen_data()
    ev.loc[ev.index[:3], "reopening"] = ~ev.loc[ev.index[:3], "reopening"]      # 3 flags disagree
    book = run_book("s", supply_legs(events(), False), EXC, YLD, RF, FOMC, CAL, nav_days(), cfg=RiskConfig())
    out = run_all.cmt_cash(ev, auc, pd.Series(True, index=ev.index), book, EXC, CAL)
    assert out["reopening_definition"]["n_in_sample_disagree_with_fiscal_data_flag"] == 3
    assert out["cash_check"]["max_abs_diff_usd"] < 1e-6
    assert set(out["reopening_control"]) == set(cs.ISSUE_TENORS) | {cs.POOLED}
    blk = run_all.cmt_block(out, {"status": "x"})
    assert blk["cash_vs_futures"] == {"status": "x"} and "summary" not in blk
