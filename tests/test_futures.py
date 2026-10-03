"""Tests for src/futures.py and src/data/databento_futures.py on synthetic data (no network, no Databento file):
quarterly roll selection, FID, DV01 regression recovery, P&L and costs, netting, the notional cap, the TN fallback,
settlement extraction, symbol years, the derived-table round trip, trial counting and the run_all futures path."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import run_all
from config.futures import SUPPLY_CONTRACT
from src import futures as fut
from src import report
from src.backtest import month_end_windows, run_strategy
from src.calendar import BondCalendar
from src.data import databento_futures as dbf
from src.flowclock import demand_legs
from src.metrics import required_metrics  # noqa: F401
from src.risk import RiskConfig, sigma_bp
from src.trial_log import trial_counts

DAYS = pd.bdate_range("2014-01-02", "2016-12-30")
CAL = BondCalendar(DAYS)
RNG = np.random.default_rng(21)
TENORS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
COMMON = np.cumsum(RNG.normal(0, 0.05, len(DAYS)))                  # yields share a level factor, as real ones do
YLD = pd.DataFrame({t: lvl + COMMON + np.cumsum(RNG.normal(0, 0.01, len(DAYS))) for t, lvl in
                    zip(TENORS, [1.0, 1.3, 1.7, 2.0, 2.3, 2.6, 2.9])}, index=DAYS)
RF = pd.Series(0.00002, index=DAYS)
PV = {"ZT": 2000.0, "ZF": 1000.0, "ZN": 1000.0, "TN": 1000.0, "ZB": 1000.0, "UB": 1000.0}
TRUE_DV01 = {"ZT": 40.0, "ZF": 50.0, "ZN": 70.0, "TN": 90.0, "ZB": 150.0, "UB": 250.0}
DRIVER = {"ZT": "DGS2", "ZF": "DGS5", "ZN": "DGS10", "TN": "DGS10", "ZB": "DGS20", "UB": "DGS30"}
CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
TN_FIRST_LISTED = pd.Timestamp("2015-06-01")
OFF = RiskConfig(fomc_half=False, dd_rule=False, notional_cap_on=False)


def synthetic_long(noise: float = 0.0):
    """Quarterly contracts of every root, delivering 2014-03..2017-06, listed 9 months before delivery and
    trading to the 20th of the delivery month; settle moves by exactly -TRUE_DV01 per bp of the driver yield."""
    rng = np.random.default_rng(5)
    s_rows, v_rows, d_rows = [], [], []
    for root in PV:
        for dlv in pd.period_range("2014-03", "2017-06", freq="Q-DEC").asfreq("M", "end"):
            listed = (dlv - 9).start_time
            if root == "TN":
                listed = max(listed, TN_FIRST_LISTED)
            last = dlv.start_time + pd.Timedelta(days=19)
            days = DAYS[(DAYS >= listed) & (DAYS <= last)]
            if len(days) == 0:
                continue
            c = f"{root}{CODES[dlv.month]}{dlv.year}"
            y = YLD[DRIVER[root]].reindex(days)
            s = 120.0 - TRUE_DV01[root] * (y - YLD[DRIVER[root]].iloc[0]) * 100.0 / PV[root]
            s = s + rng.normal(0, noise, len(s))
            front = [(d.to_period("M") < dlv) and (d.to_period("M") >= dlv - 3) for d in days]
            for d, px, f in zip(days, s, front):
                s_rows.append((c, root, dlv, d, float(px)))
                v_rows.append((c, root, dlv, d, 1000.0 if f else 100.0))
            for snap in pd.date_range(listed, last, freq="MS"):
                tick = (15.625 if snap < pd.Timestamp("2015-07-01") else 7.8125) if root == "ZT" else \
                    {"ZF": 7.8125, "ZN": 15.625, "TN": 15.625, "ZB": 31.25, "UB": 31.25}[root]
                d_rows.append((snap, c, root, dlv, c[:3] + str(dlv.year)[-1], pd.NaT, tick / PV[root],
                               PV[root] * 100, PV[root], tick))
    st = pd.DataFrame(s_rows, columns=["contract", "root", "delivery", "trade_date", "settle"])
    vo = pd.DataFrame(v_rows, columns=["contract", "root", "delivery", "date", "volume"])
    defs = pd.DataFrame(d_rows, columns=["snapshot", "contract", "root", "delivery", "raw_symbol", "expiration",
                                         "tick_size", "face", "point_value", "tick_value"])
    return st, vo, defs


ST, VO, DEFS = synthetic_long()
MKT = fut.build_market(ST, VO, DEFS, CAL, PV)


# ------------------------------------------------------------------------------------------------ roll rule

def test_first_intention_day():
    rc = fut.RollCalendar(CAL)
    assert fut.first_intention_day(pd.Period("2015-03", "M"), rc) == pd.Timestamp("2015-02-26")   # Mar 2 - 2
    assert fut.first_intention_day(pd.Period("2015-06", "M"), rc) == pd.Timestamp("2015-05-28")   # Jun 1 - 2
    # beyond the bond calendar: plain weekdays (no data after IS_END is read)
    assert fut.first_intention_day(pd.Period("2017-03", "M"), rc) == pd.Timestamp("2017-02-27")


@pytest.mark.parametrize("month", ["2015-02", "2015-05", "2015-08", "2015-11"])
def test_feb_may_aug_nov_month_ends_pick_the_next_quarterly(month):
    T = CAL.month_end(month)
    E = CAL.offset(T, -4)
    p = pd.Period(month, "M")
    front = f"ZN{CODES[(p + 1).month]}{(p + 1).year}"
    nxt = f"ZN{CODES[(p + 4).month]}{(p + 4).year}"
    assert MKT.volume.at[CAL.offset(E, -1), front] > MKT.volume.at[CAL.offset(E, -1), nxt]   # front is busier
    c = fut.select_contract(MKT, "ZN", E, T)
    assert c["contract"] == nxt and c["fid"] > CAL.offset(T, 5)
    assert MKT.fid[front] <= CAL.offset(T, 5)


@pytest.mark.parametrize("month", ["2015-01", "2015-04", "2015-07", "2015-10"])
def test_other_month_ends_pick_the_busiest_eligible(month):
    T = CAL.month_end(month)
    c = fut.select_contract(MKT, "ZN", CAL.offset(T, -4), T)
    p = pd.Period(month, "M")
    assert c["contract"] == f"ZN{CODES[(p + 2).month]}{(p + 2).year}"


def test_volume_is_read_on_the_day_before_entry():
    T = CAL.month_end("2015-01")
    E = CAL.offset(T, -4)
    vol = MKT.volume.copy()
    vol.loc[E, "ZNM2015"] = 1e9                                   # entry-day volume must not matter
    m = replace(MKT, volume=vol)
    assert fut.select_contract(m, "ZN", E, T)["contract"] == "ZNH2015"
    vol.loc[CAL.offset(E, -1), "ZNM2015"] = 1e9                    # the day before decides
    m = replace(MKT, volume=vol)
    sel = fut.select_contract(m, "ZN", E, T)
    assert sel["contract"] == "ZNM2015" and sel["volume_day"] == CAL.offset(E, -1)


# ------------------------------------------------------------------------------------------------ DV01

def test_dv01_regression_recovers_the_slope():
    st, vo, defs = synthetic_long(noise=0.002)
    m = fut.build_market(st, vo, defs, CAL, PV)
    for root, c in (("ZN", "ZNH2016"), ("ZT", "ZTH2016"), ("UB", "UBH2016")):
        r = fut.dv01_regression(m, c, YLD[DRIVER[root]], pd.Timestamp("2015-11-20"))
        assert r["n"] == 60
        assert r["dv01"] == pytest.approx(TRUE_DV01[root], rel=0.03)
        assert r["r2"] > 0.95


def test_dv01_regression_needs_50_valid_changes_ending_the_day_before_entry():
    E = pd.Timestamp("2015-11-20")
    s = MKT.settle.copy()
    s.loc[E, "ZNH2016"] = np.nan                                    # the entry day itself is not used
    m = replace(MKT, settle=s)
    assert fut.dv01_regression(m, "ZNH2016", YLD["DGS10"], E)["n"] == 60
    idx = MKT.days[MKT.days.get_loc(E) - 20: MKT.days.get_loc(E) - 14]
    s.loc[idx, "ZNH2016"] = np.nan                                  # 6 missing levels in a row remove 7 changes
    m = replace(MKT, settle=s)
    r = fut.dv01_regression(m, "ZNH2016", YLD["DGS10"], E)
    assert r["n"] == 53 and r["dv01"] == pytest.approx(70.0)
    s.loc[MKT.days[MKT.days.get_loc(E) - 40: MKT.days.get_loc(E) - 34], "ZNH2016"] = np.nan
    m = replace(MKT, settle=s)
    assert np.isnan(fut.dv01_regression(m, "ZNH2016", YLD["DGS10"], E)["dv01"])     # 46 < 50


# ------------------------------------------------------------------------------------------------ engine

def month_end(months, cfg=OFF):
    win = month_end_windows(CAL, pd.period_range(*months, freq="M"))
    legs = fut.month_end_legs(demand_legs(win, CAL))
    prep = fut.prepare_legs(legs, MKT, YLD)
    nav = DAYS[(DAYS >= "2015-01-01") & (DAYS <= "2016-11-30")]
    return fut.run_futures_book("me", legs, prep, MKT, YLD, RF, pd.DatetimeIndex([]), CAL, nav, cfg,
                                demand_per_year=12, unit="month"), legs, prep


def test_single_leg_contracts_pnl_and_costs():
    res, legs, prep = month_end(("2015-03", "2015-03"))
    L = res.legs.iloc[0]
    E, T = L["entry"], L["exit"]
    sig = sigma_bp(YLD["DGS10"], E)
    target = 0.01 * 1e7 / (sig * 2.0)
    assert L["dv01_target"] == pytest.approx(target)
    assert L["contracts"] == np.floor(target / L["dv01_contract"] + 0.5) and L["contract"] == "ZNM2015"
    dsettle = MKT.settle.at[T, "ZNM2015"] - MKT.settle.at[E, "ZNM2015"]
    assert L["gross_pnl"] == pytest.approx(L["contracts"] * dsettle * 1000.0)
    assert L["cost"] == pytest.approx(L["contracts"] * (15.625 + 2.0))
    d = res.daily
    assert d.at[E, "cost"] == pytest.approx(L["cost"] / 2) and d.at[T, "cost"] == pytest.approx(L["cost"] / 2)
    assert d["pnl"].sum() == pytest.approx(L["net_pnl"])
    assert (d["total"] - d["excess"]).to_numpy() == pytest.approx(RF.reindex(d.index).to_numpy())
    assert L["notional"] == pytest.approx(L["contracts"] * MKT.settle.at[E, "ZNM2015"] * 1000.0)


def test_cost_stress_and_metrics_helpers():
    r1, legs, prep = month_end(("2015-01", "2016-06"))
    r2, _, _ = month_end(("2015-01", "2016-06"), cfg=RiskConfig(fomc_half=False, dd_rule=False,
                                                                  notional_cap_on=False, cost_mult=2.0))
    assert r2.daily["cost"].sum() == pytest.approx(2 * r1.daily["cost"].sum())
    m = required_metrics(fut.as_strategy(r1))
    assert m["n_windows"] == 18 and m["sharpe"] == pytest.approx(
        r1.daily["excess"].mean() / r1.daily["excess"].std(ddof=1) * np.sqrt(252))


def test_cost_netting_by_contract():
    """A long leg and a short leg in the same contract: the short entry nets the long, the common exit trades 0."""
    d0, d1, d2 = pd.Timestamp("2015-10-01"), pd.Timestamp("2015-10-08"), pd.Timestamp("2015-10-15")
    legs = pd.DataFrame({"leg_id": ["a", "b"], "unit_id": ["a", "b"], "kind": ["post", "pre"],
                         "tenor": ["DGS10", "DGS10"], "entry": [d0, d1], "exit": [d2, d2], "sign": [1.0, -1.0],
                         "base_risk": [0.0025, 0.0025], "hold_days": [5, 5], "w": [1.0, 1.0],
                         "product": ["ZN", "ZN"], "tenor_years": [10.0, 10.0]})
    prep = fut.prepare_legs(legs, MKT, YLD)
    assert prep["contract"].nunique() == 1
    nav = DAYS[(DAYS >= "2015-09-01") & (DAYS <= "2015-11-30")]
    res = fut.run_futures_book("net", legs, prep, MKT, YLD, RF, pd.DatetimeIndex([]), CAL, nav, OFF)
    na, nb = res.legs["contracts"]
    half = (15.625 + 2.0) / 2
    assert res.daily.at[d0, "cost"] == pytest.approx(na * half)
    assert res.daily.at[d1, "cost"] == pytest.approx(nb * half)
    assert res.daily.at[d2, "cost"] == pytest.approx(abs(na - nb) * half)
    assert res.legs["cost"].sum() == pytest.approx((na + nb) * 2 * half)        # attributed, not netted


def test_notional_cap_scales_new_legs_and_rounds_down():
    d0, d1 = pd.Timestamp("2015-10-01"), pd.Timestamp("2015-10-08")
    legs = pd.DataFrame({"leg_id": ["a", "b"], "unit_id": ["a", "b"], "kind": ["post", "post"],
                         "tenor": ["DGS2", "DGS2"], "entry": [d0, d0], "exit": [d1, d1], "sign": [1.0, 1.0],
                         "base_risk": [0.05, 0.05], "hold_days": [5, 5], "w": [1.0, 1.0],
                         "product": ["ZT", "ZT"], "tenor_years": [2.0, 2.0]})
    prep = fut.prepare_legs(legs, MKT, YLD)
    nav = DAYS[(DAYS >= "2015-09-01") & (DAYS <= "2015-11-30")]
    res = fut.run_futures_book("cap", legs, prep, MKT, YLD, RF, pd.DatetimeIndex([]), CAL, nav,
                               RiskConfig(fomc_half=False, dd_rule=False))
    per = MKT.settle.at[d0, prep.at[0, "contract"]] * 2000.0
    assert (res.legs["cap_mult"] < 1).all()
    assert res.legs["notional"].sum() <= 3e7 + 1e-6
    f = res.legs.at[0, "cap_mult"]
    assert res.legs.at[0, "contracts"] == np.floor(np.floor(res.legs.at[0, "contracts_raw"] + 0.5) * f)
    assert res.legs.at[0, "notional"] == pytest.approx(res.legs.at[0, "contracts"] * per)


def test_tn_falls_back_to_zn_before_it_can_be_sized():
    """TN contracts list from 2015-06 (synthetic): legs before TN's first sizable entry trade ZN; later TN legs
    trade TN, and a later unsizable TN leg is not traded (no fallback)."""
    dates = pd.to_datetime(["2015-03-10", "2015-07-14", "2015-10-13", "2016-01-12", "2016-04-12"])
    legs = pd.DataFrame({"leg_id": [f"e{i}" for i in range(5)], "unit_id": [f"e{i}" for i in range(5)],
                         "kind": "post", "tenor": "DGS10", "entry": dates,
                         "exit": [CAL.offset(d, 5) for d in dates], "sign": 1.0, "base_risk": 0.0025,
                         "hold_days": 5, "w": 1.0, "product": SUPPLY_CONTRACT["DGS10"], "tenor_years": 10.0})
    s = MKT.settle.copy()
    tn = [c for c in s.columns if c.startswith("TN")]
    s.loc[CAL.offset(dates[4], -30):CAL.offset(dates[4], -1), tn] = np.nan     # last TN leg cannot be sized
    m = replace(MKT, settle=s)
    prep = fut.prepare_legs(legs, m, YLD)
    first = pd.Timestamp(prep.attrs["first_sizable_entry"]["TN"])
    assert first > TN_FIRST_LISTED
    before = legs["entry"] < first
    assert before.any() and (~before).sum() >= 2
    assert (prep.loc[before, "root"] == "ZN").all() and prep.loc[before, "fallback_used"].all()
    assert (prep.loc[~before, "root"] == "TN").all() and not prep.loc[~before, "fallback_used"].any()
    assert prep.loc[4, "status"] == "no_dv01" and prep.loc[~before & (legs.index < 4), "status"].eq("ok").all()


# ------------------------------------------------------------------------------------------------ data layer

def test_symbol_years_and_outrights():
    assert dbf.resolve_year(3, pd.Timestamp("2024-01-05")) == 2023          # a late message for ZNZ3
    assert dbf.resolve_year(5, pd.Timestamp("2024-09-03")) == 2025
    assert dbf.resolve_year(0, pd.Timestamp("2019-10-16")) == 2020
    assert dbf.parse_outright("ZNZ3") == ("ZN", "Z", 3)
    for s in ("TNG1", "ZNH5-ZNM5", "UD:Z1: TL 0219818996", "ZQH5", "ZNH"):
        assert dbf.parse_outright(s) is None
    ids = dbf.contract_ids(pd.Series(["ZTU4", "TNX0"]), pd.Series(pd.to_datetime(["2024-09-03", "2010-10-25"])))
    assert ids.loc[0, "contract"] == "ZTU2024" and ids.loc[0, "delivery"] == pd.Period("2024-09", "M")
    assert pd.isna(ids.loc[1, "contract"])


def test_settlement_is_the_last_message_per_trade_date():
    utc = lambda x: pd.Timestamp(x, tz="UTC")                                      # noqa: E731
    s = pd.DataFrame({"ts_ref": [utc("2023-04-21"), utc("2023-04-21"), utc("2023-04-21"), pd.NaT,
                                 utc("2023-04-21"), utc("2024-10-01")],
                      "price": [116.046875, 115.625, 115.625, 115.0, 99.0, 110.0],
                      "symbol": ["ZNZ3", "ZNZ3", "ZNZ3", "ZNZ3", "ZNZ3-ZNH4", "ZNZ4"]},
                     index=pd.DatetimeIndex([utc("2023-04-21 19:00"), utc("2023-04-21 20:02"),
                                             utc("2023-04-23 16:04"), utc("2023-04-24 17:00"),
                                             utc("2023-04-21 21:00"), utc("2024-10-01 00:30")], name="ts_recv"))
    out = dbf.settlements_from_stats(s, end="2024-09-30")
    assert len(out) == 1
    r = out.iloc[0]
    assert r["contract"] == "ZNZ2023" and r["trade_date"] == pd.Timestamp("2023-04-21") and r["settle"] == 115.625


def test_tick_value_is_point_in_time():
    assert MKT.tick_value("ZTZ2015", "2015-06-30") == 15.625
    assert MKT.tick_value("ZTZ2015", "2015-07-01") == 7.8125
    assert MKT.tick_value("ZTZ2015", "2014-01-02") == 15.625        # before its first snapshot: the first one


STR_COLS = ("leg_id", "unit_id", "kind", "tenor", "product", "root", "contract", "status")


def read_back(tmp_path, leg_t, day_t):
    leg_t.to_csv(tmp_path / "l.csv", index=False)
    day_t.to_csv(tmp_path / "d.csv")
    lt = pd.read_csv(tmp_path / "l.csv", dtype={c: str for c in STR_COLS}, keep_default_na=False,
                     na_values={c: [""] for c in fut.LEG_TABLE_COLS if c not in STR_COLS},
                     float_precision="round_trip")
    return lt, pd.read_csv(tmp_path / "d.csv", index_col="date", float_precision="round_trip")


def test_tables_round_trip_is_exact(tmp_path):
    """P&L survives the CSV round trip bit for bit (default float repr), so a keyless run reproduces every metric."""
    r1, _, _ = month_end(("2015-01", "2016-06"))
    leg_t, day_t = fut.to_tables({"month_end_zn": r1})
    lt, dt = read_back(tmp_path, leg_t, day_t)
    back = fut.from_tables(lt, dt, RF, {"month_end_zn": "month"}, base_cfg=OFF)["month_end_zn"]
    assert (back.daily["excess"].to_numpy() == r1.daily["excess"].to_numpy()).all()
    assert (lt["net_pnl"].to_numpy() == leg_t["net_pnl"].to_numpy()).all()
    lt2, dt2 = read_back(tmp_path, *fut.to_tables({"month_end_zn": back}))
    again = fut.from_tables(lt2, dt2, RF, {"month_end_zn": "month"}, base_cfg=OFF)["month_end_zn"]
    assert fut.month_end_metrics(back) == fut.month_end_metrics(again)          # idempotent


def test_tables_hold_no_price_or_volume():
    """Licensed data (CLAUDE.md section 15, Phase 4b): no notional, raw volume, exact cap factor or settlement level;
    daily notionals in whole $1M; contracts, DV01 and P&L kept."""
    r1, _, _ = month_end(("2015-01", "2016-06"))
    leg_t, day_t = fut.to_tables({"month_end_zn": r1})
    assert not {"notional", "volume", "cap_mult", "settle_entry", "settle_exit"} & set(leg_t.columns)
    assert {"contracts", "dv01", "dv01_contract", "gross_pnl", "cost", "net_pnl", "capped"} <= set(leg_t.columns)
    for c in ("month_end_zn__gross_notional", "month_end_zn__traded_notional"):
        v = day_t[c].to_numpy()
        assert (v == np.round(v / 1e6) * 1e6).all() and (v >= 0).all()
    assert (day_t["month_end_zn__gross_pnl"].to_numpy() == r1.daily["gross_pnl"].to_numpy()).all()
    # what the table gives for a traded leg cannot pin its entry settlement: $1M of notional spans many ticks
    t = leg_t[leg_t["contracts"] > 0].iloc[0]
    assert 1e6 / (t["contracts"] * t["point_value"]) > 50 * (15.625 / 1000.0)


def test_notional_metrics_use_traded_notional_and_days_held():
    r1, _, _ = month_end(("2015-01", "2016-06"))
    m = fut.month_end_metrics(r1)
    d = r1.daily
    years = ((d.index[-1] - d.index[0]).days + 1) / 365.25
    assert m["turnover_x_per_year"] == pytest.approx(d["traded_notional"].sum() / 1e7 / years)
    held = fut.held_overnight(r1)
    assert held.sum() == 4 * 18 and m["mean_notional_x_capital"] == pytest.approx(
        d.loc[held, "gross_notional"].mean() / 1e7)
    assert (d.loc[~held, "gross_notional"] == 0).all()


def test_capacity_curve_futures():
    nav = DAYS[(DAYS >= "2015-01-01") & (DAYS <= "2016-11-30")]
    win = month_end_windows(CAL, pd.period_range("2015-01", "2016-06", freq="M"))
    legs = fut.month_end_legs(demand_legs(win, CAL))
    prep = fut.prepare_legs(legs, MKT, YLD)
    res = fut.run_futures_book("me", legs, prep, MKT, YLD, RF, pd.DatetimeIndex([]), CAL, nav, OFF,
                               demand_per_year=12, unit="month", record_leg_daily=True)
    assert res.leg_daily.groupby("leg")["gross_pnl"].sum().to_numpy() == pytest.approx(
        res.legs.loc[res.legs["contracts"] > 0, "gross_pnl"].to_numpy())
    cap = fut.capacity_curve_futures(res, MKT, capitals=[1e-6, 1e7, 1e9, 1e11])
    g = cap["grid"]
    # impact per $ of capital falls with sqrt(K): at $1e-6 it vanishes; one leg at a time, so unnetted = netted
    assert g["sharpe"][0] == pytest.approx(cap["sharpe_strategy_netted_costs"], rel=1e-6)
    assert g["sharpe"][0] >= g["sharpe"][1] >= g["sharpe"][2] >= g["sharpe"][3]
    assert g["share_legs_capped"][-1] == 1.0 and g["max_participation"][-1] == pytest.approx(0.05)
    assert set(cap["median_adv_contracts_by_product"]) == {"ZN"}
    assert "adv_measure" not in cap                                          # the as-written block is unchanged
    # Phase 4c variant active_adv: same model, ADV from the root's busiest contract, which is >= the held one's
    act = fut.capacity_curve_futures(res, MKT, capitals=[1e-6, 1e7, 1e9, 1e11], adv_measure="active")
    ga = act["grid"]
    assert act["adv_measure"] == "active_adv" and ga["sharpe"][0] == pytest.approx(g["sharpe"][0], rel=1e-6)
    assert all(a <= h for a, h in zip(ga["share_legs_capped"], g["share_legs_capped"]))
    t = res.legs[res.legs["contracts"] > 0]
    held = np.array([fut.held_adv(MKT, c, e) for c, e in zip(t["contract"], t["entry"])])
    active = np.array([fut.active_adv(MKT, r, e) for r, e in zip(t["root"], t["entry"])])
    assert (active >= held).all() and (active > held).any()
    assert act["n_legs_held_adv_below_10pct_of_active"] == int((held < 0.1 * active).sum())
    with pytest.raises(ValueError):
        fut.capacity_curve_futures(res, MKT, adv_measure="front")


def test_active_adv_is_the_roots_busiest_contract_over_the_20_days_ending_the_day_before_entry():
    # Feb month-end: the leg holds June (FID rule); over the 20 days before entry March is the front (volume 1000)
    entry = CAL.offset(pd.Timestamp("2015-02-27"), -4)
    assert fut.select_contract(MKT, "ZN", entry, CAL.offset(entry, 7))["contract"] == "ZNM2015"
    assert fut.held_adv(MKT, "ZNM2015", entry) == 100.0
    assert fut.active_adv(MKT, "ZN", entry) == 1000.0
    p = int(MKT.days.get_loc(entry))
    win = MKT.days[p - 20:p]
    expect = MKT.volume.loc[win, MKT.contracts_of("ZN")].max(axis=1).mean()
    assert fut.active_adv(MKT, "ZN", entry) == pytest.approx(expect)
    m2 = replace(MKT, volume=MKT.volume.copy())                              # E itself and E-21 are not read
    m2.volume.loc[[entry, MKT.days[p - 21]], MKT.contracts_of("ZN")] = 1e9
    m2.volume.loc[win[0], MKT.contracts_of("ZN")] = 0.0                     # a day without any bar counts as 0
    assert fut.active_adv(m2, "ZN", entry) == pytest.approx((expect * 20 - 1000.0) / 20)


def test_trial_counts_distinct_and_rows():
    t = pd.DataFrame({"config_hash": ["a", "b", "a", "c", "b"], "sharpe_fc": [0.5, np.nan, 0.5, 0.1, np.nan],
                      "sharpe_cal": [0.4, 0.3, 0.4, np.nan, 0.3]})
    tc = trial_counts(t)
    assert tc["total_logged_runs"] == 5 and tc["distinct_variants"] == 3
    assert sorted(tc["sharpes_distinct"]) == [0.1, 0.3, 0.4, 0.5] and len(tc["sharpes_all_rows"]) == 7
    assert tc["n_configs_with_differing_sharpes"] == 0
    t.loc[2, "sharpe_fc"] = 0.6
    assert trial_counts(t)["n_configs_with_differing_sharpes"] == 1


# ------------------------------------------------------------------------------------------------ run_all

def test_run_all_futures_tables_then_block(monkeypatch, tmp_path):
    """--futures builds the derived tables; every run computes the futures block from them (identically), and with
    no tables the block is skipped."""
    (tmp_path / "tables").mkdir()
    monkeypatch.setattr(report, "OUTPUTS", tmp_path)
    monkeypatch.setattr(run_all, "FUT_START", "2015-01-01")
    monkeypatch.setattr(run_all, "IS_END", "2016-09-30")
    monkeypatch.setattr(dbf, "load_definition_snapshots", lambda: DEFS)
    monkeypatch.setattr(dbf, "load_settlements", lambda end: ST)
    monkeypatch.setattr(dbf, "load_volume", lambda end: VO)
    monkeypatch.setattr(run_all.fus, "ALIGNMENT_DATES", ["2015-06-10", "2015-11-09", "2016-03-09"])
    logged = []
    monkeypatch.setattr(run_all, "log_trials", lambda rows: logged.extend(rows) or len(rows))
    months = pd.period_range("2014-06", "2016-09", freq="M")
    win = month_end_windows(CAL, months)
    nav_cash = DAYS[(DAYS >= "2014-06-01") & (DAYS <= "2016-09-30")]
    xs = pd.Series(RNG.normal(0.0001, 0.003, len(DAYS)), index=DAYS)
    cash = run_strategy("co", win, pd.Series(1.0, index=months), xs, RF, YLD["DGS10"], YLD["DGS10"], 10.0,
                        pd.DatetimeIndex(["2015-06-17"]), CAL, nav_cash, RiskConfig())
    A = pd.to_datetime(["2015-02-10", "2015-03-24", "2015-05-12", "2015-08-11", "2015-11-09", "2016-02-09",
                        "2016-05-10", "2016-08-09"])
    tenors = ["DGS3", "DGS10", "DGS30", "DGS2", "DGS5", "DGS7", "DGS20", "DGS10"]
    ev = pd.DataFrame({"event_id": [f"{a.date()}_{t}" for a, t in zip(A, tenors)], "tenor": tenors, "A": A,
                       "pre_entry": [CAL.offset(a, -5) for a in A], "post_exit": [CAL.offset(a, 5) for a in A],
                       "w_pre": 1.0, "w_post": 1.0, "skipped": False, "window_complete": True})
    fomc = pd.DatetimeIndex(["2015-06-17"])
    sup_net = pd.Series(RNG.normal(0, 0.001, len(nav_cash)), index=nav_cash)
    kw = dict(cash_cal=cash, cash_supply_daily=sup_net, cash_supply_gross_daily=sup_net + 0.0001, rf=RF,
              git={"commit": "x", "dirty": False}, t0=0.0)
    assert run_all.futures(**kw) is None                                     # no tables yet: skipped
    run_all._build_futures_tables(CAL, win, months[months >= pd.Period("2015-01", "M")], fomc, RF, YLD["DGS10"],
                                  ev, pd.DatetimeIndex(list(ev["pre_entry"]) + list(A)), YLD, run_all.fut_paths(),
                                  0.0)
    for k in ("legs", "daily", "checks"):
        assert (tmp_path / "tables" / run_all.FUT_TABLES[k]).exists()
    b1 = run_all.futures(**kw)
    b2 = run_all.futures(**kw)
    assert report.clean({k: v for k, v in b1.items() if not k.startswith("_")}) == \
        report.clean({k: v for k, v in b2.items() if not k.startswith("_")})
    assert len(logged) == 24 and {r["window"] for r in logged} == {
        "in_sample_futures", "in_sample_futures_cost2x"} | {f"in_sample_futures_risk_{n}" for n in run_all.FUT_RISK_NAMES}
    assert len({r["config_hash"] for r in logged}) == 12                    # 12 distinct configurations, logged twice
    assert set(b1["risk_rules_on_off"]) == {"all_on", *run_all.FUT_RISK_NAMES}
    assert b1["risk_rules_on_off"]["all_off"]["metrics"]["supply_calendar"]["n_drawdown_halved"] == 0
    assert set(b1["capacity"]) == {"month_end_zn", "supply_calendar"} and "capacity" not in b1["data"]
    assert set(b1["capacity_active_adv"]) == {"month_end_zn", "supply_calendar"}
    assert "capacity_active_adv" not in b1["data"] and "adv_measure" not in b1["capacity"]["month_end_zn"]
    assert all(v["adv_measure"] == "active_adv" for v in b1["capacity_active_adv"].values())
    bc = b1["supply_calendar"]["vs_cash_supply_calendar"]["before_costs_same_days"]
    assert bc["sharpe_futures"] == b1["supply_calendar"]["metrics"]["sharpe_gross"]
    assert bc["sharpe_cash"] > bc["sharpe_cash_net"]                         # gross = net + a positive cost
    kill = b1["supply_calendar"]["kill_condition_2x_costs"]
    assert kill["met"] and kill["sharpe_1x"] == b1["supply_calendar"]["metrics"]["sharpe"]
    assert set(b1["_fig"]["fut_navs"]) == set(b1["_fig"]["fut_sharpes"]) and len(b1["_fig"]["fut_navs"]) == 4
    legs_csv = pd.read_csv(tmp_path / "tables" / run_all.FUT_TABLES["legs"], nrows=1)
    assert not {"notional", "volume", "cap_mult"} & set(legs_csv.columns)
    me = b1["month_end_zn"]
    assert me["metrics"]["n_windows"] > 0 and me["legs"]["by_product"].keys() == {"ZN"}
    assert b1["supply_calendar"]["legs"]["n_traded"] == 16
    assert set(b1["supply_calendar"]["legs"]["by_tenor_product"]) <= {f"{t}->{SUPPLY_CONTRACT[t]}" for t in tenors} \
        | {"DGS10->ZN"}
    assert b1["data"]["alignment"]["corr_dsettle_neg_dy10_by_lag"]["0"] > 0.99


def test_figure5_futures_panel(tmp_path):
    from src import figures
    a = pd.Series(np.linspace(1.0, 1.8, len(DAYS)), index=DAYS)
    b = pd.Series(np.linspace(1.0, 1.2, 300), index=DAYS[-300:])
    cap = figures.equity_curve({"cash A": a}, {"cash A": 0.6}, tmp_path / "f.png", "2014-01 to 2016-12",
                               fut_navs={"fut B": b}, fut_sharpes={"fut B": 0.3}, fut_sample="2015-10 to 2016-12")
    assert (tmp_path / "f.png").stat().st_size > 10_000
    assert "fut B 1.20x (Sharpe 0.30)" in cap and "cash A 1.80x (Sharpe 0.60)" in cap


def test_cost_block_converts_fleming_spreads_to_yield_bp():
    from src.bonds import mod_duration
    from src.report import cost_block
    d = pd.bdate_range("1996-12-02", "2000-04-28")
    y = pd.DataFrame({"DGS2": 5.8, "DGS5": 6.0, "DGS10": 6.0}, index=d)
    c = cost_block(y)
    t = c["cash"]["reference"]["by_tenor"]["DGS10"]
    dur = float(mod_duration(6.0, 6.0, 10.0))
    assert t["spread_bp_yield"] == pytest.approx((0.78 / 32) / (dur * 0.01))
    assert t["our_cost_over_spread"] == pytest.approx(0.5 / t["spread_bp_yield"])
    assert "our assumption" in c["cash"]["status"] and c["cash"]["bp_yield_per_round_trip"] == 0.5
