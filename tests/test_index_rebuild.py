"""Tests for src/index_rebuild.py on a small auction fixture (section 9 list plus point-in-time rules).

Month under test: May 2024 on a plain weekday calendar, flat 4% curve.
T = 2024-05-31, E = T-4 = 2024-05-27, previous rebalance T(m-1) = 2024-04-30.
"""
import numpy as np
import pandas as pd
import pytest

from config.settings import CURVE_KNOTS
from src.bonds import Curve, mod_duration, price
from src.calendar import BondCalendar
from src.data.auctions import load_auctions
from src.index_rebuild import BUCKETS, RebuildConfig, prepare_tranches, rebuild, rebuild_month

COLS = ["cusip", "security_type", "security_term", "auction_date", "issue_date", "maturity_date", "announcemt_date",
        "first_int_payment_date", "offering_amt", "total_accepted", "soma_accepted", "int_rate", "reopening",
        "inflation_index_security", "floating_rate", "callable"]
ROWS = [
    # A: 10y note in the index all month, pays May/Nov 15
    ["A10Y", "Note", "10-Year", "2020-05-06", "2020-05-15", "2030-05-15", "2020-04-30", "2020-11-15",
     "50000000000", "50000000000", "0", "4.000000", "No", "No", "No", "No"],
    # B: 3y note crossing 1 year: in NOW (May 1 2025 <= May 15 2025), out of NEXT (June 1 2025 > May 15 2025)
    ["B03Y", "Note", "3-Year", "2022-05-10", "2022-05-16", "2025-05-15", "2022-05-04", "2022-11-15",
     "40000000000", "40000000000", "0", "3.000000", "No", "No", "No", "No"],
    # C: new 30y bond auctioned before E: public amount = total - SOMA = 25bn, first coupon in November
    ["C30Y", "Bond", "30-Year", "2024-05-09", "2024-05-15", "2054-05-15", "2024-05-02", "2024-11-15",
     "25000000000", "26000000000", "1000000000", "4.500000", "No", "No", "No", "No"],
    # D: 7y auctioned after E (announced before): offering_amt 40bn, coupon from the curve (not the 4.6 result)
    ["D07Y", "Note", "7-Year", "2024-05-29", "2024-05-31", "2031-05-31", "2024-05-23", "2024-11-30",
     "40000000000", "44000000000", "4000000000", "4.600000", "No", "No", "No", "No"],
    # F: 2y auctioned ON T, settling next month: in under "auctioned_by_rebalance", out under "settled"
    ["F02Y", "Note", "2-Year", "2024-05-31", "2024-06-03", "2026-05-31", "2024-05-23", "2024-11-30",
     "60000000000", "61000000000", "1000000000", "4.750000", "No", "No", "No", "No"],
    # I: 5y auctioned after E and announced after E: skipped and counted
    ["I05Y", "Note", "5-Year", "2024-05-30", "2024-05-31", "2029-05-31", "2024-05-28", "2024-11-30",
     "70000000000", "70000000000", "0", "4.500000", "No", "No", "No", "No"],
    # G: TIPS and H: FRN, excluded
    ["G10T", "Note", "10-Year", "2024-05-16", "2024-05-31", "2034-01-15", "2024-05-09", "2024-07-15",
     "18000000000", "18000000000", "0", "1.750000", "No", "Yes", "No", "No"],
    ["H02F", "Note", "2-Year", "2024-05-22", "2024-05-31", "2026-04-30", "2024-05-16", "2024-07-31",
     "28000000000", "28000000000", "0", "null", "No", "No", "Yes", "No"],
]
MONTH = pd.Period("2024-05", "M")
E = pd.Timestamp("2024-05-27")


@pytest.fixture
def auctions(tmp_path):
    p = tmp_path / "auctions.csv"
    pd.DataFrame(ROWS, columns=COLS).to_csv(p, index=False)
    return load_auctions(end=None, path=p)


@pytest.fixture
def cal():
    return BondCalendar(pd.bdate_range("2024-03-01", "2024-06-28"))


@pytest.fixture
def curve(cal):
    return Curve(pd.DataFrame(4.0, index=cal.days, columns=list(CURVE_KNOTS)))


def run(auctions, curve, cal, cfg=None):
    return rebuild_month(prepare_tranches(auctions), curve, cal, MONTH, cfg or RebuildConfig())


def test_tips_and_frn_excluded(auctions):
    assert not {"G10T", "H02F"} & set(auctions["cusip"])


def test_dates(auctions, curve, cal):
    row, _ = run(auctions, curve, cal)
    assert row["T"] == "2024-05-31" and row["E"] == "2024-05-27"


def test_membership_adds_removes_and_skips(auctions, curve, cal):
    row, ch = run(auctions, curve, cal)
    kind = dict(zip(ch["cusip"], ch["change"]))
    assert kind == {"B03Y": "remove", "C30Y": "add", "D07Y": "add", "F02Y": "add"}   # I05Y skipped
    assert row["n_now"] == 2 and row["n_next"] == 4
    assert row["n_skipped"] == 1 and row["n_estimated"] == 2                              # D and F


def test_after_e_tranche_uses_offering_and_curve_coupon(auctions, curve, cal):
    _, ch = run(auctions, curve, cal)
    d = ch.set_index("cusip").loc["D07Y"]
    assert d["amount_next"] == 40e9 and not d["coupon_known"]
    years = (pd.Timestamp("2031-05-31") - E).days / 365.25
    assert d["dur"] == pytest.approx(mod_duration(4.0, 4.0, years))   # priced at the curve coupon, not 4.6
    c = ch.set_index("cusip").loc["C30Y"]
    assert c["amount_next"] == 25e9 and c["coupon_known"]             # SOMA add-on deducted


def test_auctioned_on_t_settling_next_month(auctions, curve, cal):
    _, ch = run(auctions, curve, cal)
    assert "F02Y" in set(ch["cusip"])
    _, ch_settled = run(auctions, curve, cal, RebuildConfig(inclusion_rule="settled_by_month_end"))
    assert "F02Y" not in set(ch_settled["cusip"]) and "D07Y" in set(ch_settled["cusip"])


def test_adding_30y_raises_duration(auctions, curve, cal):
    with_c, _ = run(auctions, curve, cal)
    without_c, _ = run(auctions[auctions["cusip"] != "C30Y"], curve, cal)
    assert with_c["D_next"] > without_c["D_next"] and with_c["Ext"] > without_c["Ext"]
    assert with_c["D_now"] == pytest.approx(without_c["D_now"])


def test_coupon_cash_hand_computed(auctions, curve, cal):
    row, _ = run(auctions, curve, cal)
    # NOW = {A, B}; both pay on May 15: A 4%/2 x 50bn = 1.0bn, B 3%/2 x 40bn = 0.6bn
    ya = (pd.Timestamp("2030-05-15") - E).days / 365.25
    yb = (pd.Timestamp("2025-05-15") - E).days / 365.25
    mv_now = 50e9 * price(4.0, 4.0, ya) / 100 + 40e9 * price(3.0, 4.0, yb) / 100
    assert row["c_m"] == pytest.approx(1.6e9 / mv_now, rel=1e-12)
    june, _ = rebuild_month(prepare_tranches(auctions), curve, cal, pd.Period("2024-06", "M"), RebuildConfig())
    assert june["c_m"] == 0.0                        # every June member is on a May/Nov coupon cycle


def test_fdd_identity_and_buckets(auctions, curve, cal):
    row, ch = run(auctions, curve, cal)
    assert row["FDD"] == pytest.approx(row["Ext"] + row["c_m"] * row["D_next"], abs=1e-14)
    assert sum(row[f"ext_{b}"] for b in BUCKETS) == pytest.approx(row["Ext"], abs=1e-12)
    assert sum(row[f"fdd_{b}"] for b in BUCKETS) == pytest.approx(row["FDD"], abs=1e-12)
    assert row["ext_adds"] + row["ext_removes"] + row["ext_reopens"] + row["ext_dilution"] == pytest.approx(row["Ext"])
    no_cash, _ = run(auctions, curve, cal, RebuildConfig(reinvest_coupons=False))
    assert no_cash["FDD"] == pytest.approx(no_cash["Ext"])


def test_now_universe_uses_actual_amounts_next_month(auctions, curve, cal):
    # In June, D and F (estimated at offering_amt in May) are NOW members with their actual public amounts,
    # and I (skipped in May) is in.
    june, ch = rebuild_month(prepare_tranches(auctions), curve, cal, pd.Period("2024-06", "M"), RebuildConfig())
    assert june["n_now"] == 5
    assert june["par_now_bn"] == pytest.approx((50e9 + 25e9 + 40e9 + 60e9 + 70e9) / 1e9)


def test_rebuild_refuses_test_window_in_dev(monkeypatch, auctions, curve, cal):
    import src.trial_log as tl
    monkeypatch.setattr(tl, "dev_mode", lambda: True)
    monkeypatch.setattr(tl, "_head_tags", lambda: [])
    with pytest.raises(tl.GateError):
        rebuild(auctions, curve, cal, start="2024-05", end="2024-10-31")
