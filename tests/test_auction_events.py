"""Tests for src/auction_events.py: tenor mapping, windows, skip rule, size signal (known at entry, no lookahead),
sample membership and month-end supply SA_m (PREREG_FLOWCLOCK.md). Synthetic data, no returns."""
import numpy as np
import pandas as pd
import pytest

from src.auction_events import (build_events, in_sample_mask, month_end_supply, nearest_tenor, run_starts, size_z,
                                term_years)
from src.bonds import Curve, mod_duration
from src.calendar import BondCalendar
from src.signals import past_zscore

DAYS = pd.bdate_range("2021-01-04", "2024-12-31").drop(pd.Timestamp("2023-10-09"))   # Columbus Day closed
CAL = BondCalendar(DAYS)


def auction(cusip, A, amt, term="2-Year", ann=None, reopening=False, tips=False, frn=False, sec_term=None):
    A = pd.Timestamp(A)
    return {"cusip": cusip, "auction_date": A, "announcemt_date": pd.Timestamp(ann) if ann else A - pd.Timedelta(days=14),
            "original_security_term": term, "security_term": sec_term or term, "reopening": reopening,
            "offering_amt": float(amt), "inflation_index_security": tips, "floating_rate": frn, "callable": False}


def frame(rows):
    df = pd.DataFrame(rows)
    for c in ["reopening", "inflation_index_security", "floating_rate", "callable"]:
        df[c] = df[c].astype("boolean")
    return df


def monthly(term, start, n, amts, prefix="X"):
    """n auctions on the 20th bond day of each month from `start`, announced two weeks before."""
    out = []
    for i, m in enumerate(pd.period_range(start, periods=n, freq="M")):
        out.append(auction(f"{prefix}{i:03d}", CAL.nth_bday(m, 15), amts[i], term))
    return out


@pytest.mark.parametrize("term,tenor", [
    ("2-Year", "DGS2"), ("3-Year", "DGS3"), ("4-Year", "DGS5"), ("5-Year 2-Month", "DGS5"),
    ("3-Year 11-Month", "DGS3"), ("6-Year 11-Month", "DGS7"), ("10-Year", "DGS10"), ("14-Year 10-Month", "DGS10"),
    ("15-Year 1-Month", "DGS20"), ("20-Year 1-Month", "DGS20"), ("29-Year 9-Month", "DGS30"),
    ("30-Year 3-Month", "DGS30")])
def test_original_term_maps_to_nearest_cmt_ties_longer(term, tenor):
    assert nearest_tenor(term_years(term)) == tenor


def test_term_parse():
    assert term_years("5-Year 2-Month") == pytest.approx(5 + 2 / 12)
    with pytest.raises(ValueError):
        term_years("26-Week")


def test_reopening_uses_original_term_and_tips_frn_dropped():
    rows = [auction("R1", "2023-05-10", 35e9, term="10-Year", reopening=True, sec_term="9-Year 10-Month"),
            auction("T1", "2023-05-11", 15e9, term="10-Year", tips=True),
            auction("F1", "2023-05-12", 20e9, term="2-Year", frn=True)]
    ev = build_events(frame(rows), CAL)
    assert list(ev["cusip"]) == ["R1"]
    assert ev.loc[0, "tenor"] == "DGS10" and ev.loc[0, "security_term"] == "9-Year 10-Month"
    assert ev.attrs["n_tips_dropped"] == 1 and ev.attrs["n_frn_dropped"] == 1


def test_windows_across_columbus_day():
    ev = build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL)
    # 2023-10-09 is closed: A-5 = Oct 4 (Oct 11, 10, 6, 5, 4), A+5 = Oct 19
    assert ev.loc[0, "pre_entry"] == pd.Timestamp("2023-10-04")
    assert ev.loc[0, "post_exit"] == pd.Timestamp("2023-10-19")


def test_skip_rule_uses_the_whole_window_only():
    ev = build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL)
    y = pd.DataFrame({"DGS2": 4.0}, index=DAYS)
    assert not build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL, y).loc[0, "skipped"]
    y.loc[pd.Timestamp("2023-10-03"), "DGS2"] = np.nan        # the day before A-5: outside
    y.loc[pd.Timestamp("2023-10-20"), "DGS2"] = np.nan        # the day after A+5: outside
    assert not build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL, y).loc[0, "skipped"]
    y.loc[pd.Timestamp("2023-10-16"), "DGS2"] = np.nan        # inside the post window
    assert build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL, y).loc[0, "skipped"]
    assert ev.loc[0, "window_complete"]


def test_size_z_rules():
    flat = np.array([24.0] * 6)
    assert size_z(24.0, flat) == (0.0, "sd0")
    assert size_z(26.0, flat) == (3.0, "sd0")
    assert size_z(22.0, flat) == (-3.0, "sd0")
    p = np.array([20.0, 22, 24, 26, 28, 30])
    z, f = size_z(27.0, p)
    assert f == "ok" and z == pytest.approx((27 - 25) / p.std(ddof=1))
    assert size_z(60.0, p) == (3.0, "clipped")
    assert size_z(-60.0, p) == (-3.0, "clipped")


def test_zS_post_own_amount_and_weights():
    amts = [20, 22, 24, 26, 28, 30, 27]
    ev = build_events(frame(monthly("2-Year", "2022-01", 7, [a * 1e9 for a in amts])), CAL)
    p = np.array(amts[:6], float)
    assert ev["n_prior"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert ev["zS_post"].iloc[:6].isna().all() and (ev["w_post"].iloc[:6] == 1.0).all()
    z = (27 - p.mean()) / p.std(ddof=1)
    assert ev["zS_post"].iloc[6] == pytest.approx(z)
    assert ev["w_post"].iloc[6] == pytest.approx(min(max(1 + z, 0), 2))


def test_break_of_more_than_a_year_restarts_priors():
    rows = monthly("7-Year", "2021-01", 8, [10e9] * 8, "OLD") + monthly("7-Year", "2023-03", 7, [30e9] * 6 + [32e9], "NEW")
    ev = build_events(frame(rows), CAL)
    new = ev[ev["cusip"].str.startswith("NEW")]
    assert new["n_prior"].tolist() == [0, 1, 2, 3, 4, 5, 6]            # the 2021 auctions do not count
    assert new["zS_post"].iloc[:6].isna().all()
    assert new["zS_post"].iloc[6] == 3.0 and new["zS_post_flag"].iloc[6] == "sd0"   # vs 6 x 30bn, not 10bn
    d = pd.Series(pd.to_datetime(["2021-01-01", "2021-06-01", "2022-08-01", "2022-09-01"]))
    assert run_starts(d).tolist() == [d[0], d[0], d[2], d[2]]


def test_pre_window_uses_the_size_known_at_A5():
    amts = [20e9, 22e9, 24e9, 26e9, 28e9, 30e9]
    rows = monthly("2-Year", "2022-01", 6, amts)
    A = CAL.nth_bday(pd.Period("2022-07", "M"), 15)
    a5 = CAL.offset(A, -5)
    late = build_events(frame(rows + [auction("EV", A, 40e9, ann=CAL.offset(A, -3))]), CAL).iloc[-1]
    early = build_events(frame(rows + [auction("EV", A, 40e9, ann=a5)]), CAL).iloc[-1]
    p = np.array(amts)
    assert late["S_pre_source"] == "previous" and late["S_pre"] == 30e9
    assert late["zS_pre"] == pytest.approx((30e9 - p.mean()) / p.std(ddof=1))
    assert early["S_pre_source"] == "own" and early["zS_pre"] == 3.0          # (40 - 25) / 3.74 = 4.0 -> clipped
    assert late["zS_post"] == early["zS_post"] == 3.0                          # the post leg always knows its size


def test_pre_window_priors_must_be_announced_by_A5():
    amts = [20e9, 22e9, 24e9, 26e9, 28e9, 30e9]
    rows = monthly("2-Year", "2022-01", 6, amts)
    A = CAL.nth_bday(pd.Period("2022-07", "M"), 15)
    extra = auction("XX", CAL.offset(A, -2), 50e9, ann=CAL.offset(A, -4))   # held before A, announced after A-5
    ev = build_events(frame(rows + [extra, auction("EV", A, 26e9, ann=CAL.offset(A, -3))]), CAL)
    e = ev.set_index("cusip").loc["EV"]
    p = np.array(amts)
    assert e["S_pre"] == 30e9 and e["zS_pre"] == pytest.approx((30e9 - p.mean()) / p.std(ddof=1))
    p_post = np.array(amts[1:] + [50e9])                                       # the post leg sees it
    assert e["zS_post"] == pytest.approx((26e9 - p_post.mean()) / p_post.std(ddof=1))


def test_no_lookahead_future_amounts_do_not_change_zS():
    amts = [20e9, 22e9, 24e9, 26e9, 28e9, 30e9, 27e9, 29e9, 31e9]
    rows = monthly("2-Year", "2022-01", 9, amts)
    base = build_events(frame(rows), CAL)
    rows2 = [dict(r) for r in rows]
    rows2[7]["offering_amt"], rows2[8]["offering_amt"] = 90e9, 1e9            # change auctions after event 6
    alt = build_events(frame(rows2), CAL)
    for c in ["zS_post", "zS_pre", "S_pre"]:
        assert alt.loc[6, c] == base.loc[6, c]
    rows3 = [dict(r) for r in rows]
    rows3[6]["offering_amt"] = 99e9                       # the event's own amount, announced two weeks ahead (by A-5)
    alt3 = build_events(frame(rows3), CAL)
    assert base.loc[6, "S_pre_source"] == "own" and alt3.loc[6, "zS_pre"] != base.loc[6, "zS_pre"]
    rows3[6]["announcemt_date"] = CAL.offset(rows3[6]["auction_date"], -2)     # now announced after A-5
    alt4 = build_events(frame(rows3), CAL)
    assert alt4.loc[6, "S_pre_source"] == "previous" and alt4.loc[6, "S_pre"] == 30e9
    rows3[6]["offering_amt"] = 5e9                        # unknown at A-5, so it cannot move the pre-window signal
    assert build_events(frame(rows3), CAL).loc[6, "zS_pre"] == alt4.loc[6, "zS_pre"]


def test_sample_membership_needs_the_whole_window():
    ev = build_events(frame([auction("W1", "2023-10-12", 40e9)]), CAL)
    assert in_sample_mask(ev, "2023-10-04", "2023-10-19").iloc[0]
    assert not in_sample_mask(ev, "2023-10-05", "2023-12-31").iloc[0]
    assert not in_sample_mask(ev, "2023-01-01", "2023-10-18").iloc[0]
    late = build_events(frame([auction("L1", "2024-12-26", 40e9)]), CAL)       # A+5 beyond the calendar
    assert not late.loc[0, "window_complete"] and not in_sample_mask(late, "2024-01-01", "2024-12-31").iloc[0]


def test_month_end_supply_counts_T8_to_T4():
    m = pd.Period("2024-05", "M")
    T = CAL.month_end(m)
    rows = [auction("A9", CAL.offset(T, -9), 10e9), auction("A8", CAL.offset(T, -8), 20e9, term="5-Year"),
            auction("A4", CAL.offset(T, -4), 30e9, term="30-Year"), auction("A3", CAL.offset(T, -3), 40e9)]
    ev = build_events(frame(rows), CAL)
    curve = Curve(pd.DataFrame({k: 4.0 for k in ["DGS2", "DGS5", "DGS10", "DGS30"]}, index=DAYS))
    sa = month_end_supply(ev, CAL, curve, [m])
    expect = (20e9 * float(mod_duration(4.0, 4.0, 5.0)) + 30e9 * float(mod_duration(4.0, 4.0, 30.0))) / 1e9
    assert sa.loc[m, "n_events"] == 2
    assert sa.loc[m, "SA_bn_years"] == pytest.approx(expect)


def test_zA_uses_past_months_only():
    sa = pd.Series(np.arange(50, dtype=float) ** 1.5, index=pd.period_range("2000-01", periods=50, freq="M"))
    z = past_zscore(sa)
    sa2 = sa.copy()
    sa2.iloc[45:] = 1e6
    assert z.iloc[:45].equals(past_zscore(sa2).iloc[:45])        # month 45 itself changed
    assert z.iloc[:36].isna().all() and z.iloc[36:].notna().all()
