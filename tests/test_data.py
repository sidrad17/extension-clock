"""Tests for src/data/* parsers on fixtures copied from the real responses (no network)."""
import numpy as np
import pandas as pd
import pytest

from src.data import auctions, fomc, french, fred, mspd

# ------------------------------------------------------------------------------------------------- auctions

_COLS = ["cusip", "security_type", "security_term", "auction_date", "issue_date", "maturity_date",
         "announcemt_date", "offering_amt", "total_accepted", "int_rate", "reopening", "soma_accepted",
         "inflation_index_security", "floating_rate", "callable"]
_ROWS = [  # verbatim values from the 2026-10-03 auctions_query download
    ["912827KC5", "Note", "10-Year", "1979-10-31", "1979-11-15", "1989-11-15", "1979-10-24", "2000000000",
     "2401000000", "10.750000", "No", "null", "No", "No", "No"],
    ["912810DN5", "Bond", "30-Year", "1984-11-08", "1984-11-15", "2014-11-15", "1984-10-31", "5250000000",
     "6003093000", "11.750000", "No", "null", "No", "No", "Yes"],
    ["91282CCD1", "Note", "2-Year", "2021-05-25", "2021-06-01", "2023-05-31", "2021-05-20", "60000000000",
     "71646038200", "0.125000", "No", "11646008300", "No", "No", "No"],
    ["91282CJY8", "Note", "10-Year", "2024-01-18", "2024-01-31", "2034-01-15", "2024-01-11", "18000000000",
     "18000021100", "1.750000", "No", "0", "Yes", "No", "No"],
    ["91282CJU6", "Note", "2-Year", "2024-01-24", "2024-01-31", "2026-01-31", "2024-01-18", "28000000000",
     "28000612800", "null", "No", "0", "No", "Yes", "No"],
    ["91282CRQ6", "Note", "3-Year", "2026-10-06", "2026-10-15", "2029-10-15", "2026-10-01", "58000000000",
     "null", "null", "No", "null", "No", "No", "No"],
    ["91282CRF0", "Note", "9-Year 10-Month", "2026-10-07", "2026-10-15", "2036-08-15", "2026-10-01",
     "39000000000", "null", "4.625000", "Yes", "null", "No", "No", "No"],
]


@pytest.fixture
def raw_auctions():
    return pd.DataFrame(_ROWS, columns=_COLS)


def test_clean_auctions_types(raw_auctions):
    df = auctions.clean_auctions(raw_auctions)
    assert df["auction_date"].dtype.kind == "M" and df["offering_amt"].dtype.kind == "f"
    assert df["reopening"].tolist() == [False] * 6 + [True]
    assert np.isnan(df.loc[5, "int_rate"]) and df.loc[6, "int_rate"] == 4.625   # reopening keeps its coupon
    assert np.isnan(df.loc[5, "total_accepted"])


def test_public_amount_soma(raw_auctions):
    df = auctions.clean_auctions(raw_auctions)
    pub = auctions.public_amount(df, deduct_soma=True)
    assert pub[2] == 71646038200 - 11646008300          # SOMA known: total_accepted - soma_accepted
    assert pub[0] == 2000000000                          # pre-2008, no SOMA field: offering_amt
    assert pub[3] == 18000021100                         # soma_accepted == 0
    assert np.isnan(pub[5]) and np.isnan(pub[6])         # not yet auctioned
    raw = auctions.public_amount(df, deduct_soma=False)
    assert raw[0] == 2401000000 and raw[2] == 71646038200


def test_drop_excluded(raw_auctions):
    df = auctions.drop_excluded(auctions.clean_auctions(raw_auctions))
    assert set(df["cusip"]) == {"912827KC5", "91282CCD1", "91282CRQ6", "91282CRF0"}   # no callable, TIPS, FRN


def test_load_auctions_end_filter(raw_auctions, tmp_path):
    p = tmp_path / "auctions.csv"
    raw_auctions.to_csv(p, index=False)
    df = auctions.load_auctions(end="2024-09-30", path=p)
    assert df["auction_date"].max() <= pd.Timestamp("2024-09-30") and "public_amount" in df
    assert len(auctions.load_auctions(end=None, path=p)) == 4


# ------------------------------------------------------------------------------------------------------ FRED

def test_parse_fred_csv_missing_values():
    text = "observation_date,DGS30\n2002-02-14,5.42\n2002-02-15,5.37\n2002-02-18,\n2002-02-19,.\n"
    s = fred.parse_fred_csv(text, "DGS30")
    assert s.name == "DGS30" and s.iloc[1] == 5.37
    assert s.isna().tolist() == [False, False, True, True]
    with pytest.raises(ValueError):
        fred.parse_fred_csv(text, "DGS10")


def test_known_gap_mask():
    idx = pd.to_datetime(["2002-02-15", "2002-02-19", "2004-06-01", "2006-02-08", "2006-02-09"])
    s = fred.mask_gaps(pd.Series([5.37, 5.54, 5.3, 4.67, 4.51], index=idx, name="DGS30"))
    assert s.isna().tolist() == [False, True, True, True, False]


def test_ffill_only_short_gaps():
    v = [np.nan, 1.0] + [np.nan] * 5 + [2.0] + [np.nan] * 6 + [3.0]
    s = fred.ffill_short_gaps(pd.Series(v, index=pd.bdate_range("2024-01-01", periods=len(v))))
    assert np.isnan(s.iloc[0])                        # leading NaN never filled
    assert (s.iloc[2:7] == 1.0).all()                 # run of 5: filled
    assert s.iloc[8:14].isna().all()                  # run of 6: left entirely missing
    assert s.iloc[-1] == 3.0


# ------------------------------------------------------------------------------------------------ Ken French

FRENCH_TEXT = """This file was created by using the 202608 CRSP database.
The Tbill return is the simple daily rate that, over the number of trading days
compounds to 1-month TBill rate.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
20260827,    0.72,    0.20,   -1.14,    0.01
20260828,   -0.34,   -0.51,    0.28,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_daily():
    f = french.parse_daily(FRENCH_TEXT)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and len(f) == 3
    assert f.index[-1] == pd.Timestamp("2026-08-28") and f.loc["2026-08-27", "Mkt-RF"] == 0.72
    d = french.derive_pension_input(f, start="2026-01-01")
    assert d["mkt_total_pct"].tolist() == [0.73, -0.33] and list(d.columns) == ["date", "mkt_total_pct"]
    with pytest.raises(ValueError):
        french.parse_daily(FRENCH_TEXT.replace(",Mkt-RF", ",MKT"))


# ------------------------------------------------------------------------------------------------------ MSPD

def test_mspd_notes_bonds(tmp_path):
    raw = pd.DataFrame([
        ["2026-08-31", "Marketable", "Bills", "7247855.5898", "214.4309", "7248070.0207"],
        ["2026-08-31", "Marketable", "Notes", "16214682.0126", "3738.9251", "16218420.9377"],
        ["2026-08-31", "Marketable", "Bonds", "5512728.5203", "12555.574", "5525284.0943"],
        ["2026-08-31", "Marketable", "Treasury Inflation-Protected Securities", "2151956.98", "702.86", "2152659.84"],
        ["2026-08-31", "Total Marketable", "_", "31807141.11", "20860.35", "31828001.46"],
    ], columns=mspd.KEEP_COLS)
    nb = mspd.notes_bonds(raw)
    assert nb["security_class_desc"].tolist() == ["Bonds", "Notes"]
    p = tmp_path / "m.csv"
    nb.to_csv(p, index=False)
    w = mspd.load_mspd(end=None, path=p)
    assert w.loc["2026-08-31", "notes_bonds_total"] == pytest.approx(16218420.9377 + 5525284.0943)


# ------------------------------------------------------------------------------------------------------ FOMC

@pytest.mark.parametrize("text,year,start,end,kind", [
    ("January 29-30 Meeting", 2008, "2008-01-29", "2008-01-30", "scheduled"),
    ("April/May 30-1 Meeting", 2019, "2019-04-30", "2019-05-01", "scheduled"),
    ("June 30-July 1 Meeting", 1998, "1998-06-30", "1998-07-01", "scheduled"),
    ("January 21 Conference Call", 2008, "2008-01-21", "2008-01-21", "conference_call"),
    ("March 2 (unscheduled) Meeting", 2020, "2020-03-02", "2020-03-02", "unscheduled"),
    ("October 4 (unscheduled)", 2019, "2019-10-04", "2019-10-04", "unscheduled"),
    ("March 17-18 (cancelled) Meeting", 2020, "2020-03-17", "2020-03-18", "cancelled"),
    ("March 19 (notation vote)", 2020, "2020-03-19", "2020-03-19", "notation_vote"),
    ("September 15-16* ", 2026, "2026-09-15", "2026-09-16", "scheduled"),
])
def test_parse_fomc_days(text, year, start, end, kind):
    ev = fomc.parse_days(text, year)
    assert str(ev["start_date"]) == start and str(ev["end_date"]) == end and ev["kind"] == kind


HIST_2008 = """<div class="panel"><div class="panel-heading"><h5>January 9 Conference Call - 2008</h5></div>
<p>Minutes: See end of minutes of January 29/30 meeting</p></div>
<div class="panel"><div class="panel-heading"><h5>January 21 Conference Call - 2008</h5></div>
<a href="/newsevents/press/monetary/20080122b.htm">Statement</a></div>
<div class="panel"><div class="panel-heading"><h5>January 29-30 Meeting - 2008</h5></div>
<a href="/newsevents/press/monetary/20080130a.htm">Statement</a></div>"""

HIST_2003 = """<div class="panel"><div class="panel-heading"><h5>September 15 Meeting - 2003</h5></div>
<a href="/monetarypolicy/files/FOMC20030915Agenda.pdf">Agenda (25 KB PDF)</a></div>
<div class="panel"><div class="panel-heading"><h5>September 16 Meeting - 2003</h5></div>
<a href="/boarddocs/press/monetary/2003/20030916/default.htm">Statement</a></div>"""

CURRENT = """<div class="panel"><div class="panel-heading"><h4>2025 FOMC Meetings</h4></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date">30-1</div>
<div><strong>Statement:</strong><br/><a href="/monetarypolicy/files/monetary20250501a1.pdf">PDF</a> |
<a href="/newsevents/pressreleases/monetary20250501a.htm">HTML</a></div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>August</strong></div>
<div class="fomc-meeting__date">22 (notation vote)</div>
<div><a href="/newsevents/pressreleases/monetary20250822a.htm">Statement on Longer-Run Goals and Monetary Policy
Strategy</a></div></div>
<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>December</strong></div>
<div class="fomc-meeting__date">9-10*</div></div></div>"""


def test_fomc_decision_rule():
    df = fomc.decide(fomc.parse_historical(HIST_2008, 2008) + fomc.parse_historical(HIST_2003, 2003)
                     + fomc.parse_current(CURRENT))
    got = df.set_index("label")
    assert got.loc["January 9 Conference Call - 2008", "decision"] == 0           # call, no statement
    assert str(got.loc["January 21 Conference Call - 2008", "decision_date"]) == "2008-01-22"  # statement date
    assert str(got.loc["January 29-30 Meeting - 2008", "decision_date"]) == "2008-01-30"
    assert got.loc["September 15 Meeting - 2003", "kind"] == "special_meeting"
    assert got.loc["September 15 Meeting - 2003", "decision"] == 0
    assert str(got.loc["Apr/May 30-1 - 2025", "decision_date"]) == "2025-05-01"
    assert got.loc["August 22 (notation vote) - 2025", "decision"] == 0           # longer-run goals only
    assert str(got.loc["December 9-10* - 2025", "decision_date"]) == "2025-12-10"  # future meeting, no statement yet
    assert fomc.current_page_years(CURRENT) == [2025]


def test_load_fomc_dates(tmp_path):
    df = fomc.decide(fomc.parse_historical(HIST_2008, 2008))
    p = tmp_path / "fomc.csv"
    df.to_csv(p, index=False)
    d = fomc.load_fomc_dates(p)
    assert list(d) == list(pd.to_datetime(["2008-01-22", "2008-01-30"]))
    assert list(fomc.load_fomc_dates(p, end="2008-01-25")) == [pd.Timestamp("2008-01-22")]
