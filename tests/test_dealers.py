"""Tests for H8 (src/dealers.py, src/data/pd_positions.py; PREREG_DEALERS.md) on synthetic data: no snapshot return is
computed and no dealer data is read. Publication timing, segment resets, the zS-style edge cases, the bucket checks,
the gate, and that the regression recovers a known c."""
import numpy as np
import pandas as pd
import pytest

import run_all
import src.trial_log as tl
from src import dealers
from src.calendar import BondCalendar
from src.data import pd_positions as pdp

HOLIDAYS = pd.to_datetime(["2015-11-26", "2015-12-25", "2016-01-01", "2016-07-04"])
DAYS = pd.bdate_range("2012-01-02", "2017-12-29").difference(HOLIDAYS)
CAL = BondCalendar(DAYS)


def weekly(n_a: int = 80, n_b: int = 120, seed: int = 1) -> pd.DataFrame:
    """Two segments of Wednesday releases: A (values around 100) then B (around 500)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-02", periods=n_a + n_b, freq="W-WED")
    vals = np.concatenate([100 + rng.normal(0, 5, n_a), 500 + rng.normal(0, 20, n_b)])
    return pd.DataFrame({"position_musd": vals, "segment": ["A"] * n_a + ["B"] * n_b}, index=idx)


def test_publication_is_the_first_bond_day_on_or_after_asof_plus_8():
    asof = pd.to_datetime(["2015-11-04", "2015-11-18", "2015-12-16"])
    pub = dealers.publication_dates(asof, CAL)
    assert list(pub) == list(pd.to_datetime(["2015-11-12", "2015-11-27", "2015-12-24"]))   # Thanksgiving: Friday
    assert pd.isna(dealers.publication_dates(pd.to_datetime(["2017-12-27"]), CAL)[0])      # beyond the calendar


def test_a_release_is_known_only_after_its_publication_day():
    w = weekly()
    asof = w.index[60]                                       # segment A has 80 releases: 60 earlier ones
    pub = dealers.publication_dates(pd.DatetimeIndex([asof]), CAL)[0]
    on, after = dealers.zd_at([pub], w, CAL).iloc[0], dealers.zd_at([CAL.offset(pub, 1)], w, CAL).iloc[0]
    assert on["asof"] == w.index[59] and after["asof"] == asof
    assert after["flag"] == "ok" and after["n_prior_in_segment"] == 60
    p = w["position_musd"].to_numpy()[8:60]
    z = (w["position_musd"].iloc[60] - p.mean()) / p.std(ddof=1)
    assert after["zD"] == pytest.approx(np.clip(z, -3, 3))
    first = dealers.zd_at([w.index[0]], w, CAL).iloc[0]
    assert first["flag"] == "none_known" and np.isnan(first["zD"])


def test_each_segment_restarts_the_52_release_history():
    w = weekly()
    known = lambda i: CAL.offset(dealers.publication_dates(pd.DatetimeIndex([w.index[i]]), CAL)[0], 1)   # noqa: E731
    z = dealers.zd_at([known(51), known(52), known(80 + 51), known(80 + 52)], w, CAL)
    assert list(z["flag"]) == ["segment_short", "ok", "segment_short", "ok"]
    assert z["segment"].iloc[2] == "B" and z["n_prior_in_segment"].iloc[2] == 51
    b = w["position_musd"].to_numpy()[80:132]                # only segment B values enter the B z-score
    assert z["zD"].iloc[3] == pytest.approx(np.clip((w["position_musd"].iloc[132] - b.mean()) / b.std(ddof=1), -3, 3))


def test_equal_priors_and_clip():
    idx = pd.date_range("2013-01-02", periods=60, freq="W-WED")
    vals = np.full(60, 50.0)
    vals[53], vals[54] = 50.0, 70.0
    vals[56:] = [50, 50, 50, 1e6]
    w = pd.DataFrame({"position_musd": vals, "segment": "A"}, index=idx)
    at = [CAL.offset(dealers.publication_dates(pd.DatetimeIndex([idx[i]]), CAL)[0], 1) for i in (53, 54, 59)]
    z = dealers.zd_at(at, w, CAL)["zD"].to_numpy()
    assert z[0] == 0.0 and z[1] == 3.0 and z[2] == 3.0


def events(c: float, n: int = 400, seed: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    A = pd.Series(pd.bdate_range("1999-03-01", periods=n * 15)[::15][:n])
    zs, zd = rng.normal(0, 1, n), np.clip(rng.normal(0, 1, n), -3, 3)
    post = 0.1 + 0.03 * zs + c / 2 * zd + rng.normal(0, 0.3, n)
    pre = -0.1 - 0.02 * zs - c / 2 * zd + rng.normal(0, 0.3, n)
    ev = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "A": A, "pre_entry": A - pd.Timedelta(days=7),
                       "tenor": "DGS10", "week": A.dt.to_period("W-SUN").astype(str), "skipped": False,
                       "R_pre": pre, "R_post": post, "LS": post - pre, "zS_pre": zs})
    ev.loc[0, "zS_pre"] = np.nan                                  # not an H6c event
    zdf = pd.DataFrame({"zD": zd, "flag": "ok", "segment": "SBP2013", "asof": A - pd.Timedelta(days=16),
                        "n_prior_in_segment": 60, "published": A - pd.Timedelta(days=8)},
                       index=pd.DatetimeIndex(ev["pre_entry"]))
    zdf.iloc[1, zdf.columns.get_loc("zD")] = np.nan
    zdf.iloc[1, zdf.columns.get_loc("flag")] = "segment_short"
    return ev, zdf


def test_h8_recovers_c_and_applies_the_pass_rule():
    ev, zd = events(0.2)
    r = dealers.h8(ev, zd)
    assert r["events"]["n_h6c"] == 399 and r["events"]["n_used"] == 398 == r["n"]
    assert r["events"]["n_without_zD"] == {"segment_short": 1}
    assert r["c"]["b"] == pytest.approx(0.2, abs=0.06) and r["pass"] and r["c"]["p_one_sided"] < 0.05
    sec = r["secondary"]
    assert sec["pre_leg"]["c"]["b"] + sec["post_leg"]["c"]["b"] == pytest.approx(r["c"]["b"])
    assert set(sec["by_decade"]) == set(dealers.DECADES) and "c" in sec["by_decade"]["2000-2009"]
    assert sec["by_decade"]["1993-1999"]["n"] > 0
    neg = dealers.h8(*events(-0.2))
    assert not neg["pass"] and neg["c"]["p_one_sided"] > 0.5 and neg["statement"].startswith("H8 fails")


def raw_positions(perturb: bool = False, drop: bool = False) -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2012-12-05", "2015-02-25", freq="W-WED"):
        if d < pd.Timestamp("2013-04-01"):
            cps = dict(zip(pdp.COUPON_KEYS["SBP2013"], [10, 20, -5, 7]))
            oth = {"PDPUSGTBNOP": 3, "PDPUSGTIISNOP": 2}
            tot = {"PDPUSGTNOP": sum(cps.values()) + 5 + (1 if perturb else 0)}
        else:
            sb = "SBN2013" if d < pd.Timestamp("2015-01-01") else "SBN2015"
            cps = {k: 4 for k in pdp.COUPON_KEYS[sb]}
            oth = {"PDPOSGS-B": 1, "PDPOSTIPS-L2": 1, "PDPOSTIPS-G2": 1, "PDPOSTIPS-G6L11": 1, "PDPOSTIPS-G11": 1}
            if sb == "SBN2015":
                oth["PDPOSGS-BFRN"] = 2
            tot = {"PDPOSGST-TOT": sum(cps.values()) + sum(oth.values())}
        for k, v in {**cps, **oth, **tot}.items():
            if drop and k == "PDPOSGSC-G11" and d == pd.Timestamp("2014-06-04"):
                continue
            rows.append({"asofdate": d.strftime("%Y-%m-%d"), "keyid": k, "value": str(v)})
    return pd.DataFrame(rows)


def test_parse_sums_buckets_and_checks_them_against_the_published_totals():
    p = pdp.parse(raw_positions(), end=None)
    assert p.loc["2013-03-27", "position_musd"] == 32 and p.loc["2013-03-27", "segment"] == "SBP2013"
    assert p.loc["2014-12-31", "position_musd"] == 24 and p.loc["2015-01-07", "segment"] == "SBN2015"
    assert p.attrs["checks"]["SBN"]["max_abs_diff_musd"] == 0.0
    assert pdp.parse(raw_positions(), end="2013-12-31").index.max() <= pd.Timestamp("2013-12-31")
    with pytest.raises(ValueError):
        pdp.parse(raw_positions(perturb=True), end=None)
    with pytest.raises(ValueError):
        pdp.parse(raw_positions(drop=True), end=None)


def test_loader_is_gated_by_the_prereg_dealers_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(tl, "dev_mode", lambda: True)
    monkeypatch.setattr(tl, "_tags", lambda: [tl.GATE1_TAG, tl.FLOWCLOCK_TAG])
    f = tmp_path / "p.csv"
    raw_positions().to_csv(f, index=False)
    with pytest.raises(tl.GateError):
        pdp.load_positions(end=None, path=f)
    with pytest.raises(tl.GateError):
        pdp.download()
    monkeypatch.setattr(tl, "_tags", lambda: [tl.GATE1_TAG, tl.FLOWCLOCK_TAG, tl.DEALERS_TAG])
    monkeypatch.setattr(tl, "_dealers_changed", lambda: False)
    assert len(pdp.load_positions(end=None, path=f)) > 100


@pytest.mark.filterwarnings("error")                             # no rank-deficient regression
def test_run_all_h8_block(monkeypatch):
    ev, _ = events(0.2, n=300)
    w = weekly(n_a=400, n_b=600)                               # releases cover every event (1999 to 2016)
    w.index = pd.date_range("1997-01-01", periods=len(w), freq="W-WED")
    w.attrs["checks"] = {"SBP2013": {"n_weeks": 1, "max_abs_diff_musd": 0.0}}
    monkeypatch.setattr(pdp, "load_positions", lambda end: w)
    cal = BondCalendar(pd.bdate_range("1996-01-01", "2016-12-30"))
    out = run_all.h8_block(ev, pd.Series(True, index=ev.index), cal)
    r = out["in_sample"]
    assert out["prereg"]["tag"] == "prereg-dealers" and r["n"] > 200 and "statement" in r
    assert set(r["events"]["n_without_zD"]) <= {"none_known", "segment_short"}
    assert out["data"]["checks_buckets_sum_to_published_total"]["SBP2013"]["max_abs_diff_musd"] == 0.0
