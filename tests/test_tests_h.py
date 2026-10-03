"""Tests for src/tests_h.py, src/figures.py and src/report.py on synthetic data."""
import json

import numpy as np
import pandas as pd
import pytest

from src import figures, report
from src.calendar import BondCalendar
from src.tests_h import (event_path_summary, event_paths, h1, h1_addendum, luck_candidates, luck_test,
                         tercile_labels)


@pytest.fixture
def cal():
    return BondCalendar(pd.bdate_range("2019-01-02", "2020-12-31"))


def _series(n=120, seed=0, slope=0.05):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M")
    z = pd.Series(rng.normal(size=n), index=idx)
    R = 0.05 + slope * z + pd.Series(rng.normal(0, 0.3, n), index=idx)
    return R, -R * 12.0, z


def test_h1_recovers_slope_and_terciles():
    R, Y, z = _series(2000, slope=0.1)
    r = h1(R, Y, z)
    assert r["ci"][0] < 0.1 < r["ci"][1] and r["n"] == 2000
    assert r["terciles"]["high"]["mean"] > r["terciles"]["low"]["mean"]
    assert sum(r["terciles"][k]["n"] for k in ("low", "mid", "high")) == 2000
    assert (tercile_labels(z).value_counts() > 600).all()


def test_addendum_within_group_slopes_match_split_regressions():
    R, Y, z = _series(240, seed=4)
    ref = pd.Series(R.index.month.isin([2, 5, 8, 11]).astype(int), index=R.index)
    a = h1_addendum(R, Y, z, ref)
    sel = ref == 1
    b_ref = np.polyfit(z[sel], R[sel], 1)[0]
    b_oth = np.polyfit(z[~sel], R[~sel], 1)[0]
    assert a["within_refunding"]["b"] == pytest.approx(b_ref)
    assert a["within_other"]["b"] == pytest.approx(b_oth)
    assert a["within_refunding"]["n"] == 80 and a["within_other"]["n"] == 160


def test_luck_candidates_avoid_month_end_reversal_and_15th(cal):
    x = pd.Series(0.001, index=cal.days)
    months = pd.period_range("2019-03", "2020-10", freq="M")
    cands = luck_candidates(cal, months, x)
    m = pd.Period("2019-10", "M")
    days = cal.month_days(m)
    T = cal.month_end(m)
    allowed = [d for d in days[3:] if d < cal.offset(T, -4) and d.day not in (14, 15, 16)]
    runs, cur = [], []
    for d in days:
        if d in allowed:
            cur.append(d)
        else:
            runs.append(cur)
            cur = []
    runs.append(cur)
    expected = sum(max(len(r) - 3, 0) for r in runs)
    assert len(cands[m]) == expected > 0
    assert np.allclose(cands[m], 1.001 ** 4 - 1)
    R = pd.Series(0.5, index=months)
    lt = luck_test(cands, R, n_draws=200)
    assert lt["label"] == "luck test"
    assert lt["share_random_ge_month_end"] == 0.0
    assert luck_test(cands, R, n_draws=200)["random_mean_pct"] == lt["random_mean_pct"]


def test_event_path_and_figures(cal, tmp_path):
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.0002, 0.003, len(cal.days)), index=cal.days)
    months = pd.period_range("2019-02", "2020-12", freq="M")
    paths = event_paths(cal, months, x)
    assert pd.Period("2020-12", "M") not in paths.index           # T+5 beyond the data
    assert (paths[-10] == 0).all()
    m = paths.index[0]
    T = cal.month_end(m)
    assert paths.loc[m, 0] == pytest.approx((np.prod(1 + x.loc[cal.offset(T, -10):T].iloc[1:]) - 1) * 100)
    z = pd.Series(rng.normal(size=len(months)), index=months)
    summ = event_path_summary(paths, tercile_labels(z))
    cap = figures.event_path(summ, tmp_path / "e.png", "test")
    assert (tmp_path / "e.png").stat().st_size > 0 and "T-4" in cap
    R, Y, zz = _series(90)
    cap2 = figures.terciles(h1(R, Y, zz), tmp_path / "t.png", "test")
    assert (tmp_path / "t.png").stat().st_size > 0 and "high" in cap2


def test_report_clean_rounds_and_nulls():
    out = report.clean({"a": np.float64(1.23456789), "b": np.nan, "c": [np.int64(3), True], "d": pd.Period("2020-01")})
    assert out == {"a": 1.23457, "b": None, "c": [3, True], "d": "2020-01"}
    json.dumps(out)
    sk = report.skeleton()
    assert set(sk["in_sample"]["H1_addendum"]) == {"refunding_dummy", "within_refunding", "within_other", "surprise"}
