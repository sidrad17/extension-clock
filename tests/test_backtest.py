"""Tests for src/backtest.py and src/metrics.py on synthetic data: windows, P&L, costs, NAV."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import month_end_windows, placebo_windows, run_strategy, window_return
from src.bonds import mod_duration
from src.calendar import BondCalendar
from src.metrics import max_drawdown, required_metrics
from src.risk import RiskConfig, sigma_bp


@pytest.fixture
def setup():
    days = pd.bdate_range("2019-06-03", "2020-06-30")
    cal = BondCalendar(days)
    rng = np.random.default_rng(5)
    y = pd.Series(2.0 + np.cumsum(rng.normal(0, 0.04, len(days))), index=days)
    x = pd.Series(rng.normal(0.0001, 0.004, len(days)), index=days)
    rf = pd.Series(0.00005, index=days)
    months = pd.period_range("2019-10", "2020-03", freq="M")
    return cal, y, x, rf, months


def test_windows(setup):
    cal, _, _, _, months = setup
    w = month_end_windows(cal, months)
    assert (w.loc[pd.Period("2019-11", "M"), "T"]) == pd.Timestamp("2019-11-29")
    assert w.loc[pd.Period("2019-11", "M"), "entry"] == pd.Timestamp("2019-11-25")
    p = placebo_windows(cal, months)
    # signal month 2019-11 -> window in 2019-12: enter close of bday 3 (Dec 4), exit close of bday 7 (Dec 10)
    assert p.loc[pd.Period("2019-11", "M"), "entry"] == pd.Timestamp("2019-12-04")
    assert p.loc[pd.Period("2019-11", "M"), "exit"] == pd.Timestamp("2019-12-10")


def test_window_return_excludes_entry_day(setup):
    _, _, x, _, _ = setup
    e, t = x.index[10], x.index[14]
    assert window_return(x, e, t) == pytest.approx(np.prod(1 + x.iloc[11:15]) - 1)


def test_strategy_pnl_cost_and_flat_outside(setup):
    cal, y, x, rf, months = setup
    win = month_end_windows(cal, months)
    days = cal.days[(cal.days >= "2019-10-01") & (cal.days <= "2020-03-31")]
    w = pd.Series(1.5, index=months)
    cfg = RiskConfig(fomc_half=False, dd_rule=False)
    res = run_strategy("t", win, w, x, rf, y, y, 10.0, pd.DatetimeIndex([]), cal, days, cfg)
    m = months[2]
    e, T = win.loc[m, "entry"], win.loc[m, "T"]
    sig = sigma_bp(y, e)
    dur = float(mod_duration(y[e], y[e], 10.0))
    dv01 = 0.01 * 1e7 / (sig * 2.0) * 1.5
    n = dv01 / (dur * 1e-4)
    tr = res.trades.loc[m]
    assert tr["notional"] == pytest.approx(n)
    assert tr["cost"] == pytest.approx(0.5 * dv01)
    gross = n * x.loc[e:T].iloc[1:]
    assert tr["gross_pnl"] == pytest.approx(gross.sum())
    assert res.daily.loc[e, "pnl"] == pytest.approx(-0.25 * dv01)
    assert res.daily.loc[T, "pnl"] == pytest.approx(gross.iloc[-1] - 0.25 * dv01)
    in_window = np.zeros(len(days), bool)
    for _, r in win.iterrows():
        in_window |= (days >= r["entry"]) & (days <= r["exit"])
    assert (res.daily.loc[~in_window, "pnl"] == 0).all()
    assert np.allclose(res.daily["total"], rf.loc[days] + res.daily["excess"])
    assert res.trades["net_pnl"].sum() == pytest.approx(res.daily["pnl"].sum())
    met = required_metrics(res)
    assert met["n_windows"] == len(months)
    assert met["turnover_x_per_year"] == pytest.approx(2 * res.trades["notional"].sum() / 1e7 / met["years"])
    assert met["sharpe_gross"] > met["sharpe"]


def test_overlapping_windows_refused(setup):
    cal, y, x, rf, months = setup
    win = month_end_windows(cal, months)
    win.iloc[1, win.columns.get_loc("entry")] = win.iloc[0]["exit"]
    days = cal.days[(cal.days >= "2019-10-01") & (cal.days <= "2020-03-31")]
    with pytest.raises(ValueError):
        run_strategy("t", win, pd.Series(1.0, index=months), x, rf, y, y, 10.0, pd.DatetimeIndex([]), cal, days)


def test_max_drawdown():
    nav = pd.Series([1.1, 0.99, 1.2, 0.9, 1.0])
    assert max_drawdown(nav) == pytest.approx(1 - 0.9 / 1.2)
