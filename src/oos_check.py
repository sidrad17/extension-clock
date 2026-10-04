"""Test-only check of the --oos path (CLAUDE.md section 17): the existing engines run directly on a date window of
the committed snapshot (no data view, no splice), for comparison with run_all.evaluate_window on the same window
(`python run_all.py --oos-pseudo`, tests/test_oos.py). Strategies start flat at the window's first day and signals
use all history, as the post-publication sub-sample does in run_all.month_end_phase5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.flowclock import FC_TENORS, SA_START
from config.settings import COST_STRESS, HEADLINE_TENOR, RF_SERIES, TENORS
from src import dealers
from src import futures as fut
from src.auction_events import build_events, in_sample_mask, month_end_supply
from src.backtest import month_end_windows, run_curve_allocated, run_strategy, window_returns, window_yield_change_bp
from src.bonds import KNOT_YEARS, load_curve
from src.calendar import load_calendar
from src.data import databento_futures as dbf
from src.data import pd_positions
from src.data.auctions import load_auctions
from src.data.fomc import load_fomc_dates
from src.data.fred import load_frame
from src.data.french import load_pension_input
from src.flowclock import demand_legs, event_returns, h6, h7, run_book, supply_legs
from src.index_rebuild import BUCKETS, run_default
from src.returns import tenor_returns
from src.risk import WINDOWS_PER_YEAR, RiskConfig
from src.signals import build_signals, past_zscore, pension_pressure
from src.tests_h import h1, h2, h3, h4, h5


def reference_window(start: str, end: str, with_futures: bool) -> dict:
    """{"daily": daily net excess per strategy key, "stats": {path: value}} from the existing engines."""
    cal = load_calendar(end=end)
    monthly, _ = run_default(end=end)
    total, excess = tenor_returns(end=end)
    x10 = excess[HEADLINE_TENOR]
    rf = total[HEADLINE_TENOR] - x10
    y10 = load_frame([HEADLINE_TENOR, RF_SERIES], end=end, index=cal.days, fill=True)[HEADLINE_TENOR]
    fomc = load_fomc_dates(end=end, scheduled_only=True)
    mf = pd.DataFrame({"E": pd.to_datetime(monthly["E"].to_numpy())}, index=pd.PeriodIndex(monthly["month"], freq="M"))
    sig = build_signals(monthly, fomc, pension_pressure(mf, cal, load_pension_input(end=end), total[HEADLINE_TENOR]))
    months = sig.index[(sig.index >= pd.Period(start, "M")) & (sig.index <= pd.Period(end, "M"))]
    s = sig.loc[months]
    win = month_end_windows(cal, months)
    days = cal.days[(cal.days >= pd.Timestamp(start)) & (cal.days <= pd.Timestamp(end))]
    args = dict(daily_excess=x10, rf=rf, y_tenor=y10, y10=y10, tenor_years=KNOT_YEARS[HEADLINE_TENOR],
                fomc_scheduled=fomc, cal=cal, days=days)
    ones = pd.Series(1.0, index=months)
    c1, c2 = RiskConfig(), RiskConfig(cost_mult=COST_STRESS)
    d = {"forecast_sized": run_strategy("fs", win, s["w"], cfg=c1, **args).daily["excess"],
         "calendar_only": run_strategy("co", win, ones, cfg=c1, **args).daily["excess"],
         "forecast_sized_cost_2x": run_strategy("fs2", win, s["w"], cfg=c2, **args).daily["excess"],
         "calendar_only_cost_2x": run_strategy("co2", win, ones, cfg=c2, **args).daily["excess"]}
    yall = load_frame(list(TENORS.values()), end=end, index=cal.days, fill=True)
    midx = pd.PeriodIndex(monthly["month"], freq="M")
    fdd_b = pd.DataFrame({b: monthly[f"fdd_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    ext_b = pd.DataFrame({b: monthly[f"ext_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    ca = dict(demand=fdd_b.loc[months], tenor_of=TENORS, excess=excess, yields=yall, rf=rf, y10=y10,
              fomc_scheduled=fomc, cal=cal, days=days)
    d["curve_allocated"] = run_curve_allocated("ca", win, cfg=c1, **ca).daily["excess"]
    d["curve_allocated_cost_2x"] = run_curve_allocated("ca2", win, cfg=c2, **ca).daily["excess"]
    R = window_returns(x10, win) * 100.0
    Y = -window_yield_change_bp(y10, win)
    neg_dy_b = pd.DataFrame({b: -(yall[TENORS[b]].reindex(win["exit"]).to_numpy()
                                  - yall[TENORS[b]].reindex(win["entry"]).to_numpy()) * 100.0 for b in BUCKETS},
                            index=months)
    rw = pd.DataFrame({m: {"entry": cal.month_end(m), "exit": cal.offset(cal.month_end(m), 3)} for m in months
                       if cal.days.get_loc(cal.month_end(m)) + 3 < len(cal.days)}).T
    st = {"month_end.H1.b": h1(R, Y, s["z"])["b"],
          "month_end.H2.coef": h2(neg_dy_b, fdd_b.apply(past_zscore).loc[months],
                                  ext_b.apply(past_zscore).loc[months])["coef"],
          "month_end.H3.coef": h3(window_returns(x10, rw) * 100.0, -window_yield_change_bp(y10, rw), s["z"], R)["coef"],
          "month_end.H4.diff": h4(d["forecast_sized"], d["calendar_only"])["diff"],
          "month_end.H5.coefs.z.b": h5(R, Y, s)["coefs"]["z"]["b"]}

    yld = load_frame(FC_TENORS, end=end, index=cal.days, fill=True)
    ev = build_events(load_auctions(end=end, exclude=None), cal, yld)
    _, xs = tenor_returns(end=end, tenors={t: t for t in FC_TENORS})
    ev = ev.join(event_returns(ev, xs, yld))
    mask = in_sample_mask(ev, start, end)
    sa = month_end_supply(ev, cal, load_curve(end=end), pd.period_range(SA_START, end[:7], freq="M"))
    sa["zA"] = past_zscore(sa["SA_bn_years"])
    ok = ev["window_complete"] & ~ev["skipped"]
    sup = pd.DatetimeIndex(pd.concat([ev.loc[ok, "pre_entry"], ev.loc[ok, "A"]]))
    bk = dict(excess=xs, yields=yld, rf=rf, fomc_scheduled=fomc, cal=cal, supply_entries=sup, days=days)
    e = ev[mask & ~ev["skipped"]]
    dem = demand_legs(win, cal)
    both = pd.concat([dem, supply_legs(e, False)], ignore_index=True)
    d["supply_calendar"] = run_book("s", supply_legs(e, False), cfg=c1, **bk).daily["excess"]
    d["supply_calendar_cost_2x"] = run_book("s2", supply_legs(e, False), cfg=c2, **bk).daily["excess"]
    d["supply_size_weighted"] = run_book("w", supply_legs(e, True), cfg=c1, **bk).daily["excess"]
    d["flowclock_book"] = run_book("b", both, cfg=c1, demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk) \
        .daily["excess"]
    d["flowclock_book_cost_2x"] = run_book("b2", both, cfg=c2, demand_per_year=WINDOWS_PER_YEAR, unit="month",
                                           **bk).daily["excess"]
    st["flowclock.H6.H6c.b"] = h6(ev[mask])["H6c"]["b"]
    st["flowclock.H7.c.b"] = h7(R, Y, sa.loc[months, "zA"])["c"]["b"]
    zd = dealers.zd_at(pd.DatetimeIndex(sorted(ev.loc[mask, "pre_entry"].dropna().unique())),
                       pd_positions.load_positions(end=end), cal)
    st["H8.c.b"] = dealers.h8(ev[mask], zd)["c"]["b"]

    if with_futures:
        import json

        from config.futures import SUPPLY_CONTRACT  # noqa: F401  (documents the mapping used by supply_legs_futures)
        from src.data.snapshot import REPO_ROOT
        defs = dbf.load_definition_snapshots()
        m = fut.build_market(dbf.load_settlements(end=end), dbf.load_volume(end=end), defs, cal,
                             dbf.point_values(defs))
        first = json.loads((REPO_ROOT / "outputs" / "tables" / "futures_data_checks_insample.json")
                           .read_text(encoding="utf-8"))["first_sizable_entry"]
        me_legs = fut.month_end_legs(demand_legs(win, cal))
        su_legs = fut.supply_legs_futures(supply_legs(e, False))
        me_p = fut.prepare_legs(me_legs, m, yld, first_ok_given=first["month_end"])
        su_p = fut.prepare_legs(su_legs, m, yld, first_ok_given=first["supply"])
        fa = dict(m=m, yields=yld, rf=rf, fomc_scheduled=fomc, cal=cal, days=days)
        for sfx, cfg in (("", c1), ("_cost_2x", c2)):
            d[f"futures_month_end_zn{sfx}"] = fut.run_futures_book("m", me_legs, me_p, cfg=cfg, unit="month",
                                                                    demand_per_year=WINDOWS_PER_YEAR, **fa) \
                .daily["excess"]
            d[f"futures_supply_calendar{sfx}"] = fut.run_futures_book("s", su_legs, su_p, cfg=cfg,
                                                                       supply_entries=sup, **fa).daily["excess"]
    return {"daily": d, "stats": st}


def _at(d: dict, path: str):
    for k in path.split("."):
        d = d[k]
    return d


def compare(out: dict, ref: dict) -> dict:
    """Max absolute difference per daily series (on the same days) and per statistic; a missing series or a
    different day index counts as infinite."""
    diffs = {}
    for k, r in ref["daily"].items():
        o = out["_daily"].get(k)
        if o is None or not o.index.equals(r.index):
            diffs[f"daily.{k}"] = float("inf")
            continue
        diffs[f"daily.{k}"] = float(np.nanmax(np.abs(o.to_numpy(float) - r.to_numpy(float))))
    for path, v in ref["stats"].items():
        diffs[path] = abs(float(_at(out, path)) - float(v))
    return diffs
