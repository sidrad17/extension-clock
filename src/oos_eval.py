"""Test-window labels and kill conditions (CLAUDE.md section 17), written before any test-window data was downloaded.

Criteria come from HYPOTHESIS.md and PREREG_*.md where they exist:
* H1: "the test window can confirm the sign, not significance" (power note): "sign confirmed" if b > 0.
* H4: forecast-sized beats calendar-only "in sign in the test window": "pass" if the net Sharpe difference > 0, 1x.
* H8: "pass" if c > 0 and the one-sided p < 0.05 (PREREG_DEALERS.md).
Everything else reports the estimate, its t (or its bootstrap interval for a Sharpe difference) and its sign,
labelled "consistent" when the sign matches the in-sample sign, else "not consistent"; no pass or fail. The
in-sample reference is the full in-sample block of results.json (1993-01 to 2024-09; futures 2010-07 to 2024-09).
Kill conditions apply as written: a comparison in the text is evaluated; a word without a written threshold
("centered on zero", "vanishes", "disappears") is reported with its numbers for the team; an in-sample condition is
marked not applicable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CONSISTENT, NOT_CONSISTENT, NOT_COMPUTED = "consistent", "not consistent", "not computed"


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def get(d: dict, path: str):
    """d["a"]["b"]... for path "a.b..."; None if any key is missing."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def sign_label(oos, ins) -> str:
    """"consistent" if the test-window value and the in-sample value have the same (non-zero) sign."""
    a, b = _num(oos), _num(ins)
    if a is None or b is None:
        return NOT_COMPUTED
    return CONSISTENT if np.sign(a) == np.sign(b) and a != 0 else NOT_CONSISTENT


def mean_t(daily_excess: pd.Series) -> float | None:
    """t of the mean daily net excess return: mean / (sd / sqrt(days))."""
    r = pd.Series(daily_excess, dtype=float).dropna().to_numpy()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))


# (name, oos path, in-sample path, kind) for the "consistent" rows; kind "coef" = {b, t}, "diff" = Sharpe difference
# {diff, ci, p_le_0}, "value" = a plain number. oos paths are relative to the oos blocks {"month_end", "flowclock",
# "H8", "futures"}; in-sample paths to results.json.
CONSISTENCY_ROWS = [
    ("H1 component: extension", "month_end.H1.components.ext", "in_sample.H1.components.ext", "coef"),
    ("H1 component: coupon cash", "month_end.H1.components.cash", "in_sample.H1.components.cash", "coef"),
    ("Addendum (a): b with refunding dummy", "month_end.H1_addendum.refunding_dummy.b",
     "in_sample.H1_addendum.refunding_dummy.b", "coef"),
    ("Addendum (a): refunding dummy c", "month_end.H1_addendum.refunding_dummy.c",
     "in_sample.H1_addendum.refunding_dummy.c", "coef"),
    ("Addendum (b): b within refunding months", "month_end.H1_addendum.within_refunding",
     "in_sample.H1_addendum.within_refunding", "coef"),
    ("Addendum (b): b within other months", "month_end.H1_addendum.within_other",
     "in_sample.H1_addendum.within_other", "coef"),
    ("Addendum (c): surprise extension b_s", "month_end.H1_addendum.surprise", "in_sample.H1_addendum.surprise",
     "coef"),
    ("H2: bucket demand coefficient", "month_end.H2", "in_sample.H2", "coef2"),
    ("H3: reversal slope on z", "month_end.H3", "in_sample.H3", "coef2"),
    ("H3: mean reversal R3", "month_end.H3.mean_R3", "in_sample.H3.mean_R3", "coef"),
    *[(f"H5: {k}", f"month_end.H5.coefs.{k}", f"in_sample.H5.coefs.{k}", "coef")
      for k in ("const", "z", "z_pension", "quarter_end", "year_end", "ref", "fomc")],
    ("Placebo: H1 b (bdays 4-7)", "month_end.placebo.H1", "in_sample.placebo.H1", "coef"),
    ("Placebo: H4 Sharpe difference", "month_end.placebo.H4", "in_sample.placebo.H4", "diff"),
    ("Luck test: month-end mean (%)", "month_end.random_windows.month_end_mean_pct",
     "in_sample.random_windows.month_end_mean_pct", "value"),
    ("H4 at 2x costs: Sharpe difference", "month_end.cost_stress.H4", "in_sample.cost_stress.H4", "diff"),
    ("Curve-allocated vs calendar-only, 1x", "month_end.curve_allocated.vs_calendar_only",
     "in_sample.curve_allocated.vs_calendar_only", "diff"),
    ("Curve-allocated vs calendar-only, 2x", "month_end.curve_allocated.vs_calendar_only_cost_2x",
     "in_sample.curve_allocated.vs_calendar_only_cost_2x", "diff"),
    ("H6a: mean pre-auction return", "flowclock.H6.H6a", "flowclock.in_sample.H6.H6a", "coef"),
    ("H6b: mean post-auction return", "flowclock.H6.H6b", "flowclock.in_sample.H6.H6b", "coef"),
    ("H6c: size slope beta", "flowclock.H6.H6c", "flowclock.in_sample.H6.H6c", "coef"),
    ("H7: c on zA", "flowclock.H7.c", "flowclock.in_sample.H7.c", "coef"),
    ("H7: a (month-end component)", "flowclock.H7.a", "flowclock.in_sample.H7.a", "coef"),
    ("Headline 3: book vs demand leg, 1x", "flowclock.headline3.cost_1x", "flowclock.in_sample.headline3.cost_1x",
     "diff"),
    ("Headline 3: book vs demand leg, 2x", "flowclock.headline3.cost_2x", "flowclock.in_sample.headline3.cost_2x",
     "diff"),
    ("Size-weighted vs calendar supply leg", "flowclock.supply_size_vs_calendar",
     "flowclock.in_sample.supply_size_vs_calendar", "diff"),
]

# (name, oos metrics path, in-sample metrics path, daily key) for every strategy at 1x and 2x
STRATEGY_ROWS = [
    ("forecast-sized, 1x", "month_end.metrics.forecast_sized", "in_sample.metrics.cash.forecast_sized",
     "forecast_sized"),
    ("forecast-sized, 2x", "month_end.cost_stress.metrics.forecast_sized", "in_sample.cost_stress.metrics.forecast_sized",
     "forecast_sized_cost_2x"),
    ("calendar-only, 1x", "month_end.metrics.calendar_only", "in_sample.metrics.cash.calendar_only", "calendar_only"),
    ("calendar-only, 2x", "month_end.cost_stress.metrics.calendar_only", "in_sample.cost_stress.metrics.calendar_only",
     "calendar_only_cost_2x"),
    ("curve-allocated, 1x", "month_end.metrics.curve_allocated", "in_sample.metrics.cash.curve_allocated",
     "curve_allocated"),
    ("curve-allocated, 2x", "month_end.curve_allocated.metrics_cost_2x", "in_sample.curve_allocated.metrics_cost_2x",
     "curve_allocated_cost_2x"),
    ("Flow Clock book, 1x", "flowclock.metrics.book", "flowclock.in_sample.metrics.book", "flowclock_book"),
    ("Flow Clock book, 2x", "flowclock.metrics.book_cost_2x", "flowclock.in_sample.metrics.book_cost_2x",
     "flowclock_book_cost_2x"),
    ("cash supply leg, 1x", "flowclock.metrics.supply_calendar", "flowclock.in_sample.metrics.supply_calendar",
     "supply_calendar"),
    ("cash supply leg, 2x", "flowclock.metrics.supply_calendar_cost_2x",
     "flowclock.in_sample.metrics.supply_calendar_cost_2x", "supply_calendar_cost_2x"),
    ("demand leg alone, 1x", "flowclock.metrics.demand_alone", "flowclock.in_sample.metrics.demand_alone",
     "calendar_only"),
    ("demand leg alone, 2x", "flowclock.metrics.demand_alone_cost_2x", "flowclock.in_sample.metrics.demand_alone_cost_2x",
     "calendar_only_cost_2x"),
    ("futures month-end ZN, 1x", "futures.month_end_zn.metrics", "futures.month_end_zn.metrics", "futures_month_end_zn"),
    ("futures month-end ZN, 2x", "futures.month_end_zn.metrics_cost_2x", "futures.month_end_zn.metrics_cost_2x",
     "futures_month_end_zn_cost_2x"),
    ("futures supply leg, 1x", "futures.supply_calendar.metrics", "futures.supply_calendar.metrics",
     "futures_supply_calendar"),
    ("futures supply leg, 2x", "futures.supply_calendar.metrics_cost_2x", "futures.supply_calendar.metrics_cost_2x",
     "futures_supply_calendar_cost_2x"),
]


def _coef_row(name, o, i, kind) -> dict:
    if kind == "diff":
        est, ins = get(o, "diff") if o else None, get(i, "diff") if i else None
        return {"name": name, "estimate": est, "ci": get(o, "ci") if o else None,
                "p_le_0": get(o, "p_le_0") if o else None, "in_sample": ins, "label": sign_label(est, ins),
                "rule": "net Sharpe difference, paired bootstrap by month; consistent if its sign matches in-sample"}
    if kind == "value":
        return {"name": name, "estimate": o, "in_sample": i, "label": sign_label(o, i),
                "rule": "consistent if the sign matches in-sample"}
    key = "coef" if kind == "coef2" else "b"
    est, ins = (get(o, key) if o else None), (get(i, key) if i else None)
    return {"name": name, "estimate": est, "t": get(o, "t") if o else None,
            "n": (get(o, "n") or get(o, "n_obs")) if o else None, "in_sample": ins,
            "in_sample_t": get(i, "t") if i else None, "label": sign_label(est, ins),
            "rule": "estimate, t and sign; consistent if the sign matches in-sample"}


def labels(oos: dict, ins: dict, daily: dict) -> dict:
    """The label table and the kill conditions. oos: the test-window blocks {"month_end", "flowclock", "H8",
    "futures"} (JSON-clean); ins: the committed results.json; daily: daily net excess returns per strategy key."""
    me, fc, fu = oos.get("month_end", {}), oos.get("flowclock", {}), oos.get("futures", {})
    h1, h4 = me.get("H1", {}), me.get("H4", {})
    b1, d4 = _num(h1.get("b")), _num(h4.get("diff"))
    headline = {
        "H1": {"estimate": b1, "t": h1.get("t"), "ci": h1.get("ci"), "n": h1.get("n"),
               "in_sample": get(ins, "in_sample.H1.b"),
               "label": NOT_COMPUTED if b1 is None else ("sign confirmed" if b1 > 0 else "sign not confirmed"),
               "rule": 'HYPOTHESIS.md power note: the test window "can confirm the sign, not significance"; '
                       'sign confirmed if b > 0'},
        "H4": {"sharpe_fc": h4.get("sharpe_fc"), "sharpe_cal": h4.get("sharpe_cal"), "estimate": d4,
               "ci": h4.get("ci"), "p_le_0": h4.get("p_le_0"), "in_sample": get(ins, "in_sample.H4.diff"),
               "label": NOT_COMPUTED if d4 is None else ("pass" if d4 > 0 else "fail"),
               "rule": 'HYPOTHESIS.md: forecast-sized beats calendar-only "in sign in the test window"; pass if the '
                       'net Sharpe difference (1x) > 0'},
    }
    h8 = get(oos, "H8.c")
    c8, p8 = (_num(h8.get("b")), _num(h8.get("p_one_sided"))) if isinstance(h8, dict) else (None, None)
    headline["H8"] = {"estimate": c8, "t": get(oos, "H8.c.t"), "p_one_sided": p8, "n": get(oos, "H8.n"),
                      "in_sample": get(ins, "H8.in_sample.c.b"),
                      "label": NOT_COMPUTED if c8 is None or p8 is None else (
                          "pass" if c8 > 0 and p8 < 0.05 else "fail"),
                      "rule": "PREREG_DEALERS.md: pass if c > 0 and one-sided p < 0.05"}
    rows = [_coef_row(name, get(oos, op), get(ins, ip), kind) for name, op, ip, kind in CONSISTENCY_ROWS]
    strategies = []
    for name, op, ip, key in STRATEGY_ROWS:
        o, i = get(oos, op), get(ins, ip)
        s_o, s_i = (get(o, "sharpe") if o else None), (get(i, "sharpe") if i else None)
        strategies.append({"name": name, "sharpe": s_o, "t": mean_t(daily[key]) if key in daily else None,
                           "in_sample_sharpe": s_i, "label": sign_label(s_o, s_i),
                           "rule": "net Sharpe with t = mean / (sd / sqrt(days)) of daily net excess; consistent if "
                                   "its sign matches the in-sample net Sharpe at the same cost"})
    return {"headline": headline, "tests": rows, "strategies": strategies, "kill_conditions": kill_conditions(oos)}


def kill_conditions(oos: dict) -> list[dict]:
    """HYPOTHESIS.md and PREREG_FLOWCLOCK.md kill conditions as written, on the test window (module docstring)."""
    def cmp(name, src, met, numbers, rule):
        return {"condition": name, "source": src, "met": met, "numbers": numbers, "rule": rule}
    b = _num(get(oos, "month_end.H1.b"))
    r3 = _num(get(oos, "month_end.H3.mean_R3.b"))
    s_fc, s_cal = _num(get(oos, "month_end.H4.sharpe_fc")), _num(get(oos, "month_end.H4.sharpe_cal"))
    s_book, s_dem = (_num(get(oos, "flowclock.headline3.cost_1x.sharpe_book")),
                     _num(get(oos, "flowclock.headline3.cost_1x.sharpe_demand")))
    judged = "no threshold is written; the numbers are reported for the team (as the futures kill condition, " \
             "CLAUDE.md section 15)"
    na = "an in-sample condition; not applicable to the test window"
    H, F = "HYPOTHESIS.md", "PREREG_FLOWCLOCK.md"
    return [
        cmp("b <= 0", H, None if b is None else b <= 0, {"b": b}, "met if H1 b <= 0"),
        cmp("or centered on zero", H, None, {"b": b, "t": get(oos, "month_end.H1.t"), "ci": get(oos, "month_end.H1.ci")},
            judged),
        cmp("no reversal", H, None if r3 is None else r3 >= 0, {"mean_R3_pct": r3, "H3_coef": get(oos, "month_end.H3.coef")},
            "met if the mean T..T+3 excess return R3 >= 0"),
        cmp("forecast-sized <= calendar-only", H, None if s_fc is None or s_cal is None else s_fc <= s_cal,
            {"sharpe_fc": s_fc, "sharpe_cal": s_cal}, "met if the net Sharpes compare so, 1x costs"),
        cmp("effect only in the 1990s", H, None, {}, na),
        cmp("vanishes at 2x costs", H, None,
            {"sharpe_fc_2x": get(oos, "month_end.cost_stress.metrics.forecast_sized.sharpe"),
             "sharpe_cal_2x": get(oos, "month_end.cost_stress.metrics.calendar_only.sharpe")}, judged),
        cmp("the pre or post mean has the wrong sign in-sample", F, None, {}, na),
        cmp("the effect is absent in 2009-2024", F, None, {}, na),
        cmp("the Flow Clock book does not beat the demand leg alone, net of costs", F,
            None if s_book is None or s_dem is None else s_book <= s_dem,
            {"sharpe_book": s_book, "sharpe_demand": s_dem,
             "sharpe_book_2x": get(oos, "flowclock.headline3.cost_2x.sharpe_book"),
             "sharpe_demand_2x": get(oos, "flowclock.headline3.cost_2x.sharpe_demand")},
            "met if Sharpe(book) <= Sharpe(demand leg alone), 1x costs"),
        cmp("it disappears at 2x costs", F, None,
            {"sharpe_book_2x": get(oos, "flowclock.metrics.book_cost_2x.sharpe"),
             "sharpe_supply_cash_2x": get(oos, "flowclock.metrics.supply_calendar_cost_2x.sharpe"),
             "sharpe_supply_futures_2x": get(oos, "futures.supply_calendar.metrics_cost_2x.sharpe")}, judged),
    ]
