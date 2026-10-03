"""Single entry point: --insample (default) | --futures | --oos | --refresh.

`python run_all.py` rebuilds every in-sample number from the committed snapshot (CLAUDE.md rule 4): checksums,
index rebuild, signals, 10-year cash windows, H1 (+ components, PREREG_ADDENDUM.md (a)-(b)), H4, placebo and luck
test, required metrics, figures 1-2, outputs/results.json; every run appends to runs/trials.csv (rule 7).
Phase 5 adds the addendum's (c), H2 and the curve-allocated trade, H3, H5, costs 2x, every risk rule on and off,
the post-publication sub-sample, the crowding monitor, betas, tails, cash capacity, the sensitivity grid, figures
3-6 and the Deflated Sharpe.
It then runs the Flow Clock (PREREG_FLOWCLOCK.md): auction event table, H6a-c, H7, the supply leg, the book,
Headline 3, the auction event-path figure, and in Phase 5 risk rules on and off, results by tenor and decade and a
drawdown table by year (results.json["flowclock"]).
Trials are appended to runs/trials.csv only with GQH_DEV=1; results.json reads the count and the Deflated Sharpe's
inputs from that log, so a run without GQH_DEV reproduces the committed numbers (src/trial_log.py).
Dates after IS_END are never read here (rule 2). Scope: in-sample, cash bonds only.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config.flowclock as fcs  # noqa: E402
from config.flowclock import EVENT_PATH, FC_TENORS, POST_LYZ_START, SA_START  # noqa: E402
from config.settings import (COST_STRESS, ENTRY_OFFSET, EXIT_OFFSET, HEADLINE_TENOR, IS_END, IS_START,  # noqa: E402
                             PLACEBO_BDAYS, POSTPUB_START, REVERSAL_DAYS, RF_SERIES, TENORS)
from src import figures, report  # noqa: E402
from src.auction_events import build_events, in_sample_mask, month_end_supply  # noqa: E402
from src.backtest import (month_end_windows, placebo_windows, reversal_windows, run_curve_allocated,  # noqa: E402
                          run_strategy, window_returns, window_yield_change_bp)
from src.bonds import KNOT_YEARS, load_curve  # noqa: E402
from src.calendar import load_calendar  # noqa: E402
from src.capacity import capacity_curve  # noqa: E402
from src.data.auctions import load_auctions  # noqa: E402
from src.data.fomc import load_fomc_dates  # noqa: E402
from src.data.fred import load_frame  # noqa: E402
from src.data.french import load_pension_input  # noqa: E402
from src.data.pd_volume import adv_known_at, load_volume  # noqa: E402
from src.data.snapshot import verify_checksums, verify_or_exit  # noqa: E402
from src.flowclock import (DECADES, auction_event_paths, book_metrics, by_tenor, compare_sharpe,  # noqa: E402
                           demand_legs, event_returns, h6, h7, path_summary, run_book, supply_legs, tercile_by_rank)
from src.index_rebuild import BUCKETS, RebuildConfig, config_dict, run_default  # noqa: E402
from src.metrics import (betas, crowding, equity_curves, equity_excess_on_bond_days, period_years,  # noqa: E402
                         required_metrics, slice_metrics, tails, yearly_table)
from src.returns import tenor_returns  # noqa: E402
from src.risk import WINDOWS_PER_YEAR, RiskConfig  # noqa: E402
from src.sensitivity import cell_signal, fast_daily, grid_rebuilds, run_grid, summary, to_markdown  # noqa: E402
from src.signals import build_signals, past_zscore, pension_pressure  # noqa: E402
from src.stats import deflated_sharpe  # noqa: E402
from src.tests_h import (event_path_summary, event_paths, h1, h1_addendum, h1_components, h2, h2_panel,  # noqa: E402
                         h3, h4, h5, luck_candidates, luck_test, tercile_labels)
from src.trial_log import (assert_flowclock_prereg, assert_gate1, git_state, log_trial, log_trials,  # noqa: E402
                           read_trials, trial_count, trial_row)

VERSION = "v2 (Phase 5: in-sample, cash bonds; month-end leg and Flow Clock)"
PENDING = ["oos and flowclock.oos (Gate 2 only)", "in_sample.metrics.futures and the futures layer (Phase 4)",
           "flowclock futures version and its volume-based capacity (PREREG_FLOWCLOCK.md, item 4)",
           "in_sample.tips_replication (Phase 6)", "figure 5: test window added and shaded (Gate 2)"]
FIG_DIR = report.OUTPUTS / "figures"
RISK_VARIANTS = {"fomc_off": {"fomc_half": False}, "drawdown_off": {"dd_rule": False},
                 "cap_off": {"notional_cap_on": False},
                 "all_off": {"fomc_half": False, "dd_rule": False, "notional_cap_on": False}}
GRID_CHECK_CELL = (2, 1, "DGS5")      # off-headline cell checked against run_strategy at 2x costs


def step(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def insample() -> None:
    t0 = time.time()
    git = git_state()                       # captured before this run writes anything
    verify_or_exit()
    assert_gate1()
    step(f"snapshot checksums OK; commit {git['commit']} (dirty={git['dirty']})", t0)

    cal = load_calendar(end=IS_END)
    rcfg = RebuildConfig()
    monthly, _ = run_default(cfg=rcfg)
    step(f"index rebuild: {len(monthly)} months {monthly['month'].iloc[0]}..{monthly['month'].iloc[-1]}", t0)

    tenor = HEADLINE_TENOR
    total, excess = tenor_returns(end=IS_END)
    x10 = excess[tenor]
    rf = total[tenor] - excess[tenor]
    y = load_frame([tenor, RF_SERIES], end=IS_END, index=cal.days, fill=True)
    y10 = y[tenor]
    fomc = load_fomc_dates(end=IS_END, scheduled_only=True)
    months_frame = pd.DataFrame({"E": pd.to_datetime(monthly["E"].to_numpy())},
                                index=pd.PeriodIndex(monthly["month"], freq="M"))
    pension = pension_pressure(months_frame, cal, load_pension_input(end=IS_END), total[tenor])
    sig = build_signals(monthly, fomc, pension)
    step("signals built (z, components, w, surprise, pension, dummies)", t0)

    is_months = sig.index[(sig.index >= pd.Period(IS_START, "M")) & (sig.index <= pd.Period(IS_END, "M"))]
    s_is = sig.loc[is_months]
    if s_is["z"].isna().any():
        raise RuntimeError("z_m undefined inside the in-sample period; the 36-month warm-up is missing")
    win = month_end_windows(cal, is_months, ENTRY_OFFSET, EXIT_OFFSET)
    if not ((win["entry"] == s_is["E"]).all() and (win["T"] == s_is["T"]).all()):
        raise RuntimeError("window dates differ from the rebuild's E and T")
    R = window_returns(x10, win) * 100.0
    Y = -window_yield_change_bp(y10, win)
    days = cal.days[(cal.days >= pd.Timestamp(IS_START)) & (cal.days <= pd.Timestamp(IS_END))]
    rcfg_risk = RiskConfig()
    tenor_years = KNOT_YEARS[tenor]
    common = dict(daily_excess=x10, rf=rf, y_tenor=y10, y10=y10, tenor_years=tenor_years, fomc_scheduled=fomc,
                  cal=cal, days=days, cfg=rcfg_risk)
    ones = pd.Series(1.0, index=is_months)
    strat = {"forecast_sized": run_strategy("forecast_sized", win, s_is["w"], **common),
             "calendar_only": run_strategy("calendar_only", win, ones, **common)}
    step(f"cash backtest: {len(win)} windows, {len(days)} NAV days", t0)

    res_h1 = h1(R, Y, s_is["z"])
    res_h1["components"] = h1_components(R, Y, s_is["z_ext"], s_is["z_cash"])
    res_add = h1_addendum(R, Y, s_is["z"], s_is["ref"], s_is["z_surprise"])
    res_h4 = h4(strat["forecast_sized"].daily["excess"], strat["calendar_only"].daily["excess"])
    metrics = {k: required_metrics(v) for k, v in strat.items()}

    # placebo: business days 4-7 of month m+1, paired with z_m (src/backtest.py::placebo_windows)
    p_months = is_months[is_months + 1 <= pd.Period(IS_END, "M")]
    pwin = placebo_windows(cal, p_months, PLACEBO_BDAYS)
    P = window_returns(x10, pwin) * 100.0
    PY = -window_yield_change_bp(y10, pwin)
    pstrat = {"forecast_sized": run_strategy("placebo_forecast_sized", pwin, s_is["w"], **common),
              "calendar_only": run_strategy("placebo_calendar_only", pwin, ones, **common)}
    res_placebo = {
        "window": f"business days {PLACEBO_BDAYS[0]}-{PLACEBO_BDAYS[1]} of month m+1 (enter close of bday "
                  f"{PLACEBO_BDAYS[0] - 1}, exit close of bday {PLACEBO_BDAYS[1]}), paired with z_m",
        "H1": h1(P, PY, s_is["z"].loc[p_months]),
        "components": h1_components(P, PY, s_is["z_ext"].loc[p_months], s_is["z_cash"].loc[p_months]),
        "H4": h4(pstrat["forecast_sized"].daily["excess"], pstrat["calendar_only"].daily["excess"]),
        "metrics": {k: required_metrics(v) for k, v in pstrat.items()},
    }
    luck = luck_test(luck_candidates(cal, is_months, x10), R)
    luck_draws = luck.pop("draw_means_pct")
    step("H1, components, addendum (a)-(c), H4, placebo, luck test done", t0)

    # trial log (rule 7): one row per tested configuration, before anything else can fail
    base_cfg = {"settings": report.settings_dict(), "rebuild": config_dict(rcfg), "risk": asdict(rcfg_risk),
                "tenor": tenor, "strategy": "cash"}
    head_row = log_trial({**base_cfg, "window": {"entry": -ENTRY_OFFSET, "exit": EXIT_OFFSET}}, "in_sample", {
        "strategy": "cash", "tenor": tenor, "entry": f"T-{ENTRY_OFFSET}", "exit": f"T+{EXIT_OFFSET}",
        "n": res_h1["n"], "H1_b": res_h1["b"], "H1_lo": res_h1["ci"][0], "H1_hi": res_h1["ci"][1],
        "sharpe_fc": res_h4["sharpe_fc"], "sharpe_cal": res_h4["sharpe_cal"],
        "note": "run_all --insample: pre-registered headline (cash, net of costs)"}, git=git)
    log_trial({**base_cfg, "window": {"placebo_bdays": list(PLACEBO_BDAYS)}}, "in_sample_placebo", {
        "strategy": "cash", "tenor": tenor, "entry": f"bday{PLACEBO_BDAYS[0] - 1}(m+1)",
        "exit": f"bday{PLACEBO_BDAYS[1]}(m+1)", "n": res_placebo["H1"]["n"], "H1_b": res_placebo["H1"]["b"],
        "H1_lo": res_placebo["H1"]["ci"][0], "H1_hi": res_placebo["H1"]["ci"][1],
        "sharpe_fc": res_placebo["H4"]["sharpe_fc"], "sharpe_cal": res_placebo["H4"]["sharpe_cal"],
        "note": "run_all --insample: placebo control (pre-registered), not a candidate strategy"}, git=git)
    step(f"logged 2 rows to runs/trials.csv (config {head_row['config_hash']})", t0)

    # tables
    terc = tercile_labels(s_is["z"])
    windows_tab = pd.DataFrame({"T": win["T"].dt.date, "E": win["entry"].dt.date, "FDD": s_is["FDD"],
                                "z": s_is["z"], "z_ext": s_is["z_ext"], "z_cash": s_is["z_cash"], "w": s_is["w"],
                                "tercile": terc, "ref": s_is["ref"], "fomc_scheduled_in_window": s_is["fomc"],
                                "R_pct": R, "neg_dy_bp": Y})
    windows_tab.index.name = "month"
    report.write_table(windows_tab, "windows_insample.csv")
    for k, v in strat.items():
        report.write_table(v.trades.rename_axis("month"), f"trades_cash_{k}_insample.csv")
    report.write_table(equity_curves(strat), "equity_curve_cash_insample.csv")
    report.write_table(pd.DataFrame({"window_month": pwin["window_month"], "entry": pwin["entry"].dt.date,
                                     "exit": pwin["exit"].dt.date, "z_m": s_is["z"].loc[p_months],
                                     "P_pct": P, "neg_dy_bp": PY}).rename_axis("signal_month"),
                       "placebo_windows_insample.csv")
    report.write_table(pd.DataFrame({"draw_mean_pct": luck_draws}), "luck_test_draws_insample.csv", index=False)

    # figures 1-2
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = event_paths(cal, is_months, x10)
    ev = event_path_summary(paths, terc)
    sample = f"{IS_START[:7]} to {IS_END[:7]}"
    cap1 = figures.event_path(ev, FIG_DIR / "event_path.png", sample)
    cap2 = figures.terciles(res_h1, FIG_DIR / "terciles.png", sample)
    step("figures 1-2 written", t0)

    # Phase 5, month-end leg
    eq_ex = equity_excess_on_bond_days(load_pension_input(end=IS_END), cal.days, rf)
    p5 = month_end_phase5(cal=cal, monthly=monthly, sig=sig, s_is=s_is, is_months=is_months, win=win, R=R, Y=Y,
                          x10=x10, rf=rf, y10=y10, fomc=fomc, days=days, common=common, strat=strat,
                          res_h1=res_h1, total=total, excess=excess, eq_ex=eq_ex, base_cfg=base_cfg, git=git,
                          t0=t0)

    # Flow Clock (PREREG_FLOWCLOCK.md)
    fc_block, fc_fig, fc_book = flowclock(cal, is_months, win, R, Y, strat["calendar_only"], common, git, t0,
                                          demand_variants=p5["calendar_variants"], eq_ex=eq_ex, x10=x10)

    # figure 5: equity curves (in-sample; the test window is added at Gate 2)
    navs = {"forecast-sized": (1.0 + strat["forecast_sized"].daily["excess"]).cumprod(),
            "calendar-only": (1.0 + strat["calendar_only"].daily["excess"]).cumprod(),
            "Flow Clock book": (1.0 + fc_book.daily["excess"]).cumprod(),
            "curve-allocated": (1.0 + p5["curve"].daily["excess"]).cumprod()}
    shp = {"forecast-sized": metrics["forecast_sized"]["sharpe"], "calendar-only": metrics["calendar_only"]["sharpe"],
           "Flow Clock book": fc_block["in_sample"]["metrics"]["book"]["sharpe"],
           "curve-allocated": p5["metrics_curve"]["sharpe"]}
    p5["figures"]["equity_curve"] = {"path": "outputs/figures/equity_curve.png",
                                     "caption": figures.equity_curve(navs, shp, FIG_DIR / "equity_curve.png",
                                                                     sample)}

    # Deflated Sharpe: every trial of this run is logged above, so the log is final for this run
    trials = read_trials()
    t_sh = pd.concat([trials["sharpe_fc"], trials["sharpe_cal"]]).dropna().to_numpy(float)
    n_trials = trial_count()
    dsr_inputs = {"forecast_sized": strat["forecast_sized"].daily["excess"],
                  "calendar_only": strat["calendar_only"].daily["excess"],
                  "curve_allocated": p5["curve"].daily["excess"],
                  "flowclock_book": fc_book.daily["excess"],
                  "flowclock_supply_calendar": fc_block.pop("_supply_calendar_daily")}
    dsr = {k: deflated_sharpe(v, t_sh, n_trials) for k, v in dsr_inputs.items()}
    dsr["note"] = ("Bailey & Lopez de Prado; N = rows in runs/trials.csv, V = variance of every Sharpe in the log "
                   "(both columns), skew and kurtosis from each strategy's daily excess returns (src/stats.py)")
    step(f"Deflated Sharpe: N = {n_trials} trials, {len(t_sh)} trial Sharpes", t0)

    # results.json v1
    res = report.skeleton()
    res["meta"].update({"commit": git["commit"], "dirty": git["dirty"],
                        "snapshot_checksums_ok": not verify_checksums(), "settings_hash": report.settings_hash(),
                        "version": VERSION, "pending": PENDING, "config_hash_headline": head_row["config_hash"]})
    res["validation"] = report.validation_block(monthly)
    ins = res["in_sample"]
    ins["H1"] = res_h1
    ins["H1_addendum"] = res_add
    ins["H4"] = res_h4
    ins["placebo"] = res_placebo
    ins["random_windows"] = luck
    ins["metrics"]["cash"]["forecast_sized"] = metrics["forecast_sized"]
    ins["metrics"]["cash"]["calendar_only"] = metrics["calendar_only"]
    ins["metrics"]["cash"]["curve_allocated"] = p5["metrics_curve"]
    ins["metrics"]["cash"]["equity_curve"] = "outputs/tables/equity_curve_cash_insample.csv"
    ins["event_path"] = ev
    for k in ("H2", "H3", "H5", "curve_allocated", "cost_stress", "risk_rules_on_off", "sensitivity", "betas",
              "crowding", "tails", "capacity"):
        ins[k] = p5[k]
    ins["deflated_sharpe"] = dsr
    res["post_publication"] = p5["post_publication"]
    res["figures"] = {"event_path": {"path": "outputs/figures/event_path.png", "caption": cap1},
                      "terciles": {"path": "outputs/figures/terciles.png", "caption": cap2},
                      **p5["figures"], "auction_event_path": fc_fig}
    res["flowclock"] = fc_block
    res["trials"] = {"count": trial_count()}
    report.write_results(res)
    step(f"wrote outputs/results.json (trials logged so far: {res['trials']['count']})", t0)


def _row(base_cfg: dict, git: dict, window: str, extra_cfg: dict, n: int, h: dict | None, s_fc, s_cal,
         note: str, tenor: str = HEADLINE_TENOR, entry: str = f"T-{ENTRY_OFFSET}", exit_: str = f"T+{EXIT_OFFSET}",
         strategy: str = "cash") -> dict:
    """A trial-log row for a month-end variant; h = the slope stored in H1_* ({"b", "ci"}) or None."""
    return trial_row({**base_cfg, **extra_cfg}, window, {
        "strategy": strategy, "tenor": tenor, "entry": entry, "exit": exit_, "n": n,
        "H1_b": h["b"] if h else "", "H1_lo": h["ci"][0] if h else "", "H1_hi": h["ci"][1] if h else "",
        "sharpe_fc": s_fc, "sharpe_cal": s_cal, "note": note}, git=git)


def month_end_phase5(*, cal, monthly, sig, s_is, is_months, win, R, Y, x10, rf, y10, fomc, days, common, strat,
                     res_h1, total, excess, eq_ex, base_cfg, git, t0) -> dict:
    """Phase 5 for the month-end leg (CLAUDE.md 7.10-7.13; choices in the module docstrings of src/tests_h.py,
    src/backtest.py, src/metrics.py, src/capacity.py, src/sensitivity.py). Logs its trials; returns the blocks."""
    out = {"figures": {}}
    rows: list[dict] = []
    logged = [0, 0]                          # rows built, rows written (written only with GQH_DEV=1)

    def flush() -> None:
        """Log each section's trials as soon as they exist (rule 7), so a later failure cannot drop them."""
        logged[0] += len(rows)
        logged[1] += log_trials(rows)
        rows.clear()

    sample = f"{IS_START[:7]} to {IS_END[:7]}"
    ones = pd.Series(1.0, index=is_months)

    # H3: the first 3 bond days of month m+1
    rwin = reversal_windows(cal, is_months, REVERSAL_DAYS)
    out["H3"] = h3(window_returns(x10, rwin) * 100.0, -window_yield_change_bp(y10, rwin), s_is["z"], R)
    # H5: horse race
    out["H5"] = h5(R, Y, s_is)
    step("H3, H5 done", t0)

    # H2 and the curve-allocated trade (bucket demand includes the cash term; team decision)
    yall = load_frame(list(TENORS.values()), end=IS_END, index=cal.days, fill=True)
    midx = pd.PeriodIndex(monthly["month"], freq="M")
    fdd_b = pd.DataFrame({b: monthly[f"fdd_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    ext_b = pd.DataFrame({b: monthly[f"ext_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    z_fdd_b = fdd_b.apply(past_zscore).loc[is_months]
    z_ext_b = ext_b.apply(past_zscore).loc[is_months]
    neg_dy_b = pd.DataFrame({b: -(yall[TENORS[b]].reindex(win["exit"]).to_numpy()
                                  - yall[TENORS[b]].reindex(win["entry"]).to_numpy()) * 100.0 for b in BUCKETS},
                            index=is_months)
    out["H2"] = h2(neg_dy_b, z_fdd_b, z_ext_b)
    cfg2 = RiskConfig(cost_mult=COST_STRESS)
    curve_args = dict(demand=fdd_b.loc[is_months], tenor_of=TENORS, excess=excess, yields=yall, rf=rf, y10=y10,
                      fomc_scheduled=fomc, cal=cal, days=days)
    curve = run_curve_allocated("curve_allocated", win, cfg=RiskConfig(), **curve_args)
    curve2 = run_curve_allocated("curve_allocated_cost2x", win, cfg=cfg2, **curve_args)
    out["curve"] = curve
    out["metrics_curve"] = required_metrics(curve)
    excl = pd.DataFrame(curve.excluded)
    out["curve_allocated"] = {
        "vs_calendar_only": h4(curve.daily["excess"], strat["calendar_only"].daily["excess"]),
        "vs_calendar_only_cost_2x": None, "metrics_cost_2x": required_metrics(curve2),
        "n_months_flat": len(curve.flat_months), "flat_months": [str(m) for m in curve.flat_months],
        "n_months_bucket_excluded": int(len(excl)),
        "months_bucket_excluded_by_bucket": ({} if excl.empty else
                                             {b: int(excl["buckets"].map(lambda x, b=b: b in x).sum())
                                              for b in BUCKETS}),
        "mean_weights": {b: float(curve.trades[f"a_{b}"].mean()) for b in BUCKETS},
        "note": "DV01 = calendar-only DV01 split across buckets in proportion to max(fdd_b, 0) (src/backtest.py)"}
    report.write_table(curve.trades.rename_axis("month"), "trades_cash_curve_allocated_insample.csv")
    step(f"H2 ({out['H2']['n_obs']} bucket-months) and curve-allocated trade done", t0)

    # costs 2x and every risk rule on and off (forecast-sized and calendar-only)
    me = {k: v for k, v in common.items() if k != "cfg"}
    fc2 = run_strategy("forecast_sized_cost2x", win, s_is["w"], cfg=cfg2, **me)
    cal2 = run_strategy("calendar_only_cost2x", win, ones, cfg=cfg2, **me)
    out["curve_allocated"]["vs_calendar_only_cost_2x"] = h4(curve2.daily["excess"], cal2.daily["excess"])
    h4_2 = h4(fc2.daily["excess"], cal2.daily["excess"])
    out["cost_stress"] = {"cost_mult": COST_STRESS, "H4": h4_2, "metrics": {"forecast_sized": required_metrics(fc2),
                                                                           "calendar_only": required_metrics(cal2)}}
    rows.append(_row(base_cfg, git, "in_sample_cost2x", {"risk": asdict(cfg2)}, res_h1["n"], res_h1,
                     h4_2["sharpe_fc"], h4_2["sharpe_cal"], "run_all Phase 5: costs 2x; sharpe_fc = forecast-sized, "
                     "sharpe_cal = calendar-only; H1_* = H1 (unchanged by costs)"))
    flush()
    rr = {"all_on": {"config": asdict(RiskConfig()), "H4": None,
                     "metrics": {"forecast_sized": required_metrics(strat["forecast_sized"]),
                                 "calendar_only": required_metrics(strat["calendar_only"])}}}
    cal_variants = {"all_on": strat["calendar_only"]}
    for name, kw in RISK_VARIANTS.items():
        cfgv = RiskConfig(**kw)
        fv = run_strategy(f"forecast_sized_{name}", win, s_is["w"], cfg=cfgv, **me)
        cv = run_strategy(f"calendar_only_{name}", win, ones, cfg=cfgv, **me)
        cal_variants[name] = cv
        hv = h4(fv.daily["excess"], cv.daily["excess"])
        rr[name] = {"config": asdict(cfgv), "H4": hv,
                    "metrics": {"forecast_sized": required_metrics(fv), "calendar_only": required_metrics(cv)}}
        rows.append(_row(base_cfg, git, f"in_sample_risk_{name}", {"risk": asdict(cfgv)}, res_h1["n"], res_h1,
                         hv["sharpe_fc"], hv["sharpe_cal"], f"run_all Phase 5: risk rules, {name}; sharpe_fc = "
                         "forecast-sized, sharpe_cal = calendar-only; H1_* = H1 (unchanged by risk rules)"))
        flush()
    rr["all_on"]["H4"] = h4(strat["forecast_sized"].daily["excess"], strat["calendar_only"].daily["excess"])
    out["risk_rules_on_off"] = rr
    out["calendar_variants"] = cal_variants
    rows.append(_row(base_cfg, git, "in_sample_curve_allocated", {"strategy": "curve_allocated"}, out["H2"]["n_obs"],
                     {"b": out["H2"]["coef"], "ci": out["H2"]["ci"]}, out["curve_allocated"]["vs_calendar_only"][
                         "sharpe_fc"], out["curve_allocated"]["vs_calendar_only"]["sharpe_cal"],
                     "run_all Phase 5: curve-allocated (H2 trade); sharpe_fc = curve-allocated, sharpe_cal = "
                     "calendar-only; H1_* = H2 coef; n = bucket-months", tenor="DGS2-DGS30"))
    rows.append(_row(base_cfg, git, "in_sample_curve_allocated_cost2x", {"strategy": "curve_allocated",
                                                                          "risk": asdict(cfg2)},
                     out["H2"]["n_obs"], {"b": out["H2"]["coef"], "ci": out["H2"]["ci"]},
                     out["curve_allocated"]["vs_calendar_only_cost_2x"]["sharpe_fc"],
                     out["curve_allocated"]["vs_calendar_only_cost_2x"]["sharpe_cal"],
                     "run_all Phase 5: curve-allocated at 2x costs; sharpe_fc = curve-allocated, sharpe_cal = "
                     "calendar-only at 2x; H1_* = H2 coef", tenor="DGS2-DGS30"))
    flush()
    step("costs 2x, risk rules on/off done", t0)

    # post-publication sub-sample (2019-01 to 2024-09): every H1/H4 number again, fresh strategy runs
    pp = is_months[is_months >= pd.Period(POSTPUB_START, "M")]
    pdays = days[days >= pd.Timestamp(POSTPUB_START)]
    pp_common = {**me, "days": pdays}
    pfc = run_strategy("forecast_sized_postpub", win.loc[pp], s_is["w"].loc[pp], cfg=RiskConfig(), **pp_common)
    pcal = run_strategy("calendar_only_postpub", win.loc[pp], ones.loc[pp], cfg=RiskConfig(), **pp_common)
    ph1 = h1(R.loc[pp], Y.loc[pp], s_is["z"].loc[pp])
    ph1["components"] = h1_components(R.loc[pp], Y.loc[pp], s_is["z_ext"].loc[pp], s_is["z_cash"].loc[pp])
    ph4 = h4(pfc.daily["excess"], pcal.daily["excess"])
    out["post_publication"] = {
        "sample": [str(pp[0]), str(pp[-1])], "H1": ph1,
        "H1_addendum": h1_addendum(R.loc[pp], Y.loc[pp], s_is["z"].loc[pp], s_is["ref"].loc[pp],
                                   s_is["z_surprise"].loc[pp]),
        "H4": ph4, "metrics": {"forecast_sized": required_metrics(pfc), "calendar_only": required_metrics(pcal)},
        "note": "strategies rerun from 2019-01 (drawdown rule starts flat); z_m is the full-history past-only z"}
    rows.append(_row(base_cfg, git, "post_publication", {"sample": [POSTPUB_START, IS_END]}, ph1["n"], ph1,
                     ph4["sharpe_fc"], ph4["sharpe_cal"], "run_all Phase 5: post-publication 2019-01 to 2024-09; "
                     "sharpe_fc = forecast-sized, sharpe_cal = calendar-only; H1_* = H1"))
    flush()
    step(f"post-publication: {len(pp)} months", t0)

    # crowding monitor, betas, tails (descriptive)
    terc = tercile_labels(s_is["z"])
    out["crowding"] = crowding(event_paths(cal, is_months, x10, lo=-10, hi=0), terc, entry_k=-ENTRY_OFFSET)
    bx = x10.reindex(days)
    out["betas"] = {k: betas(v.daily["excess"], bx, eq_ex) for k, v in
                    (("forecast_sized", strat["forecast_sized"]), ("calendar_only", strat["calendar_only"]),
                     ("curve_allocated", curve))}
    fomc_all = load_fomc_dates(end=IS_END, scheduled_only=False)
    out["tails"] = {k: tails(strat[k], y10, s_is, fomc_all, fomc) for k in ("forecast_sized", "calendar_only")}
    step("crowding, betas, tails done", t0)

    # capacity (cash): dealer volume in the 10-year bucket
    vol = load_volume(end=IS_END)
    cap = {}
    for k in ("forecast_sized", "calendar_only"):
        tr = strat[k].trades
        cap[k] = capacity_curve(strat[k], adv_known_at(pd.DatetimeIndex(tr["entry"]), vol))
    out["capacity"] = cap
    smp = f"{cap['calendar_only']['sample'][0][:7]} to {cap['calendar_only']['sample'][1][:7]}"
    out["figures"]["capacity"] = {"path": "outputs/figures/capacity.png", "caption": figures.capacity(
        {"forecast-sized": cap["forecast_sized"], "calendar-only": cap["calendar_only"]}, FIG_DIR / "capacity.png",
        smp)}
    step("capacity done", t0)

    # sensitivity grid (CLAUDE.md 7.12), checked against run_strategy
    rebuilds = grid_rebuilds()
    step(f"sensitivity grid: {len(rebuilds)} rebuilds", t0)
    cells, prepared = run_grid(rebuilds, cal, is_months, excess, yall, fomc, days, RiskConfig())
    for _, c in cells.iterrows():
        cell_cfg = {k: (c[k].item() if hasattr(c[k], "item") else c[k]) for k in
                    ("entry", "exit", "tenor", "inclusion_rule", "reinvest_coupons", "deduct_soma", "cost_mult")}
        rows.append(_row(base_cfg, git, "in_sample_grid", {"grid_cell": cell_cfg}, int(c["n"]),
                         {"b": c["H1_b"], "ci": [c["H1_lo"], c["H1_hi"]]}, c["sharpe_fc"], c["sharpe_cal"],
                         f"run_all Phase 5: sensitivity grid (CLAUDE.md 7.12), inclusion={c['inclusion_rule']}, "
                         f"reinvest={c['reinvest_coupons']}, deduct_soma={c['deduct_soma']}, "
                         f"cost={c['cost_mult']:g}x; sharpe_fc = forecast-sized, sharpe_cal = calendar-only",
                         tenor=c["tenor"], entry=f"T-{c['entry']}", exit_=f"T+{c['exit']}"))
    flush()
    head = cells.loc[cells["headline"]].iloc[0]
    hs = cell_signal(rebuilds[(ENTRY_OFFSET, "auctioned_by_rebalance", True)], True).loc[is_months]
    p_head = prepared[(ENTRY_OFFSET, EXIT_OFFSET, HEADLINE_TENOR)]
    gaps = {"z": float((hs["z"] - s_is["z"]).abs().max()),
            "forecast_sized": float(np.abs(fast_daily(p_head, hs["w"].to_numpy(), len(days), RiskConfig())
                                           - strat["forecast_sized"].daily["excess"].to_numpy()).max()),
            "calendar_only": float(np.abs(fast_daily(p_head, np.ones(len(p_head.months)), len(days), RiskConfig())
                                          - strat["calendar_only"].daily["excess"].to_numpy()).max()),
            "H1_b": abs(float(head["H1_b"]) - res_h1["b"])}
    ek, xk, tk = GRID_CHECK_CELL
    pc = prepared[GRID_CHECK_CELL]
    sc = cell_signal(rebuilds[(ek, "auctioned_by_rebalance", True)], True).loc[pc.months[pc.valid]]
    wc = month_end_windows(cal, pc.months[pc.valid], ek, xk)
    ref_run = run_strategy("grid_check", wc, sc["w"], excess[tk], rf, yall[tk], y10, KNOT_YEARS[tk], fomc, cal, days,
                           cfg2)
    wfull = cell_signal(rebuilds[(ek, "auctioned_by_rebalance", True)], True).loc[pc.months, "w"].to_numpy()
    gaps["off_headline_cell"] = float(np.abs(fast_daily(pc, wfull, len(days), cfg2)
                                             - ref_run.daily["excess"].to_numpy()).max())
    if max(gaps.values()) > 1e-12:
        raise RuntimeError(f"sensitivity grid differs from the reference engine: {gaps}")
    report.write_table(cells, "sensitivity.csv", index=False)
    (report.OUTPUTS / "tables" / "sensitivity.md").write_text(to_markdown(cells), encoding="utf-8")
    out["sensitivity"] = {"summary": summary(cells), "engine_check_max_abs_diff": gaps,
                          "table": "outputs/tables/sensitivity.csv", "markdown": "outputs/tables/sensitivity.md",
                          "cells": cells.to_dict(orient="records")}
    step(f"sensitivity grid: {len(cells)} cells, engine check max diff {max(gaps.values()):.1e}", t0)
    step(f"Phase 5 month-end: {logged[0]} trial rows ({logged[1]} written to runs/trials.csv)", t0)

    # figures 3 and 4
    is_mon = monthly.set_index(midx).loc[is_months]
    out["figures"]["extension_series"] = {"path": "outputs/figures/extension_series.png", "caption":
                                          figures.extension_series(list(is_months), is_mon["Ext"], s_is["ref"] == 1,
                                                                   FIG_DIR / "extension_series.png", sample)}
    panel = h2_panel(neg_dy_b, z_fdd_b)
    out["figures"]["curve_map"] = {"path": "outputs/figures/curve_map.png", "caption": figures.curve_map(
        panel["xd"], panel["yd"], out["H2"], FIG_DIR / "curve_map.png", sample)}
    report.write_table(pd.DataFrame({"month": panel["month"].to_numpy(),
                                     "bucket": panel.index.get_level_values(1).to_numpy(),
                                     "neg_dy_bp": panel["y"].to_numpy(), "z_fdd_b": panel["x"].to_numpy(),
                                     "neg_dy_bp_demeaned": panel["yd"].to_numpy(),
                                     "z_fdd_b_demeaned": panel["xd"].to_numpy()}), "h2_panel_insample.csv",
                       index=False)
    step("figures 3, 4, 6 written", t0)
    return out


def flowclock(cal, is_months, win, R, Y, demand, me_common: dict, git: dict, t0: float, demand_variants: dict,
              eq_ex: pd.Series, x10: pd.Series) -> tuple[dict, dict, object]:
    """Flow Clock in-sample (PREREG_FLOWCLOCK.md, src/auction_events.py, src/flowclock.py).

    demand: the month-end calendar_only StrategyResult as already tested (the demand leg alone, Headline 3).
    me_common: the month-end run_strategy arguments (10-year returns and yields, T-bill, FOMC dates, calendar).
    demand_variants: calendar_only rerun with each risk rule off (RISK_VARIANTS), for Phase 5's on/off table.
    Returns (results block, figure entry, the in-sample book).
    """
    assert_flowclock_prereg()
    yld = load_frame(FC_TENORS, end=IS_END, index=cal.days, fill=True)
    ev = build_events(load_auctions(end=IS_END, exclude=None), cal, yld)
    att = dict(ev.attrs)                        # DataFrame.join below does not carry attrs
    _, xs = tenor_returns(end=IS_END, tenors={s: s for s in FC_TENORS})
    ev = ev.join(event_returns(ev, xs, yld))
    masks = {"in_sample": in_sample_mask(ev, IS_START, IS_END), "post_lyz": in_sample_mask(ev, POST_LYZ_START, IS_END)}
    step(f"Flow Clock events: {len(ev)} nominal coupon auctions through {IS_END}, "
         f"{int(masks['in_sample'].sum())} in-sample, {int(masks['post_lyz'].sum())} from {POST_LYZ_START[:7]}", t0)

    sa = month_end_supply(ev, cal, load_curve(end=IS_END), pd.period_range(SA_START, IS_END[:7], freq="M"))
    sa["zA"] = past_zscore(sa["SA_bn_years"])
    if sa.loc[is_months, "zA"].isna().any():
        raise RuntimeError("zA_m undefined inside the in-sample period; the 36-month warm-up is missing")

    traded = ev["window_complete"] & ~ev["skipped"]
    sup_entries = pd.DatetimeIndex(pd.concat([ev.loc[traded, "pre_entry"], ev.loc[traded, "A"]]))
    rf = me_common["rf"]
    bk = dict(excess=xs, yields=yld, rf=rf, fomc_scheduled=me_common["fomc_scheduled"], cal=cal,
              supply_entries=sup_entries)
    me = {k: v for k, v in me_common.items() if k not in ("days", "cfg")}
    cfg1, cfg2 = RiskConfig(), RiskConfig(cost_mult=COST_STRESS)
    samples = {"in_sample": (IS_START, is_months), "post_lyz": (POST_LYZ_START, is_months[is_months >= pd.Period(
        POST_LYZ_START, "M")])}
    blocks, books = {}, {}
    for key, (start, months) in samples.items():
        days = cal.days[(cal.days >= pd.Timestamp(start)) & (cal.days <= pd.Timestamp(IS_END))]
        e = ev[masks[key] & ~ev["skipped"]]
        dem = demand_legs(win.loc[months], cal)
        ones = pd.Series(1.0, index=months)
        dem1 = demand if key == "in_sample" else run_strategy("calendar_only", win.loc[months], ones, days=days,
                                                              cfg=cfg1, **me)
        dem2 = run_strategy("calendar_only_cost2x", win.loc[months], ones, days=days, cfg=cfg2, **me)
        check = run_book("demand_check", dem, days=days, cfg=cfg1, demand_per_year=WINDOWS_PER_YEAR,
                         unit="month", **{**bk, "supply_entries": None})
        gap = float((check.daily["excess"] - dem1.daily["excess"]).abs().max())
        if gap > 1e-12:
            raise RuntimeError(f"{key}: the book engine with the demand leg alone differs from calendar_only by {gap}")
        r = {
            "supply_calendar": run_book("supply_calendar", supply_legs(e, False), days=days, cfg=cfg1, **bk),
            "supply_size_weighted": run_book("supply_size_weighted", supply_legs(e, True), days=days, cfg=cfg1, **bk),
            "book": run_book("flowclock_book", pd.concat([dem, supply_legs(e, False)], ignore_index=True), days=days,
                             cfg=cfg1, demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk),
            "supply_calendar_cost_2x": run_book("supply_calendar_cost2x", supply_legs(e, False), days=days, cfg=cfg2,
                                                **bk),
            "book_cost_2x": run_book("flowclock_book_cost2x", pd.concat([dem, supply_legs(e, False)],
                                                                        ignore_index=True), days=days, cfg=cfg2,
                                     demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk),
        }
        books[key] = {**r, "demand_alone": dem1}
        ex = {k: v.daily["excess"] for k, v in r.items()}
        metrics = {k: book_metrics(v) for k, v in r.items()}
        metrics["demand_alone"] = required_metrics(dem1)
        metrics["demand_alone_cost_2x"] = required_metrics(dem2)
        blocks[key] = {
            "sample": [str(days[0].date()), str(days[-1].date())],
            "H6": h6(ev[masks[key]]),
            "H7": h7(R.loc[months], Y.loc[months], sa.loc[months, "zA"]),
            "headline3": {"cost_1x": compare_sharpe(ex["book"], dem1.daily["excess"], "book", "demand"),
                          "cost_2x": compare_sharpe(ex["book_cost_2x"], dem2.daily["excess"], "book", "demand")},
            "supply_size_vs_calendar": compare_sharpe(ex["supply_size_weighted"], ex["supply_calendar"],
                                                      "size_weighted", "calendar"),
            "corr_daily_supply_vs_demand": float(ex["supply_calendar"].corr(dem1.daily["excess"])),
            "metrics": metrics,
            "demand_engine_check_max_abs_diff": gap,
        }
        step(f"Flow Clock {key}: H6, H7, supply leg, book, Headline 3 ({len(e)} events, {len(months)} months)", t0)

    # trial log (rule 7): the book (Headline 3), the supply leg (secondary), each at 1x and 2x costs, per sample
    fc_cfg = {"settings": report.settings_dict(), "flowclock": {k: getattr(fcs, k) for k in dir(fcs) if k.isupper()},
              "risk": asdict(cfg1)}
    for key, b in blocks.items():
        for cost_key, cm in (("cost_1x", 1.0), ("cost_2x", COST_STRESS)):
            sm_ = "supply_calendar" if cm == 1.0 else "supply_calendar_cost_2x"
            win_label = f"{key}_flowclock" + ("" if cm == 1.0 else "_cost2x")
            h3 = b["headline3"][cost_key]
            log_trial({**fc_cfg, "strategy": "flowclock_book", "cost_mult": cm, "sample": key}, win_label, {
                "strategy": "flowclock_book", "tenor": "DGS10+DGS2-DGS30", "entry": "T-4|A-5,A", "exit": "T|A,A+5",
                "n": b["H6"]["n_used"], "H1_b": b["H7"]["c"]["b"], "H1_lo": b["H7"]["c"]["ci"][0],
                "H1_hi": b["H7"]["c"]["ci"][1], "sharpe_fc": h3["sharpe_book"], "sharpe_cal": h3["sharpe_demand"],
                "note": f"run_all: Flow Clock Headline 3 (PREREG_FLOWCLOCK.md), {key}, {cm:g}x costs; sharpe_fc = "
                        f"book, sharpe_cal = demand leg alone; H1_* = H7 c"}, git=git)
            if cm == 1.0:
                sv = b["supply_size_vs_calendar"]
                s_fc, s_cal = sv["sharpe_size_weighted"], sv["sharpe_calendar"]
                note = "sharpe_fc = size-weighted supply leg, sharpe_cal = calendar supply leg"
            else:
                s_fc, s_cal = "", b["metrics"][sm_]["sharpe"]
                note = "sharpe_cal = calendar supply leg (the size-weighted leg is not stress-tested)"
            log_trial({**fc_cfg, "strategy": "supply_leg", "cost_mult": cm, "sample": key}, win_label, {
                "strategy": "supply_leg", "tenor": "DGS2-DGS30", "entry": "A-5,A", "exit": "A,A+5",
                "n": b["H6"]["n_used"], "H1_b": b["H6"]["H6c"]["b"], "H1_lo": b["H6"]["H6c"]["ci"][0],
                "H1_hi": b["H6"]["H6c"]["ci"][1], "sharpe_fc": s_fc, "sharpe_cal": s_cal,
                "note": f"run_all: Flow Clock supply leg (PREREG_FLOWCLOCK.md), {key}, {cm:g}x costs; {note}; "
                        f"H1_* = H6c beta"}, git=git)
    step("Flow Clock: logged 8 rows to runs/trials.csv", t0)

    # Phase 5 (in-sample): every risk rule on and off; by tenor, by decade, drawdown by year (descriptive)
    days_is = cal.days[(cal.days >= pd.Timestamp(IS_START)) & (cal.days <= pd.Timestamp(IS_END))]
    e_is = ev[masks["in_sample"] & ~ev["skipped"]]
    dem_is = demand_legs(win.loc[is_months], cal)
    b_is = books["in_sample"]
    m_is = blocks["in_sample"]["metrics"]
    rr = {"all_on": {"config": asdict(cfg1), "headline3": blocks["in_sample"]["headline3"]["cost_1x"],
                     "metrics": {k: m_is[k] for k in ("book", "supply_calendar", "demand_alone")}}}
    rows, n_logged = [], 0
    for name, kw in RISK_VARIANTS.items():
        cfgv = RiskConfig(**kw)
        dem_v = demand_variants[name]
        check = run_book(f"demand_check_{name}", dem_is, days=days_is, cfg=cfgv, demand_per_year=WINDOWS_PER_YEAR,
                         unit="month", **{**bk, "supply_entries": None})
        gap = float((check.daily["excess"] - dem_v.daily["excess"]).abs().max())
        if gap > 1e-12:
            raise RuntimeError(f"{name}: the book engine with the demand leg alone differs from calendar_only by "
                               f"{gap}")
        book_v = run_book(f"flowclock_book_{name}", pd.concat([dem_is, supply_legs(e_is, False)], ignore_index=True),
                          days=days_is, cfg=cfgv, demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk)
        sup_v = run_book(f"supply_calendar_{name}", supply_legs(e_is, False), days=days_is, cfg=cfgv, **bk)
        h3v = compare_sharpe(book_v.daily["excess"], dem_v.daily["excess"], "book", "demand")
        rr[name] = {"config": asdict(cfgv), "headline3": h3v,
                    "metrics": {"book": book_metrics(book_v), "supply_calendar": book_metrics(sup_v),
                                "demand_alone": required_metrics(dem_v)},
                    "demand_engine_check_max_abs_diff": gap}
        h6c, h7c = blocks["in_sample"]["H6"]["H6c"], blocks["in_sample"]["H7"]["c"]
        rows.append(trial_row({**fc_cfg, "strategy": "flowclock_book", "risk": asdict(cfgv), "sample": "in_sample"},
                              f"in_sample_flowclock_risk_{name}", {
            "strategy": "flowclock_book", "tenor": "DGS10+DGS2-DGS30", "entry": "T-4|A-5,A", "exit": "T|A,A+5",
            "n": blocks["in_sample"]["H6"]["n_used"], "H1_b": h7c["b"], "H1_lo": h7c["ci"][0], "H1_hi": h7c["ci"][1],
            "sharpe_fc": h3v["sharpe_book"], "sharpe_cal": h3v["sharpe_demand"],
            "note": f"run_all Phase 5: Flow Clock risk rules, {name}; sharpe_fc = book, sharpe_cal = demand leg "
                    f"alone (same rules); H1_* = H7 c"}, git=git))
        rows.append(trial_row({**fc_cfg, "strategy": "supply_leg", "risk": asdict(cfgv), "sample": "in_sample"},
                              f"in_sample_flowclock_risk_{name}", {
            "strategy": "supply_leg", "tenor": "DGS2-DGS30", "entry": "A-5,A", "exit": "A,A+5",
            "n": blocks["in_sample"]["H6"]["n_used"], "H1_b": h6c["b"], "H1_lo": h6c["ci"][0], "H1_hi": h6c["ci"][1],
            "sharpe_fc": "", "sharpe_cal": rr[name]["metrics"]["supply_calendar"]["sharpe"],
            "note": f"run_all Phase 5: Flow Clock risk rules, {name}; sharpe_cal = calendar supply leg; "
                    f"H1_* = H6c beta"}, git=git))
        n_logged += log_trials(rows)                 # logged per variant, as soon as it exists (rule 7)
        rows = []
    years_is = period_years(days_is)
    sup_is = b_is["supply_calendar"]
    decades = {}
    for dname, (lo_d, hi_d) in DECADES.items():
        hi_d = min(pd.Timestamp(hi_d), pd.Timestamp(IS_END))
        sel = masks["in_sample"] & (ev["A"] >= pd.Timestamp(lo_d)) & (ev["A"] <= hi_d)
        decades[dname] = {"H6": h6(ev[sel]),
                          "metrics": {k: slice_metrics(b_is[k].daily, lo_d, hi_d)
                                      for k in ("book", "supply_calendar", "demand_alone")}}
    dd_tabs = {k: yearly_table(b_is[k].daily["excess"]) for k in ("book", "supply_calendar", "demand_alone")}
    report.write_table(pd.concat(dd_tabs, names=["strategy"]), "flowclock_drawdown_by_year_insample.csv")
    bx = x10.reindex(days_is)
    blocks["in_sample"].update({
        "risk_rules_on_off": rr, "by_tenor_pnl": by_tenor(sup_is, years_is), "by_decade": decades,
        "drawdown_by_year": {k: {str(y): r for y, r in t.to_dict(orient="index").items()} for k, t in dd_tabs.items()},
        "betas": {k: betas(b_is[k].daily["excess"], bx, eq_ex) for k in ("book", "supply_calendar")}})
    step(f"Flow Clock Phase 5: risk rules on/off ({n_logged} trial rows written), by tenor, by decade, "
         f"drawdown by year", t0)

    # tables
    ins = masks["in_sample"]
    tab = ev.copy()
    tab["in_sample"], tab["post_lyz"] = masks["in_sample"], masks["post_lyz"]
    ret_cols = ["R_pre", "R_post", "LS", "dy_pre_bp", "dy_post_bp", "LS_bp"]
    tab.loc[~ins, ret_cols] = float("nan")
    for c in ["A", "announced", "pre_entry", "post_exit"]:
        tab[c] = tab[c].dt.date
    report.write_table(tab.set_index("event_id"), "auction_events_insample.csv")
    sup = sa.loc[is_months].assign(R_pct=R, neg_dy_bp=Y)
    for c in ["T", "from", "to"]:
        sup[c] = sup[c].dt.date
    report.write_table(sup.rename_axis("month"), "flowclock_supply_monthly_insample.csv")
    report.write_table(pd.DataFrame({t: {**{k: v for k, v in d.items() if k not in ("R_pre", "R_post")},
                                         "R_pre_mean": d["R_pre"]["b"], "R_pre_t": d["R_pre"]["t"],
                                         "R_post_mean": d["R_post"]["b"], "R_post_t": d["R_post"]["t"]}
                                     for t, d in blocks["in_sample"]["H6"]["by_tenor"].items()}).T.rename_axis("tenor"),
                       "flowclock_h6_by_tenor_insample.csv")
    b_is = books["in_sample"]
    for k in ("book", "supply_calendar", "supply_size_weighted"):
        lg = b_is[k].legs.copy()
        lg["entry"], lg["exit"] = lg["entry"].dt.date, lg["exit"].dt.date
        report.write_table(lg.set_index("leg_id"), f"flowclock_legs_{k}_insample.csv")
    eq = {}
    for k in ("book", "supply_calendar", "supply_size_weighted", "demand_alone"):
        eq[f"nav_total_{k}"] = (1.0 + b_is[k].daily["total"]).cumprod()
        eq[f"nav_excess_{k}"] = (1.0 + b_is[k].daily["excess"]).cumprod()
    report.write_table(pd.DataFrame(eq).rename_axis("date"), "equity_curve_flowclock_insample.csv")

    # figure: auction event path, A-10..A+10, by size-signal tercile (zS known at A-5)
    lo, hi = EVENT_PATH
    fig_ev = ev[ins & ~ev["skipped"] & ev["zS_pre"].notna()]
    rp, dp = auction_event_paths(fig_ev, xs, yld, cal, lo, hi, IS_START, IS_END)
    by_id = ev.set_index("event_id")
    lab = tercile_by_rank(by_id.loc[rp.index, "zS_pre"])
    ret_sum, dy_sum = path_summary(rp, lab, by_id["week"]), path_summary(dp, lab, by_id["week"])
    sample = f"{IS_START[:7]} to {IS_END[:7]}"
    cap = figures.auction_event_path(ret_sum, dy_sum, FIG_DIR / "auction_event_path.png", sample)
    zr = by_id.loc[rp.index, "zS_pre"]
    blocks["in_sample"]["event_path"] = {
        "excess_return_pct": ret_sum, "yield_change_bp": dy_sum,
        "tercile_zS_ranges": {t: [float(zr[lab == t].min()), float(zr[lab == t].max())] for t in ("low", "mid", "high")},
        "note": "events with A-10..A+10 inside the in-sample period and zS known at A-5; terciles by average rank"}
    step("Flow Clock tables and auction_event_path.png written", t0)

    e_is = ev[ins]
    blocks["in_sample"]["events"] = {
        "n_nominal_coupon_auctions_through_is_end": int(len(ev)), "n_tips_dropped": att["n_tips_dropped"],
        "n_frn_dropped": att["n_frn_dropped"], "n_not_bond_day": att["n_not_bond_day"],
        "n_in_sample": int(ins.sum()), "n_skipped": int(e_is["skipped"].sum()),
        "skipped": e_is.loc[e_is["skipped"], "event_id"].tolist(),
        "n_straddling_is_end": int((~ev["window_complete"] & (ev["A"] >= pd.Timestamp(IS_START))).sum()),
        "by_tenor": {k: int(v) for k, v in e_is["tenor"].value_counts().sort_index().items()},
        "zS_pre_flags": {k: int(v) for k, v in e_is["zS_pre_flag"].value_counts().sort_index().items()},
        "zS_post_flags": {k: int(v) for k, v in e_is["zS_post_flag"].value_counts().sort_index().items()},
        "S_pre_source": {(k or "none"): int(v) for k, v in e_is["S_pre_source"].value_counts().sort_index().items()},
        "SA_months_with_events_share": float((sa.loc[is_months, "n_events"] > 0).mean()),
    }
    block = {"prereg": {"file": "PREREG_FLOWCLOCK.md", "tag": "prereg-flowclock", "commit": "8c41154"},
             **blocks, "oos": {"status": "test window runs once, after gate2-frozen"},
             "_supply_calendar_daily": sup_is.daily["excess"]}
    return block, {"path": "outputs/figures/auction_event_path.png", "caption": cap}, b_is["book"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--insample", action="store_true", help="in-sample, cash bonds (default)")
    p.add_argument("--futures", action="store_true", help="add the Databento futures layer (Phase 4)")
    p.add_argument("--oos", action="store_true", help="test window, once, at Gate 2 only")
    p.add_argument("--refresh", action="store_true", help="refresh the public-data snapshot first")
    a = p.parse_args()
    if a.oos:
        sys.exit("--oos is not built yet: the test window runs once, at Gate 2 (Phase 7), on a gate2-frozen HEAD.")
    if a.refresh:
        sys.exit("Refresh the snapshot with `python scripts/download_all.py`, review and commit it, then rerun.")
    insample()
    if a.futures:
        print("--futures: the futures layer arrives in Phase 4; core (cash) results above are unchanged.")


if __name__ == "__main__":
    main()
