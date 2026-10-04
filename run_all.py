"""Single entry point: --insample (default) | --futures | --oos | --oos-pseudo | --refresh.

`python run_all.py` rebuilds every in-sample number from the committed snapshot (CLAUDE.md rule 4): checksums,
index rebuild, signals, 10-year cash windows, H1 (+ components, PREREG_ADDENDUM.md (a)-(b)), H4, placebo and luck
test, required metrics, figures 1-2, outputs/results.json; every run appends to runs/trials.csv (rule 7).
Phase 5 adds the addendum's (c), H2 and the curve-allocated trade, H3, H5, costs 2x, every risk rule on and off,
the post-publication sub-sample, the crowding monitor, betas, tails, cash capacity, the sensitivity grid, figures
3-6 and the Deflated Sharpe.
It then runs the Flow Clock (PREREG_FLOWCLOCK.md): auction event table, H6a-c, H7, the supply leg, the book,
Headline 3, the auction event-path figure, and in Phase 5 risk rules on and off, results by tenor and decade and a
drawdown table by year (results.json["flowclock"]).
Phase 4 adds the futures layer (CLAUDE.md 7.9, section 15): the month-end calendar-only leg on ZN and the Flow Clock
supply leg on the tenor's contract, at 1x and 2x costs (results.json["futures"], in_sample.metrics.futures).
`--futures` rebuilds the derived futures tables from the Databento cache (pulling it with a key if missing); without
the flag every futures number is recomputed from the committed derived tables, so a keyless run reproduces
results.json; with neither, the futures block is skipped with a message.
Phase 4d adds the CMT switch diagnostic (src/cmt_switch.py, descriptive): cash vs futures supply-leg P&L by day
around auctions (computed in the --futures build, aggregates only) and the cash reopening control
(results.json["cmt_switch_diagnostic"]).
H8 (PREREG_DEALERS.md, src/dealers.py) tests whether the auction effect is larger when primary dealers hold more
Treasury coupons (results.json["H8"]); the dealer positions are read only with the prereg-dealers tag (GQH_DEV=1).
Trials are appended to runs/trials.csv only with GQH_DEV=1; results.json reads the counts and the Deflated Sharpe's
inputs from that log, so a run without GQH_DEV reproduces the committed numbers (src/trial_log.py). The headline
Deflated Sharpe uses N = distinct variants (distinct config_hash); the one at N = every logged row is reported beside
it (CLAUDE.md section 15).
Dates after IS_END are never read by the in-sample run (rule 2).
`--oos` (Gate 2, CLAUDE.md section 17) runs every pre-registered test and strategy once on the test window
2024-10-01 to 2026-09-30 with the same code paths, from test-window data it downloads after the gate2-frozen guard
(evaluate_window, run_oos); `--oos-pseudo` runs that path on an in-sample pseudo-window as a check (run_oos_pseudo).
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

import config.dealers as dcs  # noqa: E402
import config.flowclock as fcs  # noqa: E402
import config.costs as fcost  # noqa: E402
import config.futures as fus  # noqa: E402
from config.flowclock import EVENT_PATH, FC_TENORS, POST_LYZ_START, SA_START  # noqa: E402
from config.settings import (COST_STRESS, ENTRY_OFFSET, EXIT_OFFSET, FUT_START, HEADLINE_FUTURE,  # noqa: E402
                             HEADLINE_TENOR, IS_END, IS_START, OOS_END, OOS_START, PLACEBO_BDAYS, POSTPUB_START,
                             REVERSAL_DAYS, RF_SERIES, TENORS)
from src import figures, oos_eval, report  # noqa: E402
from src import futures as fut  # noqa: E402
from src.data import databento_futures as dbf  # noqa: E402
from src.data import oos_data, pd_positions  # noqa: E402
from src.auction_events import build_events, in_sample_mask, month_end_supply  # noqa: E402
from src.backtest import (month_end_windows, placebo_windows, reversal_windows, run_curve_allocated,  # noqa: E402
                          run_strategy, window_returns, window_yield_change_bp)
from src.bonds import KNOT_YEARS, load_curve  # noqa: E402
from src.calendar import load_calendar  # noqa: E402
from src.capacity import capacity_curve  # noqa: E402
from src import cmt_switch, dealers  # noqa: E402
from src.data.auctions import load_auctions  # noqa: E402
from src.data.fomc import load_fomc_dates  # noqa: E402
from src.data.fred import load_frame  # noqa: E402
from src.data.french import load_pension_input  # noqa: E402
from src.data.pd_volume import adv_known_at, load_volume  # noqa: E402
from src.data.snapshot import REPO_ROOT, reading_from, utc_now, verify_checksums, verify_or_exit  # noqa: E402
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
from src.trial_log import (GATE2_TAG, GateError, assert_flowclock_prereg, assert_gate1,  # noqa: E402
                           assert_gate2_download, dev_mode, git_state, in_sample_trials, log_trial, log_trials,
                           read_trials, trial_count, trial_counts, trial_row)

VERSION = "v6 (H8: in-sample; cash and futures; month-end leg, Flow Clock, CMT switch diagnostic, H8 dealer test)"
PENDING = ["oos, flowclock.oos and H8.oos (Gate 2 only)",
           "in_sample.tips_replication (Phase 6)", "figure 5: test window added and shaded (Gate 2)"]
FUT_TABLES = {"legs": "futures_legs_insample.csv", "daily": "futures_daily_insample.csv",
              "checks": "futures_data_checks_insample.json"}
FUT_UNITS = {"month_end_zn": "month", "month_end_zn_cost_2x": "month", "supply_calendar": "event",
             "supply_calendar_cost_2x": "event"}
FUT_RISK_NAMES = ["fomc_off", "drawdown_off", "cap_off", "all_off"]       # = RISK_VARIANTS (Phase 4b, section 15)
FUT_UNITS.update({f"month_end_zn_{n}": "month" for n in FUT_RISK_NAMES})
FUT_UNITS.update({f"supply_calendar_{n}": "event" for n in FUT_RISK_NAMES})
FIG_DIR = report.OUTPUTS / "figures"
RISK_VARIANTS = {"fomc_off": {"fomc_half": False}, "drawdown_off": {"dd_rule": False},
                 "cap_off": {"notional_cap_on": False},
                 "all_off": {"fomc_half": False, "dd_rule": False, "notional_cap_on": False}}
GRID_CHECK_CELL = (2, 1, "DGS5")      # off-headline cell checked against run_strategy at 2x costs


def step(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def written(n: int) -> str:
    """Console text for n trial rows built by log_trial: they reach runs/trials.csv only with GQH_DEV=1."""
    return f"{n} trial row{'' if n == 1 else 's'} ({n if dev_mode() else 0} written to runs/trials.csv)"


def insample(futures_mode: str = "tables") -> None:
    """futures_mode: "rebuild" (--futures: derived futures tables from the Databento cache) or "tables" (committed
    derived tables; skipped with a message if absent)."""
    t0 = time.time()
    git = git_state()                       # captured before this run writes anything
    verify_or_exit()
    assert_gate1()
    if futures_mode == "rebuild" and not dbf.have_cache():      # before anything is computed or logged
        try:
            dbf.download()
        except dbf.NoDatabento as e:
            sys.exit(f"--futures: {e} No Databento cache in data/cache/databento/. Run without --futures to use the "
                     f"committed derived futures tables; core results are unchanged.")
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
    if futures_mode == "rebuild":           # --futures: derived futures tables first, before any trial is logged
        rebuild_futures(cal, win, is_months, fomc, rf, y10, t0)

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
    step(f"trial log: {written(2)} (config {head_row['config_hash']})", t0)

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

    # futures layer (Phase 4, CLAUDE.md 7.9 and section 15): logs its trials before the Deflated Sharpe reads the log
    fut_block = futures(cash_cal=strat["calendar_only"], cash_supply_daily=fc_block["_supply_calendar_daily"],
                        cash_supply_gross_daily=fc_block.pop("_supply_calendar_gross_daily"), rf=rf, git=git, t0=t0)
    if fut_block is not None and futures_mode == "tables":
        step("futures: every futures number above was recomputed from the committed derived tables "
             "(outputs/tables/futures_*); a full rebuild from raw Databento data needs DATABENTO_API_KEY in .env "
             "and `python run_all.py --futures`", t0)

    # figure 5: equity curves (in-sample; the test window is added at Gate 2)
    navs = {"forecast-sized": (1.0 + strat["forecast_sized"].daily["excess"]).cumprod(),
            "calendar-only": (1.0 + strat["calendar_only"].daily["excess"]).cumprod(),
            "Flow Clock book": (1.0 + fc_book.daily["excess"]).cumprod(),
            "curve-allocated": (1.0 + p5["curve"].daily["excess"]).cumprod()}
    shp = {"forecast-sized": metrics["forecast_sized"]["sharpe"], "calendar-only": metrics["calendar_only"]["sharpe"],
           "Flow Clock book": fc_block["in_sample"]["metrics"]["book"]["sharpe"],
           "curve-allocated": p5["metrics_curve"]["sharpe"]}
    fig_fut = fut_block.pop("_fig") if fut_block is not None else {}
    p5["figures"]["equity_curve"] = {"path": "outputs/figures/equity_curve.png",
                                     "caption": figures.equity_curve(navs, shp, FIG_DIR / "equity_curve.png",
                                                                     sample, **fig_fut)}

    # Deflated Sharpe: every trial of this run is logged above, so the log is final for this run
    tc = trial_counts(in_sample_trials(read_trials()))      # oos_* rows are not in-sample variants (section 17)
    dsr_inputs = {"forecast_sized": strat["forecast_sized"].daily["excess"],
                  "calendar_only": strat["calendar_only"].daily["excess"],
                  "curve_allocated": p5["curve"].daily["excess"],
                  "flowclock_book": fc_book.daily["excess"],
                  "flowclock_supply_calendar": fc_block.pop("_supply_calendar_daily")}
    if fut_block is not None:
        dsr_inputs.update({f"futures_{k}": v for k, v in fut_block.pop("_daily_excess").items()})
    dsr = {k: deflated_sharpe(v, tc["sharpes_distinct"], tc["distinct_variants"]) for k, v in dsr_inputs.items()}
    dsr["at_total_logged_runs"] = {k: deflated_sharpe(v, tc["sharpes_all_rows"], tc["total_logged_runs"])
                                   for k, v in dsr_inputs.items()}
    dsr["N_distinct_variants"], dsr["N_total_logged_runs"] = tc["distinct_variants"], tc["total_logged_runs"]
    dsr["note"] = ("Bailey & Lopez de Prado (src/stats.py). Headline (one entry per strategy): N = distinct variants "
                   "tested (distinct config_hash in runs/trials.csv), V = variance of their Sharpes (both columns, "
                   "latest row per config_hash). at_total_logged_runs: N = every row, V over every row (the Phase 5 "
                   "rule). Skew and kurtosis from each strategy's daily excess returns (CLAUDE.md section 15)")
    step(f"Deflated Sharpe: N = {tc['distinct_variants']} distinct variants ({len(tc['sharpes_distinct'])} trial "
         f"Sharpes); also at N = {tc['total_logged_runs']} logged runs ({len(tc['sharpes_all_rows'])} Sharpes)", t0)

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
    cmt_fut = fut_block.pop("_cmt_switch") if fut_block is not None else {
        "status": "skipped: no committed derived futures tables and no --futures run"}
    if fut_block is not None:
        ins["metrics"]["futures"] = fut_block.pop("_metrics_1x")
        res["futures"] = fut_block
    else:
        res["futures"] = {"status": "skipped: no committed derived futures tables and no --futures run"}
    res["cmt_switch_diagnostic"] = cmt_block(fc_block.pop("_cmt_switch"), cmt_fut)
    res["H8"] = fc_block.pop("_h8")
    res["costs"] = report.cost_block(load_frame(list(fcost.FLEMING_2003["spread_32nds"]), end=IS_END, index=cal.days,
                                                fill=True))
    res["trials"] = {"count": trial_count(), "total_logged_runs": tc["total_logged_runs"],
                     "distinct_variants": tc["distinct_variants"],
                     "n_configs_with_differing_sharpes": tc["n_configs_with_differing_sharpes"],
                     "rule": "count = total logged runs = rows of runs/trials.csv (never deleted); distinct variants "
                             "= distinct config_hash, N of the headline Deflated Sharpe (CLAUDE.md section 15)"}
    if OOS_JSON.exists():                   # the Gate 2 run's blocks (section 17), so a fresh clone reproduces them
        import json
        merge_oos(res, json.loads(OOS_JSON.read_text(encoding="utf-8")))
    report.write_results(res)
    step(f"wrote outputs/results.json (logged runs {res['trials']['count']}, distinct variants "
         f"{res['trials']['distinct_variants']})", t0)


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
    report.write_table(equity_curves({"curve_allocated": curve}), "equity_curve_curve_allocated_insample.csv")
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
    auctions = load_auctions(end=IS_END, exclude=None)
    ev = build_events(auctions, cal, yld)
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
    step(f"Flow Clock: {written(8)}", t0)

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
    # Phase 4d (descriptive, src/cmt_switch.py): the cash reopening control and the cash engine check
    cmt = cmt_cash(ev, auctions, ins, sup_is, xs, cal)
    sv = blocks["in_sample"]["supply_size_vs_calendar"]
    n_rc = cmt["reopening_control"][cmt_switch.POOLED]
    log_trial({**fc_cfg, "strategy": "supply_leg", "cost_mult": 1.0, "sample": "in_sample"},
              "in_sample_flowclock_cmt_switch_diagnostic", {
        "strategy": "supply_leg", "tenor": "DGS10,DGS20,DGS30", "entry": "A-5,A", "exit": "A,A+5",
        "n": n_rc["new_issue"]["n"] + n_rc["reopening"]["n"], "H1_b": "", "H1_lo": "", "H1_hi": "",
        "sharpe_fc": sv["sharpe_size_weighted"], "sharpe_cal": sv["sharpe_calendar"],
        "note": "run_all Phase 4d: descriptive diagnostic, not a strategy configuration (CMT switch: cash event returns "
                "of new issues vs reopenings, in-sample); config_hash and Sharpe columns repeat the in-sample 1x "
                "supply_leg row, so distinct variants and the Deflated Sharpe's V are unchanged; "
                "results.json[cmt_switch_diagnostic]"}, git=git)
    step(f"Flow Clock Phase 4d: reopening control, cash engine check; descriptive {written(1)}", t0)
    # H8 (PREREG_DEALERS.md): dealer balance sheets; an explanation test, not a strategy (one trial row, no Sharpe)
    h8b = h8_block(ev, ins, cal)
    r8 = h8b["in_sample"]
    log_trial({**fc_cfg, "test": "H8", "dealers": {k: getattr(dcs, k) for k in dir(dcs) if k.isupper()},
               "series": pd_positions.COUPON_KEYS}, "in_sample_h8", {
        "strategy": "h8_dealers", "tenor": "DGS2-DGS30", "entry": "A-5", "exit": "A+5", "n": r8["n"],
        "H1_b": r8["c"]["b"], "H1_lo": r8["c"]["ci"][0], "H1_hi": r8["c"]["ci"][1], "sharpe_fc": "", "sharpe_cal": "",
        "note": "run_all H8 (PREREG_DEALERS.md): LS on zS_pre and zD (dealer coupon positions known at A-5), week "
                "clusters; H1_* = c and its 95% CI; explanation test, not a strategy (Sharpe columns blank)"}, git=git)
    step(f"H8: c = {r8['c']['b']:.4f} (t {r8['c']['t']:.2f}), {r8['n']} events; {written(1)}", t0)
    block = {"prereg": {"file": "PREREG_FLOWCLOCK.md", "tag": "prereg-flowclock", "commit": "8c41154"},
             **blocks, "oos": {"status": "test window runs once, after gate2-frozen"}, "_cmt_switch": cmt, "_h8": h8b,
             "_supply_calendar_daily": sup_is.daily["excess"],
             "_supply_calendar_gross_daily": sup_is.daily["excess"] + sup_is.daily["cost"] / sup_is.cfg.capital}
    return block, {"path": "outputs/figures/auction_event_path.png", "caption": cap}, b_is["book"]


def cmt_cash(ev: pd.DataFrame, auctions: pd.DataFrame, ins: pd.Series, supply_book, xs: pd.DataFrame, cal) -> dict:
    """Phase 4d, cash part (src/cmt_switch.py): the reopening control on the in-sample events (ins), the reopening
    rule against Fiscal Data's flag, and the cash engine check on the in-sample calendar supply book."""
    reopen = cmt_switch.reopening_flags(ev, auctions)
    return {"reopening_control": cmt_switch.reopening_control(ev[ins], reopen[ins]),
            "reopening_definition": {"rule": "the CUSIP was auctioned before (auction records)",
                                     "n_in_sample_disagree_with_fiscal_data_flag": int(
                                         (reopen[ins] != ev.loc[ins, "reopening"].astype(bool)).sum())},
            "cash_check": cmt_switch.cash_check(supply_book.legs, xs, cal)}


def h8_block(ev: pd.DataFrame, ins: pd.Series, cal) -> dict:
    """results.json["H8"] (PREREG_DEALERS.md; src/dealers.py, src/data/pd_positions.py): zD known at each in-sample
    event's A-5 from the committed dealer-position snapshot (as-of dates <= IS_END; the loader is gated by the
    prereg-dealers tag), then H8 on the H6c events."""
    weekly = pd_positions.load_positions(end=IS_END)
    checks = dict(weekly.attrs.get("checks", {}))
    e = ev[ins]
    zd = dealers.zd_at(pd.DatetimeIndex(sorted(e["pre_entry"].dropna().unique())), weekly, cal)
    return {
        "prereg": {"file": "PREREG_DEALERS.md", "tag": "prereg-dealers", "commit": "15f67bc"},
        "in_sample": dealers.h8(e, zd),
        "oos": {"status": "test window runs once, after gate2-frozen (with the Gate 2 --oos run)"},
        "data": {"source": "NY Fed Markets Data API, primary dealer statistics (FR 2004A): net positions in Treasury "
                           "coupons excluding TIPS, all maturities (sum of the maturity buckets of each series break)",
                 "snapshot": f"data/snapshot/{pd_positions.SNAPSHOT_NAME}", "keys": pd_positions.COUPON_KEYS,
                 "series_breaks": {sb: [lo, hi] for sb, lo, hi in pd_positions.SERIES_BREAKS},
                 "checks_buckets_sum_to_published_total": checks,
                 "n_releases_through_is_end_by_segment": {k: int(v) for k, v in
                                                          weekly["segment"].value_counts().sort_index().items()},
                 "publication_rule": f"first bond business day on or after as-of + {dcs.RELEASE_LAG_DAYS} days; known "
                                     "at the close of X only if published before X (NY Fed posts Thursdays ~4:15 PM "
                                     "ET)"},
    }


def cmt_block(cash_part: dict, fut_part: dict) -> dict:
    """results.json["cmt_switch_diagnostic"] (Phase 4d, descriptive; src/cmt_switch.py): the cash vs futures day
    path (aggregates from the --futures build, read from the derived checks file) and the cash reopening control."""
    out = {"status": "descriptive diagnostic (Phase 4d, CLAUDE.md section 15): no rule, signal, sizing or cost changes; "
                     "not a strategy configuration",
           "question": "is part of the cash auction effect an artifact of the CMT input bond switching from the old to "
                       "the new issue on or right after the auction?",
           "definitions": "src/cmt_switch.py module docstring",
           "units": {"cash_vs_futures": "before-cost P&L, bp of capital, each leg at its pre-registered DV01 (LEG_RISK "
                                        "x CAPITAL / (sigma_bp x sqrt(5)) x FOMC half size; no drawdown rule or "
                                        "notional cap), the same DV01 in cash and futures; b = mean across events, "
                                        "se and t clustered by the Monday-Sunday week of A; gap = cash - futures; "
                                        "day k = close A+k-1 -> close A+k",
                     "reopening_control": "pre-registered event returns, %, before costs (R_pre, R_post, LS = R_post "
                                          "- R_pre); week-clustered"},
           "cash_vs_futures": fut_part}
    out.update(cash_part)
    if "all" in fut_part:
        a = fut_part["all"]
        out["summary"] = {"share_of_total_gap_on_A": a["day_A"]["share_of_total_gap"],
                          "share_of_total_gap_on_S": a["day_S"]["share_of_total_gap"],
                          "share_of_total_gap_on_A_and_S": a["gap_on_A_and_S"]["share_of_total_gap"],
                          "mean_total_gap_bp": a["legs"]["total"]["gap"]["b"],
                          "mean_total_gap_t": a["legs"]["total"]["gap"]["t"]}
    return out


def fut_paths() -> dict:
    return {k: report.OUTPUTS / "tables" / v for k, v in FUT_TABLES.items()}


def read_fut_tables(legs_src, daily_src) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The derived futures tables (paths or text buffers) as every futures number is computed from them."""
    str_cols = ("leg_id", "unit_id", "kind", "tenor", "product", "root", "contract", "status")
    leg_t = pd.read_csv(legs_src, dtype={c: str for c in str_cols}, keep_default_na=False,
                        na_values={c: [""] for c in fut.LEG_TABLE_COLS if c not in str_cols},
                        float_precision="round_trip")
    day_t = pd.read_csv(daily_src, index_col="date", float_precision="round_trip")
    return leg_t, day_t


def futures(*, cash_cal, cash_supply_daily, cash_supply_gross_daily, rf, git: dict, t0: float) -> dict | None:
    """Futures layer, in-sample (CLAUDE.md 7.9 and section 15; src/futures.py, src/data/databento_futures.py).

    Every futures number is computed from the derived tables (outputs/tables/futures_*), written by
    rebuild_futures() on a --futures run or committed, so a keyless run reproduces results.json; skipped (None) if
    they are absent. Logs 12 trials (4 Phase 4 + 8 risk-rule variants, Phase 4b). Phase 4c adds the capacity variant
    active_adv beside the as-written capacity and the cash supply leg's Sharpe before costs on the futures days
    (cash_supply_gross_daily: daily net excess + cost / capital), both descriptive. Phase 4d reads the CMT switch
    diagnostic's aggregates from the checks file and logs one descriptive row under the 1x supply leg's config_hash.
    Returns the results block with private keys _daily_excess (Deflated Sharpe inputs), _metrics_1x
    (in_sample.metrics.futures), _fig (figure 5's futures panel) and _cmt_switch (the diagnostic)."""
    import json
    paths = fut_paths()
    if not all(p.exists() for p in paths.values()):
        print("futures layer skipped: no derived futures tables (outputs/tables/futures_*) and no --futures run; "
              "core results unchanged.", flush=True)
        return None
    leg_t, day_t = read_fut_tables(paths["legs"], paths["daily"])
    checks = json.loads(paths["checks"].read_text(encoding="utf-8"))
    res = fut.from_tables(leg_t, day_t, rf, FUT_UNITS)
    me1, me2, su1, su2 = (res[k] for k in ("month_end_zn", "month_end_zn_cost_2x", "supply_calendar",
                                           "supply_calendar_cost_2x"))
    fdays = me1.daily.index
    m_me = {k: fut.month_end_metrics(r) for k, r in res.items() if k.startswith("month_end")}
    m_su = {k: fut.book_metrics_futures(r) for k, r in res.items() if k.startswith("supply")}

    # descriptive comparisons with the cash legs over the same days (cash runs sliced, not rerun)
    cx = cash_cal.daily["excess"].reindex(fdays)
    t_f, t_c = fut.as_strategy(me1).trades, cash_cal.trades
    both = t_f.index.intersection(t_c.index)
    pf, pc = t_f.loc[both, "net_pnl"] / me1.cfg.capital * 100, t_c.loc[both, "net_pnl"] / cash_cal.cfg.capital * 100
    vs_cash_me = {"cash_calendar_only_slice": slice_metrics(cash_cal.daily, fdays[0], fdays[-1]),
                  "sharpe_futures_vs_cash": compare_sharpe(me1.daily["excess"], cx, "futures", "cash"),
                  "corr_daily_excess": float(me1.daily["excess"].corr(cx)),
                  "windows_both": int(len(both)), "corr_window_net_pnl": float(pf.corr(pc)),
                  "mean_window_net_pnl_pct": {"futures": float(pf.mean()), "cash": float(pc.mean())},
                  "note": "cash calendar_only (10-year par bond) is the full in-sample run sliced to the futures "
                          "days; descriptive"}
    sx = cash_supply_daily.reindex(fdays)
    vs_cash_su = {"cash_supply_calendar_slice": slice_metrics(pd.DataFrame({"excess": cash_supply_daily}), fdays[0],
                                                              fdays[-1]),
                  "sharpe_futures_vs_cash": compare_sharpe(su1.daily["excess"], sx, "futures", "cash"),
                  "corr_daily_excess": float(su1.daily["excess"].corr(sx)),
                  "note": "cash calendar supply leg is the full in-sample run sliced to the futures days; descriptive"}
    cash_gross = slice_metrics(pd.DataFrame({"excess": cash_supply_gross_daily}), fdays[0], fdays[-1])
    vs_cash_su["before_costs_same_days"] = {
        "sharpe_futures": m_su["supply_calendar"]["sharpe_gross"], "sharpe_cash": cash_gross["sharpe"],
        "sharpe_futures_net": m_su["supply_calendar"]["sharpe"],
        "sharpe_cash_net": vs_cash_su["cash_supply_calendar_slice"]["sharpe"],
        "cash_supply_calendar_slice_before_costs": cash_gross,
        "note": "Phase 4c, descriptive (CLAUDE.md section 15): Sharpe of daily net excess + that day's cost / "
                "capital over the futures days; cash = the full in-sample run sliced, futures = sharpe_gross of the "
                "supply leg (costs netted by contract in the strategy)"}
    years = period_years(fdays)
    eq = {}
    for k, r in res.items():
        eq[f"nav_total_{k}"] = (1.0 + r.daily["total"]).cumprod()
        eq[f"nav_excess_{k}"] = (1.0 + r.daily["excess"]).cumprod()
    report.write_table(pd.DataFrame(eq).rename_axis("date"), "equity_curve_futures_insample.csv")
    step(f"futures: month-end {HEADLINE_FUTURE} {m_me['month_end_zn']['n_windows']} windows, supply leg "
         f"{m_su['supply_calendar']['n_legs']} legs traded, {fdays[0].date()}..{fdays[-1].date()}", t0)

    # every risk rule on and off (Phase 4b; as Phase 5 for cash), 1x costs
    rr = {"all_on": {"config": asdict(RiskConfig()), "metrics": {"month_end_zn": m_me["month_end_zn"],
                                                                 "supply_calendar": m_su["supply_calendar"]}}}
    for name in FUT_RISK_NAMES:
        rr[name] = {"config": asdict(RiskConfig(**RISK_VARIANTS[name])),
                    "metrics": {"month_end_zn": m_me[f"month_end_zn_{name}"],
                                "supply_calendar": m_su[f"supply_calendar_{name}"]}}
    kill = {"rule": 'PREREG_FLOWCLOCK.md "What kills it": "It disappears at 2x costs."',
            "sharpe_1x": m_su["supply_calendar"]["sharpe"], "sharpe_2x": m_su["supply_calendar_cost_2x"]["sharpe"],
            "met": True,
            "basis": "team judgment of Oct 3, 2026 (CLAUDE.md section 15, Phase 4b) on the Phase 4 numbers, net "
                     "Sharpe 0.289 at 1x and 0.071 at 2x",
            "same_as_phase4_numbers": bool(abs(m_su["supply_calendar"]["sharpe"] - 0.289016) < 5e-6
                                           and abs(m_su["supply_calendar_cost_2x"]["sharpe"] - 0.0711606) < 5e-7),
            "statement": f"The futures supply leg meets the pre-registered kill condition: net Sharpe "
                         f"{m_su['supply_calendar']['sharpe']:.2f} at 1x costs and "
                         f"{m_su['supply_calendar_cost_2x']['sharpe']:.2f} at 2x costs ({FUT_START[:7]} to "
                         f"{IS_END[:7]})."}

    # trial log (rule 7): month-end ZN and supply leg, 1x and 2x costs; Phase 4b: each risk rule off (1x)
    fu_cfg = {"settings": report.settings_dict(), "futures": {k: getattr(fus, k) for k in dir(fus) if k.isupper()},
              "flowclock": {k: getattr(fcs, k) for k in dir(fcs) if k.isupper()}}
    rows = []
    for key, strat_name, tenor, entry, exit_, unit_n in (
            ("month_end_zn", "futures_calendar_only", f"{HEADLINE_FUTURE} (DGS10)", f"T-{ENTRY_OFFSET}",
             f"T+{EXIT_OFFSET}", "n_windows"),
            ("supply_calendar", "futures_supply_leg", "ZT-UB by auction tenor", "A-5,A", "A,A+5", "n_units")):
        for suffix, cm in (("", 1.0), ("_cost_2x", COST_STRESS)):
            mk = m_me if key.startswith("month_end") else m_su
            mt = mk[key + suffix]
            rows.append(trial_row({**fu_cfg, "strategy": strat_name, "risk": asdict(RiskConfig(cost_mult=cm))},
                                  "in_sample_futures" + ("" if cm == 1.0 else "_cost2x"), {
                "strategy": strat_name, "tenor": tenor, "entry": entry, "exit": exit_, "n": mt[unit_n],
                "H1_b": "", "H1_lo": "", "H1_hi": "", "sharpe_fc": "", "sharpe_cal": mt["sharpe"],
                "note": f"run_all Phase 4: futures {strat_name}, {FUT_START[:7]} to {IS_END[:7]}, {cm:g}x costs; "
                        f"sharpe_cal = this calendar strategy's net Sharpe; no slope (H1_* blank)"}, git=git))
        for name in FUT_RISK_NAMES:
            mk = m_me if key.startswith("month_end") else m_su
            mt = mk[f"{key}_{name}"]
            rows.append(trial_row({**fu_cfg, "strategy": strat_name, "risk": asdict(RiskConfig(**RISK_VARIANTS[name]))},
                                  f"in_sample_futures_risk_{name}", {
                "strategy": strat_name, "tenor": tenor, "entry": entry, "exit": exit_, "n": mt[unit_n],
                "H1_b": "", "H1_lo": "", "H1_hi": "", "sharpe_fc": "", "sharpe_cal": mt["sharpe"],
                "note": f"run_all Phase 4b: futures {strat_name}, risk rules {name}, 1x costs; sharpe_cal = this "
                        f"calendar strategy's net Sharpe; no slope (H1_* blank)"}, git=git))
    cmt = checks.pop("cmt_switch_diagnostic", {"status": "not in the derived tables; rerun with --futures"})
    if "all" in cmt:            # Phase 4d, descriptive: the futures supply leg's 1x configuration, not a new one
        cfg_su = {**fu_cfg, "strategy": "futures_supply_leg", "risk": asdict(RiskConfig(cost_mult=1.0))}
        row = trial_row(cfg_su, "in_sample_futures_cmt_switch_diagnostic", {
            "strategy": "futures_supply_leg", "tenor": "ZT-UB by auction tenor", "entry": "A-5,A", "exit": "A,A+5",
            "n": cmt["n_events_used"], "H1_b": "", "H1_lo": "", "H1_hi": "", "sharpe_fc": "",
            "sharpe_cal": m_su["supply_calendar"]["sharpe"],
            "note": "run_all Phase 4d: descriptive diagnostic, not a strategy configuration (CMT switch: cash - futures "
                    "supply-leg P&L by day around auctions, same events, same DV01); config_hash and sharpe_cal "
                    "repeat the 1x futures supply leg row, so distinct variants and the Deflated Sharpe's V are "
                    "unchanged; results.json[cmt_switch_diagnostic]"}, git=git)
        base = [r for r in rows if r["window"] == "in_sample_futures" and r["strategy"] == "futures_supply_leg"]
        if len(base) != 1 or base[0]["config_hash"] != row["config_hash"]:
            raise RuntimeError("Phase 4d diagnostic row: config_hash differs from the futures supply leg's 1x row")
        rows.append(row)
    n_logged = log_trials(rows)
    step(f"futures: {len(rows)} trial rows ({n_logged} written to runs/trials.csv)", t0)

    capacity = checks.pop("capacity")
    capacity_active = checks.pop("capacity_active_adv", {"status": "not in the derived tables; rerun with --futures"})
    fig = {"fut_navs": {f"calendar-only, {HEADLINE_FUTURE} futures": (1.0 + me1.daily["excess"]).cumprod(),
                        "calendar-only, cash, same days": (1.0 + cx).cumprod(),
                        "supply leg, futures": (1.0 + su1.daily["excess"]).cumprod(),
                        "supply leg, cash, same days": (1.0 + sx).cumprod()},
           "fut_sample": f"{fdays[0].strftime('%Y-%m')} to {fdays[-1].strftime('%Y-%m')}"}
    fig["fut_sharpes"] = dict(zip(fig["fut_navs"], [m_me["month_end_zn"]["sharpe"],
                                                    vs_cash_me["cash_calendar_only_slice"]["sharpe"],
                                                    m_su["supply_calendar"]["sharpe"],
                                                    vs_cash_su["cash_supply_calendar_slice"]["sharpe"]]))
    return {
        "status": "computed from the derived tables outputs/tables/futures_*; raw Databento data stays in "
                  "data/cache/; the committed tables hold no notional, raw volume or exact cap factor and round daily "
                  "notionals to $1M (CLAUDE.md section 15, Phase 4b)",
        "sample": [str(fdays[0].date()), str(fdays[-1].date())], "years": years,
        "rules": "CLAUDE.md section 15; src/futures.py; src/data/databento_futures.py",
        "contracts": {"month_end": HEADLINE_FUTURE, "supply": dict(fus.SUPPLY_CONTRACT),
                      "fallback": dict(fus.FALLBACK)},
        "costs": {"per_contract_round_trip": f"1 tick + ${fut.FUT_COMMISSION_RT:g}", "stress_mult": COST_STRESS},
        "data": checks,
        "month_end_zn": {"metrics": m_me["month_end_zn"], "metrics_cost_2x": m_me["month_end_zn_cost_2x"],
                         "legs": fut.leg_summary(me1), "legs_cost_2x": fut.leg_summary(me2),
                         "vs_cash_calendar_only": vs_cash_me},
        "supply_calendar": {"metrics": m_su["supply_calendar"], "metrics_cost_2x": m_su["supply_calendar_cost_2x"],
                            "legs": fut.leg_summary(su1), "legs_cost_2x": fut.leg_summary(su2),
                            "by_tenor_pnl": by_tenor(fut.traded(su1), years), "vs_cash_supply_calendar": vs_cash_su,
                            "kill_condition_2x_costs": kill},
        "risk_rules_on_off": rr,
        "capacity": capacity,
        "capacity_active_adv": capacity_active,
        "tables": {k: f"outputs/tables/{v}" for k, v in FUT_TABLES.items()},
        "_daily_excess": {"month_end_zn": me1.daily["excess"], "supply_calendar": su1.daily["excess"]},
        "_metrics_1x": {"calendar_only_zn": m_me["month_end_zn"], "flowclock_supply_calendar": m_su["supply_calendar"]},
        "_fig": fig,
        "_cmt_switch": cmt,
    }


def rebuild_futures(cal, win, is_months, fomc, rf, y10, t0) -> None:
    """--futures: the auction events (src/auction_events.py, as flowclock() builds them; no returns) and the futures
    books -> the derived tables. Runs before any trial of the run is logged, so a failure here logs nothing."""
    assert_flowclock_prereg()
    yld = load_frame(FC_TENORS, end=IS_END, index=cal.days, fill=True)
    auctions = load_auctions(end=IS_END, exclude=None)
    ev = build_events(auctions, cal, yld)
    traded = ev["window_complete"] & ~ev["skipped"]
    sup_entries = pd.DatetimeIndex(pd.concat([ev.loc[traded, "pre_entry"], ev.loc[traded, "A"]]))
    fut_months = is_months[is_months >= pd.Period(FUT_START, "M")]
    issue = auctions.set_index(["auction_date", "cusip"])["issue_date"]
    cmt = {"issue_date": pd.Series(issue.reindex(pd.MultiIndex.from_arrays([ev["A"], ev["cusip"]])).to_numpy(),
                                   index=ev.index),
           "reopen": cmt_switch.reopening_flags(ev, auctions),
           "excess": tenor_returns(end=IS_END, tenors={s: s for s in FC_TENORS})[1]}
    _build_futures_tables(cal, win, fut_months, fomc, rf, y10, ev, sup_entries, yld, fut_paths(), t0, cmt=cmt)


def _build_futures_tables(cal, win, fut_months, fomc, rf, y10, ev, sup_entries, yld, paths, t0,
                          cmt: dict | None = None) -> None:
    """Settlements, volume and definitions from the Databento cache -> the futures books (Phase 4: 1x and 2x costs;
    Phase 4b: each risk rule off) -> the derived tables (no notional, raw volume, exact cap factor or settlement
    level; src/futures.py), the capacity aggregates (as written and, Phase 4c, active_adv), the data checks and,
    Phase 4d, the CMT switch diagnostic's aggregates (cmt: issue_date and reopen aligned with ev, and the cash
    excess returns per tenor; None skips it)."""
    import json

    import yaml
    defs = dbf.load_definition_snapshots()
    pv = dbf.point_values(defs)
    specs = yaml.safe_load((Path(__file__).resolve().parent / "config" / "contract_specs.yaml").read_text())
    bad = {r: (pv.get(r), float(v["point_value_usd"])) for r, v in specs.items() if pv.get(r) != v["point_value_usd"]}
    if bad:
        raise RuntimeError(f"point values in the definitions differ from config/contract_specs.yaml: {bad}")
    st, vo = dbf.load_settlements(end=IS_END), dbf.load_volume(end=IS_END)
    m = fut.build_market(st, vo, defs, cal, pv)
    step(f"futures data: {len(st)} settlements, {st['contract'].nunique()} contracts, "
         f"{st['trade_date'].min().date()}..{st['trade_date'].max().date()}", t0)
    fdays = cal.days[(cal.days >= pd.Timestamp(FUT_START)) & (cal.days <= pd.Timestamp(IS_END))]
    me_legs = fut.month_end_legs(demand_legs(win.loc[fut_months], cal))
    sel = in_sample_mask(ev, FUT_START, IS_END) & ~ev["skipped"].astype(bool)
    su_legs = fut.supply_legs_futures(supply_legs(ev[sel], False))
    me_prep, su_prep = fut.prepare_legs(me_legs, m, yld), fut.prepare_legs(su_legs, m, yld)
    common = dict(m=m, yields=yld, rf=rf, fomc_scheduled=fomc, cal=cal, days=fdays)
    variants = {"": RiskConfig(), "_cost_2x": RiskConfig(cost_mult=COST_STRESS)}
    variants.update({f"_{n}": RiskConfig(**RISK_VARIANTS[n]) for n in FUT_RISK_NAMES})     # Phase 4b, 1x costs
    res = {}
    for suffix, cfg in variants.items():
        res["month_end_zn" + suffix] = fut.run_futures_book("month_end_zn" + suffix, me_legs, me_prep, cfg=cfg,
                                                            demand_per_year=WINDOWS_PER_YEAR, unit="month",
                                                            record_leg_daily=suffix == "", **common)
        res["supply_calendar" + suffix] = fut.run_futures_book("supply_calendar" + suffix, su_legs, su_prep,
                                                               cfg=cfg, supply_entries=sup_entries,
                                                               record_leg_daily=suffix == "", **common)
    leg_t, day_t = fut.to_tables(res)
    leg_t.to_csv(paths["legs"], index=False, lineterminator="\n")       # default float repr: shortest round-trip
    day_t.to_csv(paths["daily"], lineterminator="\n")
    capacity = {k: fut.capacity_curve_futures(res[k], m) for k in ("month_end_zn", "supply_calendar")}
    capacity_active = {k: fut.capacity_curve_futures(res[k], m, adv_measure="active")      # Phase 4c variant
                       for k in ("month_end_zn", "supply_calendar")}
    step("futures: capacity curves done, as written and active_adv (aggregates only)", t0)
    if cmt is not None:                 # Phase 4d (src/cmt_switch.py): same events, same DV01, aggregates only
        ev_f = ev[sel]
        cmt_out = cmt_switch.cash_vs_futures(ev_f, cmt["issue_date"][sel], cmt["reopen"][sel], su_legs, su_prep, m,
                                             cmt["excess"], yld, fomc, cal, RiskConfig().capital,
                                             check=res["supply_calendar"])
        step(f"futures: CMT switch diagnostic, {cmt_out['n_events_used']} of {cmt_out['n_events']} events "
             f"(futures engine check max diff ${cmt_out['futures_check']['max_abs_diff_usd']:.2g})", t0)
    else:
        cmt_out = {"status": "not computed in this build"}
    cover = st.groupby("root")["trade_date"].agg(["min", "max", "size"])
    checks = {
        "source": "Databento GLBX.MDP3: statistics (settlement), ohlcv-1d (volume), definition (monthly snapshots)",
        "pull_window": {"start": dbf.DATA_START, "end": dbf.PULL_END, "trade_dates_kept_through": IS_END},
        "n_settlements": int(len(st)), "n_contracts": int(st["contract"].nunique()),
        "coverage_by_root": {r: {"first": str(c["min"].date()), "last": str(c["max"].date()), "n": int(c["size"])}
                             for r, c in cover.iterrows()},
        "n_definition_snapshots": int(defs["snapshot"].nunique()),
        "point_value_usd": pv, "tick_history": dbf.tick_history(defs),
        "first_sizable_entry": {"month_end": me_prep.attrs["first_sizable_entry"],
                                "supply": su_prep.attrs["first_sizable_entry"]},
        "alignment": fut.alignment_checks(m, y10, fus.ALIGNMENT_DATES),
        "bond_days_without_any_settlement": fut.settlement_gaps(m, fdays),
        "capacity": capacity,
        "capacity_active_adv": capacity_active,
        "cmt_switch_diagnostic": cmt_out,
    }
    paths["checks"].write_text(json.dumps(report.clean(checks), indent=2) + "\n", encoding="utf-8")
    step(f"futures: derived tables written ({len(leg_t)} leg rows, {len(day_t)} days)", t0)


# ------------------------------------------------------------------------------------------------ test window (Gate 2)

OOS_JSON = report.OUTPUTS / "results_oos.json"
OOS_LOG = REPO_ROOT / "runs" / "oos_run.log"
OOS_MARK = oos_data.CACHE_DIR / "oos" / "run_started.log"      # git-ignored: the start of a run in progress
OOS_FUT_TABLES = {"legs": "futures_legs_oos.csv", "daily": "futures_daily_oos.csv",
                  "checks": "futures_data_checks_oos.json"}
OOS_FUT_UNITS = {k: FUT_UNITS[k] for k in ("month_end_zn", "month_end_zn_cost_2x", "supply_calendar",
                                           "supply_calendar_cost_2x")}
PSEUDO_WINDOW = {"start": "2022-10-01", "end": IS_END, "split": "2022-09-30"}   # section 17, inside the in-sample
PSEUDO_TOL = 1e-12


def evaluate_window(start: str, end: str, *, git: dict, label: str = "oos", futures_src: dict | None = None) -> dict:
    """Every pre-registered test and strategy on the window [start, end] (CLAUDE.md section 17) with the in-sample
    code paths of insample(), month_end_phase5(), flowclock(), h8_block() and futures(); only the window changes, as
    for the post-publication and post-LYZ sub-samples. The data loaders read whatever snapshot is active (src/data/
    snapshot.py::reading_from: the --oos data view). Signals use all history from 1990 (past-only z); strategies
    start flat on the window's first bond day (NAV 1, drawdown rule reset); the risk look-backs reach before it.

    futures_src: None (futures skipped), {"raw": {"settle", "volume", "defs"}} (Databento long tables; the books are
    built, written to derived tables in memory and recomputed from them, as the in-sample futures() does) or
    {"tables": {"legs", "daily", "checks"}} (committed derived *_oos tables: the keyless path).

    Prints nothing and writes nothing (section 17: nothing until every block is computed). Returns the blocks
    {"month_end", "flowclock", "H8", "futures"} plus private keys: _daily (daily net excess per strategy key),
    _rows (trial rows, not logged), _tables ({file name: (DataFrame, index) or text}), _fig (equity curves)."""
    sample = [start, end]
    cal = load_calendar(end=end)
    rcfg = RebuildConfig()
    monthly, _ = run_default(end=end, cfg=rcfg)
    tenor = HEADLINE_TENOR
    total, excess = tenor_returns(end=end)
    x10 = excess[tenor]
    rf = total[tenor] - excess[tenor]
    y10 = load_frame([tenor, RF_SERIES], end=end, index=cal.days, fill=True)[tenor]
    fomc = load_fomc_dates(end=end, scheduled_only=True)
    months_frame = pd.DataFrame({"E": pd.to_datetime(monthly["E"].to_numpy())},
                                index=pd.PeriodIndex(monthly["month"], freq="M"))
    sig = build_signals(monthly, fomc, pension_pressure(months_frame, cal, load_pension_input(end=end), total[tenor]))
    months = sig.index[(sig.index >= pd.Period(start, "M")) & (sig.index <= pd.Period(end, "M"))]
    s = sig.loc[months]
    if s["z"].isna().any():
        raise RuntimeError("z_m undefined inside the window")
    win = month_end_windows(cal, months, ENTRY_OFFSET, EXIT_OFFSET)
    if not ((win["entry"] == s["E"]).all() and (win["T"] == s["T"]).all()):
        raise RuntimeError("window dates differ from the rebuild's E and T")
    R = window_returns(x10, win) * 100.0
    Y = -window_yield_change_bp(y10, win)
    days = cal.days[(cal.days >= pd.Timestamp(start)) & (cal.days <= pd.Timestamp(end))]
    cfg1, cfg2 = RiskConfig(), RiskConfig(cost_mult=COST_STRESS)
    me = dict(daily_excess=x10, rf=rf, y_tenor=y10, y10=y10, tenor_years=KNOT_YEARS[tenor], fomc_scheduled=fomc,
              cal=cal, days=days)
    ones = pd.Series(1.0, index=months)
    strat = {"forecast_sized": run_strategy("forecast_sized", win, s["w"], cfg=cfg1, **me),
             "calendar_only": run_strategy("calendar_only", win, ones, cfg=cfg1, **me)}
    fc2 = run_strategy("forecast_sized_cost2x", win, s["w"], cfg=cfg2, **me)
    cal2 = run_strategy("calendar_only_cost2x", win, ones, cfg=cfg2, **me)

    # month-end leg: H1 (+ components, addendum), H4, placebo, luck test, H2, H3, H5, curve-allocated, costs 2x
    res_h1 = h1(R, Y, s["z"])
    res_h1["components"] = h1_components(R, Y, s["z_ext"], s["z_cash"])
    res_h4 = h4(strat["forecast_sized"].daily["excess"], strat["calendar_only"].daily["excess"])
    p_months = months[months + 1 <= pd.Period(end, "M")]
    pwin = placebo_windows(cal, p_months, PLACEBO_BDAYS)
    P = window_returns(x10, pwin) * 100.0
    PY = -window_yield_change_bp(y10, pwin)
    pstrat = {"forecast_sized": run_strategy("placebo_forecast_sized", pwin, s["w"], cfg=cfg1, **me),
              "calendar_only": run_strategy("placebo_calendar_only", pwin, ones, cfg=cfg1, **me)}
    luck = luck_test(luck_candidates(cal, months, x10), R)
    luck_draws = luck.pop("draw_means_pct")
    rwin = reversal_windows(cal, months, REVERSAL_DAYS)
    yall = load_frame(list(TENORS.values()), end=end, index=cal.days, fill=True)
    midx = pd.PeriodIndex(monthly["month"], freq="M")
    fdd_b = pd.DataFrame({b: monthly[f"fdd_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    ext_b = pd.DataFrame({b: monthly[f"ext_{b}"].to_numpy(float) for b in BUCKETS}, index=midx)
    neg_dy_b = pd.DataFrame({b: -(yall[TENORS[b]].reindex(win["exit"]).to_numpy()
                                  - yall[TENORS[b]].reindex(win["entry"]).to_numpy()) * 100.0 for b in BUCKETS},
                            index=months)
    res_h2 = h2(neg_dy_b, fdd_b.apply(past_zscore).loc[months], ext_b.apply(past_zscore).loc[months])
    curve_args = dict(demand=fdd_b.loc[months], tenor_of=TENORS, excess=excess, yields=yall, rf=rf, y10=y10,
                      fomc_scheduled=fomc, cal=cal, days=days)
    curve = run_curve_allocated("curve_allocated", win, cfg=cfg1, **curve_args)
    curve2 = run_curve_allocated("curve_allocated_cost2x", win, cfg=cfg2, **curve_args)
    excl = pd.DataFrame(curve.excluded)
    h4_2 = h4(fc2.daily["excess"], cal2.daily["excess"])
    metrics = {k: required_metrics(v) for k, v in strat.items()}
    metrics["curve_allocated"] = required_metrics(curve)
    month_end = {
        "sample": [str(months[0]), str(months[-1])], "H1": res_h1,
        "H1_addendum": h1_addendum(R, Y, s["z"], s["ref"], s["z_surprise"]), "H4": res_h4,
        "H2": res_h2, "H3": h3(window_returns(x10, rwin) * 100.0, -window_yield_change_bp(y10, rwin), s["z"], R),
        "H5": h5(R, Y, s),
        "placebo": {"window": f"business days {PLACEBO_BDAYS[0]}-{PLACEBO_BDAYS[1]} of month m+1, paired with z_m",
                    "H1": h1(P, PY, s["z"].loc[p_months]),
                    "components": h1_components(P, PY, s["z_ext"].loc[p_months], s["z_cash"].loc[p_months]),
                    "H4": h4(pstrat["forecast_sized"].daily["excess"], pstrat["calendar_only"].daily["excess"]),
                    "metrics": {k: required_metrics(v) for k, v in pstrat.items()}},
        "random_windows": luck,
        "metrics": metrics,
        "cost_stress": {"cost_mult": COST_STRESS, "H4": h4_2,
                        "metrics": {"forecast_sized": required_metrics(fc2), "calendar_only": required_metrics(cal2)}},
        "curve_allocated": {
            "vs_calendar_only": h4(curve.daily["excess"], strat["calendar_only"].daily["excess"]),
            "vs_calendar_only_cost_2x": h4(curve2.daily["excess"], cal2.daily["excess"]),
            "metrics_cost_2x": required_metrics(curve2),
            "n_months_flat": len(curve.flat_months), "flat_months": [str(m) for m in curve.flat_months],
            "n_months_bucket_excluded": int(len(excl)),
            "months_bucket_excluded_by_bucket": ({} if excl.empty else
                                                 {b: int(excl["buckets"].map(lambda x, b=b: b in x).sum())
                                                  for b in BUCKETS}),
            "mean_weights": {b: float(curve.trades[f"a_{b}"].mean()) for b in BUCKETS},
            "note": "DV01 = calendar-only DV01 split across buckets in proportion to max(fdd_b, 0) (src/backtest.py)"},
    }
    daily = {"forecast_sized": strat["forecast_sized"].daily["excess"],
             "calendar_only": strat["calendar_only"].daily["excess"],
             "forecast_sized_cost_2x": fc2.daily["excess"], "calendar_only_cost_2x": cal2.daily["excess"],
             "curve_allocated": curve.daily["excess"], "curve_allocated_cost_2x": curve2.daily["excess"]}

    # Flow Clock (flowclock(), one sample) and H8 (h8_block())
    yld = load_frame(FC_TENORS, end=end, index=cal.days, fill=True)
    ev = build_events(load_auctions(end=end, exclude=None), cal, yld)
    _, xs = tenor_returns(end=end, tenors={t: t for t in FC_TENORS})
    ev = ev.join(event_returns(ev, xs, yld))
    mask = in_sample_mask(ev, start, end)
    sa = month_end_supply(ev, cal, load_curve(end=end), pd.period_range(SA_START, end[:7], freq="M"))
    sa["zA"] = past_zscore(sa["SA_bn_years"])
    if sa.loc[months, "zA"].isna().any():
        raise RuntimeError("zA_m undefined inside the window")
    traded = ev["window_complete"] & ~ev["skipped"]
    sup_entries = pd.DatetimeIndex(pd.concat([ev.loc[traded, "pre_entry"], ev.loc[traded, "A"]]))
    bk = dict(excess=xs, yields=yld, rf=rf, fomc_scheduled=fomc, cal=cal, supply_entries=sup_entries)
    me_nd = {k: v for k, v in me.items() if k != "days"}
    e = ev[mask & ~ev["skipped"]]
    dem = demand_legs(win, cal)
    dem1 = strat["calendar_only"]
    dem2 = run_strategy("calendar_only_cost2x", win, ones, days=days, cfg=cfg2, **me_nd)
    check = run_book("demand_check", dem, days=days, cfg=cfg1, demand_per_year=WINDOWS_PER_YEAR, unit="month",
                     **{**bk, "supply_entries": None})
    gap = float((check.daily["excess"] - dem1.daily["excess"]).abs().max())
    if gap > 1e-12:
        raise RuntimeError(f"the book engine with the demand leg alone differs from calendar_only by {gap}")
    r = {"supply_calendar": run_book("supply_calendar", supply_legs(e, False), days=days, cfg=cfg1, **bk),
         "supply_size_weighted": run_book("supply_size_weighted", supply_legs(e, True), days=days, cfg=cfg1, **bk),
         "book": run_book("flowclock_book", pd.concat([dem, supply_legs(e, False)], ignore_index=True), days=days,
                          cfg=cfg1, demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk),
         "supply_calendar_cost_2x": run_book("supply_calendar_cost2x", supply_legs(e, False), days=days, cfg=cfg2,
                                             **bk),
         "book_cost_2x": run_book("flowclock_book_cost2x", pd.concat([dem, supply_legs(e, False)], ignore_index=True),
                                  days=days, cfg=cfg2, demand_per_year=WINDOWS_PER_YEAR, unit="month", **bk)}
    ex = {k: v.daily["excess"] for k, v in r.items()}
    fc_metrics = {k: book_metrics(v) for k, v in r.items()}
    fc_metrics["demand_alone"] = required_metrics(dem1)
    fc_metrics["demand_alone_cost_2x"] = required_metrics(dem2)
    flow = {"sample": [str(days[0].date()), str(days[-1].date())], "H6": h6(ev[mask]),
            "H7": h7(R, Y, sa.loc[months, "zA"]),
            "headline3": {"cost_1x": compare_sharpe(ex["book"], dem1.daily["excess"], "book", "demand"),
                          "cost_2x": compare_sharpe(ex["book_cost_2x"], dem2.daily["excess"], "book", "demand")},
            "supply_size_vs_calendar": compare_sharpe(ex["supply_size_weighted"], ex["supply_calendar"],
                                                      "size_weighted", "calendar"),
            "corr_daily_supply_vs_demand": float(ex["supply_calendar"].corr(dem1.daily["excess"])),
            "metrics": fc_metrics, "demand_engine_check_max_abs_diff": gap,
            "n_events": int(mask.sum()), "n_events_traded": int(len(e))}
    daily.update({"flowclock_book": ex["book"], "flowclock_book_cost_2x": ex["book_cost_2x"],
                  "supply_calendar": ex["supply_calendar"], "supply_calendar_cost_2x": ex["supply_calendar_cost_2x"],
                  "supply_size_weighted": ex["supply_size_weighted"]})
    e8 = ev[mask]
    zd = dealers.zd_at(pd.DatetimeIndex(sorted(e8["pre_entry"].dropna().unique())),
                       pd_positions.load_positions(end=end), cal)
    h8r = dealers.h8(e8, zd)

    # futures (futures() and _build_futures_tables(), month-end ZN and supply leg, 1x and 2x)
    fut_out, fut_tables, fut_fig = {"status": "skipped: no Databento data or derived *_oos tables"}, {}, {}
    if futures_src is not None:
        import io
        import json
        if "raw" in futures_src:
            import yaml
            raw = futures_src["raw"]
            pv = dbf.point_values(raw["defs"])
            specs = yaml.safe_load((Path(__file__).resolve().parent / "config" / "contract_specs.yaml").read_text())
            bad = {k: (pv.get(k), float(v["point_value_usd"])) for k, v in specs.items()
                   if pv.get(k) != v["point_value_usd"]}
            if bad:
                raise RuntimeError(f"point values in the definitions differ from config/contract_specs.yaml: {bad}")
            mkt = fut.build_market(raw["settle"], raw["volume"], raw["defs"], cal, pv)
            first = json.loads((report.OUTPUTS / "tables" / FUT_TABLES["checks"]).read_text(encoding="utf-8"))[
                "first_sizable_entry"]
            me_legs = fut.month_end_legs(dem)
            su_legs = fut.supply_legs_futures(supply_legs(e, False))
            me_prep = fut.prepare_legs(me_legs, mkt, yld, first_ok_given=first["month_end"])
            su_prep = fut.prepare_legs(su_legs, mkt, yld, first_ok_given=first["supply"])
            fa = dict(m=mkt, yields=yld, rf=rf, fomc_scheduled=fomc, cal=cal, days=days)
            books = {}
            for sfx, cfg in (("", cfg1), ("_cost_2x", cfg2)):
                books["month_end_zn" + sfx] = fut.run_futures_book("month_end_zn" + sfx, me_legs, me_prep, cfg=cfg,
                                                                   demand_per_year=WINDOWS_PER_YEAR, unit="month",
                                                                   **fa)
                books["supply_calendar" + sfx] = fut.run_futures_book("supply_calendar" + sfx, su_legs, su_prep,
                                                                      cfg=cfg, supply_entries=sup_entries, **fa)
            leg_t, day_t = fut.to_tables(books)
            legs_csv = leg_t.to_csv(index=False, lineterminator="\n")
            daily_csv = day_t.to_csv(lineterminator="\n")
            checks = {"source": "Databento GLBX.MDP3: statistics (settlement), ohlcv-1d (volume), definition "
                                "(monthly snapshots); rows through the split from the in-sample cache",
                      "trade_dates": [str(raw["settle"]["trade_date"].min().date()),
                                      str(raw["settle"]["trade_date"].max().date())],
                      "n_definition_snapshots": int(raw["defs"]["snapshot"].nunique()),
                      "first_sizable_entry_used": first,
                      "first_sizable_entry_found_on_window_legs": {
                          "month_end": me_prep.attrs["first_sizable_entry"],
                          "supply": su_prep.attrs["first_sizable_entry"]},
                      "bond_days_without_any_settlement": fut.settlement_gaps(mkt, days)}
            checks_txt = json.dumps(report.clean(checks), indent=2) + "\n"
        else:
            tb = futures_src["tables"]
            legs_csv, daily_csv = (Path(tb[k]).read_text(encoding="utf-8") for k in ("legs", "daily"))
            checks_txt = Path(tb["checks"]).read_text(encoding="utf-8")
        leg_t, day_t = read_fut_tables(io.StringIO(legs_csv), io.StringIO(daily_csv))
        fres = fut.from_tables(leg_t, day_t, rf, OOS_FUT_UNITS)
        me1, me2, su1, su2 = (fres[k] for k in OOS_FUT_UNITS)
        m_me = {k: fut.month_end_metrics(fres[k]) for k in ("month_end_zn", "month_end_zn_cost_2x")}
        m_su = {k: fut.book_metrics_futures(fres[k]) for k in ("supply_calendar", "supply_calendar_cost_2x")}
        fdays = me1.daily.index
        fut_out = {"sample": [str(fdays[0].date()), str(fdays[-1].date())],
                   "contracts": {"month_end": HEADLINE_FUTURE, "supply": dict(fus.SUPPLY_CONTRACT),
                                 "fallback": dict(fus.FALLBACK)},
                   "data": json.loads(checks_txt),
                   "month_end_zn": {"metrics": m_me["month_end_zn"], "metrics_cost_2x": m_me["month_end_zn_cost_2x"],
                                    "legs": fut.leg_summary(me1), "legs_cost_2x": fut.leg_summary(me2)},
                   "supply_calendar": {"metrics": m_su["supply_calendar"],
                                       "metrics_cost_2x": m_su["supply_calendar_cost_2x"],
                                       "legs": fut.leg_summary(su1), "legs_cost_2x": fut.leg_summary(su2)},
                   "tables": {k: f"outputs/tables/{v}" for k, v in OOS_FUT_TABLES.items()}}
        fut_tables = {OOS_FUT_TABLES["legs"]: legs_csv, OOS_FUT_TABLES["daily"]: daily_csv,
                      OOS_FUT_TABLES["checks"]: checks_txt}
        for k, b in fres.items():
            daily[f"futures_{k}"] = b.daily["excess"]
        fut_fig = {"fut_navs": {f"calendar-only, {HEADLINE_FUTURE} futures": (1.0 + me1.daily["excess"]).cumprod(),
                                "supply leg, futures": (1.0 + su1.daily["excess"]).cumprod()},
                   "fut_sharpes": {f"calendar-only, {HEADLINE_FUTURE} futures": m_me["month_end_zn"]["sharpe"],
                                   "supply leg, futures": m_su["supply_calendar"]["sharpe"]}}

    # trial rows (section 17: 14 rows, windows oos_*), logged by the caller once every block is computed
    base_cfg = {"settings": report.settings_dict(), "rebuild": config_dict(rcfg), "risk": asdict(cfg1),
                "tenor": tenor, "strategy": "cash", "sample": sample}
    pl = month_end["placebo"]
    rows = [
        trial_row({**base_cfg, "window": {"entry": -ENTRY_OFFSET, "exit": EXIT_OFFSET}}, label, {
            "strategy": "cash", "tenor": tenor, "entry": f"T-{ENTRY_OFFSET}", "exit": f"T+{EXIT_OFFSET}",
            "n": res_h1["n"], "H1_b": res_h1["b"], "H1_lo": res_h1["ci"][0], "H1_hi": res_h1["ci"][1],
            "sharpe_fc": res_h4["sharpe_fc"], "sharpe_cal": res_h4["sharpe_cal"],
            "note": f"run_all --oos: test window {start} to {end}, pre-registered headline (cash, net of costs)"},
            git=git),
        trial_row({**base_cfg, "window": {"placebo_bdays": list(PLACEBO_BDAYS)}}, f"{label}_placebo", {
            "strategy": "cash", "tenor": tenor, "entry": f"bday{PLACEBO_BDAYS[0] - 1}(m+1)",
            "exit": f"bday{PLACEBO_BDAYS[1]}(m+1)", "n": pl["H1"]["n"], "H1_b": pl["H1"]["b"],
            "H1_lo": pl["H1"]["ci"][0], "H1_hi": pl["H1"]["ci"][1], "sharpe_fc": pl["H4"]["sharpe_fc"],
            "sharpe_cal": pl["H4"]["sharpe_cal"], "note": "run_all --oos: placebo control, test window"}, git=git),
        _row(base_cfg, git, f"{label}_cost2x", {"risk": asdict(cfg2)}, res_h1["n"], res_h1, h4_2["sharpe_fc"],
             h4_2["sharpe_cal"], "run_all --oos: costs 2x, test window; sharpe_fc = forecast-sized, sharpe_cal = "
             "calendar-only; H1_* = H1"),
    ]
    ca = month_end["curve_allocated"]
    for suffix, cfg_ca, cmp_key in (("", {}, "vs_calendar_only"), ("_cost2x", {"risk": asdict(cfg2)},
                                                                   "vs_calendar_only_cost_2x")):
        rows.append(_row(base_cfg, git, f"{label}_curve_allocated{suffix}", {"strategy": "curve_allocated", **cfg_ca},
                         res_h2["n_obs"], {"b": res_h2["coef"], "ci": res_h2["ci"]}, ca[cmp_key]["sharpe_fc"],
                         ca[cmp_key]["sharpe_cal"], f"run_all --oos: curve-allocated{suffix}, test window; sharpe_fc "
                         "= curve-allocated, sharpe_cal = calendar-only; H1_* = H2 coef; n = bucket-months",
                         tenor="DGS2-DGS30"))
    fc_cfg = {"settings": report.settings_dict(), "flowclock": {k: getattr(fcs, k) for k in dir(fcs) if k.isupper()},
              "risk": asdict(cfg1)}
    for cost_key, cm in (("cost_1x", 1.0), ("cost_2x", COST_STRESS)):
        wl = f"{label}_flowclock" + ("" if cm == 1.0 else "_cost2x")
        hh = flow["headline3"][cost_key]
        rows.append(trial_row({**fc_cfg, "strategy": "flowclock_book", "cost_mult": cm, "sample": label}, wl, {
            "strategy": "flowclock_book", "tenor": "DGS10+DGS2-DGS30", "entry": "T-4|A-5,A", "exit": "T|A,A+5",
            "n": flow["H6"]["n_used"], "H1_b": flow["H7"]["c"]["b"], "H1_lo": flow["H7"]["c"]["ci"][0],
            "H1_hi": flow["H7"]["c"]["ci"][1], "sharpe_fc": hh["sharpe_book"], "sharpe_cal": hh["sharpe_demand"],
            "note": f"run_all --oos: Flow Clock Headline 3, test window, {cm:g}x costs; sharpe_fc = book, "
                    f"sharpe_cal = demand leg alone; H1_* = H7 c"}, git=git))
        if cm == 1.0:
            sv = flow["supply_size_vs_calendar"]
            s_fc, s_cal, note = sv["sharpe_size_weighted"], sv["sharpe_calendar"], \
                "sharpe_fc = size-weighted supply leg, sharpe_cal = calendar supply leg"
        else:
            s_fc, s_cal, note = "", fc_metrics["supply_calendar_cost_2x"]["sharpe"], \
                "sharpe_cal = calendar supply leg (the size-weighted leg is not stress-tested)"
        h6c = flow["H6"]["H6c"]
        rows.append(trial_row({**fc_cfg, "strategy": "supply_leg", "cost_mult": cm, "sample": label}, wl, {
            "strategy": "supply_leg", "tenor": "DGS2-DGS30", "entry": "A-5,A", "exit": "A,A+5",
            "n": flow["H6"]["n_used"], "H1_b": h6c["b"], "H1_lo": h6c["ci"][0], "H1_hi": h6c["ci"][1],
            "sharpe_fc": s_fc, "sharpe_cal": s_cal,
            "note": f"run_all --oos: Flow Clock supply leg, test window, {cm:g}x costs; {note}; H1_* = H6c beta"},
            git=git))
    rows.append(trial_row({**fc_cfg, "test": "H8", "dealers": {k: getattr(dcs, k) for k in dir(dcs) if k.isupper()},
                           "series": pd_positions.COUPON_KEYS, "sample": label}, f"{label}_h8", {
        "strategy": "h8_dealers", "tenor": "DGS2-DGS30", "entry": "A-5", "exit": "A+5", "n": h8r["n"],
        "H1_b": h8r["c"]["b"], "H1_lo": h8r["c"]["ci"][0], "H1_hi": h8r["c"]["ci"][1], "sharpe_fc": "",
        "sharpe_cal": "", "note": "run_all --oos: H8 (PREREG_DEALERS.md), test window; H1_* = c and its 95% CI; "
                                  "explanation test, not a strategy (Sharpe columns blank)"}, git=git))
    if "metrics" in fut_out.get("month_end_zn", {}):
        fu_cfg = {"settings": report.settings_dict(), "futures": {k: getattr(fus, k) for k in dir(fus) if k.isupper()},
                  "flowclock": {k: getattr(fcs, k) for k in dir(fcs) if k.isupper()}, "sample": label}
        for key, strat_name, tnr, entry, exit_, unit_n, mk in (
                ("month_end_zn", "futures_calendar_only", f"{HEADLINE_FUTURE} (DGS10)", f"T-{ENTRY_OFFSET}",
                 f"T+{EXIT_OFFSET}", "n_windows", m_me),
                ("supply_calendar", "futures_supply_leg", "ZT-UB by auction tenor", "A-5,A", "A,A+5", "n_units",
                 m_su)):
            for suffix, cm in (("", 1.0), ("_cost_2x", COST_STRESS)):
                mt = mk[key + suffix]
                rows.append(trial_row({**fu_cfg, "strategy": strat_name, "risk": asdict(RiskConfig(cost_mult=cm))},
                                      f"{label}_futures" + ("" if cm == 1.0 else "_cost2x"), {
                    "strategy": strat_name, "tenor": tnr, "entry": entry, "exit": exit_, "n": mt[unit_n],
                    "H1_b": "", "H1_lo": "", "H1_hi": "", "sharpe_fc": "", "sharpe_cal": mt["sharpe"],
                    "note": f"run_all --oos: futures {strat_name}, test window, {cm:g}x costs; sharpe_cal = this "
                            f"calendar strategy's net Sharpe; no slope (H1_* blank)"}, git=git))

    # tables (written by the caller) and figure inputs
    terc = tercile_labels(s["z"])
    tables = {
        "windows_oos.csv": (pd.DataFrame({"T": win["T"].dt.date, "E": win["entry"].dt.date, "FDD": s["FDD"],
                                          "z": s["z"], "z_ext": s["z_ext"], "z_cash": s["z_cash"], "w": s["w"],
                                          "tercile": terc, "ref": s["ref"], "fomc_scheduled_in_window": s["fomc"],
                                          "R_pct": R, "neg_dy_bp": Y}).rename_axis("month"), True),
        "placebo_windows_oos.csv": (pd.DataFrame({"window_month": pwin["window_month"], "entry": pwin["entry"].dt.date,
                                                  "exit": pwin["exit"].dt.date, "z_m": s["z"].loc[p_months],
                                                  "P_pct": P, "neg_dy_bp": PY}).rename_axis("signal_month"), True),
        "luck_test_draws_oos.csv": (pd.DataFrame({"draw_mean_pct": luck_draws}), False),
        "equity_curve_cash_oos.csv": (equity_curves({**strat, "curve_allocated": curve}), True),
        "trades_cash_curve_allocated_oos.csv": (curve.trades.rename_axis("month"), True),
    }
    for k, v in strat.items():
        tables[f"trades_cash_{k}_oos.csv"] = (v.trades.rename_axis("month"), True)
    eq = {}
    for k in ("book", "supply_calendar", "supply_size_weighted"):
        eq[f"nav_excess_{k}"] = (1.0 + r[k].daily["excess"]).cumprod()
        lg = r[k].legs.copy()
        lg["entry"], lg["exit"] = lg["entry"].dt.date, lg["exit"].dt.date
        tables[f"flowclock_legs_{k}_oos.csv"] = (lg.set_index("leg_id"), True)
    eq["nav_excess_demand_alone"] = (1.0 + dem1.daily["excess"]).cumprod()
    tables["equity_curve_flowclock_oos.csv"] = (pd.DataFrame(eq).rename_axis("date"), True)
    evt = ev[mask].copy()
    for c in ["A", "announced", "pre_entry", "post_exit"]:
        evt[c] = evt[c].dt.date
    tables["auction_events_oos.csv"] = (evt.set_index("event_id"), True)
    tables.update(fut_tables)
    navs = {"forecast-sized": (1.0 + daily["forecast_sized"]).cumprod(),
            "calendar-only": (1.0 + daily["calendar_only"]).cumprod(),
            "Flow Clock book": (1.0 + daily["flowclock_book"]).cumprod(),
            "curve-allocated": (1.0 + daily["curve_allocated"]).cumprod()}
    shp = {"forecast-sized": metrics["forecast_sized"]["sharpe"], "calendar-only": metrics["calendar_only"]["sharpe"],
           "Flow Clock book": fc_metrics["book"]["sharpe"], "curve-allocated": metrics["curve_allocated"]["sharpe"]}
    return {"month_end": month_end, "flowclock": flow, "H8": h8r, "futures": fut_out, "_daily": daily,
            "_rows": rows, "_tables": tables, "_fig": {"navs": navs, "sharpes": shp, **fut_fig}}


def merge_oos(res: dict, oos: dict) -> None:
    """results.json gets the Gate 2 blocks of outputs/results_oos.json (CLAUDE.md section 17): oos (month-end leg and
    the label table), flowclock.oos, H8.oos and futures.oos. In place."""
    res["oos"] = oos["oos"]
    res.setdefault("flowclock", {})["oos"] = oos["flowclock"]
    res.setdefault("H8", {})["oos"] = oos["H8"]
    if isinstance(res.get("futures"), dict):
        res["futures"]["oos"] = oos["futures"]
    res.setdefault("figures", {})["equity_curve_oos"] = oos["figure"]
    res["meta"]["pending"] = [p for p in res["meta"].get("pending", []) if not p.startswith("oos,")]
    res["meta"]["oos"] = oos["meta"]


def oos_package(blocks: dict, ins: dict, meta: dict, fig: dict) -> dict:
    """outputs/results_oos.json from evaluate_window's blocks: labels and kill conditions (src/oos_eval.py)."""
    clean_blocks = report.clean({k: blocks[k] for k in ("month_end", "flowclock", "H8", "futures")})
    lab = oos_eval.labels(clean_blocks, ins, blocks["_daily"])
    return report.clean({"meta": meta, "oos": {**clean_blocks["month_end"], "labels": lab},
                         "flowclock": clean_blocks["flowclock"], "H8": clean_blocks["H8"],
                         "futures": clean_blocks["futures"], "figure": fig})


def oos_log_lines() -> list[str]:
    """runs/oos_run.log, then the start line of a run that has not completed (OOS_MARK)."""
    return [ln for p in (OOS_LOG, OOS_MARK) if p.exists() for ln in p.read_text(encoding="utf-8").splitlines()]


def oos_log(line: str, path: Path | None = None) -> None:
    path = path or OOS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {line}\n")


def oos_log_close(line: str) -> None:
    """At the end of a counted run: move the start line(s) from OOS_MARK into runs/oos_run.log, then `line`. The
    start is kept in git-ignored data/cache/ while the run computes, because a new file under runs/ would make the
    tree dirty and the Gate 2 date guard (trial_log.guard_end, GQH_DEV=1) refuses a dirty tree; a run that stops after
    its start line stays recorded there, so the next attempt needs --force-rerun."""
    OOS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OOS_LOG.open("a", encoding="utf-8") as f:
        if OOS_MARK.exists():
            f.write(OOS_MARK.read_text(encoding="utf-8"))
        f.write(f"{utc_now()} {line}\n")
    OOS_MARK.unlink(missing_ok=True)


def oos_futures_source(split: str, end: str, reproduce: bool) -> dict | None:
    """Where the futures block comes from (evaluate_window's futures_src): the committed derived *_oos tables
    (keyless), else the Databento in-sample cache through the split plus the test-window pull."""
    tabs = {k: report.OUTPUTS / "tables" / v for k, v in OOS_FUT_TABLES.items()}
    if reproduce and all(p.exists() for p in tabs.values()):
        return {"tables": tabs}
    if dbf.have_cache() and dbf.have_oos_cache():
        st, vo = dbf.load_window(split, end)
        return {"raw": {"settle": st, "volume": vo, "defs": dbf.load_definition_snapshots((dbf.DEF_DIR,
                                                                                            dbf.OOS_DEF_DIR))}}
    return None


def run_oos(databento_ok: bool = False, force_rerun: bool = False) -> None:
    """`python run_all.py --oos` (CLAUDE.md section 17). Two paths:
    * first run (no committed data/oos/): HEAD tagged gate2-frozen and a clean tree in every mode
      (trial_log.assert_gate2_download), public test-window download staged in data/cache/oos/, then the Databento
      estimate (the pull needs --databento-ok). Logged in runs/oos_run.log (oos_log_close); a second run needs
      --force-rerun.
    * keyless reproduction (data/oos/ committed and matching its checksums): no download, no key; the futures block
      from the committed derived *_oos tables. Without GQH_DEV it is not a run (judges' reproduction): runs/ is not
      touched. With GQH_DEV=1 it counts as a rerun (--force-rerun, a FORCED RERUN line, 14 trial rows).
    Nothing is printed or written until every block is computed."""
    import json
    t0 = time.time()
    git = git_state()
    verify_or_exit()
    assert_gate1()
    reproduce = oos_data.committed_ok()
    counted = (not reproduce) or dev_mode()
    started = [ln for ln in oos_log_lines() if " RUN START " in ln or " FORCED RERUN " in ln]
    if counted and started and not force_rerun:
        sys.exit(f"--oos: the test window already ran (runs/{OOS_LOG.name}: {started[0]}). It runs "
                 f"once; a second run needs --force-rerun and is disclosed (CLAUDE.md section 17).")
    if reproduce:
        new_dir = oos_data.OOS_DATA_DIR
        step("--oos: keyless reproduction from the committed data/oos/ (no download)", t0)
    else:
        try:
            assert_gate2_download()
        except GateError as e:
            sys.exit(f"--oos: {e} Nothing downloaded or computed. The test window runs once, after the team tags "
                     f"{GATE2_TAG} (CLAUDE.md section 17).")
        if not dev_mode():
            print("--oos: GQH_DEV is not 1, so the 14 trial rows of this run will not be written to runs/trials.csv "
                  "(rule 7; src/trial_log.py). The team runs it with GQH_DEV=1.", flush=True)
        if not oos_data.staged():
            step("--oos: Gate 2 guard passed; downloading the public test-window rows into data/cache/oos/", t0)
            entries = oos_data.download(dest=oos_data.STAGE_DIR, split=IS_END, end=OOS_END)
            step(f"--oos: public download complete ({', '.join(sorted(entries))})", t0)
        new_dir = oos_data.STAGE_DIR
        try:
            dbf.download_oos(approved=databento_ok)
        except dbf.NeedApproval as e:
            sys.exit(f"--oos: {e} No return computed: not a run (CLAUDE.md section 17).")
        except dbf.NoDatabento as e:
            sys.exit(f"--oos: {e} The futures block needs the test-window Databento pull; no return computed: not a "
                     f"run.")
    fsrc = oos_futures_source(IS_END, OOS_END, reproduce)
    if fsrc is None and not reproduce:
        sys.exit("--oos: the futures block needs the in-sample Databento cache (data/cache/databento/) and the "
                 "test-window pull; no return computed: not a run.")
    if counted:
        oos_log(("FORCED RERUN " if started else "RUN START ") + f"commit {git['commit']} dirty={git['dirty']} "
                f"mode={'reproduce' if reproduce else 'download'}", OOS_MARK)
    oos_data.build_view(new_dir, split=IS_END, end=OOS_END, view_dir=oos_data.VIEW_DIR)
    step("--oos: computing every block (nothing is printed or written until all are done)", t0)
    with reading_from(oos_data.VIEW_DIR):
        blocks = evaluate_window(OOS_START, OOS_END, git=git, futures_src=fsrc)
    ins = json.loads(report.RESULTS_JSON.read_text(encoding="utf-8"))
    vintage = json.loads((new_dir / oos_data.VINTAGE_NAME).read_text(encoding="utf-8"))
    meta = {"ran_utc": utc_now(), "commit": git["commit"], "dirty": git["dirty"], "window": [OOS_START, OOS_END],
            "split": IS_END, "mode": "download", "data": "data/oos/ (rows dated (IS_END, OOS_END]; vintage.json) + "
            "data/snapshot/ rows through IS_END", "vintage": vintage,
            "futures": "skipped" if fsrc is None else next(iter(fsrc)),
            "rules": "CLAUDE.md section 17; labels and kill conditions: src/oos_eval.py"}
    if reproduce and OOS_JSON.exists():     # keep the Gate 2 run's record; note the reproduction beside it
        meta = {**json.loads(OOS_JSON.read_text(encoding="utf-8"))["meta"], "reproduced_utc": meta["ran_utc"],
                "reproduced_commit": git["commit"], "reproduced_futures_from": meta["futures"]}
    sample = f"{OOS_START[:7]} to {OOS_END[:7]}"
    fig = blocks["_fig"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cap = figures.equity_curve_oos(fig["navs"], fig["sharpes"], FIG_DIR / "equity_curve_oos.png", sample,
                                   fig.get("fut_navs"), fig.get("fut_sharpes"))
    out = oos_package(blocks, ins, meta, {"path": "outputs/figures/equity_curve_oos.png", "caption": cap})
    n_written = log_trials(blocks["_rows"]) if counted else 0
    for name, tab in blocks["_tables"].items():
        if isinstance(tab, str):
            (report.OUTPUTS / "tables" / name).write_text(tab, encoding="utf-8")
        else:
            report.write_table(tab[0], name, index=tab[1])
    OOS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OOS_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if not reproduce:
        oos_data.commit_copy(oos_data.STAGE_DIR, oos_data.OOS_DATA_DIR)
    merge_oos(ins, out)
    ins["trials"]["count"] = trial_count()
    report.write_results(ins, report.RESULTS_JSON)
    if counted:
        oos_log_close(f"RUN COMPLETE commit {git['commit']} trial_rows_written={n_written}")
    step(f"--oos done: outputs/results_oos.json, merged into outputs/results.json (oos, flowclock.oos, H8.oos, "
         f"futures.oos); {len(blocks['_rows'])} trial rows ({n_written} written to runs/trials.csv)", t0)
    lab = out["oos"]["labels"]["headline"]
    for k in ("H1", "H4", "H8"):
        print(f"  {k}: {lab[k]['label']} (estimate {lab[k]['estimate']})", flush=True)


def run_oos_pseudo() -> int:
    """`python run_all.py --oos-pseudo` (CLAUDE.md section 17, test only): evaluate_window on 2022-10-01..2024-09-30
    through a data view that splits the committed snapshot at 2022-09-30 as if the later rows were downloaded, vs
    src/oos_check.reference_window (the engines run directly on the snapshot). Writes nothing to the repo (the view
    lives in a temporary directory) and logs no trial. Futures included when the in-sample Databento cache is
    present. Returns the number of series or statistics that differ by more than PSEUDO_TOL."""
    import tempfile

    from src import oos_check
    t0 = time.time()
    verify_or_exit()
    assert_gate1()
    start, end, split = PSEUDO_WINDOW["start"], PSEUDO_WINDOW["end"], PSEUDO_WINDOW["split"]
    with_fut = dbf.have_cache()
    with tempfile.TemporaryDirectory(prefix="oos_pseudo_") as tmp:
        new_dir, view_dir = Path(tmp) / "new", Path(tmp) / "view"
        new_dir.mkdir()
        for name, col in oos_data.DATE_COL.items():       # the "download": snapshot rows dated in (split, end]
            oos_data.write_csv(oos_data.rows_between(oos_data.read_str(oos_data.SNAPSHOT_DIR / name), col, split,
                                                     end), name, new_dir)
        oos_data.build_view(new_dir, split=split, end=end, view_dir=view_dir)
        fsrc = None
        if with_fut:
            st, vo = dbf.load_window(split, end, new_paths={s: dbf.raw_path(s) for s in dbf.BULK_SCHEMAS})
            fsrc = {"raw": {"settle": st, "volume": vo, "defs": dbf.load_definition_snapshots()}}
        with reading_from(view_dir):
            out = evaluate_window(start, end, git=git_state(), label="pseudo", futures_src=fsrc)
    step(f"--oos-pseudo: evaluate_window {start}..{end} (split {split}) done; reference engines next", t0)
    ref = oos_check.reference_window(start, end, with_fut)
    diffs = oos_check.compare(out, ref)
    bad = {k: v for k, v in diffs.items() if not v <= PSEUDO_TOL}
    for k, v in diffs.items():
        print(f"  {k:<40s} max abs diff {v:.3g}", flush=True)
    step(f"--oos-pseudo: {len(diffs)} series and statistics compared (futures {'in' if with_fut else 'skipped'}); "
         f"{len(bad)} differ by more than {PSEUDO_TOL:g}; nothing written, no trial logged", t0)
    return len(bad)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--insample", action="store_true", help="in-sample (default)")
    p.add_argument("--futures", action="store_true", help="rebuild the derived futures tables from the Databento cache "
                   "(pulled with DATABENTO_API_KEY if missing); without it they are read from outputs/tables/")
    p.add_argument("--oos", action="store_true", help="test window, once, at Gate 2 only (CLAUDE.md section 17)")
    p.add_argument("--databento-ok", action="store_true", help="with --oos: pull the test-window Databento data "
                   "after the team approved the printed estimate")
    p.add_argument("--force-rerun", action="store_true", help="with --oos: run the test window again (disclosed in "
                   "runs/oos_run.log)")
    p.add_argument("--oos-pseudo", action="store_true", help="check the --oos path on an in-sample pseudo-window "
                   "(writes nothing, logs no trial)")
    p.add_argument("--refresh", action="store_true", help="refresh the public-data snapshot first")
    a = p.parse_args(argv)
    if (a.databento_ok or a.force_rerun) and not a.oos:
        sys.exit("--databento-ok and --force-rerun go with --oos.")
    if a.oos_pseudo:
        sys.exit(1 if run_oos_pseudo() else 0)
    if a.oos:
        run_oos(databento_ok=a.databento_ok, force_rerun=a.force_rerun)
        return
    if a.refresh:
        sys.exit("Refresh the snapshot with `python scripts/download_all.py`, review and commit it, then rerun.")
    insample(futures_mode="rebuild" if a.futures else "tables")


if __name__ == "__main__":
    main()
