"""Single entry point: --insample (default) | --futures | --oos | --refresh.

`python run_all.py` rebuilds every in-sample number from the committed snapshot (CLAUDE.md rule 4): checksums,
index rebuild, signals, 10-year cash windows, H1 (+ components, PREREG_ADDENDUM.md (a)-(b)), H4, placebo and luck
test, required metrics, figures 1-2, outputs/results.json; every run appends to runs/trials.csv (rule 7).
Dates after IS_END are never read here (rule 2). Phase 3 scope: in-sample, cash bonds only.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from config.settings import (ENTRY_OFFSET, EXIT_OFFSET, HEADLINE_TENOR, IS_END, IS_START, PLACEBO_BDAYS,  # noqa: E402
                             RF_SERIES)
from src import figures, report  # noqa: E402
from src.backtest import (month_end_windows, placebo_windows, run_strategy, window_returns,  # noqa: E402
                          window_yield_change_bp)
from src.bonds import KNOT_YEARS  # noqa: E402
from src.calendar import load_calendar  # noqa: E402
from src.data.fomc import load_fomc_dates  # noqa: E402
from src.data.fred import load_frame  # noqa: E402
from src.data.french import load_pension_input  # noqa: E402
from src.data.snapshot import verify_checksums, verify_or_exit  # noqa: E402
from src.index_rebuild import RebuildConfig, config_dict, run_default  # noqa: E402
from src.metrics import equity_curves, required_metrics  # noqa: E402
from src.returns import tenor_returns  # noqa: E402
from src.risk import RiskConfig  # noqa: E402
from src.signals import build_signals, pension_pressure  # noqa: E402
from src.tests_h import (event_path_summary, event_paths, h1, h1_addendum, h1_components, h4,  # noqa: E402
                         luck_candidates, luck_test, tercile_labels)
from src.trial_log import assert_gate1, git_state, log_trial, trial_count  # noqa: E402

VERSION = "v1 (Phase 3: in-sample, cash bonds)"
PENDING = ["in_sample.H1_addendum.surprise (Phase 5)",
           "in_sample.H2, H3, H5 (Phase 5)", "in_sample.metrics.cash.curve_allocated (Phase 5, with H2)",
           "in_sample.metrics.futures (Phase 4)", "in_sample.risk_rules_on_off, betas, crowding, tails, capacity, "
           "deflated_sharpe (Phase 5)", "in_sample.tips_replication (Phase 6)",
           "post_publication (Phase 5)", "oos (Gate 2 only)", "figures 3-6 (Phase 5)"]
FIG_DIR = report.OUTPUTS / "figures"


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
    res_add = h1_addendum(R, Y, s_is["z"], s_is["ref"])
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
    step("H1, components, addendum (a)-(b), H4, placebo, luck test done", t0)

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
    ins["metrics"]["cash"]["equity_curve"] = "outputs/tables/equity_curve_cash_insample.csv"
    ins["event_path"] = ev
    res["figures"] = {"event_path": {"path": "outputs/figures/event_path.png", "caption": cap1},
                      "terciles": {"path": "outputs/figures/terciles.png", "caption": cap2}}
    res["trials"] = {"count": trial_count()}
    report.write_results(res)
    step(f"wrote outputs/results.json (trials logged so far: {res['trials']['count']})", t0)


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
