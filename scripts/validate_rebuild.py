"""Print index rebuild validation tables and plots for STOP 2 (CLAUDE.md 7.6).

Usage: python scripts/validate_rebuild.py
Runs the in-sample index rebuild from the committed snapshot (no returns), writes the open dataset
outputs/extension_monthly.csv, the audit trail outputs/tables/rebuild_changes.csv, the validation tables
outputs/tables/validation_*.csv + validation.md, and the plots outputs/figures/validation_*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import IS_END, IS_START  # noqa: E402
from src import validate as v  # noqa: E402
from src.calendar import load_calendar  # noqa: E402
from src.data.auctions import load_auctions  # noqa: E402
from src.data.mspd import load_mspd  # noqa: E402
from src.data.snapshot import REPO_ROOT, verify_or_exit  # noqa: E402
from src.index_rebuild import BUCKETS, OPEN_DATASET_COLUMNS, REBUILD_START, RebuildConfig, config_dict, run_default  # noqa: E402,E501

OUT = REPO_ROOT / "outputs"
TABLES, FIGS = OUT / "tables", OUT / "figures"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(title: str) -> None:
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def plot_series(monthly: pd.DataFrame, flagged: pd.DataFrame, path: Path) -> None:
    t = pd.PeriodIndex(monthly["month"], freq="M").to_timestamp()
    series = [("Ext", "Extension Ext (years of duration)"), ("cash", "Coupon term c_m x D_next (years)"),
              ("FDD", "Forced duration demand FDD = Ext + c_m x D_next (years)")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for ax, (col, title) in zip(axes, series):
        y = monthly["c_m"] * monthly["D_next"] if col == "cash" else monthly[col]
        ax.plot(t, y, color="black", lw=0.7)
        refund = pd.PeriodIndex(monthly["month"], freq="M").month.isin([2, 5, 8, 11])
        ax.scatter(t[refund], y[refund], s=6, color="0.45", label="refunding month (Feb, May, Aug, Nov)", zorder=3)
        if col in ("Ext", "FDD"):
            f = flagged[flagged["series"] == col]
            sel = monthly["month"].isin(f["month"])
            ax.scatter(t[sel], y[sel], s=40, facecolors="none", edgecolors="black", label="|z| > 4", zorder=4)
        ax.axvline(pd.Timestamp(IS_START), color="0.3", ls="--", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    axes[-1].text(pd.Timestamp(IS_START), axes[-1].get_ylim()[0], "  in-sample starts", fontsize=8, va="bottom")
    fig.suptitle("Rebuilt monthly forced duration demand, 1990-01 to 2024-09 (1990-1992 = z-score warm-up)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist(monthly: pd.DataFrame, path: Path) -> None:
    s = v.in_sample(monthly)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (col, data) in zip(axes, [("Ext", s["Ext"]), ("c_m (share of index MV)", s["c_m"]),
                                      ("FDD", s["FDD"])]):
        ax.hist(data, bins=40, color="0.5", edgecolor="black", lw=0.4)
        ax.set_title(f"{col}, in-sample months (n={len(data)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_par(par: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(par.index, par["mspd_mil"] / 1e6, color="black", lw=1.2, label="MSPD marketable notes + bonds")
    axes[0].plot(par.index, par["rebuilt_mil"] / 1e6, color="0.5", lw=1.2, ls="--",
                 label="rebuilt from auctions (gross par, MSPD definition)")
    axes[0].set_ylabel("$ trillion")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(par.index, par["gap_pct"], color="black", lw=0.9)
    axes[1].axhline(0, color="0.6", lw=0.5)
    axes[1].set_ylabel("gap, % of MSPD")
    fig.suptitle("Validation 1: rebuilt par vs MSPD (MSPD on Fiscal Data starts 2001-01)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_duration_counts(monthly: pd.DataFrame, path: Path) -> None:
    t = pd.PeriodIndex(monthly["month"], freq="M").to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t, monthly["D_now"], color="black", lw=0.9, label="D_now (index during the month)")
    axes[0].set_ylabel("modified duration (years)")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot(t, monthly["n_adds"], color="black", lw=0.7, label="adds")
    axes[1].plot(t, monthly["n_removes"], color="0.55", lw=0.7, label="removes")
    axes[1].plot(t, monthly["n_estimated"], color="0.3", lw=0.7, ls=":", label="estimated (offering_amt)")
    axes[1].plot(t, monthly["n_skipped"], color="black", lw=1.2, ls="--", label="skipped (announced after E)")
    axes[1].set_ylabel("tranches / CUSIPs per month")
    axes[1].legend(fontsize=8, frameon=False, ncol=4)
    for ax in axes:
        ax.axvline(pd.Timestamp(IS_START), color="0.3", ls="--", lw=0.8)
    fig.suptitle("Validation 2 and 4: rebuilt index duration and monthly membership changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    verify_or_exit()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = RebuildConfig()

    monthly, changes = run_default(cfg=cfg)
    monthly.insert(3, "in_sample", pd.PeriodIndex(monthly["month"], freq="M") >= pd.Period(IS_START, "M"))
    monthly[["in_sample"] + OPEN_DATASET_COLUMNS].pipe(
        lambda d: d[["month", "T", "E", "in_sample"] + [c for c in OPEN_DATASET_COLUMNS if c not in
                                                        ("month", "T", "E")]]
    ).to_csv(OUT / "extension_monthly.csv", index=False, float_format="%.6g", lineterminator="\n")
    changes.to_csv(TABLES / "rebuild_changes.csv", index=False, float_format="%.6g", lineterminator="\n")

    md = ["# Index rebuild validation (STOP 2)", "",
          f"Generated by `scripts/validate_rebuild.py`. Rebuild {REBUILD_START} to {IS_END[:7]} "
          f"(in-sample from {IS_START[:7]}); no returns used. Config: `{json.dumps(config_dict(cfg))}`.", ""]

    # 1. par vs MSPD
    section("1. Rebuilt gross par vs MSPD marketable notes + bonds ($ billions, December of each year)")
    all_auctions = load_auctions(end=IS_END, deduct_soma=False, exclude={"TIPS": True, "FRN": True,
                                                                         "callable": False})
    par = v.par_vs_mspd(all_auctions, load_mspd(end=IS_END), load_calendar(end=IS_END).days)
    par_tab = v.par_gap_summary(par)
    print(par_tab.round(2).to_string(index=False))
    by_dec = v.par_gap_by_decade(par)
    print("gap by decade (% of MSPD):", json.dumps(by_dec))
    par.to_csv(TABLES / "validation_par_gap.csv", float_format="%.4f", lineterminator="\n")
    md += ["## 1. Rebuilt par vs MSPD", "", v.md_table(par_tab.set_index("date"), "{:.2f}"), "",
           f"Gap by decade (% of MSPD): `{json.dumps(by_dec)}`", ""]

    # 2. duration
    section("2. Rebuilt index modified duration 2015-2024 (humans: add published Treasury-index ETF durations)")
    dur = v.duration_table(monthly)
    print(dur.round(3).to_string(index=False))
    dur.to_csv(TABLES / "validation_duration.csv", index=False, float_format="%.4f", lineterminator="\n")
    md += ["## 2. Index duration 2015-2024", "", v.md_table(dur.set_index("year"), "{:.3f}"), ""]

    # 3. distribution, seasonality, outliers
    section("3a. Distribution of Ext, c_m, cash term, FDD (in-sample months, years of duration)")
    stats = v.extension_stats(monthly)
    print(stats.round(4).to_string())
    print("bucket identity:", v.bucket_check(monthly))
    s = v.in_sample(monthly)
    print("corr(Ext, c_m x D_next) =", round(float(s["Ext"].corr(s["c_m"] * s["D_next"])), 3))
    stats.to_csv(TABLES / "validation_extension_stats.csv", float_format="%.6f", lineterminator="\n")
    section("3b. Mean by calendar month (in-sample)")
    seas = v.seasonality(monthly)
    print(seas.round(4).to_string())
    seas.to_csv(TABLES / "validation_seasonality.csv", float_format="%.6f", lineterminator="\n")
    section("3c. Months with |z| > 4 (full-sample diagnostic z) and the CUSIP changes behind them")
    flagged, traced = v.outliers(monthly, changes)
    print(flagged.round(4).to_string(index=False) if len(flagged) else "none")
    if len(traced):
        show = traced.assign(amount_now_bn=traced["amount_now"] / 1e9, amount_next_bn=traced["amount_next"] / 1e9,
                             maturity=pd.to_datetime(traced["maturity"]).dt.date)
        print(show[["month", "cusip", "change", "term", "maturity", "amount_now_bn", "amount_next_bn", "dur",
                    "contribution"]].round(4).to_string(index=False))
        ctx = monthly[monthly["month"].isin(flagged["month"])][
            ["month", "Ext", "ext_adds", "ext_removes", "ext_reopens", "ext_dilution", "c_m", "FDD", "D_now",
             "mv_now_bn", "n_adds", "n_removes"]]
        print(ctx.round(4).to_string(index=False))
    flagged.to_csv(TABLES / "validation_outliers.csv", index=False, float_format="%.6f", lineterminator="\n")
    section("3d. No |z| > 4 needed for a look: the 3 largest in-sample FDD months traced to CUSIP changes")
    top = v.top_months(monthly, changes)
    top_show = top.assign(amount_now_bn=top["amount_now"] / 1e9, amount_next_bn=top["amount_next"] / 1e9,
                          maturity=pd.to_datetime(top["maturity"]).dt.date)[
        ["month", "cusip", "change", "term", "maturity", "amount_now_bn", "amount_next_bn", "dur", "contribution"]]
    print(top_show.round(4).to_string(index=False))
    print(monthly[monthly["month"].isin(top["month"])][
        ["month", "Ext", "ext_adds", "ext_removes", "ext_reopens", "ext_dilution", "c_m", "D_next", "FDD"]
    ].round(4).to_string(index=False))
    md += ["## 3. Distribution (in-sample)", "", v.md_table(stats), "",
           f"Bucket identity: `{v.bucket_check(monthly)}`", "",
           "### Mean by calendar month", "", v.md_table(seas), "",
           "### Months with |z| > 4 (diagnostic z over in-sample months)", "",
           v.md_table(flagged, index=False) if len(flagged) else "none", ""]
    md += ["### The 3 largest in-sample FDD months, traced (contribution to Ext, years)", "",
           v.md_table(top_show, index=False), ""]
    if len(traced):
        md += ["Traced to CUSIP changes (contribution to Ext, years):", "",
               v.md_table(traced[["month", "cusip", "change", "term", "maturity", "amount_now", "amount_next",
                                  "dur", "contribution"]].assign(
                   amount_now=traced["amount_now"] / 1e9, amount_next=traced["amount_next"] / 1e9,
                   maturity=pd.to_datetime(traced["maturity"]).dt.date), index=False), ""]

    # 4. counts
    section("4. Membership changes per year (sums of monthly counts)")
    counts = v.count_table(monthly)
    print(counts.to_string())
    print(f"total skipped-for-lookahead tranches 1993-01..2024-09: {int(v.in_sample(monthly)['n_skipped'].sum())}")
    counts.to_csv(TABLES / "validation_counts.csv", lineterminator="\n")
    md += ["## 4. Membership changes per year", "", v.md_table(counts, "{:.0f}"), "",
           f"Skipped-for-lookahead tranches, in-sample: {int(v.in_sample(monthly)['n_skipped'].sum())}", ""]

    (TABLES / "validation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_series(monthly, flagged, FIGS / "validation_extension_series.png")
    plot_hist(monthly, FIGS / "validation_histograms.png")
    plot_par(par, FIGS / "validation_par_gap.png")
    plot_duration_counts(monthly, FIGS / "validation_duration_counts.png")
    print("\nwrote outputs/extension_monthly.csv, outputs/tables/validation_*.csv, validation.md, "
          "rebuild_changes.csv, outputs/figures/validation_*.png")


if __name__ == "__main__":
    main()
