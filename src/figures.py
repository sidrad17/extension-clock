"""Note figures (CLAUDE.md section 8): 300 dpi, readable in greyscale, captions built from the numbers they show.

Phase 3: 1. event_path.png (page-1 hero exhibit) and 2. terciles.png. Phase 5: 3. extension_series.png,
4. curve_map.png, 5. equity_curve.png (in-sample; the test window is added and shaded at Gate 2; Phase 4b adds a
futures panel), 6. capacity.png.
Flow Clock (PREREG_FLOWCLOCK.md): auction_event_path.png, A-10..A+10 by size-signal tercile.
Series are told apart by line style and grey level, so every figure reads in greyscale.
Each function returns the one-sentence caption that report.py stores in results.json beside the figure path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DPI = 300
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d9d9d9"
STYLE = {"high": {"color": "#000000", "ls": "-", "fill": "#4d4d4d", "alpha": 0.18, "label": "high forced demand"},
         "mid": {"color": "#6b6b6b", "ls": "--", "fill": "#8c8c8c", "alpha": 0.14, "label": "middle"},
         "low": {"color": "#9e9e9e", "ls": ":", "fill": "#bdbdbd", "alpha": 0.22, "label": "low forced demand"}}


def _axes_style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def event_path(summary: dict, path: Path, sample: str, entry_k: int = -4) -> str:
    """Figure 1: mean cumulative 10-year excess return from T-10 to T+5 by forced-demand tercile, 95% bands."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for t in ("low", "mid", "high"):
        s, st = summary[t], STYLE[t]
        ax.fill_between(s["k"], s["lo_pct"], s["hi_pct"], color=st["fill"], alpha=st["alpha"], lw=0)
        ax.plot(s["k"], s["mean_pct"], color=st["color"], ls=st["ls"], lw=2, label=f"{st['label']} (n={s['n']})")
    ax.axvspan(entry_k, 0, color="#f0f0f0", zorder=0)
    ax.axvline(entry_k, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ymax = ax.get_ylim()[1]
    ax.text(entry_k, ymax, " enter T-4", fontsize=8, color=INK, va="top")
    ax.text(0, ymax, " T (rebalance)", fontsize=8, color=INK, va="top")
    ax.set_xticks(summary["high"]["k"])
    ax.set_xticklabels([("T" if k == 0 else f"T{k:+d}") for k in summary["high"]["k"]], fontsize=7)
    ax.set_ylabel("cumulative excess return since T-10 (%)", fontsize=9, color=INK)
    ax.set_title(f"10-year Treasury around month-end, by forced-demand tercile ({sample})", fontsize=10,
                 loc="left", color=INK)
    _axes_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    h, lo = summary["high"], summary["low"]
    i_e, i_t, i_end = h["k"].index(entry_k), h["k"].index(0), len(h["k"]) - 1
    run = {t: summary[t]["mean_pct"][i_t] - summary[t]["mean_pct"][i_e] for t in ("high", "low")}
    after = {t: summary[t]["mean_pct"][i_end] - summary[t]["mean_pct"][i_t] for t in ("high", "low")}
    return (f"From T-4 to T the 10-year excess return path rises {run['high']:.3f}% in high-demand months and "
            f"{run['low']:.3f}% in low-demand months; from T to T+{h['k'][i_end]} it moves {after['high']:+.3f}% "
            f"and {after['low']:+.3f}% (means, 95% bands, {sample}).")


AUCTION_LABEL = {"high": "large vs recent auctions", "mid": "middle", "low": "small vs recent auctions"}


def auction_event_path(ret: dict, dy: dict, path: Path, sample: str, pre_k: int = -5, post_k: int = 5) -> str:
    """Flow Clock figure: mean cumulative excess return (left) and yield change (right) from A-10 to A+10 by
    size-signal tercile (zS known at A-5), 95% bands clustered by week (src/flowclock.py::path_summary)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, summ, ylab in ((axes[0], ret, "cumulative excess return since A-10 (%)"),
                           (axes[1], dy, "yield change since A-10 (bp)")):
        for t in ("low", "mid", "high"):
            s, st = summ[t], STYLE[t]
            ax.fill_between(s["k"], s["lo"], s["hi"], color=st["fill"], alpha=st["alpha"], lw=0)
            ax.plot(s["k"], s["mean"], color=st["color"], ls=st["ls"], lw=1.8,
                    label=f"{AUCTION_LABEL[t]} (n={s['n']})")
        ax.axvspan(pre_k, 0, color="#ececec", zorder=0)
        ax.axvspan(0, post_k, color="#f7f7f7", zorder=0)
        for k in (pre_k, 0, post_k):
            ax.axvline(k, color=MUTED, lw=0.8)
        ax.axhline(0, color=MUTED, lw=0.6)
        ks = ret["high"]["k"]
        ax.set_xticks(ks[::2])
        ax.set_xticklabels([("A" if k == 0 else f"A{k:+d}") for k in ks[::2]], fontsize=7)
        ax.set_ylabel(ylab, fontsize=9, color=INK)
        _axes_style(ax)
    ymax = axes[0].get_ylim()[1]
    axes[0].text(pre_k, ymax, " pre: short", fontsize=8, color=INK, va="top")
    axes[0].text(0, ymax, " post: long", fontsize=8, color=INK, va="top")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Treasuries around coupon auctions, by auction size vs the previous six ({sample})",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    k = ret["high"]["k"]
    i_p, i_a, i_q = k.index(pre_k), k.index(0), k.index(post_k)
    pre = {t: ret[t]["mean"][i_a] - ret[t]["mean"][i_p] for t in ("high", "low")}
    post = {t: ret[t]["mean"][i_q] - ret[t]["mean"][i_a] for t in ("high", "low")}
    return (f"From A-5 to A the excess return path moves {pre['high']:+.3f}% for auctions large vs recent ones and "
            f"{pre['low']:+.3f}% for small ones; from A to A+5 it moves {post['high']:+.3f}% and "
            f"{post['low']:+.3f}% (means, 95% bands clustered by week, {sample}).")


def terciles(h1_res: dict, path: Path, sample: str) -> str:
    """Figure 2: mean T-4 -> T excess return by forced-demand tercile with bootstrap 95% intervals."""
    labels = ["low", "mid", "high"]
    t = h1_res["terciles"]
    means = [t[k]["mean"] for k in labels]
    err = [[t[k]["mean"] - t[k]["ci"][0] for k in labels], [t[k]["ci"][1] - t[k]["mean"] for k in labels]]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(range(3), means, width=0.55, color=[STYLE[k]["fill"] for k in labels], edgecolor=INK, lw=0.6)
    ax.errorbar(range(3), means, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=5)
    for i, k in enumerate(labels):
        ax.text(i, t[k]["ci"][1], f"{t[k]['mean']:.3f}%\nn={t[k]['n']}", ha="center", va="bottom", fontsize=8,
                color=INK)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low", "middle", "high"], fontsize=9)
    ax.set_xlabel("forced duration demand tercile (z, past months only)", fontsize=9, color=INK)
    ax.set_ylabel("mean T-4 to T excess return (%)", fontsize=9, color=INK)
    ax.set_title(f"10-year window return by forced demand\n({sample})", fontsize=10, loc="left", color=INK)
    _axes_style(ax)
    ax.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return (f"Mean T-4 to T 10-year excess return: {t['high']['mean']:.3f}% in high-demand months vs "
            f"{t['low']['mean']:.3f}% in low-demand months (high minus low {h1_res['high_minus_low']:+.3f}%; "
            f"bootstrap 95% intervals, {sample}).")


def extension_series(months, ext, ref, path: Path, sample: str) -> str:
    """Figure 3: monthly index extension Ext_m (years of duration), refunding months dark, others light."""
    x = np.arange(len(months))
    ext, ref = np.asarray(ext, float), np.asarray(ref, bool)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x[~ref], ext[~ref], width=0.9, color="#bdbdbd", lw=0, label="other months")
    ax.bar(x[ref], ext[ref], width=0.9, color="#1a1a1a", lw=0, label="refunding months (Feb, May, Aug, Nov)")
    ax.axhline(0, color=MUTED, lw=0.6)
    years = np.array([m.year for m in months])
    ticks = [i for i in range(len(months)) if months[i].month == 1 and years[i] % 5 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(years[i]) for i in ticks], fontsize=8)
    ax.set_ylabel("extension at the rebalance (years)", fontsize=9, color=INK)
    ax.set_title(f"Treasury index extension by month, rebuilt from auction records ({sample})", fontsize=10,
                 loc="left", color=INK)
    ax.set_ylim(top=float(np.nanmax(ext)) * 1.3)                   # headroom so the legend sits clear of the bars
    _axes_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return (f"Mean extension is {ext[ref].mean():.3f} years in refunding months and {ext[~ref].mean():.3f} in "
            f"the other months ({int(ref.sum())} and {int((~ref).sum())} months, {sample}).")


def curve_map(xd, yd, h2_res: dict, path: Path, sample: str, n_bins: int = 10) -> str:
    """Figure 4: H2 as a binned scatter. x = a bucket's standardized predicted demand minus the month's mean across
    buckets, y = its -dy (bp) minus the month's mean; equal-count bins, mean and 95% interval, the H2 slope."""
    xd, yd = np.asarray(xd, float), np.asarray(yd, float)
    order = np.argsort(xd, kind="mergesort")
    bins = np.array_split(order, n_bins)
    bx = np.array([xd[b].mean() for b in bins])
    by = np.array([yd[b].mean() for b in bins])
    be = np.array([1.959963984540054 * yd[b].std(ddof=1) / np.sqrt(len(b)) for b in bins])
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.errorbar(bx, by, yerr=be, fmt="o", color=INK, ms=5, elinewidth=1.0, capsize=3, label="bin mean, 95% CI")
    xs = np.linspace(bx.min(), bx.max(), 50)
    ax.plot(xs, h2_res["coef"] * xs, color=MUTED, ls="--", lw=1.5, label=f"H2 slope {h2_res['coef']:+.2f} bp per sd")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("bucket's predicted demand (z) minus the month's mean", fontsize=9, color=INK)
    ax.set_ylabel("bucket's yield fall over T-4 to T (bp),\nminus the month's mean", fontsize=9, color=INK)
    ax.set_title(f"Forced demand across the curve, 5 maturity buckets\n({sample})", fontsize=10, loc="left",
                 color=INK)
    _axes_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    lo, hi = h2_res["ci"]
    return (f"Within a month, a bucket with 1 sd more predicted demand sees its yield fall {h2_res['coef']:+.2f}bp "
            f"more over T-4 to T (95% CI {lo:+.2f} to {hi:+.2f}, {h2_res['n_obs']} bucket-months, month fixed "
            f"effects, {sample}).")


EQ_STYLE = [{"color": "#000000", "ls": "-"}, {"color": "#6b6b6b", "ls": "--"}, {"color": "#000000", "ls": ":"},
            {"color": "#9e9e9e", "ls": "-."}]


def _equity_panel(ax, navs: dict, sharpes: dict, ticks: list[float]) -> None:
    for (name, nav), st in zip(navs.items(), EQ_STYLE):
        ax.plot(nav.index, nav.to_numpy(), color=st["color"], ls=st["ls"], lw=1.4,
                label=f"{name} (net Sharpe {sharpes[name]:.2f})")
    ends = sorted((float(nav.iloc[-1]), nav.index[-1]) for nav in navs.values())
    y_prev = 0.0
    for v, d in ends:                                              # end labels, nudged apart (log scale)
        y = max(v, y_prev * 1.05)
        ax.text(d, y, f" {v:.2f}", fontsize=7, color=INK, va="center")
        y_prev = y
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(1.0, color=MUTED, lw=0.6)
    ax.set_ylabel("growth of 1 (excess of T-bill, log scale)", fontsize=9, color=INK)
    _axes_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def equity_curve(navs: dict, sharpes: dict, path: Path, sample: str, fut_navs: dict | None = None,
                 fut_sharpes: dict | None = None, fut_sample: str | None = None) -> str:
    """Figure 5: excess-return NAV (strategy P&L only, log scale) of each strategy, in-sample, net of costs. With
    fut_navs, a second panel shows the futures legs and, for comparison, the cash legs over the same days (each
    rebased to 1 at the start of the futures sample)."""
    if fut_navs is None:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.2), gridspec_kw={"height_ratios": [1.1, 1.0]})
    _equity_panel(axes[0], navs, sharpes, [1.0, 1.5, 2.0, 3.0, 4.0])
    axes[0].set_title(f"Equity curves, cash, net of costs ({sample}; test window not yet run)", fontsize=10,
                      loc="left", color=INK)
    if fut_navs is not None:
        _equity_panel(axes[1], fut_navs, fut_sharpes, [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        axes[1].set_title(f"Futures legs and the cash legs over the same days, net of costs ({fut_sample})",
                          fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    parts = [f"{k} {v.iloc[-1]:.2f}x (Sharpe {sharpes[k]:.2f})" for k, v in navs.items()]
    cap = f"Growth of 1 in excess of T-bills, net of costs: {'; '.join(parts)} ({sample})."
    if fut_navs is not None:
        fp = [f"{k} {v.iloc[-1]:.2f}x (Sharpe {fut_sharpes[k]:.2f})" for k, v in fut_navs.items()]
        cap += f" Futures and same-day cash ({fut_sample}): {'; '.join(fp)}."
    return cap


def equity_curve_oos(navs: dict, sharpes: dict, path: Path, sample: str, fut_navs: dict | None = None,
                     fut_sharpes: dict | None = None) -> str:
    """Test-window equity curves (CLAUDE.md section 17): excess-return NAV of each strategy from 1 at the start of
    the window, net of costs; with fut_navs, a second panel with the futures legs."""
    ticks = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4]
    n = 1 if not fut_navs else 2
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 3.8 * n), squeeze=False)
    _equity_panel(axes[0][0], navs, sharpes, ticks)
    axes[0][0].set_title(f"Test window, cash, net of costs ({sample}; run once at Gate 2)", fontsize=10, loc="left",
                         color=INK)
    if fut_navs:
        _equity_panel(axes[1][0], fut_navs, fut_sharpes, ticks)
        axes[1][0].set_title(f"Test window, futures legs and the cash legs, net of costs ({sample})", fontsize=10,
                             loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    parts = [f"{k} {v.iloc[-1]:.2f}x (Sharpe {sharpes[k]:.2f})" for k, v in navs.items()]
    cap = f"Test window, growth of 1 in excess of T-bills, net of costs: {'; '.join(parts)} ({sample})."
    if fut_navs:
        cap += " Futures: " + "; ".join(f"{k} {v.iloc[-1]:.2f}x (Sharpe {fut_sharpes[k]:.2f})"
                                       for k, v in fut_navs.items()) + "."
    return cap


def capacity(curves: dict, path: Path, sample: str) -> str:
    """Figure 6: net Sharpe against capital (log scale) with square-root impact and the 5%-of-ADV cap."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (name, c), st in zip(curves.items(), EQ_STYLE):
        g = c["grid"]
        ax.plot(g["capital"], g["sharpe"], color=st["color"], ls=st["ls"], lw=1.6, marker="o", ms=3, label=name)
        ax.axhline(c["sharpe_at_10m"] / 2.0, color=st["color"], ls=st["ls"], lw=0.6, alpha=0.6)
        if c["capital_where_sharpe_halves"]:
            ax.axvline(c["capital_where_sharpe_halves"], color=st["color"], ls=st["ls"], lw=0.6, alpha=0.6)
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("capital ($, log scale)", fontsize=9, color=INK)
    ax.set_ylabel("net Sharpe after impact", fontsize=9, color=INK)
    ax.set_title(f"Capacity of the month-end trade, cash 10-year ({sample})", fontsize=10, loc="left", color=INK)
    _axes_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    parts = []
    for name, c in curves.items():
        k = c["capital_where_sharpe_halves"]
        parts.append(f"{name}: net Sharpe {c['sharpe_at_10m']:.2f} at $10M, halves at "
                     + (f"${k / 1e9:.2f}bn" if k else "more than $10bn"))
    return "; ".join(parts) + f" (square-root impact on dealer volume, 5% of ADV cap, {sample})."
