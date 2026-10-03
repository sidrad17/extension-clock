"""Note figures (CLAUDE.md section 8): 300 dpi, readable in greyscale, captions built from the numbers they show.

Phase 3: 1. event_path.png (page-1 hero exhibit) and 2. terciles.png. The rest arrive in Phase 5.
Flow Clock (PREREG_FLOWCLOCK.md): auction_event_path.png, A-10..A+10 by size-signal tercile.
Each function returns the one-sentence caption that report.py stores in results.json beside the figure path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

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
