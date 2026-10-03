"""Writes outputs/results.json and tables (CLAUDE.md section 8). Every number the note quotes comes from here.

results.json keeps the section-8 layout. Blocks a later phase fills stay present and empty, and meta.pending lists
them. Floats are rounded to 6 significant digits so a fresh clone on another OS reproduces the file exactly
(timestamps aside); NaN becomes null.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import IS_END
from src.data.snapshot import REPO_ROOT, utc_now

OUTPUTS = REPO_ROOT / "outputs"
RESULTS_JSON = OUTPUTS / "results.json"
SETTINGS_PY = REPO_ROOT / "config" / "settings.py"
SIG_DIGITS = 6


def settings_hash(path: Path = SETTINGS_PY) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def settings_dict() -> dict:
    """Every pre-registered parameter in config/settings.py (upper-case names)."""
    import config.settings as s
    return {k: getattr(s, k) for k in dir(s) if k.isupper()}


def skeleton() -> dict:
    """The results.json layout of CLAUDE.md section 8, plus H1_addendum (PREREG_ADDENDUM.md), flowclock
    (PREREG_FLOWCLOCK.md, CLAUDE.md section 13) and figures."""
    empty_h1_add = {"refunding_dummy": {}, "within_refunding": {}, "within_other": {}, "surprise": {}}
    return {
        "meta": {"commit": "", "dirty": None, "generated_utc": "", "snapshot_checksums_ok": None,
                 "settings_hash": "", "version": "", "pending": []},
        "validation": {"par_gap_by_decade": {}, "duration_recent": {}, "extension_stats": {}, "skipped_tranches": 0},
        "in_sample": {
            "H1": {"b": None, "ci": [None, None], "t": None, "n": 0, "terciles": {}, "components": {"ext": {},
                                                                                                    "cash": {}}},
            "H1_addendum": dict(empty_h1_add),
            "tips_replication": {},
            "H2": {"coef": None, "ci": [None, None], "n_obs": 0},
            "H3": {"coef": None, "ci": [None, None]},
            "H4": {"sharpe_fc": None, "sharpe_cal": None, "diff": None, "ci": [None, None]},
            "H5": {}, "placebo": {}, "random_windows": {},
            "metrics": {"cash": {"forecast_sized": {}, "calendar_only": {}, "curve_allocated": {}}, "futures": {}},
            "cost_stress": {}, "risk_rules_on_off": {}, "sensitivity": {}, "betas": {}, "crowding": {}, "tails": [],
            "capacity": {}, "deflated_sharpe": {},
        },
        "post_publication": {"H1": {}, "H1_addendum": dict(empty_h1_add), "H4": {}, "metrics": {}},
        "oos": {"ran_utc": None, "commit": None, "H1": {}, "H1_addendum": dict(empty_h1_add), "H4": {}, "metrics": {}},
        "flowclock": {},
        "figures": {},
        "trials": {"count": 0},
    }


def clean(obj):
    """JSON-safe copy: numpy scalars to Python, floats to SIG_DIGITS significant digits, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return 0.0 if x == 0 else float(f"{x:.{SIG_DIGITS}g}")
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj.date()) if isinstance(obj, pd.Timestamp) else str(obj)
    return obj


def validation_block(monthly: pd.DataFrame) -> dict:
    """The STOP 2 validation numbers the note quotes (CLAUDE.md 7.6), recomputed from the snapshot. No returns."""
    from src import validate as v
    from src.calendar import load_calendar
    from src.data.auctions import load_auctions
    from src.data.mspd import load_mspd
    all_auctions = load_auctions(end=IS_END, deduct_soma=False, exclude={"TIPS": True, "FRN": True,
                                                                         "callable": False})
    par = v.par_vs_mspd(all_auctions, load_mspd(end=IS_END), load_calendar(end=IS_END).days)
    dur = v.duration_table(monthly).set_index("year")[["last_month", "D_next_last", "D_now_mean"]]
    stats = v.extension_stats(monthly)
    return {"par_gap_by_decade": v.par_gap_by_decade(par),
            "duration_recent": {str(y): r.to_dict() for y, r in dur.iterrows()},
            "extension_stats": {k: r.to_dict() for k, r in stats.iterrows()},
            "seasonality_mean_by_calendar_month": {str(k): r.to_dict() for k, r in v.seasonality(monthly).iterrows()},
            "skipped_tranches": int(v.in_sample(monthly)["n_skipped"].sum())}


def cost_block(yields: pd.DataFrame) -> dict:
    """Cost assumptions with the published reference (Phase 4b, CLAUDE.md section 15; config/costs.py).

    The cash cost (CASH_COST_BP per round trip) is our assumption. Fleming (2003)'s mean interdealer bid-ask spreads
    of the on-the-run notes (32nds of a point) are converted to yield bp with the modified duration of a par bond of
    that maturity at the tenor's mean CMT yield over the paper's sample (src/bonds.py); a round trip crosses one full
    spread, so the ratio our cost / spread compares like with like. yields: the reference tenors' CMT yields (%)."""
    from config.costs import FLEMING_2003 as ref
    from config.settings import CASH_COST_BP, COST_STRESS, FUT_COMMISSION_RT
    from src.bonds import KNOT_YEARS, mod_duration
    lo, hi = (pd.Timestamp(d) for d in ref["sample"])
    by = {}
    for t, s32 in ref["spread_32nds"].items():
        y = float(yields.loc[lo:hi, t].mean())
        d = float(mod_duration(y, y, KNOT_YEARS[t]))
        pts = s32 / 32.0
        bp = pts / (d * 0.01)                         # 1 bp of yield moves a par bond's price by D x 0.01 points
        by[t] = {"spread_32nds": s32, "spread_points": pts, "mean_cmt_yield_pct": y, "par_mod_duration": d,
                 "spread_bp_yield": bp, "our_cost_over_spread": CASH_COST_BP / bp}
    return {
        "cash": {"bp_yield_per_round_trip": CASH_COST_BP, "stress_bp": CASH_COST_BP * COST_STRESS,
                 "status": "0.5 bp of yield per round trip is our assumption (config/settings.py, gate1-prereg); no "
                           "source fixes it",
                 "reference": {"citation": ref["citation"], "url": ref["url"], "sample": ref["sample"],
                               "data": ref["data"], "by_tenor": by,
                               "note": "interdealer spreads are the narrowest in the market; customer trades, "
                                       "off-the-run issues, 1993-1996 and stressed days cost more"}},
        "futures": {"per_contract_round_trip": f"1 tick + ${FUT_COMMISSION_RT:g}",
                    "status": "team decision (CLAUDE.md section 15); tick value from the exchange definitions",
                    "stress_mult": COST_STRESS},
    }


def write_results(results: dict, path: Path = RESULTS_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean(results)
    out["meta"]["generated_utc"] = utc_now()
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path


def write_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = OUTPUTS / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format="%.8g", lineterminator="\n")
    return path
