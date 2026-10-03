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
            "risk_rules_on_off": {}, "betas": {}, "crowding": {}, "tails": [], "capacity": {}, "deflated_sharpe": {},
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
