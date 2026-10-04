"""runs/trials.csv trial logging and Gate 1 / Gate 2 guards (CLAUDE.md 7.14).

Guards act only when GQH_DEV=1 (team .env); judges' runs are never blocked (CLAUDE.md section 5).
log_trial() / log_trials() append one row per tested configuration to runs/trials.csv on every in-sample run made
with GQH_DEV=1 (rule 7; team decision of Oct 3, 2026, Phase 5). Without GQH_DEV the log is left as committed, so a
judge's run reproduces results.json, whose trials.count and Deflated Sharpe read the committed log (trial_count(),
read_trials()). Rows are never deleted, and reruns of an unchanged configuration are logged again (same
config_hash).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import IS_END
from src.data.snapshot import REPO_ROOT

TRIALS_CSV = REPO_ROOT / "runs" / "trials.csv"
TRIAL_COLUMNS = ["timestamp_utc", "git_commit", "dirty", "config_hash", "window", "strategy", "tenor", "entry", "exit",
                 "n", "H1_b", "H1_lo", "H1_hi", "sharpe_fc", "sharpe_cal", "note"]
GATE1_TAG = "gate1-prereg"
GATE2_TAG = "gate2-frozen"
PREREG_FILES = ["HYPOTHESIS.md", "config/settings.py"]
FLOWCLOCK_TAG = "prereg-flowclock"
FLOWCLOCK_FILES = ["PREREG_FLOWCLOCK.md", "PREREG_ADDENDUM.md", "HYPOTHESIS.md", "config/settings.py"]
DEALERS_TAG = "prereg-dealers"
DEALERS_FILES = ["PREREG_DEALERS.md", *FLOWCLOCK_FILES]


class GateError(RuntimeError):
    """A pre-registration gate (CLAUDE.md rules 1-2) would be broken."""


def dev_mode() -> bool:
    """True when GQH_DEV=1 in the environment or in the repo's .env (environment wins)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # python-dotenv is pinned in requirements.txt; tolerate its absence
        pass
    return os.environ.get("GQH_DEV", "").strip() == "1"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _tags() -> list[str]:
    return _git("tag", "--list").split()


def _head_tags() -> list[str]:
    return _git("tag", "--points-at", "HEAD").split()


def _dirty() -> bool:
    return bool(_git("status", "--porcelain").strip())


def _prereg_changed() -> bool:
    return bool(_git("diff", GATE1_TAG, "--", *PREREG_FILES).strip())


def assert_gate1() -> None:
    """Rule 1: no return analysis before the gate1-prereg tag; pre-registered files unchanged since the tag."""
    if not dev_mode():
        return
    if GATE1_TAG not in _tags():
        raise GateError(f"Gate 1: tag {GATE1_TAG!r} not found; no return analysis before pre-registration.")
    if _prereg_changed():
        raise GateError(f"Gate 1: {', '.join(PREREG_FILES)} differ from {GATE1_TAG}; they are locked.")


def _flowclock_changed() -> bool:
    return bool(_git("diff", FLOWCLOCK_TAG, "--", *FLOWCLOCK_FILES).strip())


def assert_flowclock_prereg() -> None:
    """PREREG_FLOWCLOCK.md: no auction-window return before the prereg-flowclock tag; the pre-registration files
    (that file, PREREG_ADDENDUM.md, HYPOTHESIS.md, config/settings.py) unchanged since the tag."""
    if not dev_mode():
        return
    if FLOWCLOCK_TAG not in _tags():
        raise GateError(f"Flow Clock: tag {FLOWCLOCK_TAG!r} not found; no auction-window return before it.")
    if _flowclock_changed():
        raise GateError(f"Flow Clock: {', '.join(FLOWCLOCK_FILES)} differ from {FLOWCLOCK_TAG}; they are locked.")


def _dealers_changed() -> bool:
    return bool(_git("diff", DEALERS_TAG, "--", *DEALERS_FILES).strip())


def assert_dealers_prereg() -> None:
    """PREREG_DEALERS.md (H8): no dealer-position data downloaded or read before the prereg-dealers tag; the
    pre-registration files (that file and FLOWCLOCK_FILES) unchanged since the tag."""
    if not dev_mode():
        return
    if DEALERS_TAG not in _tags():
        raise GateError(f"H8: tag {DEALERS_TAG!r} not found; no dealer-position data before it.")
    if _dealers_changed():
        raise GateError(f"H8: {', '.join(DEALERS_FILES)} differ from {DEALERS_TAG}; they are locked.")


def assert_gate2() -> None:
    """Rule 2: dates after IS_END only when HEAD carries gate2-frozen and the working tree is clean."""
    if not dev_mode():
        return
    if GATE2_TAG not in _head_tags():
        raise GateError(f"Gate 2: dates after IS_END={IS_END} need HEAD tagged {GATE2_TAG!r}.")
    if _dirty():
        raise GateError("Gate 2: working tree is dirty; commit or stash before touching the test window.")


def assert_gate2_download() -> None:
    """Every test-window download (public data or Databento), in every mode, not only with GQH_DEV=1: HEAD tagged
    gate2-frozen and a clean working tree (tightens rule 2; Oct 4, 2026). The keyless reproduction from the committed
    data/oos/ downloads nothing and needs no tag."""
    try:
        tagged, dirty = GATE2_TAG in _head_tags(), _dirty()
    except (OSError, subprocess.CalledProcessError) as e:
        raise GateError(f"Gate 2: cannot read the git state ({e}); no test-window download outside a git checkout.")
    if not tagged:
        raise GateError(f"Gate 2: a test-window download needs HEAD tagged {GATE2_TAG!r}.")
    if dirty:
        raise GateError("Gate 2: working tree is dirty; a test-window download needs a clean tree.")


def guard_end(end) -> None:
    """Call before producing any return or statistic: end=None (no cut-off) or end > IS_END needs Gate 2."""
    if end is None or pd.Timestamp(end) > pd.Timestamp(IS_END):
        assert_gate2()


# ------------------------------------------------------------------------------------------------- trial log

def git_state() -> dict:
    """Commit (short hash) and dirty flag of the working tree; 'unknown' outside a git checkout."""
    try:
        return {"commit": _git("rev-parse", "--short", "HEAD").strip(), "dirty": _dirty()}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": None}


def config_hash(cfg: dict) -> str:
    """Short sha256 of a JSON-serialisable configuration (sorted keys)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def trial_count(path=TRIALS_CSV) -> int:
    if not path.exists():
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def trial_row(cfg: dict, window: str, results: dict, git: dict | None = None) -> dict:
    """One runs/trials.csv row (CLAUDE.md 7.14). `results` supplies strategy, tenor, entry, exit, n, H1_b, H1_lo,
    H1_hi, sharpe_fc, sharpe_cal and note; `git` is the state captured at the start of the run.

    Flow Clock rows (window "*_flowclock*") keep the same columns: H1_b/lo/hi hold that row's pre-registered slope
    (H7 c for the book, H6c beta for the supply leg), sharpe_fc the candidate's net Sharpe (the book; the
    size-weighted supply leg) and sharpe_cal its benchmark's (the demand leg alone; the calendar supply leg). The
    note says which. Phase 5 rows say in `note` what H1_* and the two Sharpe columns hold."""
    git = git or git_state()
    row = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "git_commit": git["commit"], "dirty": git["dirty"], "config_hash": config_hash(cfg), "window": window}
    for k in TRIAL_COLUMNS[5:]:
        row[k] = results.get(k, "")
    return row


def append_rows(rows: list[dict], path=TRIALS_CSV) -> int:
    """Append rows to runs/trials.csv only when GQH_DEV=1 (module docstring). Returns the number written."""
    if not rows or not dev_mode():
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def log_trial(cfg: dict, window: str, results: dict, git: dict | None = None, path=TRIALS_CSV) -> dict:
    """Build one row (trial_row) and append it when GQH_DEV=1 (append_rows). Returns the row either way."""
    row = trial_row(cfg, window, results, git)
    append_rows([row], path)
    return row


def log_trials(rows: list[dict], path=TRIALS_CSV) -> int:
    """Append many trial_row() rows in one write (the sensitivity grid); only when GQH_DEV=1."""
    return append_rows(rows, path)


def read_trials(path=TRIALS_CSV) -> pd.DataFrame:
    """The committed trial log as a DataFrame (empty if absent); Sharpe columns as floats (blank = NaN)."""
    if not path.exists():
        return pd.DataFrame(columns=TRIAL_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("sharpe_fc", "sharpe_cal"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    return df


OOS_WINDOW_PREFIX = "oos"


def in_sample_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """The log without the Gate 2 rows (window "oos*", CLAUDE.md section 17): the Deflated Sharpe deflates for
    in-sample selection, so total_logged_runs and distinct_variants count in-sample rows only."""
    return trials[~trials["window"].astype(str).str.startswith(OOS_WINDOW_PREFIX)]


def trial_counts(trials: pd.DataFrame) -> dict:
    """Distinct variants tested vs total logged runs (CLAUDE.md section 15, team decision of Oct 3, 2026).

    A variant is a distinct config_hash: identical configurations re-run (and re-logged) count once. Each
    config_hash contributes the Sharpe columns of its latest row (the log is in time order); reruns logged so far
    have identical Sharpes, and n_configs_with_differing_sharpes counts any that do not (to 1e-9). Returns the two
    counts and the trial Sharpes for each: every row (total logged runs) and the latest row per config_hash."""
    cols = ["sharpe_fc", "sharpe_cal"]
    latest = trials.groupby("config_hash", sort=False).tail(1)
    differ = 0
    for _, g in trials.groupby("config_hash", sort=False):
        for c in cols:
            v = g[c].dropna().to_numpy(float)
            if len(v) > 1 and np.ptp(v) > 1e-9:
                differ += 1
                break
    return {"total_logged_runs": int(len(trials)), "distinct_variants": int(trials["config_hash"].nunique()),
            "n_configs_with_differing_sharpes": int(differ),
            "sharpes_all_rows": pd.concat([trials[c] for c in cols]).dropna().to_numpy(float),
            "sharpes_distinct": pd.concat([latest[c] for c in cols]).dropna().to_numpy(float)}
