"""runs/trials.csv trial logging and Gate 1 / Gate 2 guards (CLAUDE.md 7.14).

Phase 1 implements the guards only (returns.py needs them); log_trial() arrives with the first in-sample run.
Guards act only when GQH_DEV=1 (team .env); judges' runs are never blocked (CLAUDE.md section 5).
"""
from __future__ import annotations

import os
import subprocess

import pandas as pd

from config.settings import IS_END
from src.data.snapshot import REPO_ROOT

GATE1_TAG = "gate1-prereg"
GATE2_TAG = "gate2-frozen"
PREREG_FILES = ["HYPOTHESIS.md", "config/settings.py"]


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


def assert_gate2() -> None:
    """Rule 2: dates after IS_END only when HEAD carries gate2-frozen and the working tree is clean."""
    if not dev_mode():
        return
    if GATE2_TAG not in _head_tags():
        raise GateError(f"Gate 2: dates after IS_END={IS_END} need HEAD tagged {GATE2_TAG!r}.")
    if _dirty():
        raise GateError("Gate 2: working tree is dirty; commit or stash before touching the test window.")


def guard_end(end) -> None:
    """Call before producing any return or statistic: end=None (no cut-off) or end > IS_END needs Gate 2."""
    if end is None or pd.Timestamp(end) > pd.Timestamp(IS_END):
        assert_gate2()
