"""Write and verify the committed data snapshot, CHECKSUMS.sha256 and VINTAGE.md (CLAUDE.md 7.1).

Layout (repo-relative paths, so `shasum -a 256 -c data/snapshot/CHECKSUMS.sha256` works from the repo root):
  data/snapshot/*.csv            public U.S.-government data as published, plus derived pension_pressure.csv
  config/fomc_dates.csv          FOMC decision dates (scraped by scripts/fetch_fomc.py), also checksummed
  data/snapshot/vintage.json     machine-readable download log (source, URL, UTC timestamp, rows, date range)
  data/snapshot/VINTAGE.md       human-readable rendering of vintage.json plus per-source notes
"""
from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshot"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CHECKSUMS = SNAPSHOT_DIR / "CHECKSUMS.sha256"
VINTAGE_JSON = SNAPSHOT_DIR / "vintage.json"
VINTAGE_MD = SNAPSHOT_DIR / "VINTAGE.md"
FOMC_CSV = REPO_ROOT / "config" / "fomc_dates.csv"

_READING: list[Path] = []


def active_dir() -> Path:
    """The directory the data loaders read: the committed snapshot, or inside reading_from() another directory with
    the same file layout (the --oos data view, CLAUDE.md section 17)."""
    return _READING[-1] if _READING else SNAPSHOT_DIR


@contextmanager
def reading_from(directory: Path):
    """Point every snapshot loader (src/data/*.py) at `directory` for the duration of the block."""
    _READING.append(Path(directory))
    try:
        yield
    finally:
        _READING.pop()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_text(text: str, name: str, directory: Path = SNAPSHOT_DIR) -> Path:
    """Write text with LF line endings (checksums must not depend on the OS)."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace("\r\n", "\n"))
    return path


def write_csv(df: pd.DataFrame, name: str, directory: Path = SNAPSHOT_DIR) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def snapshot_files(snapshot_dir: Path = SNAPSHOT_DIR, extra: tuple[Path, ...] = (FOMC_CSV,)) -> list[Path]:
    """Every file the checksums cover: snapshot CSVs plus config/fomc_dates.csv when present."""
    files = sorted(snapshot_dir.glob("*.csv"))
    files += [p for p in extra if p.exists()]
    return files


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_checksums(files: list[Path] | None = None, path: Path = CHECKSUMS, root: Path = REPO_ROOT) -> Path:
    files = snapshot_files() if files is None else files
    lines = [f"{sha256(p)}  {_rel(p, root)}" for p in files]
    return write_text("\n".join(lines) + "\n", path.name, path.parent)


def verify_checksums(path: Path = CHECKSUMS, root: Path = REPO_ROOT,
                     snapshot_dir: Path | None = None) -> list[str]:
    """Return a list of problems (empty list = snapshot intact)."""
    if not path.exists():
        return [f"missing checksum file {path}"]
    problems, listed = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        listed.add(rel)
        target = root / rel
        if not target.exists():
            problems.append(f"missing file {rel}")
        elif sha256(target) != digest:
            problems.append(f"checksum mismatch {rel}")
    snapshot_dir = path.parent if snapshot_dir is None else snapshot_dir
    for p in sorted(snapshot_dir.glob("*.csv")):
        if _rel(p, root) not in listed:
            problems.append(f"file not in checksums {_rel(p, root)}")
    return problems


def verify_or_exit(path: Path = CHECKSUMS) -> None:
    """Stop with a clear message if the committed snapshot was altered (CLAUDE.md 7.1, rule 4)."""
    problems = verify_checksums(path)
    if problems:
        msg = "\n  ".join(problems)
        sys.exit(f"Snapshot checksum check FAILED:\n  {msg}\n"
                 "The committed public-data snapshot differs from data/snapshot/CHECKSUMS.sha256. "
                 "Restore it with `git checkout -- data/snapshot config/fomc_dates.csv`, or refresh it "
                 "deliberately with `python scripts/download_all.py` and commit the new checksums.")


# ----------------------------------------------------------------------------------------------- vintage log

def read_vintage(path: Path = VINTAGE_JSON) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "notes": {}}


def record_vintage(entries: dict[str, dict], notes: dict[str, str] | None = None,
                   path: Path = VINTAGE_JSON) -> dict:
    """Merge download records ({file: {source, url, downloaded_utc, rows, first, last}}) and notes, then re-render."""
    v = read_vintage(path)
    v["files"].update(entries)
    v["notes"].update(notes or {})
    write_text(json.dumps(v, indent=2, sort_keys=True) + "\n", path.name, path.parent)
    render_vintage_md(v, path.parent / VINTAGE_MD.name)
    return v


def render_vintage_md(v: dict, path: Path = VINTAGE_MD) -> Path:
    out = ["# Data vintage", "",
           "Generated by `src/data/snapshot.py` from `vintage.json`. Every CSV listed here is covered by",
           "`CHECKSUMS.sha256` (verify from the repo root: `shasum -a 256 -c data/snapshot/CHECKSUMS.sha256`).", "",
           "| file | source | downloaded (UTC) | rows | first | last |", "|---|---|---|---|---|---|"]
    for name in sorted(v["files"]):
        e = v["files"][name]
        out.append(f"| `{name}` | [{e.get('source', '')}]({e.get('url', '')}) | {e.get('downloaded_utc', '')} | "
                   f"{e.get('rows', '')} | {e.get('first', '')} | {e.get('last', '')} |")
    if v.get("notes"):
        out += ["", "## Notes", ""]
        for key in sorted(v["notes"]):
            out += [f"### {key}", "", v["notes"][key].strip(), ""]
    return write_text("\n".join(out).rstrip() + "\n", path.name, path.parent)
