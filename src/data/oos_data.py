"""Public test-window data for `python run_all.py --oos` (CLAUDE.md section 17).

Rows dated up to the split (IS_END) come from the committed snapshot (data/snapshot/). Rows dated after it, through
the window end (OOS_END), come only from data that --oos downloads after the Gate 2 guard. The committed snapshot
also holds rows after IS_END (the Oct 3, 2026 download); they are never read here. The two parts are joined into a
data view with the snapshot's file names and columns, which every loader reads through
src/data/snapshot.py::reading_from, so the in-sample code runs unchanged on the view.

Files and the date column that splits them:
  fred_<SERIES>.csv          observation_date   CURVE_KNOTS + DTB3 (src/data/fred.py)
  auctions.csv               auction_date       Fiscal Data auctions_query (src/data/auctions.py)
  soma_asof_dates.csv        asOfDate           NY Fed SOMA as-of dates (src/data/soma.py)
  soma_notesbonds.csv        asOfDate           NY Fed SOMA holdings, only the as-of dates the rebuild needs
  pd_treasury_positions.csv  asofdate           NY Fed primary dealer positions, H8 (src/data/pd_positions.py)
  pension_pressure.csv       date               Ken French daily factors, derived (src/data/french.py)
Not downloaded (no test or strategy uses them in the window): mspd_notes_bonds.csv, pd_treasury_volume.csv. FOMC dates
stay the committed config/fomc_dates.csv.

Every value is kept as the published string (read with dtype=str), so a row of the view parses exactly as the same
row of the snapshot. The download keeps only rows dated in (split, end]; the Ken French raw zip stays in the
git-ignored cache (CLAUDE.md rule 6).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import requests

from config.settings import IS_END, OOS_END
from src.data import auctions, french, fred, pd_positions, soma
from src.data.snapshot import (CACHE_DIR, REPO_ROOT, SNAPSHOT_DIR, utc_now, verify_checksums, write_checksums,
                               write_csv, write_text)

OOS_DATA_DIR = REPO_ROOT / "data" / "oos"            # committed after the Gate 2 run: the downloaded rows
STAGE_DIR = CACHE_DIR / "oos" / "download"           # git-ignored staging during the run
VIEW_DIR = CACHE_DIR / "oos" / "view"                # git-ignored: snapshot rows <= split + downloaded rows
CHECKSUMS_NAME = "CHECKSUMS.sha256"
VINTAGE_NAME = "vintage.json"
COMPLETE = ".complete"

DATE_COL = {**{fred.snapshot_name(s): "observation_date" for s in fred.SERIES},
            auctions.SNAPSHOT_NAME: "auction_date", soma.ASOF_NAME: "asOfDate", soma.SNAPSHOT_NAME: "asOfDate",
            pd_positions.SNAPSHOT_NAME: "asofdate", french.SNAPSHOT_NAME: "date"}


def read_str(path: Path) -> pd.DataFrame:
    """A snapshot CSV with every value as the published string ("" for blanks)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def rows_between(df: pd.DataFrame, col: str, after, through) -> pd.DataFrame:
    """Rows with after < df[col] <= through (dates as YYYY-MM-DD strings; after=None means no lower bound)."""
    d = pd.to_datetime(df[col], format="%Y-%m-%d")
    keep = d <= pd.Timestamp(through)
    if after is not None:
        keep &= d > pd.Timestamp(after)
    return df[keep]


def splice(base: pd.DataFrame, new: pd.DataFrame, col: str, split, end) -> pd.DataFrame:
    """Rows of `base` dated <= split, then rows of `new` dated in (split, end]. The columns must match."""
    if list(base.columns) != list(new.columns):
        raise ValueError(f"columns differ: {list(base.columns)} vs {list(new.columns)}; re-inspect the source (rule 8)")
    return pd.concat([rows_between(base, col, None, split), rows_between(new, col, split, end)], ignore_index=True)


def build_view(new_dir: Path, split: str = IS_END, end: str = OOS_END, base_dir: Path = SNAPSHOT_DIR,
               view_dir: Path = VIEW_DIR) -> dict:
    """Write the data view: for every file in DATE_COL, base rows dated <= split + new rows dated in (split, end].
    Returns per file the row counts from each part and the first and last new date."""
    view_dir.mkdir(parents=True, exist_ok=True)
    for old in view_dir.glob("*.csv"):
        old.unlink()
    out = {}
    for name, col in DATE_COL.items():
        base, new = read_str(base_dir / name), read_str(new_dir / name)
        v = splice(base, new, col, split, end)
        write_csv(v, name, view_dir)
        n = rows_between(new, col, split, end)[col]
        out[name] = {"rows_through_split": int(len(v) - len(n)), "rows_after_split": int(len(n)),
                     "first_after_split": n.min() if len(n) else None, "last_after_split": n.max() if len(n) else None}
    return out


# ------------------------------------------------------------------------------------------------ download

def _entry(source: str, url: str, when: str, df: pd.DataFrame, col: str) -> dict:
    return {"source": source, "url": url, "downloaded_utc": when, "rows": int(len(df)),
            "first": df[col].min() if len(df) else None, "last": df[col].max() if len(df) else None}


def download(dest: Path = STAGE_DIR, split: str = IS_END, end: str = OOS_END,
             session: requests.Session | None = None, raw_zip: Path | None = None) -> dict:
    """Download every public source in DATE_COL and keep the rows dated in (split, end] (module docstring).

    Called by run_all.py --oos only after the Gate 2 guard. FRED goes first: the SOMA holdings needed are the as-of
    dates usable at T-0..T-6 of the window's months (src/data/soma.py), on the bond calendar of the view's FRED rows.
    Writes dest/*.csv, dest/vintage.json and a completion marker; returns the vintage entries. raw_zip: where the Ken
    French zip goes (default data/cache/oos/). A download that
    reaches past IS_END needs HEAD tagged gate2-frozen and a clean tree in every mode
    (src/trial_log.py::assert_gate2_download); before any request."""
    from src.calendar import BondCalendar
    from src.trial_log import assert_gate2_download
    if pd.Timestamp(end) > pd.Timestamp(IS_END):
        assert_gate2_download()
    s = session or requests.Session()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / COMPLETE).unlink(missing_ok=True)
    entries = {}
    for sid in fred.SERIES:
        when = utc_now()
        text, _ = fred.fetch_series(sid, s)
        name = fred.snapshot_name(sid)
        df = rows_between(read_str_text(text), "observation_date", split, end)
        write_csv(df, name, dest)
        entries[name] = _entry(f"FRED {sid}", f"{fred.FRED_CSV_URL}?id={sid}", when, df, "observation_date")
    when = utc_now()
    df = rows_between(auctions.fetch_auctions(s), "auction_date", split, end)
    write_csv(df, auctions.SNAPSHOT_NAME, dest)
    entries[auctions.SNAPSHOT_NAME] = _entry("Fiscal Data auctions_query", auctions.AUCTIONS_URL, when, df,
                                             "auction_date")
    when = utc_now()
    asof_new = rows_between(soma.fetch_asof_dates(s), "asOfDate", split, end)
    write_csv(asof_new, soma.ASOF_NAME, dest)
    entries[soma.ASOF_NAME] = _entry("NY Fed Markets Data API, SOMA as-of dates", soma.ASOF_LIST_URL, when, asof_new,
                                     "asOfDate")
    dgs10 = splice(read_str(SNAPSHOT_DIR / fred.snapshot_name("DGS10")), read_str(dest / fred.snapshot_name("DGS10")),
                   "observation_date", split, end)
    cal = BondCalendar.from_dgs10(fred.parse_fred_csv(dgs10.to_csv(index=False, lineterminator="\n"), "DGS10"))
    asof_all = pd.concat([rows_between(read_str(SNAPSHOT_DIR / soma.ASOF_NAME), "asOfDate", None, split), asof_new])
    asof_idx = pd.DatetimeIndex(pd.to_datetime(asof_all["asOfDate"], format="%Y-%m-%d"))
    first_month = (pd.Timestamp(split) + pd.Timedelta(days=1)).strftime("%Y-%m")
    need = soma.needed_asof_dates(asof_idx, cal, first_month=first_month, last_month=pd.Timestamp(end).strftime("%Y-%m"))
    need = need[(need > pd.Timestamp(split)) & (need <= pd.Timestamp(end))]
    frames = [soma.fetch_holdings(d.date().isoformat(), s) for d in need]
    h = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=soma.FIELDS)) \
        .sort_values(["asOfDate", "cusip"], kind="mergesort").reset_index(drop=True)
    write_csv(h, soma.SNAPSHOT_NAME, dest)
    entries[soma.SNAPSHOT_NAME] = _entry("NY Fed Markets Data API, SOMA notes and bonds by CUSIP",
                                         soma.HOLDINGS_URL.format(date="{date}"), when, h, "asOfDate")
    when = utc_now()
    df = rows_between(pd_positions.download(s), "asofdate", split, end)
    write_csv(df, pd_positions.SNAPSHOT_NAME, dest)
    entries[pd_positions.SNAPSHOT_NAME] = _entry("NY Fed Markets Data API, primary dealer positions",
                                                 pd_positions.SERIES_URL.format(keyid="{keyid}"), when, df, "asofdate")
    when = utc_now()
    raw_zip = raw_zip or CACHE_DIR / "oos" / french.RAW_ZIP.name       # git-ignored (rule 6)
    french.download_raw(s, path=raw_zip)
    p = french.derive_pension_input(french.parse_daily(french.read_zip_text(raw_zip)))
    df = rows_between(p.astype(str), "date", split, end)
    write_csv(df, french.SNAPSHOT_NAME, dest)
    entries[french.SNAPSHOT_NAME] = _entry("Ken French daily factors (derived)", french.FRENCH_URL, when, df, "date")
    vint = {"split": split, "end": end, "files": entries,
            "note": "rows dated in (split, end] only; rows through the split come from data/snapshot/ "
                    "(CLAUDE.md section 17)"}
    write_text(json.dumps(vint, indent=2, sort_keys=True) + "\n", VINTAGE_NAME, dest)
    (dest / COMPLETE).write_text(utc_now() + "\n", encoding="utf-8")
    return entries


def read_str_text(text: str) -> pd.DataFrame:
    """A FRED CSV body as published strings (the header is checked by fred.parse_fred_csv on load)."""
    import io
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def staged() -> bool:
    return (STAGE_DIR / COMPLETE).exists()


def committed_ok() -> bool:
    """data/oos/ exists and matches its checksums."""
    path = OOS_DATA_DIR / CHECKSUMS_NAME
    return path.exists() and not verify_checksums(path, REPO_ROOT, OOS_DATA_DIR)


def commit_copy(src: Path = STAGE_DIR, dest: Path = OOS_DATA_DIR) -> list[Path]:
    """Copy the downloaded rows and their vintage into data/oos/ and write its checksums (after the run)."""
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for name in DATE_COL:
        shutil.copyfile(src / name, dest / name)
        files.append(dest / name)
    shutil.copyfile(src / VINTAGE_NAME, dest / VINTAGE_NAME)
    write_checksums(files, dest / CHECKSUMS_NAME, REPO_ROOT)
    return files
