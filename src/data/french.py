"""Ken French daily factors loader; derives pension_pressure.csv (CLAUDE.md 7.1).

Live file inspected 2026-10-03 (rule 8): the zip holds one member, `F-F_Research_Data_Factors_daily.csv`
(latin-1 text). Layout: three description lines ("This file was created by using the 202608 CRSP database." ...),
a blank line, the header `,Mkt-RF,SMB,HML,RF`, rows `YYYYMMDD,    0.09,   -0.23,   -0.28,    0.01` in percent
(1926-07-01 to 2026-08-31 in that file), a blank line and a copyright line.

What is committed: the raw zip stays in data/cache/ (git-ignored, CLAUDE.md rule 6). The snapshot file
`pension_pressure.csv` holds the one derived input the pension signal needs (CLAUDE.md 7.7): the daily equity total
return `mkt_total_pct = Mkt-RF + RF` (percent), from 1990-01-01. The monthly pension-pressure signal itself also
needs the 10-year cash-bond return, so it is computed in src/signals.py (Phase 3) behind the gate guards.
The file ends at the last CRSP update (2026-08-31 in this vintage), so test-window month 2026-09 has no input.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config.settings import IS_END
from src.data.snapshot import CACHE_DIR, active_dir

FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "F-F_Research_Data_Factors_daily_CSV.zip")
RAW_ZIP = CACHE_DIR / "F-F_Research_Data_Factors_daily_CSV.zip"
SNAPSHOT_NAME = "pension_pressure.csv"
HEADER = ["", "Mkt-RF", "SMB", "HML", "RF"]
DERIVED_START = "1990-01-01"


def download_raw(session: requests.Session | None = None, path: Path = RAW_ZIP) -> Path:
    s = session or requests.Session()
    r = s.get(FRENCH_URL, timeout=120)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def read_zip_text(path: Path = RAW_ZIP) -> str:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError(f"Ken French zip: expected one member, found {names}")
        return z.read(names[0]).decode("latin-1")


def parse_daily(text: str) -> pd.DataFrame:
    """Daily factor table (percent) indexed by date; skips the description header and copyright footer."""
    lines = text.splitlines()
    try:
        h = next(i for i, l in enumerate(lines) if [c.strip() for c in l.split(",")] == HEADER)
    except StopIteration:
        raise ValueError("Ken French file: header ',Mkt-RF,SMB,HML,RF' not found; re-inspect the file (rule 8)")
    body = []
    for l in lines[h + 1:]:
        first = l.split(",", 1)[0].strip()
        if not (len(first) == 8 and first.isdigit()):
            break  # blank line before the copyright footer (or the start of an annual table)
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join(body)), header=None, names=["date", "Mkt-RF", "SMB", "HML", "RF"],
                     dtype={"date": str}, skipinitialspace=True)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), format="%Y%m%d"), name="date")
    return df.astype(float)


def derive_pension_input(factors: pd.DataFrame, start: str = DERIVED_START) -> pd.DataFrame:
    """date, mkt_total_pct (= Mkt-RF + RF, percent), the only French quantity used downstream (CLAUDE.md 7.7)."""
    f = factors.loc[start:]
    out = pd.DataFrame({"date": f.index.strftime("%Y-%m-%d"), "mkt_total_pct": (f["Mkt-RF"] + f["RF"]).round(4)})
    return out.reset_index(drop=True)


def load_pension_input(end: str | None = IS_END, path: Path | None = None) -> pd.Series:
    """Daily equity total return in percent from the snapshot, dates <= end."""
    path = path or active_dir() / SNAPSHOT_NAME
    df = pd.read_csv(path)
    s = pd.Series(df["mkt_total_pct"].to_numpy(float),
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"], format="%Y-%m-%d"), name="date"),
                  name="mkt_total_pct")
    return s if end is None else s.loc[: pd.Timestamp(end)]
