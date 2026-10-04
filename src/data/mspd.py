"""MSPD marketable notes and bonds outstanding, for rebuild validation only (CLAUDE.md 7.1).

Live response inspected 2026-10-03 (rule 8): `/v1/debt/mspd/mspd_table_1` rows have keys record_date,
security_type_desc, security_class_desc, debt_held_public_mil_amt, intragov_hold_mil_amt, total_mil_amt,
src_line_nbr, record_fiscal_year, record_fiscal_quarter, record_calendar_year, record_calendar_quarter,
record_calendar_month, record_calendar_day (all strings, amounts in $ millions). Coverage: 2001-01-31 to
2026-08-31 at month-ends, 4,659 rows, one page at page[size]=10000. So the par validation can start only in 2001.
Marketable classes: Bills, Notes, Bonds, TIPS ("Inflation-Indexed Notes" / "Inflation-Indexed Bonds" through
2004-05, "Treasury Inflation-Protected Securities" from 2004-06), Floating Rate Notes (from 2014-01), Federal
Financing Bank. "Notes" and "Bonds" exclude TIPS and FRNs throughout, which matches our nominal universe.
MSPD "debt held by the public" INCLUDES Federal Reserve (SOMA) holdings; intragovernmental holdings are
government accounts. The rebuild's par before SOMA deduction compares to total_mil_amt.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from config.settings import IS_END
from src.data.snapshot import active_dir

MSPD_TABLE1_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_1"
SNAPSHOT_NAME = "mspd_notes_bonds.csv"
KEEP_COLS = ["record_date", "security_type_desc", "security_class_desc", "debt_held_public_mil_amt",
             "intragov_hold_mil_amt", "total_mil_amt"]
CLASSES = ["Notes", "Bonds"]


def fetch_table1(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(MSPD_TABLE1_URL, params={"page[size]": 10_000, "sort": "record_date"}, timeout=120)
    r.raise_for_status()
    j = r.json()
    if j["meta"]["total-pages"] != 1:
        raise RuntimeError("MSPD table 1 no longer fits one page; add pagination")
    df = pd.DataFrame(j["data"])
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"MSPD: response lacks expected fields {missing}; re-inspect the API (rule 8)")
    return df


def notes_bonds(raw: pd.DataFrame) -> pd.DataFrame:
    """Marketable Notes and Bonds lines only, as published (strings)."""
    m = raw[(raw["security_type_desc"] == "Marketable") & raw["security_class_desc"].isin(CLASSES)]
    return m[KEEP_COLS].sort_values(["record_date", "security_class_desc"]).reset_index(drop=True)


def load_mspd(end: str | None = IS_END, path: Path | None = None) -> pd.DataFrame:
    """Wide monthly table ($ millions): notes_total, bonds_total, notes_public, bonds_public, notes_bonds_total."""
    path = path or active_dir() / SNAPSHOT_NAME
    df = pd.read_csv(path, dtype={"security_class_desc": str})
    df["record_date"] = pd.to_datetime(df["record_date"], format="%Y-%m-%d")
    if end is not None:
        df = df[df["record_date"] <= pd.Timestamp(end)]
    wide = df.pivot(index="record_date", columns="security_class_desc",
                    values=["total_mil_amt", "debt_held_public_mil_amt"])
    out = pd.DataFrame({
        "notes_total": wide[("total_mil_amt", "Notes")],
        "bonds_total": wide[("total_mil_amt", "Bonds")],
        "notes_public": wide[("debt_held_public_mil_amt", "Notes")],
        "bonds_public": wide[("debt_held_public_mil_amt", "Bonds")],
    })
    out["notes_bonds_total"] = out["notes_total"] + out["bonds_total"]
    return out
