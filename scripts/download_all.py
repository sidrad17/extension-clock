"""Refresh the public-data snapshot and rewrite checksums (CLAUDE.md 7.1).

Usage: python scripts/download_all.py
Downloads Fiscal Data auctions, FRED yields (CURVE_KNOTS + DTB3), MSPD table 1 and the Ken French daily factors;
writes data/snapshot/*.csv (public data as published; Ken French only as the derived pension_pressure.csv, raw zip
kept in git-ignored data/cache/), updates vintage.json / VINTAGE.md and rewrites CHECKSUMS.sha256 (which also
covers config/fomc_dates.csv; refresh that one with scripts/fetch_fomc.py). Normal runs never call this: they
read the committed snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.data import auctions, french, fred, mspd  # noqa: E402
from src.data.snapshot import (record_vintage, utc_now, verify_checksums, write_checksums,  # noqa: E402
                               write_csv, write_text)

NOTES = {
    "auctions.csv": """
All Note and Bond auctions from the Fiscal Data `auctions_query` endpoint, every column as published (strings,
missing = "null"). Includes auctions announced but not yet held at download time (no results yet).
SOMA check (details in `src/data/auctions.py::public_amount`): on all 1,121 auctions with `soma_accepted > 0`
(2008-04 onward), `total_accepted = offering_amt + soma_accepted` to within 0.003%, so `total_accepted` includes the
Fed's add-on and the public amount is `total_accepted - soma_accepted`. Before 2008-04-10 the feed has no SOMA
field and `total_accepted` exceeds `offering_amt` by ~14% on average, so the public amount there is
`offering_amt`. Secondary-market SOMA (QE) holdings are not in this feed.
""",
    "fred_*.csv": """
FRED `fredgraph.csv` files as published (blank = missing). Known gaps: DGS20 is blank 1987-01-01 to 1993-09-30.
DGS30 has values 2002-02-19 to 2006-02-08 in this vintage, but they splice badly at both ends (+17bp vs DGS20
+3bp on 2002-02-19; -16bp vs -3bp on 2006-02-09), consistent with an extrapolated 30-year rate when no 30-year
bond was issued; `src/data/fred.py` masks that window by default (the file keeps the published values).
Loaders never forward-fill a run of more than 5 missing business days.
""",
    "mspd_notes_bonds.csv": """
Monthly Statement of the Public Debt, table 1 (Fiscal Data `/v1/debt/mspd/mspd_table_1`), marketable Notes and Bonds
lines only, $ millions as published. Coverage starts 2001-01-31, so the par validation (CLAUDE.md 7.6) runs from
2001. "Notes"/"Bonds" exclude TIPS and FRNs. "Debt held by the public" includes Federal Reserve holdings.
""",
    "pension_pressure.csv": """
Derived from the Ken French daily factors (raw zip kept in git-ignored `data/cache/`, never committed):
`mkt_total_pct = Mkt-RF + RF` (percent per day), from 1990-01-01. This is the only French input the pension
signal uses (CLAUDE.md 7.7); the monthly signal is computed in `src/signals.py`. The file ends at the last CRSP
update in the downloaded vintage.
""",
}


def main() -> None:
    s = requests.Session()
    entries = {}

    when = utc_now()
    raw = auctions.fetch_auctions(s)
    write_csv(raw, auctions.SNAPSHOT_NAME)
    entries[auctions.SNAPSHOT_NAME] = {"source": "Fiscal Data auctions_query", "url": auctions.AUCTIONS_URL,
                                       "downloaded_utc": when, "rows": int(len(raw)),
                                       "first": raw["auction_date"].min(), "last": raw["auction_date"].max()}
    print(f"auctions: {len(raw)} rows, {raw['auction_date'].min()} to {raw['auction_date'].max()}")

    for sid in fred.SERIES:
        when = utc_now()
        text, series = fred.fetch_series(sid, s)
        write_text(text, fred.snapshot_name(sid))
        valid = series.dropna()
        entries[fred.snapshot_name(sid)] = {"source": f"FRED {sid}", "url": f"{fred.FRED_CSV_URL}?id={sid}",
                                            "downloaded_utc": when, "rows": int(len(series)),
                                            "first": str(valid.index.min().date()),
                                            "last": str(valid.index.max().date())}
        print(f"{sid}: {len(series)} rows, {valid.index.min().date()} to {valid.index.max().date()}")

    when = utc_now()
    m = mspd.notes_bonds(mspd.fetch_table1(s))
    write_csv(m, mspd.SNAPSHOT_NAME)
    entries[mspd.SNAPSHOT_NAME] = {"source": "Fiscal Data MSPD table 1", "url": mspd.MSPD_TABLE1_URL,
                                   "downloaded_utc": when, "rows": int(len(m)),
                                   "first": m["record_date"].min(), "last": m["record_date"].max()}
    print(f"mspd: {len(m)} rows, {m['record_date'].min()} to {m['record_date'].max()}")

    when = utc_now()
    french.download_raw(s)
    p = french.derive_pension_input(french.parse_daily(french.read_zip_text()))
    write_csv(p, french.SNAPSHOT_NAME)
    entries[french.SNAPSHOT_NAME] = {"source": "Ken French daily factors (derived)", "url": french.FRENCH_URL,
                                     "downloaded_utc": when, "rows": int(len(p)),
                                     "first": p["date"].min(), "last": p["date"].max()}
    print(f"pension_pressure (derived): {len(p)} rows, {p['date'].min()} to {p['date'].max()}")

    record_vintage(entries, {k: v for k, v in NOTES.items()})
    write_checksums()
    problems = verify_checksums()
    print("checksums written;", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
