"""Scrape FOMC decision dates into config/fomc_dates.csv (CLAUDE.md 7.1).

Usage: python scripts/fetch_fomc.py
Fetches fomccalendars.htm (recent years) and fomchistorical<YEAR>.htm for every earlier year from 1993, applies
the decision rule in src/data/fomc.py, writes config/fomc_dates.csv (all events, with a `decision` flag) and
records the download in data/snapshot/vintage.json. Then run scripts/download_all.py (or
`python -c "from src.data.snapshot import write_checksums; write_checksums()"`) to refresh the checksums.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.data import fomc  # noqa: E402
from src.data.snapshot import FOMC_CSV, record_vintage, utc_now, write_csv  # noqa: E402

UA = {"User-Agent": "extension-clock research (Gator Quant Hacks 2026)"}

NOTE = """
All events on the Fed's FOMC calendar pages from 1993, one row each, with `decision` = 1 for decision dates. Rule
(details in `src/data/fomc.py`): every scheduled meeting counts on its last day; an unscheduled meeting, conference
call or notation vote counts only if it links a policy "Statement", dated by that statement; cancelled meetings
and special meetings without a statement (2000 on) do not count. 1993 conference calls predate statements and do
not count. Future scheduled meetings on the calendar page are included.
Use: only scheduled meetings (`kind == "scheduled"`) feed risk rule 2 and the FOMC dummy, because their dates are
public about a year ahead; unscheduled actions are not known at entry and appear only in descriptive tables
(`src/data/fomc.py::load_fomc_dates(scheduled_only=True)`, CLAUDE.md 7.1).

Hand-check (2026-10-03, by Claude Code; humans to confirm at STOP 2): 12 decision dates were each confirmed by
opening the Fed press release linked from the calendar page and finding the same date and the policy action on
it: 1994-02-04 (first statement), 1994-04-18 (call), 1998-10-15 (call), 2001-01-03 (call, -50bp), 2001-09-17
(call, -50bp), 2008-01-22 (call on 01-21, -75bp), 2008-10-08 (call on 10-07, coordinated cut), 2008-12-16
(0-1/4%), 2015-12-16 (liftoff), 2020-03-15 (unscheduled, to 0-1/4%), 2022-03-16 (first hike), 2024-09-18 (-50bp).
"""


def get(session: requests.Session, url: str) -> str:
    r = session.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    time.sleep(0.3)  # be polite to federalreserve.gov
    return r.text


def main() -> None:
    s = requests.Session()
    when = utc_now()
    current = get(s, fomc.CALENDAR_URL)
    years_current = fomc.current_page_years(current)
    events = fomc.parse_current(current)
    for year in range(fomc.FIRST_YEAR, min(years_current)):
        url = fomc.HISTORICAL_URL.format(year=year)
        evs = fomc.parse_historical(get(s, url), year, source=url)
        if not evs:
            raise RuntimeError(f"no FOMC events parsed from {url}; re-inspect the page (rule 8)")
        events += evs
    df = fomc.decide(events)
    write_csv(df, FOMC_CSV.name, FOMC_CSV.parent)
    dec = df[df["decision"] == 1]
    print(f"{len(df)} events, {len(dec)} decisions, {dec['decision_date'].min()} to {dec['decision_date'].max()}")
    print(df["kind"].value_counts().to_string())
    record_vintage({FOMC_CSV.name: {
        "source": "Federal Reserve FOMC calendars", "url": fomc.CALENDAR_URL, "downloaded_utc": when,
        "rows": int(len(df)), "first": str(dec["decision_date"].min()), "last": str(dec["decision_date"].max()),
    }}, {FOMC_CSV.name: NOTE})


if __name__ == "__main__":
    main()
