"""FOMC decision dates loader from config/fomc_dates.csv (CLAUDE.md 7.1).

Pages inspected 2026-10-03 (rule 8):
* `fomccalendars.htm`: one `div.panel` per year headed "<YEAR> FOMC Meetings" (2021-2027 on that day). Each
  meeting is a `div.fomc-meeting` with `.fomc-meeting__month` ("January", "Apr/May") and `.fomc-meeting__date`
  ("27-28", "17-18*" where * marks a projections meeting, "30-1", "22 (notation vote)"). The policy statement is a
  block with `<strong>Statement:</strong>` followed by PDF/HTML links whose href carries the date
  (".../monetary20260128a.htm").
* `fomchistorical<YEAR>.htm` (1993-2020): one `div.panel` per event; `.panel-heading` reads e.g.
  "January 29-30 Meeting - 2008", "January 21 Conference Call - 2008", "April/May 30-1 Meeting - 2019",
  "October 4 (unscheduled) - 2019", "March 2 (unscheduled) Meeting - 2020", "March 17-18 (cancelled) Meeting - 2020",
  "March 19 (notation vote) - 2020". A policy statement is a link whose text is exactly "Statement"; its href
  carries the release date ("/fomc/19940204default.htm", "/boarddocs/press/general/2001/20010103/",
  "/newsevents/press/monetary/20080122b.htm", "/newsevents/pressreleases/monetary20200303a.htm").
  Other links ("Press Release", "Statement on Longer-Run Goals and Monetary Policy Strategy") are not decisions.

Decision rule (our choice, pre-registered intent of risk rule 2 and the H5 dummy, CLAUDE.md 7.7-7.8):
* every scheduled meeting is a decision on its last day, whether or not a statement was issued (statements
  start in 1994; meetings with no change before ~1999 issued none);
* an unscheduled meeting, conference call or notation vote is a decision only if it links a policy "Statement",
  dated by that statement (e.g. the 2008-01-21 call was announced 2008-01-22);
* on the historical pages from 2000 on, a "Meeting" with no statement link is a special meeting, not a decision
  (the FOMC has released a statement after every regular meeting since May 1999; on the 2026-10-03 pages this
  affects only "September 15 Meeting - 2003", the day before the regular 2003-09-16 meeting). The June 27-28, 2007
  panel links a statement dated 2007-06-18 (a link error on the Fed page); scheduled meetings use the end date;
* cancelled meetings are dropped. Before 1994 no statements exist, so 1993 conference calls cannot be classified
  and are not decisions; the funds-rate target was unchanged through 1993.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.data.snapshot import FOMC_CSV

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
FIRST_YEAR = 1993
STATEMENT_EVERY_MEETING_FROM = 2000
COLUMNS = ["decision_date", "decision", "kind", "start_date", "end_date", "statement_date", "label", "source"]

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                       "nov", "dec"], start=1)}
_DAYS_RE = re.compile(r"^(?P<m1>[A-Za-z]+)(?:/(?P<m2>[A-Za-z]+))?\s+(?P<d1>\d{1,2})"
                      r"(?:\s*-\s*(?:(?P<m3>[A-Za-z]+)\s+)?(?P<d2>\d{1,2}))?\s*\*?\s*(?P<rest>.*)$")
_DATE_IN_HREF = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")


def _month(name: str) -> int:
    key = name.strip().lower()[:3]
    if key not in _MONTHS:
        raise ValueError(f"unknown month {name!r}")
    return _MONTHS[key]


def _kind(tag: str, type_word: str) -> str:
    tag, type_word = tag.lower(), type_word.lower()
    if "cancel" in tag:
        return "cancelled"
    if "notation" in tag:
        return "notation_vote"
    if "unscheduled" in tag:
        return "unscheduled"
    if "conference call" in type_word:
        return "conference_call"
    if "meeting" in type_word or type_word == "":
        return "scheduled"
    raise ValueError(f"cannot classify FOMC event tag={tag!r} type={type_word!r}")


def parse_days(text: str, year: int, default_type: str = "") -> dict:
    """'April/May 30-1 Meeting', 'June 30-July 1 Meeting', 'March 2 (unscheduled) Meeting', '22 (notation vote)'
    (month given separately on the current page) -> start/end dates and kind."""
    m = _DAYS_RE.match(text.strip())
    if not m:
        raise ValueError(f"cannot parse FOMC dates from {text!r}")
    m1 = _month(m["m1"])
    end_month = _month(m["m3"]) if m["m3"] else (_month(m["m2"]) if m["m2"] else m1)
    d1 = int(m["d1"])
    d2 = int(m["d2"]) if m["d2"] else d1
    start, end = date(year, m1, d1), date(year, end_month, d2)
    if end < start:
        raise ValueError(f"FOMC event ends before it starts: {text!r}")
    rest = m["rest"]
    tag = " ".join(re.findall(r"\(([^)]*)\)", rest))
    type_word = re.sub(r"\([^)]*\)", "", rest).strip() or default_type
    return {"start_date": start, "end_date": end, "kind": _kind(tag, type_word)}


def _href_date(href: str | None) -> date | None:
    m = _DATE_IN_HREF.search(href or "")
    return pd.Timestamp(m.group(1)).date() if m else None


def parse_historical(html: str, year: int, source: str = "") -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for panel in soup.select("div.panel"):
        head = panel.select_one(".panel-heading")
        if head is None:
            continue
        label = head.get_text(" ", strip=True)
        m = re.match(r"^(?P<body>.+?)\s+-\s+(?P<year>\d{4})$", label)
        if not m or int(m["year"]) != year:
            continue
        ev = parse_days(m["body"], year)
        stmt = [a for a in panel.select("a") if a.get_text(strip=True).lower() == "statement"]
        ev.update(label=label, source=source,
                  statement_date=_href_date(stmt[0].get("href")) if stmt else None)
        if ev["kind"] == "scheduled" and not stmt and year >= STATEMENT_EVERY_MEETING_FROM:
            ev["kind"] = "special_meeting"
        events.append(ev)
    return events


def current_page_years(html: str) -> list[int]:
    soup = BeautifulSoup(html, "html.parser")
    years = []
    for panel in soup.select("div.panel"):
        h = panel.find(["h4", "h3", "h5"])
        m = re.match(r"^(\d{4}) FOMC Meetings$", h.get_text(strip=True)) if h else None
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def parse_current(html: str, source: str = CALENDAR_URL) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for panel in soup.select("div.panel"):
        h = panel.find(["h4", "h3", "h5"])
        m = re.match(r"^(\d{4}) FOMC Meetings$", h.get_text(strip=True)) if h else None
        if not m:
            continue
        year = int(m.group(1))
        for row in panel.select("div.fomc-meeting"):
            month = row.select_one(".fomc-meeting__month").get_text(strip=True)
            days = row.select_one(".fomc-meeting__date").get_text(" ", strip=True)
            first, _, rest = days.partition(" ")
            months = month.split("/")
            if "-" in first and len(months) == 2:
                text = f"{months[0]}/{months[1]} {first} {rest}"
            else:
                text = f"{months[0]} {first} {rest}"
            ev = parse_days(text, year, default_type="Meeting")
            stmt_date = None
            for strong in row.select("strong"):
                if strong.get_text(strip=True).lower().startswith("statement:"):
                    dates = [_href_date(a.get("href")) for a in strong.parent.select("a")]
                    dates = [d for d in dates if d]
                    stmt_date = dates[0] if dates else None
                    break
            ev.update(label=f"{month} {days} - {year}", source=source, statement_date=stmt_date)
            events.append(ev)
    return events


def decide(events: list[dict]) -> pd.DataFrame:
    """Apply the decision rule in the module docstring; one row per event, sorted by end date."""
    rows = []
    for ev in events:
        if ev["kind"] == "scheduled":
            decision, ddate = True, ev["end_date"]
        elif ev["kind"] in ("cancelled", "special_meeting"):
            decision, ddate = False, None
        else:
            decision, ddate = ev["statement_date"] is not None, ev["statement_date"]
        rows.append({**ev, "decision": int(decision), "decision_date": ddate})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["end_date", "label"], kind="mergesort").reset_index(drop=True)


def load_fomc_dates(path: Path = FOMC_CSV, end: str | None = None, scheduled_only: bool = True) -> pd.DatetimeIndex:
    """Sorted unique FOMC decision dates (announcement days).

    scheduled_only=True (default) keeps scheduled meetings only: their dates are public about a year ahead, so they
    are known at entry E. Unscheduled actions are not, and using them in a signal or risk rule would be lookahead
    (CLAUDE.md 7.1, 7.7, 7.8); pass False only for descriptive tables such as the worst windows.
    """
    df = pd.read_csv(path)
    keep = df["decision"] == 1
    if scheduled_only:
        keep &= df["kind"] == "scheduled"
    d = pd.to_datetime(df.loc[keep, "decision_date"], format="%Y-%m-%d")
    if end is not None:
        d = d[d <= pd.Timestamp(end)]
    return pd.DatetimeIndex(sorted(d.unique()), name="fomc_decision")
