"""Optional Databento CME Treasury futures layer; databento imported lazily (CLAUDE.md 7.1).

Pull (CLAUDE.md 7.1; Phase 4 choices in CLAUDE.md section 15):
* Dataset GLBX.MDP3, parents ZT.FUT, ZF.FUT, ZN.FUT, TN.FUT, ZB.FUT, UB.FUT (`stype_in="parent"`), from the dataset's
  first day (2010-06-06; the 60-day DV01 regression needs history before FUT_START = 2010-07-01) through IS_END.
* `statistics` (daily settlement, `databento.StatType.SETTLEMENT_PRICE`) runs to 2024-10-01 12:00 UTC, so a
  settlement for 2024-09-30 sent after midnight UTC is not cut off; `ohlcv-1d` (volume for the roll rule) ends
  2024-10-01 (exclusive). The loaders drop every trade date after IS_END.
* `definition`: one snapshot (a one-day request) on the first weekday of each month, plus a day-by-day search between
  two snapshots whenever a product's tick size differs between them, so each contract's tick size is known on every
  date. A single 14-year request streamed at about 1 KB/s (it was stopped after 1.8 years); every contract lives about
  nine months, so monthly snapshots see each one several times.
* `client.metadata.get_cost(...)` is called first and printed; the pull aborts if the total is above
  DATABENTO_BUDGET_USD.
* The API key is read only through python-dotenv (repo `.env`, DATABENTO_API_KEY) and is never printed or logged.
  Without a key the download is skipped with a clear message (CLAUDE.md rule 4).
* Raw files go to data/cache/databento/ (git-ignored, CLAUDE.md rule 6); only derived per-trade and daily P&L tables
  are committed (src/futures.py, outputs/tables/futures_*.csv).

Fields, from the files pulled on 2026-10-03 (rule 8):
* statistics -> ts_recv (index), ts_event, rtype, publisher_id, instrument_id, ts_ref, price, quantity, sequence,
  ts_in_delta, stat_type, channel_id, update_action, stat_flags, symbol. For SETTLEMENT_PRICE, `ts_ref` is the trade
  date (00:00 UTC) and each trade date has several messages: a preliminary price around 19:00-21:00 UTC, repeats, and
  a final one (stat_flags 3 from 2015) after about 22:40 UTC or on the weekend. In 256 of 73,619 contract-days the
  price changes between messages, so the settlement is the LAST message for (contract, ts_ref). 2,936 messages have
  no ts_ref (weekend replays of a known price) and are dropped. stat_flags is 0 or 4 before 2015 (older feed), so it
  is not used.
* ohlcv-1d -> ts_event (index, 00:00 UTC), rtype, publisher_id, instrument_id, open, high, low, close, volume,
  symbol. Bars are UTC days (Sunday bars hold the Sunday evening session), so the bar dated d is the session of
  trade date d plus the first hours of d + 1's evening session.
* definition -> raw_symbol, asset, security_type ("FUT"), instrument_class ("F"), maturity_year, maturity_month,
  expiration, activation, min_price_increment (points), unit_of_measure_qty (face value: 200,000 for ZT, 100,000
  otherwise; blank in the 2010-06-07 and 2010-07-01 snapshots, filled with the root's only value).
  `min_price_increment_amount` is not a dollar tick value in this feed (0.001 for ZN, 15.625 for TN on 2024-09-03),
  so tick value = min_price_increment x unit_of_measure_qty / 100. Tick sizes found (2010-06 to 2024-09): ZT 1/128
  until 2019-01-11 and 1/256 from 2019-01-14 (first snapshot with the new value, by bisection); ZF 1/128; ZN and TN
  1/64; ZB and UB 1/32 throughout.
* Symbols: the parents also hold calendar spreads (raw symbols with "-"), user-defined spreads ("UD:..."), and in
  2010-2012 a different "TN" product listed in serial months (TNG1, TNK1, TNQ1, TNX0, ...; the Ultra 10-year note
  starts with TNH6 on 2016-01-10). Only quarterly outrights (root + H/M/U/Z + one year digit) are kept; the year digit
  is resolved to the one year in [observation year - 1, observation year + 8] that ends in it, and the result is
  checked against the definitions' maturity_year. CME reuses security ids over the years, so contracts are keyed by
  symbol + year (e.g. ZNZ2023), not by instrument_id.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from config.settings import FUTURES, IS_END
from src.data.snapshot import CACHE_DIR, REPO_ROOT

DATASET = "GLBX.MDP3"
DATA_START = "2010-06-06"                     # first day of GLBX.MDP3 (metadata.get_dataset_range, 2026-10-03)
PULL_END = {"statistics": "2024-10-01T12:00", "ohlcv-1d": "2024-10-01"}
BULK_SCHEMAS = list(PULL_END)
DATABENTO_BUDGET_USD = 5.00                   # abort the pull above this estimate (team, Oct 3, 2026)
RAW_DIR = CACHE_DIR / "databento"
DEF_DIR = RAW_DIR / "definition"
PARENTS = [f"{p}.FUT" for p in FUTURES]
QUARTER_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
N_WORKERS = 4


class NoDatabento(RuntimeError):
    """No key (or no databento package) and no cached raw data: the futures layer cannot be rebuilt."""


# ------------------------------------------------------------------------------------------------ client, pull

def api_key() -> str | None:
    """DATABENTO_API_KEY from the repo .env via python-dotenv (environment wins). Never printed."""
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
    k = os.environ.get("DATABENTO_API_KEY", "").strip()
    return k or None


def client():
    key = api_key()
    if key is None:
        raise NoDatabento("DATABENTO_API_KEY is not set (.env); the futures layer needs it to download.")
    try:
        import databento as db
    except ImportError as e:                                   # optional extra (requirements.txt)
        raise NoDatabento("the databento package is not installed (pip install -r requirements.txt)") from e
    return db.Historical(key)


def raw_path(schema: str):
    return RAW_DIR / f"glbx_{schema}_{DATA_START}_{PULL_END[schema].replace(':', '')}.dbn.zst"


def def_path(day: pd.Timestamp):
    return DEF_DIR / f"glbx_definition_{day.date()}.dbn.zst"


def request(schema: str, start=DATA_START, end=None) -> dict:
    return {"dataset": DATASET, "start": str(start), "end": str(end or PULL_END[schema]), "symbols": PARENTS,
            "schema": schema, "stype_in": "parent"}


def def_request(day: pd.Timestamp) -> dict:
    return request("definition", day.date(), (day + pd.Timedelta(days=1)).date())


def snapshot_days(start=DATA_START, end=IS_END) -> list[pd.Timestamp]:
    """First weekday of each month (the dataset's first weekday for its first month)."""
    out = []
    for p in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M"):
        d = pd.bdate_range(max(p.start_time, pd.Timestamp(start)), p.end_time)[0]
        out.append(d)
    return out


def _parallel(fn, items):
    with ThreadPoolExecutor(N_WORKERS) as ex:
        return list(ex.map(fn, items))


def cost_estimate(c, days: list[pd.Timestamp], bulk: list[str]) -> dict:
    """USD cost and billable MB of the missing pieces (metadata calls; free)."""
    out = {s: {"usd": float(c.metadata.get_cost(**request(s))),
               "mb": float(c.metadata.get_billable_size(**request(s))) / 1e6} for s in bulk}
    if days:
        usd = _parallel(lambda d: float(c.metadata.get_cost(**def_request(d))), days)
        mb = _parallel(lambda d: float(c.metadata.get_billable_size(**def_request(d))) / 1e6, days)
        out[f"definition ({len(days)} one-day snapshots)"] = {"usd": float(sum(usd)), "mb": float(sum(mb))}
    return out


def print_cost(est: dict) -> float:
    total = sum(v["usd"] for v in est.values())
    print(f"Databento cost estimate ({DATASET}, {', '.join(PARENTS)}, {DATA_START} to {IS_END}):")
    for s, v in est.items():
        print(f"  {s:38s} {v['mb']:9.2f} MB  ${v['usd']:.4f}")
    print(f"  total{'':43s}${total:.4f}  (budget ${DATABENTO_BUDGET_USD:.2f})")
    return total


def _pull(c, req: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    c.timeseries.get_range(**req, path=tmp)
    tmp.replace(path)


def download(budget: float = DATABENTO_BUDGET_USD) -> None:
    """Pull every missing raw file into data/cache/databento/ after a cost check (module docstring)."""
    bulk = [s for s in BULK_SCHEMAS if not raw_path(s).exists()]
    days = [d for d in snapshot_days() if not def_path(d).exists()]
    if bulk or days:
        c = client()
        total = print_cost(cost_estimate(c, days, bulk))
        if total > budget:
            raise RuntimeError(f"Databento estimate ${total:.2f} is above the budget ${budget:.2f}; nothing pulled.")
        for s in bulk:
            _pull(c, request(s), raw_path(s))
            print(f"  pulled {s} ({raw_path(s).stat().st_size / 1e6:.1f} MB)")
        _parallel(lambda d: _pull(c, def_request(d), def_path(d)), days)
        if days:
            print(f"  pulled {len(days)} definition snapshots")
    refine_tick_changes()


def refine_tick_changes() -> list[pd.Timestamp]:
    """Between two snapshots whose tick sizes differ for a product, find the first weekday with the new tick by
    bisection (one-day requests, cached). Returns the change dates found."""
    found = []
    c = None
    while True:
        changes = _tick_change_intervals(load_definition_snapshots())
        todo = []
        for lo, hi in changes:
            mid_days = pd.bdate_range(lo + pd.Timedelta(days=1), hi - pd.Timedelta(days=1))
            if len(mid_days) == 0:
                found.append(hi)
                continue
            todo.append(mid_days[len(mid_days) // 2])
        if not todo:
            return sorted(set(found))
        c = c or client()
        est = sum(_parallel(lambda d: float(c.metadata.get_cost(**def_request(d))), todo))
        if est > DATABENTO_BUDGET_USD:
            raise RuntimeError(f"definition refinement estimate ${est:.2f} is above the budget")
        _parallel(lambda d: _pull(c, def_request(d), def_path(d)), todo)


def _tick_change_intervals(defs: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(snapshot, next snapshot) pairs between which a product's set of tick sizes changes and no snapshot lies
    in between yet."""
    out = []
    for _, g in defs.groupby("root"):
        ticks = g.groupby("snapshot")["tick_size"].apply(lambda s: tuple(sorted(set(s.round(12)))))
        snaps = ticks.index
        for a, b in zip(snaps[:-1], snaps[1:]):
            if ticks[a] != ticks[b] and len(pd.bdate_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1))):
                out.append((a, b))
    return out


def have_cache() -> bool:
    return all(raw_path(s).exists() for s in BULK_SCHEMAS) and all(def_path(d).exists() for d in snapshot_days())


# ------------------------------------------------------------------------------------------------ symbols

def resolve_year(digit: int, observed: pd.Timestamp) -> int:
    """The one year in [observed.year - 1, observed.year + 8] whose last digit is `digit` (module docstring)."""
    for y in range(observed.year - 1, observed.year + 9):
        if y % 10 == digit:
            return y
    raise AssertionError("unreachable")


def parse_outright(symbol: str) -> tuple[str, str, int] | None:
    """('ZN', 'Z', 3) for 'ZNZ3'; None for spreads, other roots and non-quarterly months."""
    if not isinstance(symbol, str) or len(symbol) != 4 or not symbol[3].isdigit():
        return None
    root, code = symbol[:2], symbol[2]
    if root not in FUTURES or code not in QUARTER_CODES:
        return None
    return root, code, int(symbol[3])


def contract_ids(symbols: pd.Series, observed: pd.Series) -> pd.DataFrame:
    """root, contract (e.g. ZNZ2023), delivery (monthly Period) for quarterly outrights; NaN rows otherwise."""
    parsed = [parse_outright(s) for s in symbols]
    rows = []
    for p, d in zip(parsed, observed):
        if p is None:
            rows.append((None, None, pd.NaT))
            continue
        root, code, digit = p
        y = resolve_year(digit, pd.Timestamp(d))
        rows.append((root, f"{root}{code}{y}", pd.Period(year=y, month=QUARTER_CODES[code], freq="M")))
    return pd.DataFrame(rows, columns=["root", "contract", "delivery"], index=symbols.index)


# ------------------------------------------------------------------------------------------------ loaders

def _read(path) -> pd.DataFrame:
    import databento as db
    return db.DBNStore.from_file(path).to_df(map_symbols=True)


def load_settlements(end=IS_END) -> pd.DataFrame:
    """One settlement per (contract, trade date <= end) from the cached statistics file (settlements_from_stats)."""
    import databento as db
    s = _read(raw_path("statistics"))
    return settlements_from_stats(s[s["stat_type"] == db.StatType.SETTLEMENT_PRICE], end)


def settlements_from_stats(s: pd.DataFrame, end=IS_END) -> pd.DataFrame:
    """SETTLEMENT_PRICE messages (index ts_recv; columns ts_ref, price, symbol) -> the last message per (contract,
    trade date = ts_ref), trade dates <= end. Columns: contract, root, delivery, trade_date, settle (points)."""
    s = s[s["ts_ref"].notna()].reset_index()
    s["trade_date"] = s["ts_ref"].dt.tz_localize(None).dt.normalize()
    ids = contract_ids(s["symbol"], s["trade_date"])
    s = pd.concat([s, ids], axis=1)
    s = s[s["contract"].notna() & (s["trade_date"] <= pd.Timestamp(end))]
    s = s.sort_values(["ts_recv"], kind="mergesort")
    last = s.groupby(["contract", "trade_date"], sort=True).tail(1)
    return last[["contract", "root", "delivery", "trade_date", "price"]].rename(columns={"price": "settle"}) \
        .sort_values(["contract", "trade_date"]).reset_index(drop=True)


def load_volume(end=IS_END) -> pd.DataFrame:
    """Daily volume per contract from the UTC-day ohlcv-1d bars, dates <= end. Columns: contract, root, delivery,
    date, volume."""
    o = _read(raw_path("ohlcv-1d")).reset_index()
    o["date"] = o["ts_event"].dt.tz_localize(None).dt.normalize()
    o = pd.concat([o, contract_ids(o["symbol"], o["date"])], axis=1)
    o = o[o["contract"].notna() & (o["date"] <= pd.Timestamp(end))]
    out = o.groupby(["contract", "root", "delivery", "date"], as_index=False)["volume"].sum()
    return out.sort_values(["contract", "date"]).reset_index(drop=True)


def load_definition_snapshots() -> pd.DataFrame:
    """Every cached definition snapshot, quarterly outrights of FUTURES only. Columns: snapshot, contract, root,
    delivery, raw_symbol, expiration, tick_size (points), face, point_value ($ per point), tick_value ($)."""
    frames = []
    for p in sorted(DEF_DIR.glob("glbx_definition_*.dbn.zst")):
        d = _read(p)
        if d.empty:
            continue
        d = d.reset_index()
        d["snapshot"] = pd.Timestamp(p.name.split("_")[-1].split(".")[0])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["security_type"] == "FUT") & (d["instrument_class"] == "F")]
    ids = contract_ids(d["raw_symbol"], d["snapshot"])
    d = pd.concat([d, ids], axis=1)
    d = d[d["contract"].notna() & (d["asset"] == d["root"])]
    bad = d[(d["maturity_year"] != d["delivery"].map(lambda p: p.year))
            | (d["maturity_month"] != d["delivery"].map(lambda p: p.month))]
    if len(bad):
        raise ValueError(f"symbol year resolution disagrees with the definitions on {len(bad)} rows: "
                         f"{bad[['snapshot', 'raw_symbol', 'maturity_year', 'maturity_month']].head().to_dict('records')}")
    d["face"] = d["unit_of_measure_qty"].astype(float)
    face = d.dropna(subset=["face"]).groupby("root")["face"].unique()
    if (face.map(len) != 1).any():
        raise ValueError(f"more than one contract size per root in the definitions: {face.to_dict()}")
    # the 2010-06-07 and 2010-07-01 snapshots (older feed) leave unit_of_measure_qty blank: use the root's only value
    d["face"] = d["face"].fillna(d["root"].map(face.map(lambda v: float(v[0]))))
    d["point_value"] = d["face"] / 100.0
    d["tick_size"] = d["min_price_increment"].astype(float)
    d["tick_value"] = d["tick_size"] * d["point_value"]
    d["expiration"] = d["expiration"].dt.tz_localize(None)
    cols = ["snapshot", "contract", "root", "delivery", "raw_symbol", "expiration", "tick_size", "face", "point_value",
            "tick_value"]
    return d[cols].drop_duplicates(["snapshot", "contract"], keep="last").sort_values(["contract", "snapshot"]) \
        .reset_index(drop=True)


def tick_value_asof(defs: pd.DataFrame, contract: str, date: pd.Timestamp) -> float:
    """$ tick value of `contract` on `date`: its latest snapshot on or before the date, else its first snapshot."""
    g = defs[defs["contract"] == contract]
    if g.empty:
        raise KeyError(f"no definition for {contract}")
    before = g[g["snapshot"] <= date]
    return float((before.iloc[-1] if len(before) else g.iloc[0])["tick_value"])


def point_values(defs: pd.DataFrame) -> dict[str, float]:
    """$ per 1.00 price point by root; the face value must be the same for every contract of a root."""
    pv = defs.groupby("root")["point_value"].unique()
    out = {}
    for r, v in pv.items():
        if len(v) != 1:
            raise ValueError(f"{r}: more than one contract size in the definitions: {v}")
        out[r] = float(v[0])
    return out


def tick_history(defs: pd.DataFrame) -> dict:
    """Tick size by root and the snapshot dates where it changes (for results.json and contract_specs.yaml)."""
    out = {}
    for r, g in defs.groupby("root"):
        t = g.groupby("snapshot")["tick_size"].apply(lambda s: sorted(set(np.round(s, 12))))
        hist, prev = [], None
        for d, v in t.items():
            if v != prev:
                hist.append({"from_snapshot": str(d.date()), "tick_size": v})
                prev = v
        out[r] = hist
    return out
