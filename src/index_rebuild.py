"""Point-in-time Treasury index rebuild: universe, duration, Ext, coupon cash, FDD, bucket demand (CLAUDE.md 7.5).

Index rules (Bloomberg US Treasury Index as summarized for SPTB; parameters in config/settings.py, pre-registered):
* fixed-coupon nominal Treasury notes and bonds; TIPS, FRNs, callable bonds and bills excluded (EXCLUDE);
* at least MIN_MATURITY_YEARS (1 year) to maturity from the first day of the month after the rebalance;
* at least MIN_PUBLIC_AMOUNT ($300M) public amount: par outstanding less Fed SOMA holdings (DEDUCT_SOMA), see
  "Fed holdings" below;
* membership changes only at the rebalance T (last bond business day); a tranche counts once auctioned by T,
  settled or not (INCLUSION_RULE = "auctioned_by_rebalance"; sensitivity "settled_by_month_end": issue_date on or
  before the last calendar day of the month).

Point in time (CLAUDE.md rule 3): every number for month m uses only what is known at the close of E = T - 4.
* NEXT universe (after the rebalance at T): tranches auctioned on or before E use their public amount; tranches
  auctioned in (E, T] use offering_amt if announced by E, and are skipped (and counted) if announced after E.
  A CUSIP with no tranche auctioned by E has no coupon yet: coupon = curve_yield(E, maturity).
* NOW universe (members during month m) = the NEXT rule applied at T(m-1), evaluated at E(m): every tranche in it
  was auctioned by T(m-1) < E(m), so its result is known and actual public amounts are used.
* Both universes are priced at E on the same FRED par curve (bonds.py), so only membership and amounts differ:
  D = sum_i w_i D_i with market-value weights, Ext = D_next - D_now.

Fed holdings (DEDUCT_SOMA; the index deducts SOMA holdings, "both purchases at issuance and net secondary market
transactions", Bloomberg US Treasury Index methodology). Two rules, chosen per month (PREREG_ADDENDUM.md):
* SOMA by CUSIP (cfg.soma_by_cusip, months from SOMA_FIRST_CUSIP_MONTH = 2003-08): amount of CUSIP i =
  sum over its tranches of [total_accepted if auctioned by E and issued on or before the SOMA date s;
  total_accepted - Fed add-on (= public_amount) if auctioned by E but issued after s, since the add-on is not in
  the holdings yet; offering_amt if auctioned after E] minus H_i(s), the NY Fed's holding of i on s
  (src/data/soma.py). s = the latest as-of date known at the close of the evaluation date (soma.usable_asof).
  NEXT(m) is evaluated at E(m). NOW(m) is the index during month m, whose amounts were fixed at the previous
  rebalance, so by default (soma_now_date = "rebalance") it uses s known at T(m-1); with "entry" it uses s known
  at E(m) (diagnostic). The difference is the Fed's net buying between the two dates, which the index passes to
  trackers at T(m). A negative result (holdings above our par, a data mismatch) is set to 0 and counted.
* Auction-only (before 2003-08: the NY Fed history starts 2003-07-09, and the first month whose NOW date has
  holdings is 2003-08; or soma_by_cusip = False): public_amount from the auctions feed = total_accepted -
  soma_accepted (2008-04 on), offering_amt before (src/data/auctions.py::public_amount).
One rule per month for both universes, so no month mixes the two.

Exact decomposition used for tracing: since each CUSIP has the same D_i in both universes and both weight vectors
sum to one, Ext = sum_i (w_next_i - w_now_i) (D_i - D_now). Adds contribute w_next_i (D_i - D_now), removals
-w_now_i (D_i - D_now), reopenings (a new tranche of a member CUSIP) their weight change, and every other member
contributes the remainder (ext_dilution: weight changes from other amounts, and under SOMA by CUSIP the Fed's net
buying or selling of that CUSIP).

Coupon cash c_m (CLAUDE.md 7.5): coupons paid during month m by NOW members, int_rate / 2 x amount, on the
maturity-month schedule (coupon months = maturity month +/- 6; the first coupon month is first_int_payment_date's
month, which matches that cycle for every eligible auction in the 2026-10-03 snapshot), divided by NOW market value
at E. FDD_m = Ext_m + c_m x D_next (REINVEST_COUPONS; False gives FDD = Ext).

Buckets (CLAUDE.md 7.5) by remaining maturity at E: 1-3, 3-7, 7-10, 10-20, 20+ years (a NOW bond under 1 year is
counted in 1-3). dC_b = sum_{next in b} w_i D_i - sum_{now in b} w_i D_i (sums to Ext); cash_b = c_m x
sum_{next in b} w_i D_i (sums to c_m x D_next); fdd_b = dC_b + cash_b.

REBUILD_START = 1990-01 (our choice, to confirm at STOP 2): 36 months before IS_START, so the expanding z-score
(ZSCORE_MIN_MONTHS = 36, CLAUDE.md 7.7) is defined from the first in-sample month. Auction records start in
1979-10, so securities issued before then are missing from every month (see validate.py, par check).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from config.settings import (DEDUCT_SOMA, ENTRY_OFFSET, EXCLUDE, INCLUSION_RULE, IS_END, MIN_MATURITY_YEARS,
                             MIN_PUBLIC_AMOUNT, REINVEST_COUPONS)
from src.bonds import Curve, mod_duration, price
from src.calendar import BondCalendar
from src.data.soma import SomaHoldings
from src.trial_log import guard_end

REBUILD_START = "1990-01"
BUCKETS = ["1-3y", "3-7y", "7-10y", "10-20y", "20y+"]
BUCKET_EDGES = [-np.inf, 3.0, 7.0, 10.0, 20.0, np.inf]
RULES = ("auctioned_by_rebalance", "settled_by_month_end")
SOMA_NOW_DATES = ("rebalance", "entry")
SOMA_FIRST_CUSIP_MONTH = "2003-08"   # first month whose NOW date (T of 2003-07) has NY Fed holdings


@dataclass(frozen=True)
class RebuildConfig:
    inclusion_rule: str = INCLUSION_RULE
    reinvest_coupons: bool = REINVEST_COUPONS
    deduct_soma: bool = DEDUCT_SOMA
    entry_offset: int = ENTRY_OFFSET
    min_maturity_years: float = MIN_MATURITY_YEARS
    min_public_amount: float = MIN_PUBLIC_AMOUNT
    exclude: dict = field(default_factory=lambda: dict(EXCLUDE))
    soma_by_cusip: bool = True          # NY Fed holdings by CUSIP from 2003-08 (PREREG_ADDENDUM.md); False = auction-only
    soma_now_date: str = "rebalance"    # SOMA date for NOW: known at T(m-1) ("rebalance") or at E(m) ("entry")

    def __post_init__(self):
        if self.inclusion_rule not in RULES:
            raise ValueError(f"inclusion_rule must be one of {RULES}")
        if self.soma_now_date not in SOMA_NOW_DATES:
            raise ValueError(f"soma_now_date must be one of {SOMA_NOW_DATES}")


def prepare_tranches(auctions: pd.DataFrame) -> pd.DataFrame:
    """One row per tranche (original issue or reopening) with the fields the rebuild needs.

    `auctions` comes from src.data.auctions.load_auctions (exclusions applied, public_amount per DEDUCT_SOMA).
    """
    cols = ["cusip", "security_type", "security_term", "auction_date", "issue_date", "announcemt_date",
            "maturity_date", "int_rate", "offering_amt", "total_accepted", "public_amount", "reopening",
            "first_int_payment_date"]
    tr = auctions[cols].copy()
    tr["first_coupon_month"] = tr["first_int_payment_date"].dt.to_period("M")
    tr["maturity_month"] = tr["maturity_date"].dt.to_period("M")
    return tr.sort_values(["auction_date", "cusip"], kind="mergesort").reset_index(drop=True)


def holdings(tr: pd.DataFrame, T: pd.Timestamp, E: pd.Timestamp, rule: str, soma: pd.Series | None = None,
             soma_date: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Amount per CUSIP for the rebalance at T, as known at the close of E. Returns (by_cusip, skipped_tranches).

    soma = None: auction-only rule (public_amount per tranche). Otherwise `soma` holds the Fed's par by CUSIP on
    `soma_date` and the SOMA-by-CUSIP rule applies (module docstring, "Fed holdings").
    """
    if rule == "auctioned_by_rebalance":
        inc = tr["auction_date"] <= T
    else:
        inc = tr["issue_date"] <= T.to_period("M").end_time.normalize()
    t = tr[inc]
    known = t["auction_date"] <= E
    announced = t["announcemt_date"] <= E
    use = known | announced
    skipped = t[~use]
    t, known = t[use], known[use]
    if soma is None:
        amt = np.where(known, t["public_amount"], t["offering_amt"])
    else:
        settled = t["issue_date"] <= soma_date
        amt = np.where(known, np.where(settled, t["total_accepted"], t["public_amount"]), t["offering_amt"])
    t = t.assign(amt=amt, estimated=~known, known=known)
    g = t.groupby("cusip", sort=True).agg(
        amount=("amt", "sum"), n_tranches=("amt", "size"), n_estimated=("estimated", "sum"),
        coupon_known=("known", "any"), int_rate=("int_rate", "first"), maturity=("maturity_date", "first"),
        maturity_month=("maturity_month", "first"), first_coupon_month=("first_coupon_month", "first"),
        term=("security_term", "first"), first_auction=("auction_date", "min"))
    g["soma"] = 0.0 if soma is None else soma.reindex(g.index).fillna(0.0).to_numpy(float)
    g["soma_excess"] = g["soma"] > g["amount"] + 1.0
    g["amount"] = (g["amount"] - g["soma"]).clip(lower=0.0)
    return g, skipped


def members(h: pd.DataFrame, first_day_next: pd.Timestamp, cfg: RebuildConfig) -> pd.DataFrame:
    """Apply the maturity and size rules (index methodology, settings.py)."""
    min_mat = first_day_next + pd.DateOffset(months=int(round(12 * cfg.min_maturity_years)))
    return h[(h["maturity"] >= min_mat) & (h["amount"] >= cfg.min_public_amount)]


def price_at(m: pd.DataFrame, curve: Curve, E: pd.Timestamp) -> pd.DataFrame:
    """Price, modified duration and market value at E on the par curve; coupon from the curve if not yet set."""
    years = (m["maturity"] - E).dt.days.to_numpy() / 365.25
    y = np.atleast_1d(curve.yields(E, years)) if len(m) else np.array([])
    coupon = np.where(m["coupon_known"].to_numpy(bool), m["int_rate"].to_numpy(float), y)
    px = np.atleast_1d(price(coupon, y, years)) if len(m) else np.array([])
    dur = np.atleast_1d(mod_duration(coupon, y, years)) if len(m) else np.array([])
    out = m.assign(years=years, yld=y, coupon=coupon, price=px, dur=dur, mv=m["amount"].to_numpy() * px / 100.0)
    if out[["price", "dur"]].isna().any().any():
        raise ValueError(f"pricing failed at {E.date()} for {list(out.index[out['price'].isna()])}")
    out["w"] = out["mv"] / out["mv"].sum()
    out["bucket"] = pd.cut(out["years"], BUCKET_EDGES, labels=BUCKETS, right=False)
    return out


def coupon_cash(now: pd.DataFrame, month: pd.Period) -> float:
    """Coupons paid in `month` by NOW members, as a share of NOW market value (CLAUDE.md 7.5)."""
    lag = (month.ordinal - now["maturity_month"].map(lambda p: p.ordinal)) % 6
    pays = (lag == 0) & (now["first_coupon_month"] <= month)
    coupons = (now["int_rate"] / 100.0 / 2.0 * now["amount"])[pays].sum()
    return float(coupons / now["mv"].sum())


def soma_dates(soma: SomaHoldings | None, cal: BondCalendar, month: pd.Period, T_prev: pd.Timestamp,
               E: pd.Timestamp, cfg: RebuildConfig) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """(SOMA date for NOW, for NEXT), or (None, None) when month m uses the auction-only rule."""
    if not (cfg.deduct_soma and cfg.soma_by_cusip and soma is not None) or month < pd.Period(SOMA_FIRST_CUSIP_MONTH,
                                                                                            "M"):
        return None, None
    s_next = soma.usable(cal, E)
    s_now = soma.usable(cal, T_prev if cfg.soma_now_date == "rebalance" else E)
    if s_now is None or s_next is None:
        raise ValueError(f"{month}: no usable SOMA date (NOW {s_now}, NEXT {s_next})")
    return s_now, s_next


def rebuild_month(tr: pd.DataFrame, curve: Curve, cal: BondCalendar, month: pd.Period,
                  cfg: RebuildConfig, soma: SomaHoldings | None = None) -> tuple[dict, pd.DataFrame]:
    """All outputs for month m, plus a per-CUSIP table of membership changes and Ext contributions.

    soma = None means the auction-only Fed deduction for every month (module docstring, "Fed holdings").
    """
    T = cal.month_end(month)
    E = cal.offset(T, -cfg.entry_offset)
    T_prev = cal.month_end(month - 1)
    first_day = month.start_time.normalize()
    first_day_next = (month + 1).start_time.normalize()

    s_now, s_next = soma_dates(soma, cal, month, T_prev, E, cfg)
    h_now, _ = holdings(tr, T_prev, E, cfg.inclusion_rule, None if s_now is None else soma.on(s_now), s_now)
    h_next, skipped = holdings(tr, T, E, cfg.inclusion_rule, None if s_next is None else soma.on(s_next), s_next)
    if h_now["n_estimated"].sum():
        raise AssertionError(f"{month}: NOW universe used an estimated amount; NOW must be fully known at E")
    now = price_at(members(h_now, first_day, cfg), curve, E)
    nxt = price_at(members(h_next, first_day_next, cfg), curve, E)

    d_now = float((now["w"] * now["dur"]).sum())
    d_next = float((nxt["w"] * nxt["dur"]).sum())
    ext = d_next - d_now
    c_m = coupon_cash(now, month)
    fdd = ext + (c_m * d_next if cfg.reinvest_coupons else 0.0)

    wd_now = (now["w"] * now["dur"]).groupby(now["bucket"], observed=False).sum().reindex(BUCKETS, fill_value=0.0)
    wd_next = (nxt["w"] * nxt["dur"]).groupby(nxt["bucket"], observed=False).sum().reindex(BUCKETS, fill_value=0.0)
    d_ext_b = wd_next - wd_now
    cash_b = c_m * wd_next if cfg.reinvest_coupons else 0.0 * wd_next

    # per-CUSIP changes and exact Ext contributions (module docstring)
    allc = now[["w", "dur", "amount", "n_tranches", "term", "maturity", "first_auction"]].join(
        nxt[["w", "dur", "amount", "n_tranches", "term", "maturity", "first_auction", "n_estimated", "coupon_known"]],
        how="outer", lsuffix="_now", rsuffix="_next")
    for c in ["w_now", "w_next", "amount_now", "amount_next", "n_tranches_now", "n_tranches_next"]:
        allc[c] = allc[c].fillna(0.0)
    allc["dur"] = allc["dur_next"].fillna(allc["dur_now"])
    allc["term"] = allc["term_next"].fillna(allc["term_now"])
    allc["maturity"] = allc["maturity_next"].fillna(allc["maturity_now"])
    allc["contribution"] = (allc["w_next"] - allc["w_now"]) * (allc["dur"] - d_now)
    in_now, in_next = allc.index.isin(now.index), allc.index.isin(nxt.index)
    allc["change"] = np.select(
        [in_next & ~in_now, in_now & ~in_next, in_now & in_next & (allc["n_tranches_next"] > allc["n_tranches_now"])],
        ["add", "remove", "reopen"], default="")
    if not np.isclose(allc["contribution"].sum(), ext, atol=1e-10):
        raise AssertionError(f"{month}: Ext decomposition does not add up")
    changes = allc[allc["change"] != ""].reset_index().rename(columns={"index": "cusip"})
    changes.insert(0, "month", str(month))
    changes = changes[["month", "cusip", "change", "term", "maturity", "amount_now", "amount_next", "dur", "w_now",
                       "w_next", "contribution", "n_estimated", "coupon_known"]]
    dilution = float(allc.loc[allc["change"] == "", "contribution"].sum())

    row = {
        "month": str(month), "T": T.date().isoformat(), "E": E.date().isoformat(),
        "Ext": ext, "c_m": c_m, "FDD": fdd, "D_now": d_now, "D_next": d_next,
        "mv_now_bn": now["mv"].sum() / 1e9, "mv_next_bn": nxt["mv"].sum() / 1e9,
        "par_now_bn": now["amount"].sum() / 1e9, "par_next_bn": nxt["amount"].sum() / 1e9,
        "n_now": len(now), "n_next": len(nxt),
        "n_adds": int((changes["change"] == "add").sum()), "n_removes": int((changes["change"] == "remove").sum()),
        "n_reopens": int((changes["change"] == "reopen").sum()),
        "n_estimated": int(h_next["n_estimated"].sum()), "n_skipped": int(len(skipped)),
        "soma_rule": "auction" if s_next is None else "cusip",
        "soma_date_now": "" if s_now is None else s_now.date().isoformat(),
        "soma_date_next": "" if s_next is None else s_next.date().isoformat(),
        "soma_now_bn": float(h_now.loc[now.index, "soma"].sum() / 1e9),
        "soma_next_bn": float(h_next.loc[nxt.index, "soma"].sum() / 1e9),
        "n_soma_excess": int(h_now["soma_excess"].sum() + h_next["soma_excess"].sum()),
        "ext_adds": float(changes.loc[changes["change"] == "add", "contribution"].sum()),
        "ext_removes": float(changes.loc[changes["change"] == "remove", "contribution"].sum()),
        "ext_reopens": float(changes.loc[changes["change"] == "reopen", "contribution"].sum()),
        "ext_dilution": dilution,
    }
    for b in BUCKETS:
        row[f"ext_{b}"] = float(d_ext_b[b])
    for b in BUCKETS:
        row[f"cash_{b}"] = float(cash_b[b])
    for b in BUCKETS:
        row[f"fdd_{b}"] = float(d_ext_b[b] + cash_b[b])
    return row, changes


def rebuild(auctions: pd.DataFrame, curve: Curve, cal: BondCalendar, start: str = REBUILD_START,
            end: str = IS_END, cfg: RebuildConfig | None = None,
            soma: SomaHoldings | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly rebuild from `start` to `end` (months whose T <= end). Returns (monthly, changes)."""
    guard_end(end)
    cfg = cfg or RebuildConfig()
    tr = prepare_tranches(auctions)
    months = cal.month_ends(start, pd.Timestamp(end).to_period("M")).index
    rows, changes = [], []
    for m in months:
        if cal.month_end(m) > pd.Timestamp(end):
            continue
        r, ch = rebuild_month(tr, curve, cal, m, cfg, soma)
        rows.append(r)
        changes.append(ch)
    monthly = pd.DataFrame(rows)
    return monthly, pd.concat(changes, ignore_index=True)


def run_default(start: str = REBUILD_START, end: str = IS_END,
                cfg: RebuildConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild from the committed snapshot (in-sample by default)."""
    from src.bonds import load_curve
    from src.calendar import load_calendar
    from src.data.auctions import load_auctions
    from src.data.soma import load_soma
    cfg = cfg or RebuildConfig()
    guard_end(end)
    auctions = load_auctions(end=end, deduct_soma=cfg.deduct_soma, exclude=cfg.exclude)
    soma = load_soma(end=end) if cfg.deduct_soma and cfg.soma_by_cusip else None
    return rebuild(auctions, load_curve(end=end), load_calendar(end=end), start=start, end=end, cfg=cfg, soma=soma)


OPEN_DATASET_COLUMNS = (["month", "T", "E", "Ext", "c_m", "FDD", "D_now", "D_next", "mv_now_bn", "mv_next_bn",
                         "n_now", "n_next", "n_adds", "n_removes", "n_reopens", "n_estimated", "n_skipped",
                         "soma_rule", "soma_date_now", "soma_date_next", "soma_now_bn", "soma_next_bn"]
                        + [f"ext_{b}" for b in BUCKETS] + [f"cash_{b}" for b in BUCKETS]
                        + [f"fdd_{b}" for b in BUCKETS])


def config_dict(cfg: RebuildConfig) -> dict:
    return asdict(cfg)
