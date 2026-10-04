"""H8: dealer balance sheets and the auction effect (PREREG_DEALERS.md, tag `prereg-dealers`, commit 15f67bc).

An explanation test, not a trade: no rule, signal, sizing or cost changes either way. Parameters: config/dealers.py.
Data: src/data/pd_positions.py (weekly primary dealer net positions in Treasury coupons excluding TIPS, all
maturities, $ millions, with the NY Fed series break of each week).

* Publication (PREREG_DEALERS.md "Point in time"): the NY Fed posts on Thursdays at about 4:15 PM ET, after the
  ~3:30 PM CMT marks. The API has no release dates, so a release's publication date = the first bond business day
  on or after its as-of date + RELEASE_LAG_DAYS (8) calendar days. A release is known at the close of X only if its
  publication date is before X. Releases whose publication date falls after the bond calendar are never known.
* zD at X (= A-5): X_0 = the latest release known at X; X_1..X_52 = the ZD_PRIOR (52) releases before it in the same
  series segment (each NY Fed series break starts a segment). zD = (X_0 - mean) / sd (ddof 1), clipped to
  [-ZD_CLIP, ZD_CLIP]; if the 52 values are all equal, 0 when X_0 equals them and +/-ZD_CLIP otherwise (as zS).
  Flags: "ok", "none_known" (no release known yet), "segment_short" (fewer than 52 earlier releases in X_0's segment).
* Events: the H6c events (in-sample, not skipped, R_pre and R_post defined, zS_pre known), of which H8 uses those
  with a zD.
* H8: LS = a + b zS_pre + c zD by OLS, standard errors clustered by the Monday-Sunday week of A
  (src/flowclock.cluster_ols: statsmodels CR1, normal intervals, as H6). One-sided p = 1 - Phi(t_c). Pass: c > 0 and
  p < P_PASS (0.05).
* Secondary (reported, not pass conditions): the same regressors with the short pre leg's return (-R_pre) and the
  post leg's return (R_post) as the dependent variable (c_pre + c_post = c, since LS = -R_pre + R_post), and the H8
  regression within each decade of A (src/flowclock.DECADES; 2020-2024 ends at IS_END). A cell with fewer than 5
  events, fewer than 3 clusters or a single zD value reports its counts only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.dealers import P_PASS, RELEASE_LAG_DAYS, ZD_CLIP, ZD_PRIOR
from src.flowclock import DECADES, cluster_ols

MIN_CELL_EVENTS, MIN_CELL_CLUSTERS = 5, 3


def publication_dates(asof: pd.DatetimeIndex, cal, lag_days: int = RELEASE_LAG_DAYS) -> pd.DatetimeIndex:
    """First bond business day on or after as-of + lag_days; NaT beyond the bond calendar."""
    target = pd.DatetimeIndex(asof) + pd.Timedelta(days=lag_days)
    k = cal.days.searchsorted(target, side="left")
    out = [cal.days[i] if i < len(cal.days) else pd.NaT for i in k]
    return pd.DatetimeIndex(out)


def zd_at(dates, weekly: pd.DataFrame, cal, n_prior: int = ZD_PRIOR, clip: float = ZD_CLIP) -> pd.DataFrame:
    """zD known at the close of each date (module docstring). weekly: position_musd and segment by as-of date.
    Returns, per date: zD, flag, asof (of X_0), segment, n_prior_in_segment, published (X_0's publication date)."""
    w = weekly.sort_index()
    pub = publication_dates(w.index, cal)
    if not pub.dropna().is_monotonic_increasing:
        raise ValueError("publication dates are not increasing")
    pub_ns = pub.to_numpy(dtype="datetime64[ns]")
    known_ok = ~np.isnat(pub_ns)
    vals = w["position_musd"].to_numpy(float)
    seg = w["segment"].to_numpy(object)
    first_of_seg = pd.Series(np.arange(len(w))).groupby(seg).transform("min").to_numpy()
    rows = []
    for x in pd.DatetimeIndex(dates):
        k = int(np.searchsorted(pub_ns[known_ok], np.datetime64(x), side="left"))       # published strictly before x
        if k == 0:
            rows.append({"zD": np.nan, "flag": "none_known", "asof": pd.NaT, "segment": None,
                         "n_prior_in_segment": 0, "published": pd.NaT})
            continue
        i = int(np.flatnonzero(known_ok)[k - 1])
        n_seg = i - int(first_of_seg[i])
        r = {"asof": w.index[i], "segment": seg[i], "n_prior_in_segment": n_seg, "published": pub[i]}
        if n_seg < n_prior:
            rows.append({**r, "zD": np.nan, "flag": "segment_short"})
            continue
        p = vals[i - n_prior:i]
        x0 = vals[i]
        if np.ptp(p) == 0:
            z = 0.0 if x0 == p[0] else float(np.sign(x0 - p[0]) * clip)
        else:
            z = float(np.clip((x0 - p.mean()) / p.std(ddof=1), -clip, clip))
        rows.append({**r, "zD": z, "flag": "ok"})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


def _reg(y: pd.Series, e: pd.DataFrame) -> dict:
    X = np.column_stack([np.ones(len(e)), e["zS_pre"].to_numpy(float), e["zD"].to_numpy(float)])
    r = cluster_ols(y.to_numpy(float), X, e["week"].to_numpy(), ["a", "zS", "zD"])
    c = r["params"]["zD"]
    return {"c": {**{k: c[k] for k in ("b", "se", "t", "ci")}, "p_one_sided": float(norm.sf(c["t"]))},
            "b_zS": {k: r["params"]["zS"][k] for k in ("b", "se", "t", "ci")},
            "a": {k: r["params"]["a"][k] for k in ("b", "se", "t", "ci")},
            "n": r["n"], "n_clusters": r["n_clusters"],
            "sample": [str(e["A"].min().date()), str(e["A"].max().date())]}


def h8(ev: pd.DataFrame, zd: pd.DataFrame) -> dict:
    """H8 on the in-sample events (rows of the Flow Clock event table with event_returns() columns, A, week,
    zS_pre, skipped, pre_entry). zd: zd_at() indexed by each event's A-5 (pre_entry)."""
    h6c = ev[~ev["skipped"].astype(bool) & ev["R_pre"].notna() & ev["R_post"].notna() & ev["zS_pre"].notna()]
    z = zd.reindex(h6c["pre_entry"])
    e = h6c.assign(zD=z["zD"].to_numpy(), zd_flag=z["flag"].to_numpy(), zd_segment=z["segment"].to_numpy(),
                   zd_asof=z["asof"].to_numpy())
    use = e[e["zD"].notna()]
    main = _reg(use["LS"], use)
    c = main["c"]
    passed = bool(c["b"] > 0 and c["p_one_sided"] < P_PASS)
    decades = {}
    for name, (lo, hi) in DECADES.items():
        d = use[(use["A"] >= pd.Timestamp(lo)) & (use["A"] <= pd.Timestamp(hi))]
        if len(d) < MIN_CELL_EVENTS or d["week"].nunique() < MIN_CELL_CLUSTERS or d["zD"].nunique() < 2:
            decades[name] = {"n": int(len(d)), "n_clusters": int(d["week"].nunique()),
                             "status": "too few events, weeks or distinct zD values for the regression"}
        else:
            decades[name] = _reg(d["LS"], d)
    word = "passes" if passed else "fails"
    statement = (f"H8 {word}: across {main['n']} auctions from {main['sample'][0]} to {main['sample'][1]}, a 1-sd "
                 f"higher primary dealer coupon position at A-5 goes with a {c['b']:+.3f} percentage-point change in "
                 f"the auction long-short return (t = {c['t']:.2f}, one-sided p = {c['p_one_sided']:.3f}; pass needs "
                 f"c > 0 and p < {P_PASS:g}).")
    zu = use["zD"]
    return {
        "prediction": f"c > 0, one-sided p < {P_PASS:g}",
        "model": "LS = a + b x zS_pre + c x zD; LS = R_post - R_pre (%), cash constant-maturity excess returns, before "
                 "costs; SEs clustered by the Monday-Sunday week of A (CR1, normal); p = 1 - Phi(t_c)",
        **main, "pass": passed, "statement": statement,
        "secondary": {"pre_leg": {**_reg(-use["R_pre"], use), "y": "-R_pre (the short pre leg's return, %)"},
                      "post_leg": {**_reg(use["R_post"], use), "y": "R_post (%)"},
                      "by_decade": decades,
                      "note": "reported, not pass conditions; c_pre + c_post = c"},
        "events": {"n_h6c": int(len(h6c)), "n_used": int(len(use)),
                   "n_without_zD": {k: int(v) for k, v in e.loc[e["zD"].isna(), "zd_flag"].value_counts()
                                    .sort_index().items()},
                   "by_segment_of_X0": {k: int(v) for k, v in use["zd_segment"].value_counts().sort_index().items()},
                   "by_tenor": {k: int(v) for k, v in use["tenor"].value_counts().sort_index().items()}},
        "zD_used": {"mean": float(zu.mean()), "sd": float(zu.std(ddof=1)), "min": float(zu.min()),
                    "max": float(zu.max()), "share_clipped": float((zu.abs() >= ZD_CLIP).mean()),
                    "latest_asof_used": str(pd.Timestamp(use["zd_asof"].max()).date())},
    }
