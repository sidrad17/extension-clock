# Pre-registration addendum (before any return)

Written on Oct 3, 2026, after the STOP 2 review of the index rebuild. The commit tagged `prereg-addendum` is the
timestamp of record.

**When.** This addendum was written after we saw the structure of forced duration demand (FDD) in the STOP 2
validation output, and before any return was computed. At the tagged commit, no month-end window return, return
regression, Sharpe ratio or strategy result exists in the repository: `src/signals.py`, `src/tests_h.py`,
`src/stats.py` and `src/backtest.py` are still empty stubs, and `runs/trials.csv` does not exist.

**What does not change.** `HYPOTHESIS.md` and `config/settings.py` are unchanged since `gate1-prereg`
(`git diff gate1-prereg -- HYPOTHESIS.md config/settings.py` prints nothing). The headline result stays the
pre-registered H1: the slope b in R_m = a + b z_m + e_m, with its Newey-West 95% interval. The second headline (net
Sharpe, forecast-sized vs calendar-only), the predictions, the kill conditions, every parameter and the test window
are also unchanged. This addendum only adds analyses reported beside H1 and records two data decisions.

## 1. Why: FDD is concentrated in refunding months

At STOP 2, mean FDD by calendar month (`outputs/tables/validation_seasonality.csv`, and "Mean by calendar month"
in `outputs/tables/validation.md`) is far larger in the refunding months (February, May, August, November) than in
the other eight months. Both parts contribute: the quarterly refundings add the new long issues, and the coupons of
past refunding issues are paid in the same months. So a positive H1 slope could reflect a
refunding-month calendar effect rather than a dose-response between months of similar type. The analyses below
separate the two. They were chosen before any return was seen.

## 2. Analyses reported beside the pre-registered H1

Common to all three: R_m (10-year window excess return, T-4 to T, in %), z_m (FDD standardized on past months only,
as in H1), Newey-West standard errors with 3 lags (`HAC_LAGS`), monthly observations. Each is reported for the
in-sample period (1993-01 to 2024-09), the post-publication sub-sample (2019-01 to 2024-09) and the test window
(2024-10 to 2026-09, once, after `gate2-frozen`), with the same outputs as H1: coefficient, 95% CI, t, n, and the
same model in yield terms (-Δy, bp). REF_m = 1 in February, May, August and November (the refunding-month dummy of
CLAUDE.md 7.7), else 0.

**(a) H1 with a refunding-month dummy.**
R_m = a + b z_m + c REF_m + e_m. Report b and c.
Prediction: b > 0.

**(b) Slope within refunding months and within other months.**
R_m = a_0 + a_1 REF_m + b_ref (z_m × REF_m) + b_oth (z_m × (1 − REF_m)) + e_m, estimated on the full monthly series
so the Newey-West lags run in calendar time. The point estimates equal those of separate regressions within each
group. z_m is the H1 z, not re-standardized within groups. Report b_ref and b_oth with n per group.
Prediction: b_ref > 0 and b_oth > 0.

**(c) Surprise extension (moved from Tier 3 to Tier 2).**
surprise_m = Ext_m − mean(Ext in the same calendar month over the 3 prior years) (CLAUDE.md 7.7). zs_m = surprise
standardized on past months only, with at least `ZSCORE_MIN_MONTHS` (36) prior values, the same rule as z_m.
R_m = a + b_s zs_m + e_m. The surprise exists from 1993-01 (the rebuild starts 1990-01), so zs_m, and this test,
start in 1996-01.
Prediction: b_s > 0.

**How we will read them.** These predictions follow from the same mechanism as H1. They are not new kill conditions
and they do not replace the headline. If H1's b > 0 while (a)'s b and both within-group slopes in (b) are centered
on zero, we will report that the H1 slope is explained by the refunding calendar, not by the size of forced demand
within comparable months.

Results go to `outputs/results.json` under `H1_addendum` with keys `refunding_dummy`, `within_refunding`,
`within_other` and `surprise`, in `in_sample`, `post_publication` and `oos`.

## 3. Index rebuild: Fed holdings deducted by CUSIP

The Bloomberg US Treasury Index deducts Fed SOMA holdings from amounts outstanding, "both purchases at issuance and
net secondary market transactions" (index methodology). The STOP 2 rebuild deducted only the Fed's purchases at
auction (the auctions feed), so amounts in the QE years were too large. Before any return was computed, we changed
the rebuild (`src/index_rebuild.py`, "Fed holdings"; data in `src/data/soma.py`):

- **Data.** NY Fed Markets Data API, SOMA Treasury notes-and-bonds holdings by CUSIP (keyless), weekly from
  2003-07-09. Stored in `data/snapshot/soma_notesbonds.csv` and `soma_asof_dates.csv`, covered by
  `CHECKSUMS.sha256` and described in `VINTAGE.md`. The snapshot keeps every as-of date the rebuild can use, from
  T−0 to T−6 of each month: the NOW universe, E = T−4, and the pre-registered entry grid T−2 to T−6.
- **Rule.** A CUSIP's index amount = its auctioned par − its SOMA holding on the SOMA date s. A tranche auctioned by
  E but issued after s counts net of the Fed's auction add-on, which is not in the holdings yet. A tranche auctioned
  after E still counts at `offering_amt`.
- **Point in time.** The NY Fed releases holdings the day after the as-of date and does not publish the time of day.
  So a SOMA date counts as known at the close of a day X only if it falls before the bond business day preceding X.
- **Which date for each universe.** NEXT(m), the forecast of the index after the rebalance, uses the SOMA date known
  at E(m). NOW(m) is the index during month m, whose amounts were fixed at the previous rebalance, so it uses the SOMA
  date known at T(m−1). Ext therefore includes the duration the Fed bought or sold during the month, which the index
  passes to trackers at T(m). A diagnostic with NOW also at E(m) is reported in the validation only
  (`*_now_at_entry` in `outputs/tables/validation_soma_eras.csv`). It is not tested against returns.
- **Before 2003-08.** The NY Fed history starts 2003-07-09. The first month whose NOW date has holdings is 2003-08.
  Earlier months keep the auction-only rule: `total_accepted − soma_accepted` from 2008-04, `offering_amt` before.
  For 1993-01 to 2003-07 this deducts no secondary-market Fed holdings. Each month uses one rule for both universes,
  and the open dataset records it per month (`soma_rule`, `soma_date_now`, `soma_date_next`).
- **Sensitivity.** `DEDUCT_SOMA = False` (no deduction) stays the grid cell it was. The auction-only rule is not
  added to the grid. Its series is the STOP 2 output (commit `0c06441`) and the "before" columns of the validation.
- **Validation.** Re-run in `outputs/tables/validation.md`: the par gap table (gross par, unchanged by construction),
  the Fed's holdings beside it, a CUSIP-level check against the Fed's own percent-outstanding field, the 2015–2024
  duration table before vs after, and mean Ext and FDD by era before vs after. The CUSIP check shows that the
  remaining par gap against MSPD is Treasury buybacks (2000–2002, and again from 2024), which auction records do not
  show. We do not correct for buybacks.

## 4. Sample

The rebuild starts in 1990-01, so the 36-month z warm-up (`ZSCORE_MIN_MONTHS`) is complete before the first
in-sample month. Months 1990-01 to 1992-12 serve only as the past for z_m and for the surprise's 3-year mean. They
never enter a test. The in-sample period stays 1993-01 to 2024-09 (`IS_START`, `IS_END`). The test window stays
2024-10 to 2026-09.
