# Pre-registration addendum 3: dealer balance sheets (H8)

**When.** Written on Oct 3, 2026, after the Flow Clock and Phase 4d results and before any dealer-position data was
downloaded, loaded or plotted. The commit tagged `prereg-dealers` is the timestamp of record. Until that tag exists,
no code reads dealer positions.

The Flow Clock found the auction dose-response in cash (H6c: b = 0.135, t = 3.43, 1,665 events, 1993-01 to
2024-09). Phase 4d found that the cash-futures gap is not driven by the CMT input switch: only 11% of it sits on days
A and S, and it is larger at reopenings, where the input bond does not change. Those results stand exactly as
reported. `HYPOTHESIS.md`, `PREREG_ADDENDUM.md`, `PREREG_FLOWCLOCK.md` and `config/settings.py` are unchanged. This
file adds one explanation test, H8. It changes no rule, signal, sizing, cost or trade.

## Hypothesis

The pre-auction dip is payment to dealers for warehousing new supply on limited balance sheets. If so, the auction
effect should be larger when primary dealers already hold a lot of Treasuries compared with their recent past.

## Data

- **Source.** NY Fed Primary Dealer Statistics (FR 2004A), keyless Markets Data API
  (`https://markets.newyorkfed.org/api/pd`). Weekly, as of each Wednesday. Primary dealers' net outright positions
  in U.S. Treasury coupon securities excluding TIPS, all maturities combined, in $ millions. FRNs are excluded where
  they are reported separately.
- **Series.** In each NY Fed series break, use the key for that total if the API publishes one. Otherwise use the
  sum of that break's coupon (excluding TIPS) maturity-bucket keys. Current keys are identified from the API's
  descriptions. Older keys have no descriptions, so they are identified from their names and checked by summing the
  buckets to a published total, as `src/data/pd_volume.py` did for transactions (rule 8). The keys and that check
  are written into the build commit before any H8 return is computed. A break with no key matching this definition
  is not used.
- **Snapshot.** Public U.S. government data, committed as `data/snapshot/pd_treasury_positions.csv` with checksums
  and a `VINTAGE.md` entry.

## Point in time

- **Publication.** The NY Fed posts the weekly data on Thursdays at about 4:15 PM ET, with the previous week's
  statistics. That is after the ~3:30 PM CMT marks that define the close. The API carries no release dates
  (`src/data/pd_volume.py`). So a release's publication date is the first bond business day on or after its as-of
  date + 8 calendar days.
- **Known at A-5.** At an event, use the latest release whose publication date is before A-5. A release published
  on A-5 itself comes after that day's close.

## Dealer signal zD

- zD_e = (X_0 - mean(X_1..X_52)) / sd(X_1..X_52).
  - X_0 is the latest release known at A-5 (point-in-time rule above).
  - X_1..X_52 are the 52 releases before X_0 in the same series segment.
  - The sd uses ddof = 1, as zS does. zD is clipped to [-3, 3].
  - If the 52 values are all equal, zD = 0 when X_0 equals them and +3 or -3 (the sign of the change) otherwise, as
    for zS.
- **Breaks.** Each NY Fed series break (`/api/pd/list/seriesbreaks.json`) starts a new segment, as the issuance-gap
  rule restarts a tenor's zS history.
  - The breaks were recorded from that endpoint on Oct 3, 2026, in `src/data/pd_volume.py`: SBP2001 (1998-01-28 to
    2001-06-30), SBP2013 (2001-07-01 to 2013-03-31), SBN2013, SBN2015, SBN2022, and SBN2024 (from 2024-07-03).
  - The FR 2004 changed at each break. Before seeing the data we cannot check that the all-maturity total kept its
    definition, so every break counts as a definition break.
  - An event whose X_0 has fewer than 52 earlier releases in its segment has no zD and is left out of H8. Its counts
    are reported.

## Test

- **Events.** The H6c events: in-sample, not skipped, both window returns defined, and zS known at A-5 (`zS_pre`).
  Of these, H8 uses the events that also have a zD.
- **Event return.** The pre-registered cash supply-leg event return, as in H6c: LS_e = R_post - R_pre, in %. That is
  short A-5 -> A plus long A -> A+5, using the tenor's constant-maturity excess return, before costs.
- **H8.** LS_e = a + b x zS_e + c x zD_e + e_e, by OLS.
  - zS_e is `zS_pre`, the size known at A-5, as in H6c.
  - Standard errors are clustered by the Monday-Sunday calendar week of A, as in H6 (statsmodels CR1, normal
    distribution).
  - One-sided p = 1 - Phi(t_c).
  - **Pass: c > 0 and one-sided p < 0.05.** Otherwise H8 fails and is reported as failed.
- **Secondary (reported, not pass conditions).**
  - The same regression with the short pre leg's return (-R_pre) and with the post leg's return (R_post) as the
    dependent variable. Both use the same zS_pre and the zD known at A-5, so c_pre + c_post = c.
  - The H8 regression within each decade of A: 1993-1999, 2000-2009, 2010-2019 and 2020 to 2024-09. Each cell is
    reported with its n.
  - Also reported: a, b, N events, N weeks, first and last A, and the number of H6c events left out for lack of zD.
- **Samples.**
  - **In-sample:** events whose window A-5..A+5 lies within 1993-01-01..2024-09-30 and that have a zD. In practice
    the sample starts at the first event with 52 prior releases.
  - **Breaks:** about a year of events after each break has no zD. The SBN2024 break means in-sample H8 events end
    around 2024-06.
  - **Test window (2024-10-01 to 2026-09-30):** untouched. H8 joins the Gate 2 `--oos` run, once, after
    `gate2-frozen`. Because of the 2024-07-03 break, test-window zD exists only from about mid-July 2025.

## Rules

- **Gate.** With `GQH_DEV=1`, the dealer-position loader refuses to download or read positions unless the tag
  `prereg-dealers` exists and this file, `PREREG_FLOWCLOCK.md`, `PREREG_ADDENDUM.md`, `HYPOTHESIS.md` and
  `config/settings.py` are unchanged since that tag. Gate 1 and Gate 2 apply unchanged.
- **Trial log.** Each run appends one row to `runs/trials.csv`.
  - H1_* holds c and its 95% interval, and n holds the event count.
  - The Sharpe columns are blank because H8 is not a strategy.
  - The row has a new `config_hash`, so distinct variants rise by one.
- **No trading-rule changes either way.** H8 is an explanation test, not a new trade. A pass supports the
  dealer-warehousing explanation of the auction effect. A failure is reported as failed. The Flow Clock results and
  strategies stand unchanged in both cases.
