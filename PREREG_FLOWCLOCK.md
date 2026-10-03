# Pre-registration addendum 2: the supply side ("Flow Clock")

**When.** Written on Oct 3, 2026, after the Phase 3 results and before any return around a Treasury auction was
computed. The commit tagged `prereg-flowclock` is the timestamp of record.

Phase 3 found that the month-end rally is real (mean 0.194% per T-4..T window, t = 4.48, larger than all 1,000
random windows) but that forced index demand does not predict its size (H1 b = 0.011, t = 0.17), and forecast-sized
does not beat calendar-only (net Sharpe 0.57 vs 0.63). Those results stand exactly as reported. `HYPOTHESIS.md`,
`PREREG_ADDENDUM.md` and `config/settings.py` are unchanged. This file adds a second, separate hypothesis and a
combined strategy; it changes nothing that was already tested.

## Hypothesis (structural constraint + liquidity provision)

Primary dealers must take down new Treasury supply at each coupon auction and hedge it in advance, and their balance
sheets are limited, so the market pays them a concession to absorb it. We expect the yield of the auctioned maturity
to rise over the 5 bond business days before each nominal coupon auction and to fall over the 5 days after, by more
when the auction is larger than recent auctions of the same maturity.

- **Who is on the other side:** dealers and other bidders who must absorb and then distribute new supply, and who pay
  for immediacy. After the auction we buy the inventory they need to distribute.
- **Why it persists:** the auction calendar is public, but warehousing duration around auctions uses scarce dealer
  balance sheet and carries rate risk several times the size of the concession.
- **Prior evidence (not our contribution):** Lou, Yan & Zhang (2013, Review of Financial Studies): 2-year note yields
  rise 2.53bp in the 5 days before auctions and fall 2.32bp in the 5 days after; the effect grows with offering
  size; a long-short strategy has an annualized Sharpe of 1.08 over 1980-2008. Fleming, Liu & Nguyen (NY Fed Staff
  Report 1188, 2026): yields still rise before auctions and reverse after; price pressure has not increased recently.

## What is new (extend, then cite)

1. A test after Lou-Yan-Zhang's sample (2009-01 to 2024-09) and on the untouched test window (2024-10 to 2026-09).
2. The interaction with month-end: the 2-, 5- and 7-year auctions now sit in the last week of the month, so part of
   the month-end rally may be post-auction reversal. This could also explain why forced index demand did not matter.
3. One netted book that trades both sides of the Treasury flow calendar: auction supply and month-end demand.
4. A tradeable futures version and a volume-based capacity estimate (as for the month-end leg).

## Definitions (fixed now)

- **Events:** every nominal fixed-rate coupon auction in `data/snapshot/auctions.csv`, new issues and reopenings;
  exclude TIPS (`inflation_index_security` = Yes), FRNs (`floating_rate` = Yes) and bills. The auction date A must be a
  bond business day. Tenor = the security's original term (not the remaining term of a reopening), mapped to the
  nearest CMT series among DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 (ties go to the longer). An event is skipped if
  its tenor's yield is missing anywhere in its windows; skipped events are counted and reported.
- **Windows:** pre = close of A-5 to close of A; post = close of A to close of A+5 (bond business days). Return =
  constant-maturity excess return of that tenor from `src/returns.py`, in %.
- **Size signal:** zS_e = (offering_amt - mean of the previous 6 auctions of the same tenor) / their standard deviation,
  past auctions only, at least 6 prior. Auctions before a gap of more than one year between consecutive auctions of
  the tenor do not count as prior (in the data: 7-year 1993-2009, 30-year 2001-2006, 3-year 1998-2003 and 2007-2008,
  20-year 1986-2020), so a restarted tenor is compared only with recent auctions. If the 6 prior amounts are all equal
  (standard deviation 0, about a quarter of in-sample auctions), zS_e = 0 when the amount is unchanged and +3 or -3
  (the sign of the change) otherwise; every zS_e is clipped to [-3, 3]. An event with fewer than 6 prior auctions has
  no zS_e: it stays in H6a-b and the calendar leg, is left out of H6c, and gets w_e = 1. For the pre window the size
  must be known at the close of A-5: use the event's offering_amt only if `announcemt_date` is on or before A-5;
  otherwise use the previous same-tenor auction's.
- **Month-end interaction:** SA_m = sum over events with A in [T-8, T-4] of offering_amt x the tenor's modified duration
  (supply whose post-auction week overlaps the month-end window). zA_m = SA_m standardized on past months only, with
  the same 36-month minimum as z_m.

## Tests

- **H6a:** mean pre-window excess return < 0 (yields rise before auctions).
- **H6b:** mean post-window excess return > 0 (yields fall after).
- **H6c (dose-response):** event long-short return (post minus pre) = alpha + beta x zS_e + e; prediction beta > 0.
  The long-short enters at A-5, so zS_e here is the size known at the close of A-5 (the pre-window rule above).
- H6a-c use standard errors clustered by calendar week, since events in the same week share rate shocks.
- **H7 (interaction):** R_m = a + c x zA_m + e_m, with R_m as in H1 and Newey-West with 3 lags. Prediction: c > 0 and
  a > 0 (part of the month-end rally is post-auction reversal, and a month-end component remains).
- **Samples:** in-sample 1993-01 to 2024-09; after Lou-Yan-Zhang 2009-01 to 2024-09; test window 2024-10 to 2026-09,
  run once after `gate2-frozen`. An event belongs to a sample when its whole window, A-5 to A+5, falls inside it, so
  no in-sample window reaches past 2024-09-30.

## Strategy: the Flow Clock book

- **Supply leg:** for every event, short the tenor over the pre window and long it over the post window, the same size
  every event (calendar version). Each leg is sized in DV01 so a 1-sd 5-day move equals 0.25% of capital, using that
  tenor's daily yield-change volatility over the 60 bond business days ending the day before entry.
- **Demand leg:** the month-end calendar-only trade exactly as already tested. Under the pre-registered rule, when the
  forecast adds nothing we carry the base effect.
- **Book:** positions summed in DV01 by tenor each day; costs charged on net DV01 traded at the cash cost in
  `settings.py` (and at 2x as a stress test). The same risk rules apply: scheduled-FOMC half size, drawdown rule, 3x
  notional cap.
- **Headline 3:** net Sharpe of the Flow Clock book vs the demand leg alone. Secondary: the size-weighted supply leg,
  w_e = clip(1 + zS_e, 0, 2), vs the calendar supply leg.

## What kills it

- The pre or post mean has the wrong sign in-sample.
- The effect is absent in 2009-2024.
- The Flow Clock book does not beat the demand leg alone, net of costs.
- It disappears at 2x costs.
