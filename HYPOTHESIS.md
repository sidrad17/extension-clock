# Hypothesis (committed before any backtest)

Edge source: structural constraint + liquidity provision.

We expect US Treasury notes and bonds to rise in price over the last four trading days of the month, by more when
the forced duration demand is larger, because index-tracking bond funds must buy duration at month-end, when the
Treasury index adds new issues and reinvests the month's coupon cash. The edge persists because the index rules are
fixed and trackers are judged on matching the index, not on when they trade. If true, returns should scale with the
predicted forced demand and partly reverse in the first days of the next month. It fails if the size of the forced
demand does not matter, or the gain does not reverse.

Forced duration demand: FDD_m = Ext_m + c_m × D_next, where Ext_m is the jump in index duration at the rebalance
(new issues auctioned by the rebalance date added, bonds under 1 year removed) and c_m is the month's coupon cash
as a share of index value.

Headline numbers (fixed in advance):
1. Dose-response slope b in R_m = a + b z_m + e_m (10-year window excess return on the standardized forced
   duration demand, Newey-West 95% interval). Prediction: b > 0, and both components point the same way.
2. Net Sharpe of forecast-sized (w_m = clip(1 + z_m, 0, 2)) vs calendar-only, in-sample (1993-01 to 2024-09) and
   test window (2024-10 to 2026-09).

Predictions: H1 b > 0; H2 maturity buckets receiving more predicted buying see larger yield falls; H3 partial
reversal in the first 3 business days of the next month, larger after months of heavy forced demand; H4 forecast-sized beats
calendar-only in-sample, after publication (2019-01 to 2024-09) and in sign in the test window; H5 pension pressure
adds a little, quarter-end adds nothing beyond forced demand. Placebo: windows on business days 4–7 show neither
the gain nor the relation. Stretch: the same slope appears in the TIPS index, which has a different issuance calendar.

Power note: with a true Sharpe near 1, 24 test-window months give t ≈ 1.4; the test window can confirm the sign,
not significance.

Kill conditions: b ≤ 0 or centered on zero; no reversal; forecast-sized ≤ calendar-only; effect only in the 1990s;
vanishes at 2× costs.

Known prior evidence (not our contribution): Hartley & Schwarz document the base month-end effect for 1990–2018.
