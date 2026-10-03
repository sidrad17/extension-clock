"""Flow Clock parameters (PREREG_FLOWCLOCK.md, tag `prereg-flowclock`, commit 8c41154).

`config/settings.py` stays locked at `gate1-prereg`, so the second pre-registration's numbers live here. Every value
below is stated in PREREG_FLOWCLOCK.md; shared parameters (VOL_LOOKBACK, CAPITAL, CASH_COST_BP, COST_STRESS,
NOTIONAL_CAP, FOMC_HALF, DD_RULE, DD_MULT, ZSCORE_MIN_MONTHS, HAC_LAGS, BOOT_N, SEED) come from settings.py unchanged.
"""

FC_TENORS = ["DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]   # CMT series an auction maps to
PRE_DAYS = 5              # pre window: close of A-5 to close of A (bond business days)
POST_DAYS = 5             # post window: close of A to close of A+5
ZS_PRIOR = 6              # zS_e uses the previous 6 auctions of the same tenor
ZS_BREAK_DAYS = 365       # a gap longer than this between consecutive auctions of a tenor restarts its prior list
ZS_CLIP = 3.0             # zS_e clipped to [-3, 3]; +/-3 when the 6 prior amounts are all equal and the amount differs
WS_CLIP = (0.0, 2.0)      # size-weighted supply leg: w_e = clip(1 + zS_e, 0, 2); w_e = 1 when zS_e is missing
SA_FROM_T = (8, 4)        # SA_m: auctions with A in [T-8, T-4]
SA_START = "1990-01"      # first month of SA_m (as the index rebuild), so zA_m starts 1993-01 after 36 months
LEG_RISK = 0.0025         # each supply leg: a 1-sd 5-day move of the tenor's yield = 0.25% of capital
LEG_HOLD_DAYS = 5         # holding period used in that sizing (both legs hold 5 bond days)
POST_LYZ_START = "2009-01-01"   # after Lou, Yan & Zhang's 1980-2008 sample, through IS_END
EVENT_PATH = (-10, 10)    # auction event-path figure: A-10 .. A+10
