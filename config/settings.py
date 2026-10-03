IS_START, IS_END   = "1993-01-01", "2024-09-30"   # in-sample (cash bonds)
FUT_START          = "2010-07-01"                 # in-sample futures start (GLBX.MDP3 history begins 2010-06)
POSTPUB_START      = "2019-01-01"                 # post-publication sub-sample, through IS_END
OOS_START, OOS_END = "2024-10-01", "2026-09-30"   # test window: last 2 years (track rule), run once
ZSCORE_MIN_MONTHS  = 36                           # before this, w_m = 1

ENTRY_OFFSET = 4          # enter at close of T-4 (T = last bond-market business day of the month)
EXIT_OFFSET  = 0          # exit at close of T
REVERSAL_DAYS = 3         # H3: first 3 business days of next month
PLACEBO_BDAYS = (4, 7)    # placebo window: enter close of bday 3, exit close of bday 7 (clear of reversal and the 15th)
N_RANDOM_PLACEBO = 1000; SEED = 7; BOOT_N = 10_000; HAC_LAGS = 3

HEADLINE_TENOR = "DGS10"
TENORS = {"1-3y": "DGS2", "3-7y": "DGS5", "7-10y": "DGS10", "10-20y": "DGS20", "20y+": "DGS30"}
CURVE_KNOTS = ["DGS1","DGS2","DGS3","DGS5","DGS7","DGS10","DGS20","DGS30"]   # interpolation, flat beyond ends
RF_SERIES = "DTB3"

# Index rules (Bloomberg US Treasury Index, as summarized for SPTB)
MIN_MATURITY_YEARS = 1.0          # measured at the rebalance date
MIN_PUBLIC_AMOUNT  = 300e6        # after Fed (SOMA) holdings
INCLUSION_RULE = "auctioned_by_rebalance"  # methodology: issued (auctioned) on/before the rebalance date T qualifies, settled or not
                                           # sensitivity: "settled_by_month_end" (issue_date <= last calendar day of month)
REINVEST_COUPONS = True   # FDD includes coupon cash held by the index during the month (sensitivity: False = extension only)
DEDUCT_SOMA = True                # sensitivity: False
EXCLUDE = {"TIPS": True, "FRN": True, "callable": True}

W_CLIP = (0.0, 2.0)               # w_m = clip(1 + z_m, 0, 2)
CAPITAL = 10_000_000
RISK_PER_TRADE = 0.01             # 1-sigma 4-day loss = 1% of capital
VOL_LOOKBACK = 60                 # business days of 10y yield changes, ending T-5
NOTIONAL_CAP = 3.0                # x capital
FOMC_HALF = True                  # risk rule 2
DD_RULE = True; DD_MULT = 2.0     # risk rule 5: drawdown > 2x expected yearly vol -> half size until new high
CASH_COST_BP = 0.5                # yield bp per round trip (cash version)
FUT_COMMISSION_RT = 2.0           # USD per contract round trip; plus 1 tick
COST_STRESS = 2.0
ROLL_BUFFER_BDAYS = 5             # trade contract whose first intention day > exit + 5 bdays
DV01_LOOKBACK = 60
ADV_LOOKBACK = 20; ADV_CAP = 0.05
FUTURES = ["ZT","ZF","ZN","TN","ZB","UB"]; HEADLINE_FUTURE = "ZN"
FUT_YIELD_MAP = {"ZT":"DGS2","ZF":"DGS5","ZN":"DGS10","TN":"DGS10","ZB":"DGS20","UB":"DGS30"}
SEC_UA = None  # not used in this project
