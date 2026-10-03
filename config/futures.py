"""Futures-layer parameters (Phase 4, CLAUDE.md section 15; team decisions of Oct 3, 2026, before any futures result).

`config/settings.py` stays locked at `gate1-prereg`; its futures parameters (FUT_START, FUT_COMMISSION_RT,
COST_STRESS, ROLL_BUFFER_BDAYS, DV01_LOOKBACK, FUTURES, HEADLINE_FUTURE, FUT_YIELD_MAP) are used unchanged.
"""

# Flow Clock supply leg: auction tenor (CMT series) -> contract (team). The month-end leg uses
# settings.HEADLINE_FUTURE = "ZN" as pre-registered (team: no TN month-end variant).
SUPPLY_CONTRACT = {"DGS2": "ZT", "DGS3": "ZT", "DGS5": "ZF", "DGS7": "ZN", "DGS10": "TN", "DGS20": "ZB",
                   "DGS30": "UB"}
FALLBACK = {"TN": "ZN", "UB": "ZB"}   # used before the product's first sizable entry date (section 15)
DV01_MIN_OBS = 50                     # valid daily changes needed among the DV01_LOOKBACK (60) ending E-1
VOLUME_LAG_BDAYS = 1                  # roll rule: highest volume on the bond day before entry
VOLUME_MAX_BACK = 5                   # if the product has no volume that day, step back up to this many bond days
FID_BDAYS_BEFORE = 2                  # FID = 2 business days before the first business day of the delivery month
ALIGNMENT_DATES = ["2016-11-09", "2020-03-09", "2022-11-10"]   # known large 10-year moves (trade-date check)
