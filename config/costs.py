"""Cost assumptions and the published reference they are compared with (Phase 4b, team decision of Oct 3, 2026).

The cash cost, settings.CASH_COST_BP = 0.5 bp of yield per round trip (pre-registered at gate1-prereg), is OUR
ASSUMPTION; no source fixes it. For comparison, results.json["costs"] converts a published on-the-run spread into
yield terms with the par-bond duration of src/bonds.py (src/report.py::cost_block).

Reference (read from the paper on 2026-10-03): Fleming, M. J. (2003), "Measuring Treasury Market Liquidity",
Federal Reserve Bank of New York Economic Policy Review 9(3), 83-108, Table 3: mean daily interdealer bid-ask
spreads of the on-the-run notes, December 30, 1996 to March 31, 2000, from GovPX data, in 32nds of a point (a point
is 1% of par). https://www.newyorkfed.org/medialibrary/media/research/epr/03v09n3/0309flempdf.pdf
A round trip crosses one full spread (buy at the offer, sell at the bid), so the published spread is compared with
CASH_COST_BP directly. Interdealer spreads are the narrowest in the market; customer trades, off-the-run issues,
1993-1996 and stressed days cost more. The 2x cost results (1 bp) are reported beside the 1x ones.
"""

FLEMING_2003 = {
    "citation": "Fleming, M. J. (2003). Measuring Treasury Market Liquidity. FRBNY Economic Policy Review 9(3), "
                "83-108, Table 3",
    "url": "https://www.newyorkfed.org/medialibrary/media/research/epr/03v09n3/0309flempdf.pdf",
    "sample": ["1996-12-30", "2000-03-31"],
    "data": "GovPX, interdealer, on-the-run notes, mean of daily average spreads",
    "spread_32nds": {"DGS2": 0.21, "DGS5": 0.39, "DGS10": 0.78},     # mean; medians 0.20, 0.37, 0.73
}
