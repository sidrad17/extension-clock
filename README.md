# The Flow Clock

Gator Quant Hacks 2026, Systematic Trading track. **Quant note:** [`docs/Flow-Clock-Note.pdf`](docs/Flow-Clock-Note.pdf).

Two groups must trade US Treasuries on dates known in advance: primary dealers absorbing new bonds at coupon auctions,
and bond index funds rebalancing at month-end. Their forced trades move prices for a few days. The Flow Clock is a
calendar of trades around those dates. It never forecasts the level of rates. Every hypothesis was committed and
tagged in git before its first return was computed. The 2-year test window (2024-10 to 2026-09) was run once, on
code frozen at the `gate2-frozen` tag.

**What we claim.** A month-end trade, long 10-year note futures (ZN) from T−4 to T, kept a positive net Sharpe out of
sample: 0.79 in-sample (2010-07 to 2024-09) and
0.64 in the test window (0.55 at 2× costs,
t = 0.91). The window confirms the sign; 24 months cannot establish significance.
Capacity: net Sharpe halves at about $1.4B.

**What we do not claim.**
- That the auction trade is tradable. It failed its pre-registered kill condition in futures in-sample, and the cash
  version lost money in the test window.
- That dealer inventory drives the auction effect. H8 failed in both samples.
- That we know which forced flow drives the month-end rally. In-sample, index funds' forced duration demand (FDD) did
  not predict the rally's size, and late-month auction supply explained part of it (H7). In the test window both
  flipped: FDD had the predicted sign (b = 0.88, t = 2.99), and H7 reversed
  (t = −2.21). With 24 months against 381, we report the labels as written and leave the
  mechanism open.

## Results

Net of costs. Max DD = maximum drawdown of the excess-return NAV, in-sample. "Curve" = cash constant-maturity returns
from FRED yields: a fitted curve, not directly tradable. Test-window labels follow the rules committed before the run
(`CLAUDE.md` section 17): "consistent" means the sign matches in-sample.

| Strategy | In-sample | Sharpe 1× | Sharpe 2× | Max DD | Test 1× | Test 2× | Test label | Tradable? |
|---|---|--:|--:|--:|--:|--:|---|---|
| **Month-end, ZN futures** | 2010-07 to 2024-09 | 0.79 | 0.71 | 6.7% | 0.64 | 0.55 | consistent | **yes** |
| Month-end, cash, calendar-only | 1993-01 to 2024-09 | 0.63 | 0.45 | 8.0% | 0.71 | 0.51 | consistent | curve |
| Month-end, cash, forecast-sized (FDD) | 1993-01 to 2024-09 | 0.57 | 0.42 | 10.7% | 1.20 | 1.02 | consistent | curve |
| Month-end, cash, curve-allocated | 1993-01 to 2024-09 | 0.69 | 0.53 | 7.9% | 0.73 | 0.52 | consistent | curve |
| Auction supply leg, cash | 1993-01 to 2024-09 | 0.80 | 0.46 | 9.5% | −0.68 | −1.11 | not consistent | curve |
| Flow Clock book, cash | 1993-01 to 2024-09 | 0.87 | 0.51 | 11.1% | −0.18 | −0.55 | not consistent | curve |
| Auction supply leg, futures | 2010-07 to 2024-09 | 0.29 | 0.07 | 6.5% | 0.60 | 0.37 | consistent | no (kill condition met in-sample) |

| Pre-registered test | Prediction | In-sample (1993-01 to 2024-09) | Verdict | Test window (2024-10 to 2026-09) | Label |
|---|---|---|---|---|---|
| Month-end rally, 10-year, T−4 to T | mean > 0 | +0.194%, t = 4.48; beats all 1,000 random windows | passed | +0.151%, t = 0.84 | consistent |
| H1: index funds' forced demand (FDD) sets its size | b > 0 | b = 0.011, t = 0.17 | **failed** | b = 0.875, t = 2.99, n = 24 | sign confirmed |
| H4: forecast-sized beats calendar-only | Sharpe diff > 0 | 0.57 vs 0.63 | **failed** | 1.20 vs 0.71 (p = 0.017) | pass |
| H3: reversal after month-end | R3 < 0, slope < 0 | R3 −0.063% (t = −1.23); slope +0.079 | not supported | −0.023% (t = −0.21) | consistent |
| H6a: dip before coupon auctions | mean < 0 | −0.102%, t = −3.31 | passed | −0.137% (t = −1.37) | consistent |
| H6b: bounce after | mean > 0 | +0.114%, t = 3.44 | passed | −0.057% (t = −0.74) | not consistent |
| H6c: larger auctions, larger swing | β > 0 | β = 0.135, t = 3.43 | passed | −0.161 (t = −0.75) | not consistent |
| H7: late-month auctions explain part of the month-end rally | c > 0 | c = 0.056, t = 2.35 | passed | −0.246 (t = −2.21) | not consistent |
| Flow Clock book beats the demand leg alone | Sharpe diff > 0 | 0.87 vs 0.63 (p = 0.06) | passed | −0.18 vs 0.71 | **kill condition met** |
| Auction leg survives costs in futures | not gone at 2× | 0.29 at 1×, 0.07 at 2× | **failed (kill condition)** | 0.60 at 1×, 0.37 at 2× | consistent |
| H8: dealer inventory drives the auction effect | c > 0, p < 0.05 | c = 0.006, t = 0.15, p = 0.44 | **failed** | c = −0.021, p = 0.57, n = 101 | fail |

All numbers come from `outputs/results.json` (test window: `outputs/results_oos.json`, merged into it). Figures are in
`outputs/figures/` (`equity_curve.png` in-sample, `equity_curve_oos.png` test window).

**Selection and robustness.** 519 distinct variants were tested and logged
(3,115 in-sample runs in `runs/trials.csv`). Deflated Sharpe at N = 519:
0.987 for the cash book, 0.874 for ZN month-end,
0.236 for the futures auction leg. Cash calendar-only is profitable in all 480
sensitivity-grid cells. The placebo window (business days 4–7) shows no rally in-sample.

## Model

No machine learning: about 380 monthly observations are too few to fit one without overfitting, and every rule can be
read in the code. All rules were fixed in a tagged file before their first result.

1. **Signals.** Auction size against the previous six auctions of the same maturity, known at A−5:
   zSₑ = (Sₑ − mean₆) / sd₆, clipped to ±3. Late-month supply zAₘ: standardized size × duration of auctions from T−8
   to T−4. Forced duration demand FDDₘ = Extₘ + cₘ × D(next), rebuilt point-in-time from public auction records with
   Fed SOMA holdings deducted; zₘ is its past-only z-score.
2. **Legs.** Supply: short the auctioned maturity from close A−5 to close A, long from A to A+5. Each leg is sized in
   DV01 so a 1-sd 5-day move costs 0.25% of capital. Demand: long the 10-year (ZN in futures) from T−4 to T, sized so
   a 1-sd 4-day move costs 1%. Forecast-sized variant: wₘ = min(max(1 + zₘ, 0), 2).
3. **Book.** Positions netted in DV01 by maturity each day. Futures map: 2y/3y→ZT, 5y→ZF, 7y→ZN, 10y→TN (ZN before
   2016), 20y→ZB, 30y→UB.
4. **Risk rules (each tested on and off).** 60-day volatility targeting; half size when a scheduled FOMC decision falls
   in the window; half size after a drawdown above 2× expected yearly volatility, until a new high; gross notional
   ≤ 3× capital.
5. **Costs.** Cash: 0.5bp of yield per round trip (1.4–1.8× Fleming 2003 interdealer spreads). Futures: 1 tick + $2 per
   contract. Every result is also shown at 2×.

## Known flaws (disclosed, not fixed after seeing results)

- 66 of 662 2-year (ZT) futures auction legs, in the zero-rate years (2011–14, 2020–21), got a weak DV01 fit
  (R² < 0.5), and 62 hit the 3× cap. The month-end leg is unaffected.
- ZN month-end capacity is $1.41B with most-active-contract volume and $333M with the rule
  as first written. The first reads the thin new contract in roll months. Both are reported.
- Databento flagged a few test-window days as reduced quality (for example 2025-09-17, 2025-09-24 and 2025-11-28).
  They were used as delivered.
- The intermittent sensitivity-grid difference described under Reproduce.

## Pre-registration

The hypothesis ([HYPOTHESIS.md](HYPOTHESIS.md)) and every parameter ([config/settings.py](config/settings.py)) were committed
and tagged `gate1-prereg` (commit `745354e`) at 1:02 AM ET on Oct 3, 2026, before any data download or return analysis.
Neither file has changed since. To check: `git diff gate1-prereg -- HYPOTHESIS.md config/settings.py` prints nothing.

After reviewing the index rebuild and before computing any return, we added
[PREREG_ADDENDUM.md](PREREG_ADDENDUM.md) (tag `prereg-addendum`). It adds three analyses reported beside H1, because
forced demand is concentrated in refunding months, and records the Fed-holdings deduction by CUSIP. The headline
stays the pre-registered H1.

Every tag is on GitHub; `git show <tag>` gives its commit and time:

| Tag | Commit | Committed (ET) | Fixed before any related result |
|---|---|---|---|
| `gate1-prereg` | 745354e | Oct 3, 1:00 AM | Month-end hypothesis H1–H5 and every parameter |
| `prereg-addendum` | dbfe85e | Oct 3, 3:34 AM | Fed SOMA deduction by CUSIP; refunding-month analyses |
| `prereg-flowclock` | 8c41154 | Oct 3, 4:27 AM | Auction hypothesis H6a–c, H7, the Flow Clock book and its kill conditions |
| `prereg-dealers` | 15f67bc | Oct 3, 10:27 PM | H8 dealer-inventory test |
| `gate2-frozen` | 825135d | Oct 4, 1:15 AM | Code frozen before the one-time test-window run (rules: f2d3531) |

## Test window (Gate 2)

Ran once: `RUN START` 2026-10-04T05:59:52Z at 825135d, results committed in 51fb1e1 (`runs/oos_run.log`, 14 trial rows).


The test window is 2024-10-01 to 2026-09-30. It is evaluated once by `python run_all.py --oos`, with the same code
as the in-sample run (rules: `CLAUDE.md` section 17). The team runs it after tagging the frozen code `gate2-frozen`:

```bash
GQH_DEV=1 python run_all.py --oos                  # downloads the public test-window rows, prints the Databento estimate, stops
GQH_DEV=1 python run_all.py --oos --databento-ok   # after the team approves the estimate: pulls the futures data and runs
```

Every test-window download refuses unless HEAD carries the `gate2-frozen` tag and the working tree is clean, in every
mode. The run happens once and is recorded in `runs/oos_run.log`. A second run needs `--force-rerun` and is logged
as a forced rerun. Nothing is printed or written until every block is computed.

Results land in `outputs/results_oos.json`, which is merged into `outputs/results.json` as `oos`, `flowclock.oos`,
`H8.oos` and `futures.oos`. The run also writes the `outputs/tables/*_oos.csv` tables (futures: derived tables only)
and `outputs/figures/equity_curve_oos.png`. The downloaded public rows are committed to `data/oos/` with checksums
and a vintage. A keyless clone then reproduces the block with no download and no key: the plain `python run_all.py`
merges the committed `results_oos.json`, and `python run_all.py --oos` recomputes it from `data/oos/` and the
committed futures `*_oos` tables.

The public-data snapshot (downloaded Oct 3, 2026) includes rows after 2024-09-30. Every in-sample loader cuts at
2024-09-30, the date guard refused later dates, and no test-window statistic was computed before the gate2-frozen
tag. `--oos` downloads its test-window rows fresh and never reads those rows.

## Reproduce

You need git and Python 3.11 or later (checked with 3.12). No API key and no `.env` file.

```bash
git clone https://github.com/sidrad17/flow-clock.git && cd flow-clock
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python run_all.py
```

`python run_all.py` first verifies the checksums of the committed public-data snapshot (`data/snapshot/`). It then
rebuilds `outputs/results.json` and every table in `outputs/tables/` and figure in `outputs/figures/`. To check that
it reproduced the committed numbers:

```bash
git status --short                # only outputs/results.json is listed
git diff outputs/results.json     # only meta.commit and meta.generated_utc change
```

On a fresh clone (macOS, Python 3.12.0, no `.env`) those two lines were the only change, and every table and figure
was byte-identical. On another OS the PNG bytes may differ even though no number changes.
One intermittent difference is not explained: in 1 of about 95 runs, 24 of the sensitivity grid's 480 cells, all in
one rebuild variant (entry T-5, settled by month-end, Fed holdings deducted), came out different; the cause is
unknown, and the headline numbers and every test were unaffected.

Runtime on an Apple M2 (8 cores, 16 GB): `pip install` about 20 s with a warm pip cache, `pytest -q` (no network)
about 1 min, and `python run_all.py` about 2 min. Most of that is the sensitivity grid's 20 index rebuilds, which
run in parallel (`GQH_WORKERS`, default min(8, CPUs)).

**What needs a key.** Nothing above does.

| Part | Key? | Rebuilt from |
|---|---|---|
| Index rebuild, cash results, H1-H8, Flow Clock, sensitivity grid, Deflated Sharpe, figures | no | committed public snapshot `data/snapshot/` (Fiscal Data, FRED, NY Fed, FOMC, Ken French-derived) |
| Futures numbers (`results.json["futures"]`, the futures metrics, figure 5's futures panel, the CMT switch diagnostic's cash-vs-futures part) | no | committed derived tables `outputs/tables/futures_*`, which hold no prices or raw volume (licensed data); the run prints that it used them |
| Rebuilding those derived tables from raw Databento data | **yes**: `DATABENTO_API_KEY` | `python run_all.py --futures` (below) |

To rebuild the futures tables from raw data:

```bash
cp .env.example .env              # set DATABENTO_API_KEY=...; leave GQH_DEV empty
python run_all.py --futures
```

The first `--futures` run downloads about 110 MB of CME futures data (Databento GLBX.MDP3: `statistics`,
`ohlcv-1d` and 176 one-day `definition` snapshots, 2010-06 to 2024-09) into `data/cache/`, which is git-ignored.
It prints Databento's cost estimate first and stops if the estimate is above $5 (our pull cost about $1.6). The
download is the slow part: on Oct 3, 2026 it streamed at about 10-20 KB/s, so allow 2-3 hours. With the raw files
cached, `python run_all.py --futures` takes about 2.5 min and rewrites the derived tables byte-identically.
Without a key and without the cache, `--futures` stops with a message and the plain `python run_all.py` still
reproduces every number.

`GQH_DEV=1` is for the team only. It switches on the pre-registration guards and appends every run to
`runs/trials.csv`, which changes `results.json["trials"]`. Leave it unset to reproduce.

## Repository history

The project started as "The Extension Clock". That first hypothesis failed its test, and the Flow Clock is what
survived. This repository carries the full commit and tag history of the original private repository, minus one
commit that held licensed Databento data and was force-pushed away. The original repository's GitHub activity log,
with the push time of every pre-registration tag, is available to judges on request.
