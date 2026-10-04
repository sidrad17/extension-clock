# The Extension Clock

Gator Quant Hacks 2026, Systematic Trading track.

> Placeholder. The results table, sources and the open dataset description are filled in at Phase 8 from
> `outputs/results.json`.

## Reproduce

You need git and Python 3.11 or later (checked with 3.12). No API key and no `.env` file.

```bash
git clone https://github.com/sidrad17/extension-clock.git && cd extension-clock
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

## Model

A systematic strategy with three parts, all fixed in advance:

1. **Signal model:** a structural forecast of the duration that Treasury index funds are forced to buy at each
   month-end (FDDₘ = Extₘ + cₘ × D(U next)), rebuilt from public Treasury auction records.
2. **Predictive model:** Rₘ = a + b · zₘ + εₘ, where zₘ is forced demand standardized on past months only. Tested
   in-sample, after publication, and once on an untouched 2-year test window.
3. **Sizing model:** wₘ = min( max(1 + zₘ, 0), 2 ), applied to a position scaled so a normal 4-day move costs 1% of capital.

No machine learning, by design: about 380 monthly observations are too few to fit one without overfitting, and every
rule here can be read and checked in the code.

## Pre-registration

The hypothesis ([HYPOTHESIS.md](HYPOTHESIS.md)) and every parameter ([config/settings.py](config/settings.py)) were committed
and tagged `gate1-prereg` (commit `745354e`) at 1:02 AM ET on Oct 3, 2026, before any data download or return analysis.
Neither file has changed since. To check: `git diff gate1-prereg -- HYPOTHESIS.md config/settings.py` prints nothing.

After reviewing the index rebuild and before computing any return, we added
[PREREG_ADDENDUM.md](PREREG_ADDENDUM.md) (tag `prereg-addendum`). It adds three analyses reported beside H1, because
forced demand is concentrated in refunding months, and records the Fed-holdings deduction by CUSIP. The headline
stays the pre-registered H1.
