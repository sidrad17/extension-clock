# The Extension Clock

Gator Quant Hacks 2026, Systematic Trading track.

> Placeholder. Results table, setup commands, the reproduce command, sources, runtime and the open dataset
> description are filled in at Phase 8 from `outputs/results.json`.

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
