# Cortical decoders trained on other individuals age slower than your own

**A neural decoder built from *other* animals loses less accuracy over two weeks than one built
from the subject's own earlier data. This is the opposite of the standard assumption, and it
replicates across two cohorts, two recording modalities, and three architectures.**

Paper: `paper_neurips/pooling.pdf` · ICLR-format variant: `paper_iclr/`

## The result

A decoder fit to one animal's cortex drifts, losing accuracy over days. The usual remedy is to
recalibrate on fresh data from the same subject. We measured whether that is actually the right
move.

On **BraiDyn-BC** (25 mice, 44 Allen-atlas cortical parcels, ~15-day operant protocol,
DANDI:001425), decoding four task events and measuring per-mouse AUC lost over two weeks:

| Comparison | Effect | Significance |
|---|---|---|
| Personal decoder ages more than pooled | **+0.017 AUC** | p = 5e-5, 73/100 mouse-event pairs |
| Replication, Allen Visual Behavior (23 mice, two-photon, VISp) | **+0.023 AUC** | p = 0.003, 74% of mice |
| Count-matched control (self vs one other mouse) | **+0.011 AUC** | holds on both datasets |

It is **not a data-volume effect**: at a matched training-event count, a decoder from a single
other mouse still ages less. It survives a 1-D CNN and a GRU, so it is not an artifact of one
model class. And it reproduces on an independent two-photon dataset from a different laboratory,
so it is not specific to widefield calcium or to this cohort.

The interpretation: a personal decoder overfits the individual's own drift-prone idiosyncrasies.
The early-week accuracy you give up by decoding across individuals is repaid as late-week
stability, which makes population priors a real alternative to per-subject recalibration.

## Layout

```
braidyn_*.py / cardin_*.py     per-dataset decoding experiments
pooling_*.py                   pooled vs personal decoder comparisons
aging_slope.py, full_decay.py  drift-over-time measurement
*_leakfix.py                   event-overlap leakage controls
event_overlap_check.py         verifies chained events are handled honestly
nonlinear_*.py                 1-D CNN and GRU replications
make_*_figs.py                 figure generation
*.json                         committed results for every run above
paper_neurips/ paper_iclr/     manuscripts and figures
PILOT_HIT.md                   full-study result tables
```

Every experiment writes its results to a matching `.json`, all of which are committed, so the
numbers in the paper can be traced to the run that produced them without re-executing anything.

## Data

Both datasets are public and streamed rather than downloaded:

- **BraiDyn-BC** — DANDI:001425 (Kondo et al., *Sci Data* 2025). Cortex-wide widefield calcium,
  parcellated dF/F at 30 Hz plus behavior, CC-BY. Streamed via `remfile`, so no raw-movie
  download is required.
- **Allen Visual Behavior** — single-plane two-photon VISp populations, used as an independent
  replication cohort from a different lab.

## Honest caveats

The four task events are chained within each trial (every reward onset has a lick within
0.12 s) and differ substantially in sample size. Per-event effects are therefore reported as
overall variation rather than attributed to individual behaviors. `event_overlap_check.py` and
the `*_leakfix.py` scripts exist specifically to keep that from quietly inflating the result.
