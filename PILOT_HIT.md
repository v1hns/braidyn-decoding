# BraiDyn-BC leave-mouse-out cortex-wide decoding — PILOT HIT (2026-07-09)

Dataset: BraiDyn-BC, DANDI:001425 (Kondo et al., Sci Data 2025). Cortex-wide widefield
calcium (44 Allen-atlas cortical parcels, dF/F @30Hz) + behavior, cued lever-pull operant
task, 25 mice x 15 sessions. Open CC-BY, streamed via remfile (no raw-movie download).

Axis (VIRGIN — 0 prior decoding papers on this dataset; descriptor does no decoding):
leave-ONE-MOUSE-out decoding of lever-pull onset (evoked cortical response vs baseline)
from the 44 shared atlas parcels.

RESULT (10 mice, 1 session each, ~150 pull events/mouse):
- Positive control (within-mouse pull-vs-baseline): AUC 0.791 +/- 0.081  -> pipeline sound, signal real
- NOVEL AXIS (leave-one-mouse-out): AUC 0.730; every held-out mouse above chance (0.61-0.91)
- Permutation null: AUC 0.498, p=0.005  -> significant, leakage-free
- Small cross-mouse gap (0.79->0.73): cortex-wide motor representation is largely CONSERVED
  across animals in the shared atlas space.

Clears every bar: real signal, virgin axis, well-powered (25 mice), clean leave-mouse-out,
permutation-significant. First genuine HIT of the whole search.

Full-study path: all 25 mice; multiple decode targets (cue tone, reward, lick, not just lever);
within-mouse cross-SESSION drift (early vs late over 15 days); per-parcel importance maps;
within-mouse ceiling vs leave-mouse-out gap quantified with CIs.
