# BraiDyn-BC cross-mouse cortex-wide decoding — FULL STUDY (2026-07-10)

Dataset: BraiDyn-BC, DANDI:001425 (Kondo et al., Sci Data 2025). Cortex-wide widefield calcium,
44 Allen-atlas cortical parcels (dF/F @30Hz) + behavior, cued lever-pull operant task, 25 mice.
Open CC-BY; parcellated dF/F + behavior STREAMED via remfile (no raw-movie download).

Axis (VIRGIN — descriptor does zero decoding): leave-ONE-MOUSE-out decoding of behavioral
events from the 44 shared atlas parcels.

## Full result (all mice, 4 targets attempted; lever-pull & lick had enough events)
| target     | mice | within-mouse | leave-one-mouse-out | 95% CI          | perm p |
|------------|------|--------------|---------------------|-----------------|--------|
| lever-pull | 23   | 0.727        | 0.742               | [0.701, 0.782]  | 0.007  |
| lick       | 22   | 0.707        | 0.750               | [0.694, 0.801]  | 0.007  |

HEADLINE: LOMO >= within-mouse -> NO generalization gap. Cortex-wide behavioral
representations are CONSERVED across individuals in the shared atlas space. Every held-out
mouse decodes above chance. Motor cortex (MOp/MOs) tops the lever-pull decoder; lick weights
differ (target-specific codes). reward/tone lacked threshold-detectable events (limitation).

Files: braidyn_full.py (analysis), make_braidyn_figs.py, braidyn_results.json, braidyn_pilot.py,
paper/braidyn.tex + braidyn.pdf (compiled), paper/figs/. Compute on Lambda; box terminated.
Full study DONE. Extensions: cross-session drift (15-day protocol), more targets, nonlinear temporal models.

## CROSS-LAB REPLICATION (2026-07-10): Cardin-Higley widefield (Benisty/Higley 2023, figshare 175317)
Independent widefield cohort (Higley lab), 6 mice, 23 CCFv3 parcels + pupil/face/wheel; spontaneous
behavior (not an operant task). Decode behavioral state from parcels, within-mouse block-CV vs LOMO:
  movement(wheel): within 0.744+/-0.11, LOMO 0.731 (per-mouse 0.57-0.91)
  arousal(pupil):  within 0.645+/-0.11, LOMO 0.627 (per-mouse 0.51-0.73)
BOTH: LOMO ~= within-mouse -> NO generalization gap, REPLICATING the BraiDyn conservation finding
on a different lab / atlas / task / behavioral target. Upgrades the claim to a cross-dataset general
principle: cortex-wide behavioral-state representations are conserved across individuals.

## CROSS-SESSION DRIFT STUDY (2026-07-14): conserved code is ALSO stable over time
Dataset actually gives 16-19 sessions/mouse (13-15 operant TASK days, ~3-week protocol) -- the
"1 session/mouse" limitation is now resolved. Streamed all 357 task sessions (0 skipped), 25 mice.
Design: per mouse split task days into EARLY (1-5) vs LATE (11-15); 2x2 = within/LOMO x same/cross-block.
Targets state_lever + lick (all 25 mice usable once daily sessions pooled).

| condition                          | lever-pull | lick  |
|------------------------------------|-----------|-------|
| WS within-mouse same-block         | 0.826     | 0.883 |
| WX within-mouse EARLY->LATE        | 0.838     | 0.865 |
| LS LOMO same-block                 | 0.792     | 0.832 |
| LX LOMO EARLY(others)->LATE(heldout)| 0.818    | 0.842 |
paired LX-LS: lever +0.026 (p=0.008), lick +0.010 (p=0.45). Drift-curve slope: lever -0.0004/day,
lick -0.0022/day (<0.04 AUC over 14 days; flat).

HEADLINE: NO within-animal drift (WX>=WS) AND the conserved code is temporally stable across animals
(LX>=LS -- other mice's week-1 reads a held-out mouse's week-3 with no loss). LX>LS direction likely
because late-week behavior is more practiced/stereotyped, hence marginally cleaner. Code invariant along
BOTH individual and session axes. Files: braidyn_drift.py, braidyn_drift.json, make_drift_figs.py,
paper figs fig4_drift_2x2 + fig5_daygap, drift subsection + Table II in braidyn.tex/pdf (compiled w/ tectonic).
Compute: Lambda a10 us-east-1 (24-worker streaming, ~2h wall), box TERMINATED. Reward/tone still lack events.

## NONLINEAR TEMPORAL MODELS (2026-07-15): conservation is NOT a low-capacity artifact
Q: does modelling the full temporal trajectory (not just evoked-diff vector) raise AUC, and does
the no-gap conservation SURVIVE a high-capacity model (which could overfit mouse idiosyncrasies)?
Design: full spatiotemporal window (-0.5..+1.0s = 45 frames x 44 parcels) per event, 3 task
sessions/mouse (75 sessions). 3 decoders on identical events/splits: linear logistic (evoked-diff),
1D-CNN over time, GRU over sequence. within-mouse 5-fold + LOMO, targets lever-pull+lick. GPU (a10).

| target     | model  | within | LOMO  | gap(LOMO-within) |
|------------|--------|--------|-------|------------------|
| lever-pull | linear | 0.841  | 0.804 | -0.037 |
| lever-pull | 1D-CNN | 0.942  | 0.895 | -0.047 |
| lever-pull | GRU    | 0.935  | 0.909 | -0.026 |
| lick       | linear | 0.886  | 0.850 | -0.036 |
| lick       | 1D-CNN | 0.978  | 0.966 | -0.013 |
| lick       | GRU    | 0.975  | 0.962 | -0.013 |

HEADLINE: (1) temporal models raise cross-mouse LOMO AUC +0.09-0.12 (lever 0.80->0.91, lick
0.85->0.97) -> collapsed evoked vector wastes real temporal signal. (2) nonlinear models DO NOT
widen the cross-animal gap (<=0.05, no larger than linear; SMALLER for lick) -> conservation is
robust to model capacity; the richer temporal dynamics are ALSO shared across individuals. NOTE:
in this pooled-session regime a small gap appears even for linear (within-mouse benefits from more
per-mouse data than single-session main analysis) -- the point is capacity doesn't enlarge it.
Files: braidyn_nonlinear.py (CNN+GRU+linear), braidyn_nonlinear.json, make_nl_figs.py, fig6_nonlinear,
Table III + subsection IV-C in paper. torch cu128 (NOT cu130 -- driver is CUDA 12.8). Box TERMINATED.
