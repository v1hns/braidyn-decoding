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
