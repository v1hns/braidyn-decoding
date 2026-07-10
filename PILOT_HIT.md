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
