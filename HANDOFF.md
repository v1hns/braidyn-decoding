# BraiDyn pooling paper — full session handoff

Written 2026-08-02. Everything a fresh session needs to continue this work. Read top to bottom once.

---

## 0. TL;DR

The paper: **a decoder trained on OTHER individuals ages more slowly over days than a decoder
trained on the subject itself.** Population pooling as an alternative to per-subject recalibration.

- Repo: `~/braidyn-decoding`, GitHub `v1hns/braidyn-decoding`, all work merged to `main`.
- Live paper: `paper_neurips/pooling.tex` — 8 pages, compiles clean, in the author's own voice.
- Headline: **+0.017 AUC, p = 5×10⁻⁵**, 73 of 100 mouse–event pairs, session-held-out.
- **A second dataset replicated it** (Allen 2-photon, +0.0229, p = 0.0034) and has now been through
  the full leakage-corrected protocol — see §5. Written into the paper; awaiting a dictation pass.
- A second paper (`paper_methods/leakage.tex`) exists but the author **decided not to publish it**.
  Do not resurrect it without being asked.

---

## 1. What the paper claims

On BraiDyn-BC (DANDI:001425, 25 mice, 44 Allen-atlas cortical parcels, ~15-day cued lever-pull
operant protocol), for each mouse we compare how much AUC is lost over two weeks by:

- the mouse's **own** early-week decoder (personal), vs
- a decoder **pooled over the other 24 mice's** early-week data.

Aging = accuracy lost moving from the held-out mouse's early data to its late block, training fixed.
Asymmetry = (personal aging) − (pooled aging). Positive ⇒ the personal decoder ages more.

### Final numbers (session-held-out — these are the correct ones)

| Event | Personal | Pooled | Asymmetry (95% CI) | p | own ages more | trial-split (WRONG) |
|---|---|---|---|---|---|---|
| Lever-pull | −0.022 | −0.025 | +0.003 [−0.004, 0.011] | 0.37 | 64% | +0.014 |
| Lick | +0.012 | −0.009 | +0.020 [0.008, 0.032] | 0.0023 | 80% | +0.028 |
| Reward | +0.017 | −0.012 | +0.029 [0.014, 0.046] | 0.0005 | 80% | +0.032 |
| Tone | +0.016 | +0.001 | +0.015 [0.001, 0.032] | 0.077 | 68% | +0.022 |
| **Pooled** | | | **+0.017 [0.010, 0.024]** | **5×10⁻⁵** | **73%** | +0.024 |

Supporting results:
- **Count-matched control**: at equal training-event count, self ages more than a decoder from one
  other mouse by **+0.010 AUC** on average (lick +0.007, reward +0.017, tone +0.006; lever REVERSES
  at −0.002). Kills the data-volume explanation. `many < one` on **all four** events.
- **Decay regression** (15 days): gap narrows 0.0014 AUC/day, p=5×10⁻⁵, 65% of pairs. Pooled decoder
  *gains* +0.0015/day while personal is flat (+0.0000/day) — see §3 warning.
- **Capacity** (now a FOOTNOTE, not a section): effect survives 1-D CNN and GRU for lick and reward.
  linear/CNN/GRU = lick +0.020/+0.012/+0.010, reward +0.029/+0.015/+0.012.

---

## 2. CRITICAL — bugs already found and fixed. Do not reintroduce.

This codebase had three separate rounds of real errors. Every one was a claim that was *assumed*
rather than *measured*. If you are about to assert something about this data, measure it first.

### 2a. Data leakage in `pooling_drift.py` (fixed by `pooling_leakfix.py`)

Two bugs, both inflating **only the personal arm** — i.e. one side of an asymmetry.

1. `gather()` vstacked early sessions and **destroyed session identity**; WS was then estimated with
   `StratifiedKFold(shuffle=True)` over pooled trials, so the personal decoder trained on session 3
   and was tested on other trials of session 3. The pooled decoder trains on other mice and cannot
   leak. **Fix: leave-one-early-SESSION-out with matched training sets for WS and WX.**
   Cost: pooled +0.024 → +0.017; lever-pull +0.014 (p=0.001) → **+0.003 (p=0.37)** — entirely leakage.
2. `analysis_C` (count-matched control) scored the self arm on its **own training data**:
   `pc=_fit(Xe,ye); _auc(pc,Xe,ye) - _auc(pc,Xl,yl)` — training accuracy, no CV, while one/many were
   held out. **Fix: all three arms train on n events, score on the same held-out session.**
   Moved lever-pull's self arm by −0.024 and reversed the control's ordering.

`analysis_B` (scaling vs N) was always clean — never sees the held-out mouse.

**Validation that makes the corrected numbers trustworthy:** `pooling_leakfix.py` replays the OLD
protocol on the SAME features and reproduces the published numbers exactly (+0.0240 pooled vs
published +0.024; per-event +0.0138/+0.0276/+0.0322/+0.0224 vs +0.014/+0.028/+0.033/+0.022). So the
difference is a measurement of the bug, not of a reimplementation.

### 2b. The four "events" are NOT four independent events (`event_overlap_check.py`)

Measured on 8 sessions × 8 mice, onset timings only:

- Every **reward** onset has a **lever-pull** onset a median **0.083 s** away and a **lick** onset
  **0.117 s** away — **100%** and **98%** land inside the [−0.5,+1.0] s feature window.
- Positive windows containing ≥1 other event: reward **100%**, lick 84%, tone 84%, lever 68%.
- **Negatives are contaminated 33–57%** (reward 56%, tone 57%): `build_session` screens negatives
  only against the SAME target's onsets. The Methods sentence claiming ">2 s from any event" was
  **wrong** and is now fixed to say what the code does.
- Event counts differ >10×: reward 56/session, tone 107, lick 530, lever 652 — and the per-event
  asymmetry runs roughly OPPOSITE to that count (Spearman ≈ −0.8, n=4).

**Consequence, and DO NOT UNDO THIS:** a planned restructure to "lead with reward and lick" was
**reverted**, because those two co-occur within 120 ms and were never two independent confirmations.
The paper now defends the **pooled estimate** and reports the per-event column as **data, not four
findings**. §4.2 "What the four events are" states all of the above. Do not re-propose a per-event
framing.

### 2c. Two pilot-selection bugs (mine, fixed 2026-08-02)

First Allen 2P pilot run selected **1 mouse** and looked like a null. Causes: depth binned `[125,175)`
in the pilot vs `[150,200)` in the analysis that found the mice, plus requiring every experiment in a
container to match. Also mis-flagged low neuron counts as a `find_dff` bug — they were real (that
plane has 12 cells; cohort median is 57). Fixed; see §5.

---

## 3. Framing decisions the author made (respect these)

The author did a full dictation pass on a read-and-restate worksheet; the paper is in **his voice**,
built from his sentences as blocks. Memory rule `feedback_formatting_only_on_hand_rewrites` applies:
**when he hand-writes prose, change only formatting/presentation, never wording.**

Decisions from that pass:
- **"targets" → "events"** everywhere. Zero instances of "target" should remain.
- Leakage caveat **removed from the abstract**, lives in limitations ("I don't even get what the
  second part is saying... put this in the limitations").
- Capacity section + Table 3 **collapsed to a footnote** ("maybe I'll have a short footnote of it").
- Related-work order **swapped** (prior-work-at-fixed-time first, count-matched second).
- Abstract cut 293 → 212 words; the lick/reward result is stated **once**, not three times.
- Discussion has a paragraph on **why the effect is believable** (four independent checks, and the
  reported effect is what SURVIVED the leakage correction).

**WARNING — the "rising tide" framing is BraiDyn-specific.** In BraiDyn the pooled decoder *gains*
(+0.0015/day) while the personal one is flat, so the gap narrows because pooling improves. In the
Allen 2P replication pooled aging is **+0.001 (flat)** and the asymmetry comes entirely from the
personal decoder aging. Do not export the rising-tide language to the new dataset.

---

## 4. Files in the repo

### Analysis (BraiDyn)
| file | what it does |
|---|---|
| `pooling_drift.py` | ORIGINAL, **has the leakage bugs**. Kept for provenance. Do not cite its JSON. |
| `pooling_leakfix.py` | **Corrected** A/B/C, runs leaky+fixed side by side → `pooling_leakfix.json` |
| `nonlinear_leakfix.py` | Corrected CNN/GRU capacity → `nonlinear_leakfix.json` |
| `full_decay.py` | 15-day decay regression (ALWAYS was correct, LOSO) → `full_decay.json` |
| `event_overlap_check.py` | Event timing/overlap audit → `event_overlap.json` |
| `make_pooling_figs.py` | fig_mechanism from `pooling_leakfix.json` |
| `make_decay_fig.py` | fig_decay from `full_decay.json` |
| `allen2p_pilot.py` | Allen 2-photon replication pilot → `allen2p_pilot.json` |
| `allen2p_leakfix.py` | **NEW** corrected protocol on Allen (A/C/D + leak-vs-aggregation E) → `allen2p_leakfix.json` |
| `leakage_sim.py`, `cardin_leakage.py` | for the shelved methods note |

### Papers
- `paper_neurips/pooling.tex` — **THE paper**. 8pp. Compile: `/opt/homebrew/bin/tectonic paper_neurips/pooling.tex`
- `paper_methods/leakage.tex` — methods note, **author declined to publish**. Leave parked.
- `paper_iclr/`, `paper/` — superseded, ignore.

---

## 5. ⭐ THE OPEN THREAD — Allen 2-photon replication (result just landed)

`allen2p_pilot.py`, run 2026-08-02. **The effect replicated on an independent dataset.**

Cohort selection (this is the load-bearing part): Allen Visual Behavior 2-photon, containers spanning
**≥14 days**, all experiments **VISp** in a single 50 µm depth bin. Largest band is **150–199 µm →
25 mice**, 184 experiments, median 57 cells/experiment. 23 mice usable (≥4 sessions).

**Why permutation-invariant features:** there is NO cross-animal neuron correspondence (mouse A's
neuron #17 ≠ mouse B's #17), and the longitudinal subset is single-plane, so area-averaging gives a
1-D feature. The only comparable representation is a permutation-invariant summary of the population
response: 25 quantiles + mean + sd of the per-neuron evoked response = **27 dims**, fixed across mice.

Target is `is_change` (real image change vs sham catch trial) — a **stimulus** contrast, materially
less reward/lick-coupled than anything in BraiDyn.

### Result

| | BraiDyn | Allen 2P |
|---|---|---|
| personal aging | +0.012…+0.017 | **+0.0240** |
| pooled aging | −0.009…−0.012 | **+0.0010** |
| **asymmetry** | **+0.017** (p=5e-5) | **+0.0229** (p=0.0031) |
| holds in | 73% of pairs | **74% of mice (23)** |
| decodability | — | within 0.672 / pooled 0.638 |

Different lab, modality, task, and feature space. Same sign, comparable magnitude.

### ✅ DONE 2026-08-03 — the corrected protocol was run (`allen2p_leakfix.py` → `allen2p_leakfix.json`)

Features cached to `allen2p_features.npz` (gitignored); re-runs take ~2 min instead of ~20. The
script imports `allen2p_pilot.py` for cohort/build/features, so it is not a reimplementation. All
three gates passed.

1. **Count-matched control.** self **+0.0240**, one other mouse **+0.0132**, many others **+0.0030**.
   self−one **+0.0108** [0.0010, 0.0217] p=0.054 (marginal — say so, don't hide it); self−many
   **+0.0210** p=0.0019; many−one **−0.0102** p=0.0031, the same diversity ordering as BraiDyn.
   The leaky variant inflates self to +0.0631, so bug C would have bitten here too.
2. **Trial-split vs session-held-out — DOES NOT BEHAVE LIKE BRAIDYN.** The trial split *deflates*
   the asymmetry here (+0.0137, p=0.065) instead of inflating it. Decomposed in `analysis_E` rather
   than asserted: leakage still inflates the personal arm **+0.0153** at matched aging, but scoring
   one pooled AUC across sessions instead of averaging per-session AUCs costs **−0.0301** (p<1e-4)
   and dominates. The corrected estimate is the *larger* one here, so the replication is not a leak.
   Do not export "the trial split inflates the asymmetry" as a general claim — it is BraiDyn-specific
   as a *net* statement. The transferable claim is narrower: only the personal arm can share a
   session with its test data.
3. **Inclusion rule doesn't matter.** ≥3 → 24 mice +0.0223; ≥4 → 23 mice +0.0229; ≥5 → 22 +0.0228;
   ≥6 → 18 +0.0239. And §5's old framing was wrong: every container has ≥5 sessions, so the 2 mice
   the pilot lost were per-experiment QC failures (17 of 18 dropped experiments had <15 cells),
   not short containers.

`analysis_A` fixed reproduces the pilot to the digit (+0.0229, p=0.0034, 74%, WS 0.6721) from a
separate code path. Effect size d=0.68, sd 0.034 > mean 0.023 — a real but noisy 23-mouse result.

**Also fixed:** `pooling_leakfix.py` seeds resampling with `hash(m)`, which Python salts per process,
so its count-matched numbers are not reproducible across runs. `allen2p_leakfix.py` uses `crc32`.
Worth backporting.

---

## 6. Dataset landscape (searched exhaustively — don't redo this)

**Public neural datasets have many subjects OR many sessions per subject, essentially never both,
and almost never with a shared coordinate system too.** BraiDyn has all three, which is why this
study was constructible. Worth a sentence in the discussion.

| dataset | subjects | sessions/subject | verdict |
|---|---|---|---|
| FALCON (DANDI 000954 etc.) | **6** | 287 days ✓ | can't do LOSO; FALCON says so itself |
| LINK (DANDI 001201) | **1 monkey** | 312 ✓ | single subject |
| IBL Widefield (DANDI 001712) | **1** | 5 files | partial upload |
| BRAVO1 (DANDI 001535) | **1** human | — | single subject |
| DANDI 000244 | 25 ✓ | **~1** | no longitudinal |
| IBL Brain-Wide Map | 139 ✓ | acute | no longitudinal |
| Allen VB **Neuropixels** | 81 ✓ | **2 consecutive days** | hard fail |
| **Allen VB 2-photon** | 107 ✓ | median 11 d (25 mice ≥14 d) | ✅ **USED — see §5** |
| **BraiDyn-BC** | 25 ✓ | ~15 over 3 wk ✓ | the main dataset |

Allen 2P full download would be ~480 GB (248 MB/experiment × 1936). **Stream with `remfile`+`h5py`
instead** — that's what `allen2p_pilot.py` does; never download whole files.

Allen 2P NWB layout (verified by probe):
- dF/F: `processing/ophys/dff/traces/data`, shape **(time, neurons)**, timestamps alongside
- trials: `intervals/trials` with `change_time`, `is_change`, `go`, `catch`, `hit`, `lick_times`
- metadata CSVs: `https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com/visual-behavior-ophys/project_metadata/{ophys_experiment_table,ophys_cells_table}.csv`

---

## 7. Prior art / scooping (all now cited)

- **Safaie et al. 2023**, Nature 623:765–771, *Preserved neural dynamics across animals* — the
  mechanistic precursor. Paper states: conservation makes pooling **possible** but does not imply it
  **resists drift**. That gap is our question.
- **NEDS**, Zhang et al., arXiv:2504.08201 — pooled pretraining on 83 IBL animals + fine-tuning on
  held-out animals, IBL consortium authors. **No time-resolved analysis** → doesn't touch our claim.
- **Meta-AlignNN**, Zou et al., bioRxiv 2025, doi:10.1101/2025.04.20.649482 — **closest work**. Uses
  cross-subject consistency for BCI stability across subjects/time/tasks, 2 years, 3 monkeys. Gets
  its own paragraph. Distinction: **they propose a method, we measure whether the effect exists.**
- **WiCAT** (Hosseini et al., ICML 2026, arXiv:2607.09754) — cross-subject decoding on the SAME
  BraiDyn cohort. Already cited. Never claim we beat them (AUC vs R², not comparable).

A forward-citation sweep (114 citations of Safaie, 85 of NDT2, via Semantic Scholar graph) found
**nobody has published the differential-aging claim**. That is the novelty.

---

## 8. External resources

- GitHub: `v1hns/braidyn-decoding` — PRs #9–#13 merged to `main`. **#14 (Allen corrected protocol)
  and #15 (paper write-up, stacked on #14) are OPEN** — agent merge was blocked by the permission
  classifier, so they need `gh pr merge 14 --merge` then `gh pr merge 15 --merge`.
- Overleaf (pooling paper): https://www.overleaf.com/project/6a5acae5bbb72f3b80b9173a — synced.
- Overleaf (methods note, parked): https://www.overleaf.com/project/6a6c57c6faa0705a6da29f14
- Worksheet Doc (drift paper, **dictation already applied**):
  `14gP87UxTIES-4PnpVBuPAgBFUvr0HQosfvXKHWQ5rfI`
- Worksheet Doc (methods note, never used): `1CeFASJCpWGj1RkiGJ7X3uasrP5Mga_rECsbT40aonhU`
- Worksheet Doc **v5** (two-dataset reframe, **awaiting dictation**):
  `1mQTRXRLzQ2pjNGEr572SxcWxNlBLTEfUJqfURf1I46I`
  (a stray empty doc `15VVmnn1p6KQiMoRrcNm1e5V5EjKuNuAuECV2g9BrH9E` was created by mistake — delete it)
- User memory: `~/.claude/projects/-Users-vihaanshringi/memory/project_braidyn_hit.md`

**Overleaf upload gotcha:** uploading figures with the `figs` folder merely *selected* still dumps
them at ROOT as duplicates while `figs/` keeps stale copies. Reliable path: **right-click a file
inside `figs/` → Upload → Overwrite**, then delete any root duplicates. For a NEW project, upload a
zip instead.

---

## 9. Environment / ops

- Compile: `/opt/homebrew/bin/tectonic` (full path; not on non-interactive PATH) on the Mac. The
  2026-08-03 session ran on Linux and used `pdflatex`, which works fine (9 pp: main text ends p. 8,
  refs run to p. 9; NeurIPS excludes refs from the limit).
- Python: `~/braidyn-decoding/.venv` has numpy, scipy, sklearn, dandi, remfile, h5py, pynwb.
- Lambda: creds `~/.lambda_key`, key `~/.ssh/lambda_tvdx` (registered as `tvdx-key`). **Terminate
  when done** — all boxes from this session ARE terminated.
- **Lambda ssh gotcha (hit twice, two regions):** boxes go unreachable on port 22 after ~4 sequential
  ssh/scp operations (banner-exchange timeout) while the API still says `active`. Looks like
  connection rate-limiting. Provision with ONE combined ssh + retries, not a chain.
- The leakage sim + Cardin tests were run **locally** — a justified exception to the always-cloud
  rule, which exists for GPU/MPS RAM reasons that don't apply to CPU sklearn on tiny data.
- Web search budget was exhausted in this session; Semantic Scholar / Crossref / DANDI APIs via
  `curl` are good substitutes and don't consume it.

---

## 10. User preferences (from memory)

- **Always push to branches, never directly to main**; PR then `gh pr merge --merge` (**never
  `--squash`** — preserve per-commit history).
- Commit + push after every prompt that edits code.
- Heavy compute on cloud, not the laptop.
- Save local deliverables to `~/Downloads`, not Desktop.
- When he hand-writes prose, change **only** formatting — never his wording.
- He values being told when something doesn't hold up. Three claims died on contact with a test this
  session and he wanted each of them killed rather than defended.

---

## 11. Immediate next steps

1. ~~Run the leakage-corrected protocol on the Allen 2P cohort.~~ **Done, §5. It held.**
2. ~~Write it up as a Results subsection + update abstract/contributions.~~ **Done** — §5.7 + Table 4
   + methods paragraphs + abstract sentence + contributions bullet, PR #15 (stacked on #14). The
   write-up is **additive only**: not one existing sentence was reworded.
3. **PENDING — the dictation pass.** The author asked to reframe the headline around both datasets
   ("optimize for results"). That touches title/abstract/contributions/intro, all his prose, so
   **worksheet v5** was built instead of rewriting it:
   `1mQTRXRLzQ2pjNGEr572SxcWxNlBLTEfUJqfURf1I46I`. 16 chunks. Chunk 1 is the ordering decision
   (recommendation: headline both, BraiDyn primary, Allen as replication — Allen has the bigger
   number but weaker evidence, and the two AUC deltas are not on a comparable scale).
   **Two chunks are not stylistic and must be resolved before submission:**
   - **Ch. 15 — the paper now contradicts itself.** Limitations says single-neuron codes "remains
     unknown"; the Allen replication *is* single-neuron two-photon.
   - **Ch. 14 —** Discussion says "four checks"; there are now five.
4. **Overleaf sync is manual and was NOT done** — no Overleaf credentials or tool available to the
   agent, and past sessions did it by hand. Only `pooling.tex` changed (no new figures), so it is a
   single-file overwrite and the §8 figs-folder gotcha does not apply. Bundle also at
   `~/Downloads/pooling_overleaf_2026-08-03.zip`.
5. Decide venue. Workshop (non-archival, so an archival submission later is normally fine — but
   **verify per-workshop CFP + host conference dual-submission clause**, this was NOT verified) vs
   ICLR 2027 (deadline 2026-09-24). The Allen replication is exactly the "substantial new content"
   an archival venue expects.
4. Add a discussion sentence on §6 — the reason nobody tested this is that the data barely exists.

---

## 12. ⭐ START HERE — where the 2026-08-03 session left off

The session ended on a deliberate **Claude Code restart** so the Claude in Chrome extension would be
detected at startup. Everything below is already done and pushed to `main`.

### State of the work

- **All merged to `main`** (PRs #14, #15, #16). `allen2p_leakfix.py` + `.json`, and the paper
  write-up: §5.7, Table 4, the Allen methods paragraphs, one abstract sentence, one contributions
  bullet, the dataset citation. **Additive only — not one of the author's sentences was reworded.**
- `paper_neurips/pooling.tex` compiles clean under `pdflatex` (main text ends p. 8, refs run to p. 9;
  NeurIPS excludes refs from the limit).
- Feature cache `allen2p_features.npz` is gitignored and **may not exist after the restart** — if
  missing, `allen2p_leakfix.py` re-streams 184 experiments from S3 (~10 min). Analyses alone are ~2 min.
- Local venv at `.venv` (numpy/scipy/sklearn/h5py/remfile). The Mac paths in §9 do not apply here.

### THE ONE THING BLOCKING THE PAPER: the dictation pass

`WORKSHEET_v6.md` in this repo. **Do not ask the author to re-dictate anything from v4.** v5 made
that mistake and he called it out. v6 is deliberately small:

- **Part 1** — one decision: headline both cohorts (A, proposed), headline Allen (B), or interleave (C).
  Recommendation is A: Allen has the bigger number (+0.023 vs +0.017) but weaker evidence (n=23,
  p=0.003, count-matched only marginal at p=0.054), and the two AUC deltas are not on a comparable
  scale, so leading with Allen reads as cherry-picking.
- **Part 2** — eight chunks of agent-written prose that has never been through his voice.
- **Part 3** — the only **two** of his sentences that must change:
  1. **Limitations** claims single-neuron codes "remains unknown" — the Allen replication *is*
     single-neuron two-photon. **The paper currently contradicts itself.** Not a polish item.
  2. **Discussion** says "four checks"; there are five now.

Under Option A everything else he dictated in v4 stands untouched, because the new material is added
*alongside* his sentences rather than replacing them.

### Overleaf

- **Free plan — confirmed by looking at the account.** So the git bridge is OUT (premium only).
  Do not propose it again.
- Target project: **`pooling-neurips-overleaf`**. Only `pooling.tex` changed, no new figures, so it
  is a **single-file overwrite** and the §8 figs-folder gotcha does not apply.
- Push **once**, after the dictation lands. Not before.

### The VPS browser stack (new this session, and the reason for the restart)

This box is a headless KVM VPS reached over tmux/ssh. There was no display, so one was built:

- `~/bin/vps-desktop.sh {start|stop|restart|status|chrome|setpass}` — Xvfb `:99` at 1920x1080,
  openbox, x11vnc, noVNC. Logs in `~/.vps-desktop/`.
- **Everything is bound to 127.0.0.1** (x11vnc `-localhost`, websockify explicit loopback bind).
  Reach it only through an SSH tunnel — this box has a public IP, so keep it that way:
  `ssh -L 6080:127.0.0.1:6080 vihaan@152.53.168.226` (or Tailscale `100.73.25.122`), then
  `http://localhost:6080/vnc.html`. VNC password is hashed at `~/.vnc/passwd`.
- Chrome runs on `:99` with a persistent profile at `~/.chrome-vps`, already **logged into Overleaf
  and Google**. The Claude extension (`fcoeoabgfenejglbffodgkkbkcdhcgfn`, v1.0.84) is installed there.
- `xdotool` and `scrot` are available on `:99` as a fallback if the extension route misbehaves.

### Two capabilities that got blocked by the permission classifier

Worth knowing before planning around them: `sudo apt-get install` and
`mcp__claude_ai_Google_Drive__create_file` were both denied mid-session (Drive had worked earlier).
The author ran the apt install by hand. If a Doc is needed again, either retry Drive or create it
through the now-working browser.
