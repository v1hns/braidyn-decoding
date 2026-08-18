# Cortical Decoders — Worksheet v7 (Vibha restructure)

**This one is different from v4/v6.** Those were about your voice. This one is about *structure*,
and it comes from Vibha Singhal's 2026-08-17 email. You've said her advice leads. So this worksheet
does not propose sentences for you to react to — it tells you what each slot needs to contain and
why she wants it, and you write it. I have not written any of your prose.

**Already done, no dictation needed** (mechanical, from her general guidance):
- Acronyms defined on first use in the main text — CCF, AUC, VISp, CNN, GRU were all undefined.
- Figure 1 and both table captions expanded into self-contained summaries.

**Her two reference papers** — I could not open these (OpenReview is behind a bot check I won't
complete). Open them yourself before dictating; she picked them as structural models:
- https://openreview.net/pdf?id=fyp34w19N2
- https://openreview.net/pdf?id=REIo9ZLSYo

---

## PART 0 — One decision that gates everything

Her structural advice describes a **full-length conference paper**. The Brain-and-Body submission is
capped at **5 pages** of main text. A page-long introduction plus prose Methods subsections plus
thematic Results blocks does not fit in 5 pages.

- **Option A (proposed).** Restructure the **full paper** (`paper_neurips/pooling.tex`, 8 pp, the
  ICLR 2027 version) to her template in full. Then re-derive the 5-page workshop cut from it,
  keeping the new *ordering* and *framing* but compressed. You get her structure where there's room
  for it, and the workshop version inherits the improved framing.
- **Option B.** Apply it to both at full strength and let the workshop version lose content
  elsewhere (probably the replication detail or the count-matched control) to buy intro space.
- **Option C.** Workshop version first, since Sept 5 is the live deadline; restructure the full
  paper afterwards.

▶ YOUR CALL (A / B / C):

---

## PART 1 — Abstract

**Her template.** A good abstract covers five things, in order:
1. Broad context & importance — the overarching problem, and why it matters
2. The specific unaddressed gap — what prior work overlooks
3. Core empirical finding — what you did, exact quantitative results
4. Theoretical insight / mechanism — what principle explains why this happens
5. Actionable community takeaway — what tool or guideline you're providing

**Her diagnosis of your other paper was that (2), (3), (4) were fine and (1) and (5) were weak.**
The same reading applies here. Your current abstract opens on the mechanism ("A decoder fit to a
singular animal's cortex drifts") and closes on a suggestion ("suggests population priors as an
alternative to per-subject recalibration").

What each slot needs from you:

**1.1 — Broad context & importance (currently missing).** One or two sentences before your existing
opening. Why anyone outside this subfield should care: decoders deployed over months, recalibration
requiring labelled data from the subject, the cost of that. Not the drift mechanism — the stakes.

▶ YOUR VERSION:

**1.2 — Actionable takeaway (currently a suggestion, she wants a deliverable).** You have two
concrete things a reader can use: population priors instead of per-subject recalibration, *and* the
session-held-out validation protocol. The second is a guideline the community can adopt tomorrow
and it is currently buried. Consider ending on both.

▶ YOUR VERSION:

**1.3 — Anything in the middle you want to change while you're in there?** Slots (2), (3), (4) she'd
call adequate. Your call whether to touch them.

▶ YOUR NOTES:

---

## PART 2 — Introduction

**Her instruction:** "Your current introduction is very small. Typically this should be about a page
or more and should be written like a story so that the reader is hooked and gets to know everything
about your paper from here. After this section if they read the paper it is more for details."

Yours is currently three short paragraphs plus the contributions list. Her four-paragraph structure,
mapped onto this paper:

**2.1 — Paragraph 1: the big picture and motivation.** Why anyone cares whether a decoder trained on
other individuals holds up over time. The deployment story: BCIs and decoders that must work for
months, and what recalibration costs in practice.

▶ YOUR VERSION:

**2.2 — Paragraph 2: the scientific tension.** For your other paper this was Aw et al. vs Gao et al.
Here it is not a disagreement but an untested assumption: prior work established that population
structure is *shared* (Safaie, MacDowell, Musall) and that pooling *works* at a fixed time (NEDS,
WiCAT, HTNet), and Meta-AlignNN builds methods on that premise — but nobody measured whether pooling
resists drift. State what the field currently assumes and on what evidence.

▶ YOUR VERSION:

**2.3 — Paragraph 3: the core discovery / the gap.** The insight itself: a decoder that never saw
the subject cannot overfit that subject's drift-prone idiosyncrasies, so it should age more slowly —
and that this is measurable as an asymmetry. This is where the reader should understand the whole
paper.

▶ YOUR VERSION:

**2.4 — Contributions.** You already have four explicit bullets, which is the part of her advice you
were already doing. Do you want them reworded given the new paragraphs above, or left alone?

▶ YOUR CALL:

---

## PART 3 — Data and methods

**Her instruction:** "your current writing style is bulleted and a reviewer who is away from the
domain will have to first do work outside your manuscript to understand details." She wants full
prose subsections with narrative connections, and named ones.

Your Methods is currently four bold run-in headings (Dataset / Features and events / Decoders /
Temporal blocks and aging / External validation dataset). Her equivalent restructure here:

**3.1 — Datasets and preprocessing.** BraiDyn-BC and the Allen cohort described in continuous prose,
including *why* the two differ in subject count and session structure — she explicitly asked for the
"why they differ" narrative, not just the numbers.

▶ YOUR VERSION:

**3.2 — Features and decoders.** Feature construction, the linear readout, and why a linear readout
is the right primary choice. Currently this is stated compactly; she'd want the reasoning spelled
out for an outside reviewer.

▶ YOUR VERSION:

**3.3 — Temporal blocks, aging, and the session-held-out protocol.** The WS/WX/LS/LX definitions and
the asymmetry. Critically: why holding out whole sessions rather than trials is *essential* rather
than a detail. This is currently split between Methods and Results §4.5.

▶ YOUR VERSION:

**3.4 — The external validation cohort.** Cohort selection and the permutation-invariant feature,
with the load-bearing argument (no neuron correspondence across animals) given room.

▶ YOUR VERSION:

---

## PART 4 — Results reorganization

**Her instruction:** group related subsections into thematic blocks rather than a long list of small
ones. You currently have seven. Proposed grouping — this is a structural call, not prose:

| Block | Merges | Story it tells |
|---|---|---|
| A. The effect | §4.1 asymmetry + §4.3 day-by-day decay | Two different estimators, same conclusion |
| B. It is not an artifact | §4.2 event overlap + §4.4 count-matched | Alternative explanations, ruled out |
| C. Measuring it correctly | §4.5 protocol | The validation result, which is also a community contribution |
| D. It replicates | §4.6 Allen | Independent cohort |

▶ YOUR CALL on the grouping (approve / change):

▶ Any new connecting sentences between blocks are yours to write:

---

## PART 5 — Discussion

**Her instruction:** "your current discussion is dense and your key contribution is getting eclipsed
by limitations."

**5.1 — Best-practices checklist.** She wants the actionable guidance as a formal box or numbered
list, not buried in prose. For this paper the natural items are: hold out whole sessions rather than
trials; count-match the training sets before comparing arms; report both arms' aging separately, not
only the difference. You'd know if there are others.

▶ YOUR VERSION (and tell me if you want it as a numbered list or a boxed float):

**5.2 — Limitations, split into separate paragraphs.** Currently one dense paragraph carrying four
distinct concessions: what the four events are, mesoscale vs single-neuron, the two-week/one-species
scope, and the trial-level inflation caution. She wants these broken out so the contribution isn't
eclipsed.

▶ YOUR CALL on the split, and any rewording:

---

## PART 6 — The one place her advice contradicts your v6 instruction

**Her general guidance #2:** "Eliminate short, abruptly phrased sentences (e.g. 'Neither obvious
escape works.') with standard transitions (e.g. 'Furthermore, neither of the proposed mitigation
strategies resolves the underlying measurement gap.')."

**Your v6 Part 4 instruction to me was:** "Just be concise. Don't have these fluffy claims... Be very
concise, data-driven, numerics."

You've now said her advice wins. Flagging it because these are the specific sentences it affects —
all of them yours, and all of them deliberate:

1. "The pooled estimate is the claim we defend; the per-event column is data."
2. "We cannot separate that explanation from a behavioral one with four events, and we do not try."
3. "One thing does not transfer."
4. "Every number above holds out whole sessions."
5. "Each of these could have removed the result and none did."
6. "The reward decoder is therefore never reading reward in isolation."

▶ YOUR CALL — all of them, some of them, or none:

---

## PART 7 — Anything else

Her closing note: "Please don't feel the need to accept each and every suggestion above. Overall,
your results seem solid and some writing fine tuning will significantly increase acceptance chances."

▶ YOUR NOTES:
