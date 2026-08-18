# Cortical Decoders — Worksheet v7 (Vibha restructure)

**What changed from the first v7:** every slot now shows **what we currently have**, so you can see
exactly what you're restating instead of guessing. Current text is quoted verbatim.

**The rule for this pass:** follow Vibha's method over your v6 instruction. Where the two conflict —
her transitions and narrative flow versus your terse, numeric style — hers wins. Conciseness still
matters, but not at the cost of the structure she's asking for.

**Already done, no dictation needed:** acronyms defined on first use in main text (CCF, AUC, VISp,
CNN, GRU — all were undefined); Figure 1 and both table captions expanded into self-contained
summaries.

**Her two reference papers** — open these yourself, OpenReview is behind a bot check I won't
complete: https://openreview.net/pdf?id=fyp34w19N2 and https://openreview.net/pdf?id=REIo9ZLSYo

**Deadlines (verified 2026-08-18):** NeurReps **Aug 24**, ICBINB-BIO **Aug 29** (tentative),
Brain-and-Body **Sept 5**.

---

## PART 0 — One decision that gates everything

Her structure describes a full-length conference paper. Brain-and-Body caps main text at **5 pages**.
A page-plus introduction, prose Methods subsections and thematic Results blocks do not fit in 5.

- **Option A (proposed).** Restructure the **full paper** (8 pp, also the ICLR 2027 version) to her
  template, then re-derive the 5-page cut from it so the workshop version inherits the framing in
  compressed form.
- **Option B.** Apply at full strength to both, and let the 5-page version drop content elsewhere.
- **Option C.** 5-page version first (Sept 5 is the live deadline), full paper after.

▶ YOUR CALL (A / B / C):

---

## PART 1 — Abstract

**Her template:** (1) broad context & importance → (2) the specific unaddressed gap → (3) core
empirical finding with exact numbers → (4) theoretical insight / mechanism → (5) actionable
community takeaway. She judged (2)(3)(4) fine and (1)(5) weak on your other paper. Same here.

### WHAT WE CURRENTLY HAVE

> A decoder fit to a singular animal's cortex drifts, losing accuracy over days. Usually the remedy
> is recalibrating on fresh data from the same subject. We report a different kind of robustness: a
> decoder built from one other individual, or pooled from a set of other individuals, ages more
> slowly than the subject's own. On the open BraiDyn-BC cohort (25 mice, 44 Allen-atlas cortical
> parcels, a ~15-day operant protocol) we decode four task events and measure, per mouse, the AUC
> lost over two weeks by its own early-week decoder against one pooled over the other 24 mice and one
> built from a single other mouse. The personal decoder ages more by +0.017 AUC (p = 5×10⁻⁵, 73 of
> 100 mouse–event pairs). The delta from each effect varies among them, but they are chained within
> each trial — every reward onset has a lick within 0.12 s — and they differ highly in sample size,
> so we report the variation in general without attributing it to the individual behaviors. This
> also is not a data-volume effect: at a matched training-event count a decoder from a single other
> mouse still ages less. Hence we conclude this is rather an effect of not overfitting an
> individual's drift-prone idiosyncrasies. It also survives a 1-D CNN and a GRU. The effect holds
> past this cohort and recording modality. On an independent two-photon dataset from a different
> laboratory — Allen Visual Behavior, 23 mice, single-plane VISp populations decoding a stimulus
> change — the same session-held-out protocol gives +0.023 AUC (p = 0.003, 74% of mice), and the
> count-matched control reproduces there as well (self − one other mouse = +0.011 AUC). The
> early-week accuracy cost of decoding across individuals is repaid as late-week stability, which
> suggests population priors as an alternative to per-subject recalibration. Code and a
> streamed-feature pipeline are released.

**Mapping onto her five slots:**

| Slot | Where it currently is | Verdict |
|---|---|---|
| (1) Broad context | Missing — you open on the mechanism ("A decoder fit to a singular animal's cortex drifts") | **Needs writing** |
| (2) The gap | "We report a different kind of robustness…" | OK |
| (3) Finding + numbers | +0.017, +0.023, count-matched | OK |
| (4) Mechanism | "not overfitting an individual's drift-prone idiosyncrasies" | OK |
| (5) Actionable takeaway | "suggests population priors as an alternative" — a suggestion, not a deliverable | **Needs strengthening** |

### 1.1 — Add slot (1): broad context and importance
One or two sentences *before* your current opening. The stakes, not the mechanism: decoders that
must keep working for months, and what per-subject recalibration costs to obtain.

▶ YOUR VERSION:

### 1.2 — Strengthen slot (5): actionable takeaway
You have two deliverables, not one: population priors instead of recalibration, **and** the
session-held-out validation protocol. The second is a guideline anyone can adopt immediately and it
currently doesn't appear in the abstract at all.

▶ YOUR VERSION:

### 1.3 — The middle, if you want it
Her method would also cut the long "The delta from each effect varies among them…" sentence down —
it's a caveat sitting where the finding should be.

▶ YOUR VERSION (or "leave"):

---

## PART 2 — Introduction

**Her instruction:** "Your current introduction is very small. Typically this should be about a page
or more and should be written like a story so that the reader is hooked and gets to know everything
about your paper from here."

### WHAT WE CURRENTLY HAVE — three short paragraphs, then the contributions list

> **¶1** Representational drift, the reorganization of an animal's neural code over days with fixed
> behavior, is now documented across sensory, motor, and association cortex and in the hippocampus.
> This has a stark impact on neural decoding: a decoder trained on an animal today degrades on that
> same animal weeks later. The dominant response is recalibration, periodically refitting the decoder
> on fresh within-subject data, which requires ongoing labelled data from the individual and treats
> each subject in isolation.
>
> **¶2** The code's cross-individual structure offers an alternate route to temporal robustness.
> Cortex-wide behavioral representations are considerably shared among individuals, which already
> allows models to decode behavior from unseen individuals. The question we pose is temporal: does a
> decoder built from a population of other individuals resist drift better than a decoder built from
> the subject itself? This prediction, to our knowledge, has not been examined.
>
> **¶3** We test it on BraiDyn-BC, a widefield calcium-imaging resource whose ~15-day operant
> protocol and shared 44-region Allen parcellation make within-subject and cross-subject decoders
> directly comparable across time. Our contributions:

Then four bullets: Pooling resists drift / It is not a data-volume artifact / It replicates on an
independent dataset / Session-level validation is load-bearing.

### 2.1 — ¶1 rewritten as "the big picture and motivation"
Your current ¶1 opens on the phenomenon. Hers opens on why the reader should care. The deployment
story — decoders in service for months, recalibration needing fresh labelled data from the subject
every time — is currently a subordinate clause at the end of ¶1. It should probably lead.

▶ YOUR VERSION:

### 2.2 — ¶2 rewritten as "the scientific tension"
Currently ¶2 states the question. Hers states what the field *assumes* and on what evidence, so the
gap has force. The material: conservation across individuals is established (Safaie, MacDowell,
Musall); pooling is known to work at a fixed time (NEDS, WiCAT, HTNet); Meta-AlignNN builds methods
on that premise — and nobody measured whether pooling resists drift over time.

▶ YOUR VERSION:

### 2.3 — ¶3 as "the core discovery", with a new home for the dataset sentence
Your current ¶3 is dataset logistics. Hers wants the insight here: a decoder that never saw the
subject cannot overfit that subject's drift-prone idiosyncrasies, so it should age more slowly, and
that difference is measurable as an asymmetry. The BraiDyn-BC sentence can move down or into Methods.

▶ YOUR VERSION:

### 2.4 — Contributions
You already have four explicit bullets — the part of her advice you were already doing. Reword given
the new paragraphs, or leave?

▶ YOUR CALL:

---

## PART 3 — Data and methods

**Her instruction:** "your current writing style is bulleted and a reviewer who is away from the
domain will have to first do work outside your manuscript to understand details." She wants named
prose subsections with narrative connections.

### WHAT WE CURRENTLY HAVE — one section, five bold run-in headings, no subsections

> **Dataset.** BraiDyn-BC (DANDI:001425, CC-BY) provides, for each mouse and session,
> hemodynamically corrected dorsal-cortex ΔF/F traces parcellated into 44 Allen Common Coordinate
> Framework regions at ~30 Hz, together with time-aligned behavioral channels. The dataset includes
> 25 mice, each of which completed ~15 daily operant sessions over ~3 weeks. We stream the 44-region
> traces and behavioral channels without downloading the raw movies.
>
> **Features and events.** We detect the onset of four events (lever pull, lick, reward, and tone) as
> rising edges, with a 1 s refractory period. For each onset we construct a 44-dimensional evoked
> feature by subtracting the mean pre-onset baseline (−0.5–0 s) from the mean post-onset ΔF/F
> response (0–1 s) in each parcel. Negative samples are drawn from matched-count windows more than
> 2 s from any onset *of the same event*; they are not screened against the other three events…
> Every classification is a binary decode of one event against these matched negatives, scored by
> area under the ROC curve (AUC). The four decoders are therefore four independent binary detectors,
> not a four-way classifier.
>
> **Decoders.** The primary decoder is a standardized ℓ2-regularized logistic regression on the
> 44-dimensional evoked feature. This is a linear readout of the parcel-space code… and it is the
> natural comparison to prior cross-subject decoders, which are overwhelmingly linear. Every
> condition below uses this same readout, so the only variable is which data the decoder was trained
> on. [+ long footnote on CNN/GRU]
>
> **Temporal blocks and aging.** We group each mouse's sessions into an early block (days 1–5) and a
> late block (days 11–15)… The personal decoder is estimated by leaving out one whole early *session*
> at a time… [WS/WX/LS/LX definitions] … Aging is the accuracy lost when testing moves from the
> held-out mouse's early data to its late block while training stays fixed… We assess significance
> with a paired sign-flip permutation test across mice (20,000 permutations) and compute 95%
> confidence intervals with a cluster bootstrap over mice.
>
> **External validation dataset.** [Allen cohort selection, the permutation-invariant feature, and
> the is_change contrast.]

Plus a standalone paragraph: *"Holding out sessions rather than trials is essential and not a
detail…"* — currently sitting between the aging definitions.

### 3.1 — Datasets and preprocessing
Both cohorts in continuous prose. She specifically asked for the **why they differ** narrative:
why subject counts and session structure differ between BraiDyn-BC and Allen, not just the numbers.

▶ YOUR VERSION:

### 3.2 — Features and decoders
Merge the current "Features and events" and "Decoders". She'd want the *why linear* reasoning spelled
out for an outside reviewer rather than compressed into one clause.

▶ YOUR VERSION:

### 3.3 — Temporal blocks, aging, and the session-held-out protocol
The WS/WX/LS/LX definitions plus the "holding out sessions is essential" argument, which is currently
orphaned mid-section and duplicated in Results §4.5.

▶ YOUR VERSION:

### 3.4 — The external validation cohort
Cohort selection and the permutation-invariant feature, with the load-bearing argument (no neuron
correspondence across animals) given room rather than compressed.

▶ YOUR VERSION:

---

## PART 4 — Results reorganization

**Her instruction:** thematic blocks, not a long list of small subsections.

### WHAT WE CURRENTLY HAVE — seven subsections

1. A population-pooled decoder ages more slowly than a personal one
2. What the four events are
3. The personal decoder's advantage erodes over the protocol
4. The robustness is not a data-volume artifact
5. Diversity: a secondary scaling with the number of training individuals
6. How much the validation protocol matters
7. The effect replicates on an independent dataset

### PROPOSED REGROUPING

| Block | Merges | Story |
|---|---|---|
| A. The effect | 1 + 3 | Two different estimators, same conclusion |
| B. Not an artifact | 2 + 4 + 5 | Alternative explanations ruled out |
| C. Measuring it correctly | 6 | The protocol result, also a community contribution |
| D. It replicates | 7 | Independent cohort |

(In the 5-page cut, 5 is already folded into 4.)

▶ YOUR CALL on the grouping:

▶ Connecting sentences between blocks are yours:

---

## PART 5 — Discussion

**Her instruction:** "your current discussion is dense and your key contribution is getting eclipsed
by limitations."

### WHAT WE CURRENTLY HAVE — two paragraphs, then one dense Limitations block

> **¶1** A subject's own decoder overfits idiosyncratic features of its early sessions, some of which
> later drift. A decoder that has never seen the subject must instead rely on components of the code
> that are shared across individuals and more stable over time. The count-matched control supports
> this reading directly: at equal data, other-individual data ages less than the subject's own.
> Population structure is therefore not only a resource for cross-subject transfer but also for
> temporal robustness… Unlike within-subject recalibration, this requires no newly labelled data from
> the individual at test time — which is the practical point, since obtaining that data is the
> expensive part of deploying a decoder over months.
>
> **¶2** The effect is small in absolute terms, and we think it is worth being explicit about why we
> believe it anyway. It rests on 25 animals and 357 recording sessions… supported by five checks that
> are not variations on one analysis. [two-block vs day-by-day / count-matched / CNN+GRU /
> session-held-out / second cohort] Each of these could have removed the result and none did.
>
> **Limitations.** The clearest limitation is what the four events are… We therefore cannot always
> say which behavior the effect belongs to, nor can we exclude the idea that the per-event ranking is
> a tracker of how much data each decoder had to overfit. A design with temporally separated,
> count-matched events would settle both, and is the experiment we would run next.
>
> We also measure at the mesoscale, using parcel-averaged activity, and the analysis covers a
> two-week interval in one species. The two-photon replication reaches single neurons only at one
> depth and one visual area. Finally, the trial-level inflation of §4.5 is worth stating as a caution
> as much as a result: it is large enough to manufacture an effect that is not there, and any
> comparison of this shape should be reported with sessions held out.

### 5.1 — Best-practices checklist (new; she wants a formal box or numbered list)
Nothing like this exists yet. The guidance is currently scattered: the session-held-out
recommendation is in the contributions and §4.5, the count-matched design is in §4.4, and the
"report both arms" point is implicit. Candidate items: hold out whole sessions, not trials;
count-match training sets before comparing arms; report each arm's aging separately, not only the
difference.

▶ YOUR VERSION (and: numbered list, or boxed float?):

### 5.2 — Split Limitations into separate paragraphs
Four distinct concessions currently share two paragraphs: what the four events are; mesoscale vs
single-neuron; two-week / one-species scope; and the trial-level inflation caution. Her point is
that the contribution gets eclipsed — so the split matters, and possibly the order.

▶ YOUR CALL on the split and any rewording:

---

## PART 6 — Terse sentences

You've said to follow her method here, so the default is **rewrite these with transitions**. Listed
so you can veto individually rather than losing them all by default — they're yours and deliberate.

1. "The pooled estimate is the claim we defend; the per-event column is data."
2. "We cannot separate that explanation from a behavioral one with four events, and we do not try."
3. "One thing does not transfer."
4. "Every number above holds out whole sessions."
5. "Each of these could have removed the result and none did."
6. "The reward decoder is therefore never reading reward in isolation."

▶ ANY YOU WANT KEPT AS-IS:

---

## PART 7 — Anything else

Her closing note: "Please don't feel the need to accept each and every suggestion above. Overall,
your results seem solid and some writing fine tuning will significantly increase acceptance chances."

▶ YOUR NOTES:
