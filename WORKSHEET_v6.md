# Cortical Decoders — Worksheet v6 (additive only)

**Use this one. Ignore v5.** v5 asked you to re-dictate sentences you already dictated in v4, which
was wasted work. This version contains only what is genuinely new.

**The key realisation, which shrinks this a lot.** If you take Option A below, the two-dataset
reframe is almost entirely *additive*. Your abstract sentence, your contribution bullets and your
intro sentence all stay exactly as you dictated them in v4 — the new material goes *after* them
rather than replacing them. Only **two** sentences you wrote actually have to change, and they are
in Part 3, with the edits already drafted so you can just approve or strike them.

So: Part 1 is one decision. Part 2 is my prose that needs your voice. Part 3 is two of your
sentences. That's it.

## PART 1 — The one decision

**Option A (proposed).** Headline both cohorts. BraiDyn-BC stays the primary analysis and keeps
every Results section it has; Allen becomes the external-validation section. Your existing prose is
untouched — new sentences are added alongside it.

**Option B.** Headline Allen because the effect is larger there (+0.023 vs +0.017). This inverts the
paper, since every Results section is BraiDyn, and trades your strongest statistics (p = 5e-5, 100
pairs, four supporting checks) for your largest point estimate (p = 0.003, 23 mice, count-matched
only marginal at p = 0.054). It would also mean re-dictating most of what you did in v4.

**Option C.** Headline both but interleave the cohorts per claim rather than BraiDyn-then-Allen.
Reads more like a genuine two-dataset paper; costs a restructure of Results.

▶ YOUR CALL (A / B / C):

## PART 2 — My prose. This is what needs your voice.

All of this is text I wrote. None of it has been through you.

### 2.1 — Abstract, the new sentence (goes after your existing +0.017 sentence, which does not change)

MINE: The effect is not confined to that cohort or that recording modality: on an independent
two-photon dataset from a different laboratory (Allen Visual Behavior, 23 mice, single-plane VISp
populations decoding a stimulus change) the same session-held-out protocol gives +0.023 AUC
(p = 0.003, 74% of mice), and the count-matched control reproduces there as well.

⚠ AI-tells: "not confined to" is a stock hedge-flip, and the colon-reveal is the shape you cut twice
in v4.

▶ YOUR VERSION:

### 2.2 — Contributions, the new bullet (goes after your existing three, which do not change)

MINE: **It replicates on an independent dataset.** The same protocol applied to the Allen Visual
Behavior two-photon cohort — a different laboratory, recording modality, task and feature space, at
cellular rather than mesoscale resolution — gives an asymmetry of +0.023 AUC (p = 0.003, 74% of 23
mice), with the count-matched ordering also reproduced.

⚠ AI-tells: em-dash parenthetical stacking four nouns; "with X also reproduced" as a trailing
absolute.

▶ YOUR VERSION:

### 2.3 — Results 5.7, opening

MINE: Everything above rests on one cohort, one imaging modality and one task. To test whether the
asymmetry is a property of that setting we repeat the analysis on the Allen Visual Behavior
two-photon dataset: a different laboratory, cellular two-photon rather than mesoscale widefield
imaging, a visual change-detection task rather than a cued lever pull, and a 27-dimensional
permutation-invariant population summary rather than a 44-parcel atlas feature. The decoded contrast
is a stimulus change, so it is much less coupled to reward and licking than anything decoded on
BraiDyn-BC. Little is shared between the two analyses except the question and the protocol.

⚠ AI-tells: the last sentence is a closing flourish of exactly the kind you cut in v4.

▶ YOUR VERSION:

### 2.4 — Results 5.7, the result

MINE: The asymmetry reproduces at comparable magnitude. Across 23 mice the personal decoder ages by
+0.024 AUC and the pooled decoder by +0.001, an asymmetry of +0.023 [0.010, 0.037], p = 0.003, with
the mouse's own decoder ageing more in 74% of animals. Early-block accuracy is 0.672 for the personal
decoder and 0.638 for the pooled one, so the familiar ordering holds at the start; by the late block
the personal decoder leads by only 0.012, about a third of its initial advantage.

NOTE: numbers locked. Worth keeping the raw AUCs in — 0.67 is an honest decode, not a spectacular
one, and saying so beats letting a reviewer find it.

▶ YOUR VERSION:

### 2.5 — Results 5.7, the mechanism difference (this one carries a real constraint)

MINE: The mechanism is not identical, and the difference is worth stating. On BraiDyn-BC the gap
narrows mostly because the pooled decoder gains over days as the animal becomes more practised,
while the personal decoder stays flat. Here the pooled decoder is flat (+0.001) and the entire
asymmetry comes from the personal decoder losing accuracy. The shared finding is that the personal
decoder ages faster, not the route by which the two arms separate.

NOTE: this is your "rising tide / can't ride the wave" framing from v4. It does **not** transfer to
Allen. It has to stay marked as BraiDyn-specific or it becomes a false claim about the second
dataset.

▶ YOUR VERSION:

### 2.6 — Results 5.7, count-matched + inclusion rule

MINE: The count-matched control reproduces as well. At equal training-event count the self decoder
ages by +0.024, a decoder from one other mouse by +0.013 and one drawn across the other mice by
+0.003. The self-minus-one gap is +0.011 [0.001, 0.022], close to the +0.010 measured on BraiDyn-BC,
though on 23 animals it is marginal by the sign-flip test (p = 0.054); the self-minus-many gap is
+0.021 [0.010, 0.033], p = 0.002. The diversity ordering is again the most robust part: many other
mice age less than one other mouse by −0.010 [−0.016, −0.004], p = 0.003. The result also does not
depend on where the inclusion threshold is drawn, giving +0.022 at three or more sessions per mouse
(24 mice), +0.023 at four (23), +0.023 at five (22) and +0.024 at six (18).

NOTE: the p = 0.054 is deliberately not buried. It is the soft spot in the replication and conceding
it flatly is worth more than hiding it — same logic as the v4 retreat you were happy with.

▶ YOUR VERSION:

### 2.7 — Results 5.7, the protocol reversal (the subtlest paragraph in the paper)

MINE: One thing does not transfer, and it sharpens rather than weakens the recommendation of the
protocol section. Repeating the trial-level split here lowers the asymmetry, to +0.014 (p = 0.07),
rather than raising it as it does on BraiDyn-BC. Two things differ between the protocols and only one
of them is leakage, so we separate them on the same trials. Holding the scoring fixed, trial
shuffling still inflates the personal decoder's within-block accuracy, by +0.015, in the same
direction and of the same size as on BraiDyn-BC. But the trial-split protocol also scores a single
pooled AUC across sessions rather than averaging per-session values, and on this dataset that costs
−0.030 (p < 1e-4), because between-session shifts in the decision variable depress a pooled AUC in a
way a within-session one never sees. The larger of the two effects is the scoring convention, which
is why the naive comparison points the wrong way. The leakage itself behaves exactly as before, and
the corrected estimate is here the larger one, so the replication is not an artifact of the split.

NOTE: the claim is — leakage still runs the same direction (+0.015), a second and larger scoring
effect (−0.030) masks it, and the corrected number is the bigger one, so the replication survives.
If that is unclear to you it will be unclear to a reviewer; say so and I will restructure the
argument rather than reword it.

▶ YOUR VERSION:

### 2.8 — Methods, the Allen dataset paragraph

MINE (condensed): We select containers spanning at least 14 days whose experiments are all in VISp
within a single 50 µm depth bin; the largest such band, 150–199 µm, holds 25 mice and 184
experiments. We require at least 15 segmented cells and 40 usable trials, leaving 166 experiments and
23 mice with at least four sessions. There is no neuron correspondence across animals — mouse A's
cell 17 is unrelated to mouse B's cell 17 — and the longitudinal subset is single-plane, so averaging
over the field of view would give a one-dimensional feature. The comparable representation is
therefore a permutation-invariant summary: per neuron the mean dF/F over 0–1 s minus the mean over
−0.5–0 s, then that distribution described by 25 quantiles plus mean and sd, giving a fixed 27-dim
feature regardless of how many cells were segmented.

NOTE: mostly mechanical, but the permutation-invariance argument is load-bearing — it is why this
dataset is usable at all.

▶ YOUR VERSION:

## PART 3 — The only two sentences of yours that have to change

Edits already drafted. Say "yes" to take mine, or overwrite in your own words. Nothing else you
dictated in v4 is touched.

### 3.1 — Limitations. This one is a factual contradiction, not a style call.

YOURS, CURRENTLY: "We also measure at the mesoscale, using parcel-averaged activity, so whether the
same result holds for single-neuron codes — where representational drift is best characterized —
remains unknown, and the analysis covers a two-week interval in one species."

THE PROBLEM: the Allen replication *is* single-neuron two-photon data. As written, the paper
contradicts its own Section 5.7.

MY PROPOSED MINIMAL EDIT — delete the clause that is now false, keep the rest of your sentence
intact, and add one sentence that turns the concession into a narrower and more honest one:

"We also measure at the mesoscale, using parcel-averaged activity, and the analysis covers a
two-week interval in one species. The two-photon replication does reach single neurons, but only at
one depth in one visual area."

▶ YES / YOUR VERSION:

### 3.2 — Discussion. One number, one added clause.

YOURS, CURRENTLY: "...it is supported by four checks that are not variations on one analysis. [...]
Each of these could have removed the result and none did."

THE PROBLEM: there are five now, and the fifth — an independent cohort, another lab, another
instrument, another task — is the strongest of them. Leaving it at four undersells the paper, which
is the thing you said in v4 you did not want.

MY PROPOSED MINIMAL EDIT — "four" becomes "five", and one sentence goes in before your closing line:

"And it reproduces on a second cohort recorded in another laboratory, with a different instrument
and a different task."

▶ YES / YOUR VERSION:

## PART 4 — Anything else

Anything in Part 2 that does not sound like you, or any claim now over- or under-sold?

▶ YOUR NOTES:
