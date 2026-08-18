# Workshop submission builds

Source of truth is `paper_neurips/pooling.tex` (8 pp main text). These are derived from it.

| dir | venue | deadline | limit | template | build | main text now | cut needed |
|---|---|---|---|---|---|---|---|
| `brainbodyfm_2026/` | Foundation Models for the Brain and Body | Sept 5 | 5 pp excl. refs + appendices | their `neurips_2025.sty` | ✅ clean | **5 pp exactly** | done |
| ~~`neuroai_2026/`~~ | ~~Closed-Loop NeuroAI~~ | — | — | — | — | — | **workshop NOT accepted for 2026** |
| `neurreps_2026/` | NeurReps Extended Abstract | **Aug 22** | 4 pp excl. refs | `jmlr` `mlabstract` | ✅ clean | ~9 pp | ~56% |

Both venues are double-blind; the source is already anonymised.

**All mechanical work is done.** What remains for both is a content cut, which is an
editorial decision on the author's prose, not a formatting task.

## Notes

- **NeuroAI needed no reformatting** — its CFP requires the NeurIPS 2026 template, which the
  source already uses. `pooling.tex` is copied verbatim.
- **NeurReps required a real conversion**: `\documentclass[mlabstract,onecolumn]{jmlr}`, a
  single `.tex` (`abstract.tex`), and references in a `.bib`. The source's embedded
  `thebibliography` (21 `\bibitem`s) was converted to `refs.bib` with **citation keys
  preserved**, so every `\citep{}` still resolves — verified, 0 undefined citations.
- **The jmlr format is less dense than NeurIPS style**: identical content renders at ~9 pp
  here versus ~8 pp there. So NeurReps needs the larger cut *and* has the earlier deadline.
- Building `neurreps_2026/` requires `texlive-science` (for `algorithm2e.sty`, auto-loaded by
  the jmlr class). Installed on this box 2026-08-06.

## Build

    cd neuroai_2026  && pdflatex pooling.tex && pdflatex pooling.tex
    cd neurreps_2026 && pdflatex abstract.tex && bibtex abstract && pdflatex abstract.tex && pdflatex abstract.tex


## The 5-page cut (`brainbodyfm_2026/pooling_5pp.tex`)

Main text lands on **exactly 5 pages**; references start on p6; appendices run p7-9.
Brain-and-Body excludes **both** references *and appendices* from the limit, so detail was
**moved, not deleted**:

| Moved to appendix | Kept in main text |
|---|---|
| Full related work (App. A) | Headline asymmetry +0.017 and Table 1 |
| Full methods (App. B) | Event-overlap honesty (short form) |
| Day-by-day decay regression (App. C) | Count-matched control + Fig. 1 |
| Event timing/overlap detail (App. D) | Protocol caution (short form) |
| Validation-protocol detail (App. E) | Allen replication in full + Table 2 |

**Genuinely deleted**, not moved: `fig_decay`; the §5.5 scaling subsection (folded into §5.4 as
two sentences); the standalone Reproducibility section (folded into Limitations as one sentence);
and roughly six sentences of the author's prose, removed whole rather than reworded.

`neuroai_2026/` is retained only for provenance — that workshop was not among the 102 accepted.
