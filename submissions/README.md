# Workshop submission builds

Source of truth is `paper_neurips/pooling.tex` (8 pp main text). These are derived from it.

| dir | venue | deadline | limit | template | build | main text now | cut needed |
|---|---|---|---|---|---|---|---|
| `neuroai_2026/` | NeuroAI: Closed-Loop NeuroAI | Aug 29 | 5 pp excl. refs | NeurIPS 2026 (**same as source**) | ✅ clean | ~8 pp | ~37% |
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
