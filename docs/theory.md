# Theory note — moved to LaTeX

> **AI-generated, for agentic coding.** Written by an AI agent as reference
> material for AI agents (and humans) working on this repository. It may contain
> errors — check it against the code and tests rather than trusting it blindly.
> It is **not normative**: [`docs/gauge.md`](gauge.md) and
> [`docs/gauge_soN.md`](gauge_soN.md) remain the frozen authority for every
> basis, phase, ordering and normalization convention.

The theory material is a self-contained mathematical note covering the objects
`racah` computes — compact groups and highest weights, Gelfand–Tsetlin bases and
the B/C/D generator bootstrap, Clebsch–Gordan coefficients with their
orthonormality and completeness relations, gauge freedom and what freezing a
gauge means, the F and R symbols as explicit CGC contractions in the crate's
index convention, the pentagon and hexagon identities, Frobenius–Schur
indicators, and global forms — together with the prior-literature results the
implementation rests on, cited.

**The full note is [`docs/theory.tex`](theory.tex), built to
[`docs/theory.pdf`](theory.pdf)** (16 pages; GitHub renders the PDF in blob
view). Rebuild with `pdflatex theory.tex` twice — no bibtex, no external
packages beyond a standard TeX Live install.

Porting provenance and the shared bibliography: [`docs/references.md`](references.md).
