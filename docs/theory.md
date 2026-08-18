# Theory note — moved to LaTeX

> **AI-generated, for agentic coding.** Written by an AI agent as reference
> material for AI agents (and humans) working on this repository. It may contain
> errors — check it against the code and tests rather than trusting it blindly.
> It is **not normative**: [`docs/gauge.md`](gauge.md) and
> [`docs/gauge_soN.md`](gauge_soN.md) remain the frozen authority for every
> basis, phase, ordering and normalization convention.

**Role in the documentation set.** Mathematics, not usage. For learning the
library see the [User Guide](user-guide/README.md); documentation map:
[`docs/README.md`](README.md).

The theory material is a self-contained mathematical note organised around the
**Killing–Cartan classification**. It starts from the nine families of simple
Lie algebras (A_r, B_r, C_r, D_r, G2, F4, E6, E7, E8) with their Dynkin
diagrams, derives each family's compact simply connected group and centre and
the global forms as centre quotients, records which entries of the
classification `racah` actually implements and by which code path — mirroring
`RootSystem` and `GroupId = RootSystem × GlobalForm` in `src/group.rs` — and
then covers the objects the crate computes: highest weights, Gelfand–Tsetlin
bases and the B/C/D generator bootstrap, Clebsch–Gordan coefficients with their
orthonormality and completeness relations, gauge freedom and what freezing a
gauge means, the F and R symbols as explicit CGC contractions in the crate's
index convention, the pentagon and hexagon identities, and Frobenius–Schur
indicators — together with the prior-literature results the implementation
rests on, cited.

**The full note is [`docs/theory.tex`](theory.tex), built to
[`docs/theory.pdf`](theory.pdf)** (21 pages; GitHub renders the PDF in blob
view). Rebuild with `pdflatex theory.tex` twice — no bibtex, no external
packages beyond a standard TeX Live install.

Porting provenance and the shared bibliography: [`docs/references.md`](references.md).
