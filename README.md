# racah

Racah–Wigner calculus for compact Lie groups, in Rust.

Irrep labels, dimensions, duals, product decomposition (fusion
multiplicities), Clebsch–Gordan coefficients, and recoupling coefficients
(6j / F / R) for:

- **SU(2)** — exact closed-form 3j/6j/CGC (big-rational Racah sums, no
  doubled-spin ceiling), in the base crate with minimal dependencies.
- **SU(N)** — runtime generation via the Gelfand–Tsetlin construction
  (`cgc-gen` feature).
- **SO(N) / Sp(2N)** — runtime generation via defining-representation seeds
  and a family-generic decomposition loop (`cgc-gen` feature).

Self-check functions (orthogonality, unitarity, pentagon/hexagon) ship with
the coefficients and gate generation.

## Design principles

- **Pure mathematics.** No fusion-category traits, no tensor-network
  concepts. Consumers translate the output into their own interfaces.
- **Computational exactness over value exactness.** Combinatorial structure
  and discrete data are exact (integer/rational); gauge fixing is a
  deterministic, specified function of the subspace; floating-point stages
  are verification-gated. Coefficient values are floating point, as in every
  production reference.
- **No hand-rolled kernels.** Dense factorizations and contractions behind
  `cgc-gen` route through a selectable backend.
- **Versioned gauge.** The gauge and generation algorithm are part of this
  crate's semantic-versioning contract: a change that can alter coefficient
  values is a breaking change.

## References

- WignerSymbols.jl (exact SU(2) model)
- SUNRepresentations.jl / Alex, Kalus, Huckleberry, von Delft,
  J. Math. Phys. 52, 023507 (2011) (Gelfand–Tsetlin SU(N) construction)
- QSpace v4 (SO(N)/Sp(2N) generation and production discipline)

## Status

Early scaffold; API unstable.

## License

MIT OR Apache-2.0
