# racah

Racah–Wigner calculus for compact Lie groups, in Rust: irrep data,
Clebsch–Gordan coefficients, and recoupling coefficients (6j / F / R) for
SU(2), SU(N), SO(N), and Sp(2N) — computed at runtime, with no precomputed
irrep ceiling.

## Why this crate exists

Symmetric tensor libraries need, for every symmetry group they support, the
coefficients of its representation theory: fusion multiplicities, dimensions,
duals, Frobenius–Schur indicators, and above all the recoupling data (the
F- and R-symbols, i.e. generalized 6j coefficients) that drive every basis
transformation of a symmetric tensor.

Two supply models exist today, and both have a ceiling:

- **Precomputed tables** (offline generation, checked-in blobs) are complete
  for *finite* fusion categories — anyon models have a fixed, small sector
  set, so a table is the whole truth. But for compact Lie groups the sector
  set is infinite: a growing simulation produces ever-larger irreps through
  fusion closure, and any table has a cut that will eventually be exceeded.
- **External coefficient crates** solve one group at a fixed scope (e.g.
  exact SU(2) 3j/6j with a bounded label domain) and cannot be extended to
  SU(N≥3) or SO(N), where no closed-form expressions exist.

`racah` removes both ceilings: coefficients for any irrep pair are computed
on demand, inside the process, in pure Rust. It consolidates the roles of
three production references into one crate:

| Reference | What is taken from it |
|---|---|
| WignerSymbols.jl | the exact SU(2) model: big-rational Racah sums, prime-factorized factorials as the measured-need upgrade |
| SUNRepresentations.jl (Alex–Kalus–Huckleberry–von Delft, J. Math. Phys. 52, 023507 (2011)) | the SU(N) pipeline: Gelfand–Tsetlin patterns, exact ladder matrices, highest-weight nullspace, deterministic gauge canonicalization, weight-ladder descent |
| QSpace v4 (Weichselbaum) | the SO(N)/Sp(2N) pipeline: per-family defining-representation seeds feeding one family-generic decomposition loop; and the production discipline — abort on tolerance violation, per-representation error recording, precision tiers |

These are complementary, not competing: the Gelfand–Tsetlin construction is
fundamentally SU(N)-specific, while QSpace's generator-based decomposition is
the only production reference that generates SO(N) and Sp(2N).

## What it computes

- Irrep labels, dimensions, duals, Frobenius–Schur indicators.
- Product decomposition: fusion multiplicities N^c_ab (exact combinatorics).
- Clebsch–Gordan coefficients C^{ab→c} (m-basis tensors, outer multiplicity
  as a trailing index).
- Recoupling coefficients: F-symbols (contraction of four CGC over all
  magnetic indices, leaving the multiplicity indices) and R-symbols
  (symmetric braiding phases). For SU(2) these reduce to the closed-form
  Racah/6j expressions and are computed exactly.
- Self-check functions: CGC orthogonality, F-unitarity, R-orthogonality,
  pentagon/hexagon identities — shipped as public API so they double as
  generation gates and as oracle harnesses for downstream users.

## What it deliberately is not

- **No fusion-category trait vocabulary.** No sector-identity types, no
  tensor-network concepts, no dependency on any tensor engine. `racah`
  answers "what are the correct numbers"; consumers translate them into
  their own categorical interfaces. (The category of representations of a
  compact group is one fusion category among many; a consumer's engine
  should not be able to tell whether an F-block came from this crate, a
  closed form, or a checked-in table.)
- **No pentagon solving for finite fusion categories.** Anyon models
  (Fibonacci, Ising, …) have complete exact F/R data published (e.g. the
  AnyonWiki classification, all multiplicity-free categories up to rank 7);
  those are a data-conversion problem for the consumer, not a computation
  problem for this crate.
- **No symbolic algebraic-number coefficients.** See the exactness contract
  below.

## Design

### Layering

```
racah
├─ base (minimal dependencies)
│   └─ SU(2): exact 3j/6j/CGC — closed-form big-rational Racah sums,
│      canonical Regge keys, bounded publication cache, no doubled-spin
│      ceiling; a single final rounding to floating point
└─ feature "cgc-gen"
    ├─ SU(N):  GT-pattern basis → exact Rational ladder matrices →
    │          highest-weight nullspace → gauge canonicalization
    │          (positive-diagonal QR ∘ column-pivoted reduced echelon,
    │          pivot rules part of the specification) → ladder descent
    ├─ SO(N)/Sp(2N): per-family defining-rep seeds (simple-root raising
    │          operators + Cartan generators) → shared decomposition loop
    │          (raising-operator seed → Gram–Schmidt sweep → column QR)
    ├─ CGC → F/R contraction (m-indices contracted, multiplicity indices
    │          [μ,ν,κ,λ] remain)
    ├─ verification gates (orthogonality, unitarity, pentagon/hexagon)
    └─ bounded provider-internal coefficient caches
```

The feature boundary is mathematical, not organizational: SU(2) has closed
forms and needs no matrix computation; every other family must be generated
numerically. Consumers that only need abelian or SU(2) symmetries never pull
a linear-algebra stack.

### Kernel routing

All dense numerical work behind `cgc-gen` — the nullspace/QR/least-squares
factorizations and the CGC contractions producing F/R — routes through a
selectable dense backend. `racah` contains no hand-rolled numeric kernels:
the backend a consumer selects for its tensor computations is the backend
used for coefficient generation. An extended-precision tier (the QSpace
model: compute in ~128-bit precision, tighten tolerances, store f64) is a
future backend capability with an explicit unsupported boundary until
implemented, not a private arithmetic stack inside this crate.

### Exactness contract

Coefficient *values* are floating point — as in every production reference
(the Julia SU(N) stack is Float64 end-to-end after the ladder matrices;
QSpace is double or MPFR-128; exact algebraic-number coefficients exist only
in research-scale tools). What is exact is the *computation*:

1. **Combinatorial structure is exact.** Pattern enumeration, fusion
   multiplicities, weight systems, and multiplicity dimensions use
   integer/rational arithmetic only.
2. **Discrete data is exact.** Duals, Frobenius–Schur phases, signs, and
   basis ordering are combinatorial facts, never numerical results.
3. **Gauge fixing is deterministic.** The canonicalization is a specified,
   deterministic function of the nullspace subspace (pivot rules and sign
   conventions included); a discrete gauge flip across runs, builds, or
   backends is a defect, not a tolerance event.
4. **Versioned values.** The generation algorithm and gauge are part of this
   crate's semantic-versioning contract: any change that can alter
   coefficient values is a breaking change, so consumers can key caches and
   persisted data on the crate version.
5. **Verification-gated floating point.** Orthogonality, unitarity, and
   pentagon/hexagon checks run at generation time; a tolerance violation is
   a typed error, never a silently degraded coefficient.

This generalizes the exact-SU(2) tradition (compute in rationals, round
once): for generated families the single rounding point moves earlier — into
the nullspace solve — while structure, gauge, and verification stay at the
same standard.

### Gauge continuity

The SU(N) pipeline reproduces the gauge of its reference implementation by
construction: the canonical gauge is a deterministic function of the GT basis
order and the nullspace subspace, so a faithful port reproduces
reference-generated coefficient tables to numerical tolerance. Existing
table-based deployments can therefore demote their tables from authority to
oracle fixtures. SO(N)/Sp(2N) carry their own gauge tag; cross-checks against
QSpace numbers go through an explicit gauge-transformation harness.

## Verification strategy

Oracles are independent of the code under test:

- exhaustive agreement with the existing exact SU(2) crate over its label
  domain, plus reference-generated fixtures beyond it;
- regeneration diffs against reference-generated SU(N) tables (gauge
  continuity makes this a direct comparison);
- Regge/tetrahedral symmetries, pentagon/hexagon identities, and
  orthogonality as internal consistency gates;
- QSpace numbers for SO(N)/Sp(2N) after gauge alignment.

## Status

Scaffold only; no implementation yet. Planned order: exact SU(2) base →
GT combinatorics (exact layer) → SU(N) generation → F/R contraction and
verification gates → SO(N)/Sp(2N).

## License

MIT OR Apache-2.0
