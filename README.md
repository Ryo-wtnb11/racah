# racah

Racah–Wigner calculus for compact Lie groups, in Rust: irreducible
representations, Clebsch–Gordan coefficients, and recoupling coefficients
(3j / 6j / F / R) for SU(2), SU(N), SO(N), and Sp(2N).

Coefficients for any admissible labels are computed on demand in exact or
verification-gated arithmetic — there is no precomputed table and no label
ceiling.

## Installation

Not yet published to crates.io: the `cgc-gen` feature depends on
[tenferro-rs](https://github.com/tensor4all/tenferro-rs), which is not itself
published, so a crates.io release is blocked upstream. The supported path today
is a git dependency:

```toml
[dependencies]
racah = { git = "https://github.com/Ryo-wtnb11/racah" }
# generated SU(N)/SO(N)/Sp(2N) families:
# racah = { git = "https://github.com/Ryo-wtnb11/racah", features = ["cgc-gen"] }
```

MSRV: latest stable Rust (CI pins no minimum version; it builds and tests on
`stable`).

## Feature flags

| Feature | Adds | Pulls in |
|---|---|---|
| *(default)* | Exact SU(2): 3j / 6j / Clebsch–Gordan / F / R, closed-form big-rational | `num-bigint`, `num-rational`, `num-traits` only |
| `cgc-gen` | Runtime CGC / F / R generation for SU(N) (Gelfand–Tsetlin) and SO(N)/Sp(2N) (generator bootstrap) | `tenferro-linalg` / `-cpu` / `-runtime` (the dense factorization + contraction backend) |

The `cgc-gen` dependencies are pinned to an exact `tenferro-rs` git revision
(see `Cargo.toml`), so a fresh checkout resolves without a sibling tenferro on
disk. Consumers enabling `cgc-gen` inherit that pinned revision; bumping it is
an ordinary reviewed commit. The default build stays dependency-light and needs
no linear-algebra stack.

## Quick start

One minimal example per layer. Each is a literal copy of a crate doctest, so it
compiles against the current API (`cargo test` / `cargo test --all-features`).

Exact SU(2) 6j (base, no features). Spins are doubled (`dj = 2j`), so `2` means
spin 1; `{1 1 1; 1 1 1} = 1/6`:

```rust
use racah::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

SU(N) F-symbol (`cgc-gen`). Irreps are built from Dynkin labels (length `N-1`);
this is the SU(3) sextet `1 ⊗ 3 ⊗ 3 → 6`, a `1×1×1×1` identity move:

```rust
use racah::sun::{f_symbol, Irrep};

let triv = Irrep::trivial(3).unwrap(); // SU(3) singlet
let three = Irrep::from_dynkin(&[1, 0]).unwrap(); // fundamental
let six = Irrep::from_dynkin(&[2, 0]).unwrap();

let block = f_symbol(&triv, &three, &three, &six, &three, &six).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
```

SO(N)/Sp(2N) F-symbol (`cgc-gen`). Generation runs through a per-(series, rank)
`CanonicalCatalog` that caches the aligned CGC; this is an Sp(4) (`C_2`) block:

```rust
use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4)
let triv = Irrep::trivial(Series::C, 2).unwrap();
let v = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // vector
let adj = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap(); // in v ⊗ v

let block = f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

## Why this crate exists

No library — in Rust, and essentially nowhere as a standalone component —
computes the *full* representation-theory coefficient set for the compact Lie
groups on demand with unbounded labels. By "full set" we mean, for a given
group and any admissible irreps: fusion multiplicities, dimensions, duals,
Frobenius–Schur indicators, Clebsch–Gordan coefficients, and the recoupling
data (3j / 6j and the F- and R-symbols). `racah` is that standalone library. It
covers SU(2), SU(N), SO(N), and Sp(2N); it is pure representation mathematics
with no tensor-network vocabulary and no dependency on any tensor engine; and it
is usable by anyone who needs these numbers — atomic and molecular spectroscopy,
nuclear and quantum-chemistry coupling, lattice and continuum models, symmetric
tensor networks, and more.

The existing supply of these coefficients stops short of that, in two ways:

- **Precomputed tables** (offline generation, checked-in data) are complete for
  *finite* symmetry sets — a fixed, small collection of irreps means a table is
  the whole truth. But a compact Lie group has infinitely many irreps, and
  taking tensor products only makes them larger, so any table has a cut that a
  large-enough calculation will exceed.
- **Single-group coefficient packages** solve one group at a fixed scope (for
  example exact SU(2) 3j/6j over a bounded label range) and do not extend to
  SU(N≥3), SO(N), or Sp(2N), where no closed-form expressions exist and the
  coefficients must be *constructed*.

`racah` removes both limits: coefficients for any admissible labels are computed
on demand, inside the process, in pure Rust, with no label ceiling. To do this
faithfully it consolidates the algorithms of three production references, one
per family (full provenance in [`docs/references.md`](docs/references.md)):

| Reference | What is taken from it |
|---|---|
| WignerSymbols.jl | the exact SU(2) model: big-rational Racah sums, prime-factorized factorials as the measured-need upgrade |
| SUNRepresentations.jl (Alex–Kalus–Huckleberry–von Delft, J. Math. Phys. 52, 023507 (2011)) | the SU(N) pipeline: Gelfand–Tsetlin patterns, exact ladder matrices, highest-weight nullspace, deterministic gauge canonicalization, weight-ladder descent |
| QSpace v4 (Weichselbaum) | the SO(N)/Sp(2N) pipeline: per-family defining-representation seeds feeding one family-generic decomposition loop; and the production discipline — abort on tolerance violation, per-representation error recording, precision tiers |

These are complementary, not competing: the Gelfand–Tsetlin construction is
fundamentally SU(N)-specific, while QSpace's generator-based decomposition is
the only production reference that generates SO(N) and Sp(2N). For the
representation-theory background behind these objects, see
[`docs/theory.md`](docs/theory.md).

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

### Why each family gets its algorithm

The construction per family is forced by the group's branching structure, not
chosen for convenience (the full argument is in
[`docs/theory.md`](docs/theory.md) §5):

- **SU(2)** — closed forms exist (Racah), so the 3j/6j/CGC/F/R are evaluated
  directly in exact big-rational arithmetic with a single final rounding; there
  is nothing to generate.
- **SU(N)** — the unitary chain U(N) ⊃ U(N-1) ⊃ … ⊃ U(1) has multiplicity-free
  branching (the intermediate U(1) charge at each step separates copies that the
  SU chain alone would repeat), so basis states of an SU(N) irrep are labelled
  uniquely by Gelfand–Tsetlin patterns and the ladder operators have exact
  closed-form matrix elements (Alex–Kalus–Huckleberry–von Delft). That closed
  form is what makes the direct GT construction possible, and it is SU(N)-specific.
- **SO(N) / Sp(2N)** — the symplectic chain Sp(2r) ⊃ Sp(2r-2) has branching
  multiplicities, so no GT-type basis with practical closed-form matrix elements
  exists (the SO chains are multiplicity-free, and explicit GT-type matrix
  elements for them do exist — Gelfand–Tsetlin 1950; Molev — but they are
  substantially more involved and no production implementation exists). So these
  families use the generator bootstrap —
  defining-representation seeds (writable explicitly per series), tensor
  products, numeric highest-weight decomposition, harvest, recurse — which needs
  almost no family-specific structure. Its price, a gauge fixed by procedural
  determinism rather than a formula, is what
  [`docs/gauge_soN.md`](docs/gauge_soN.md) pins down.

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

For the base SU(2) provider this convention set is exposed as an opaque
fingerprint that changes only on the value-affecting breaking release of point 4
above; see [Provider contract](#provider-contract) below.

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

## Provider contract

The base SU(2) provider (default build, no features) exposes a small, stable
contract so a consumer can use it as one coefficient authority without
duplicating convention identity, representation validation, or cache accounting.
Every item below is base-SU(2)-only and pulls no linear-algebra stack.

### Authority fingerprint

`su2_authority_fingerprint() -> &'static [u8]` returns opaque bytes that
identify the *convention set* every returned SU(2) coefficient (3j, 6j,
Clebsch–Gordan, F, R, Frobenius–Schur) is computed in.

- **What it is** — an identifier for the value-fixing conventions, not a
  document. Treat it as opaque: compare by equality only, never parse it.
- **When it changes** — not on a rebuild, dependency bump, or additive release;
  it is derived from none of the crate version, source, docs, or process state.
  It changes only on a value-affecting *breaking* release — a change that can
  alter a returned coefficient, its normalization, or its canonical convention,
  the same event class point 4 of the exactness contract declares breaking. So
  "fingerprint changed ⇔ value-affecting breaking release" is one reviewable
  invariant, pinned by the compatibility-policy test `tests/su2_fingerprint.rs`.
- **How a consumer uses it** — persist the bytes next to anything derived from
  these coefficients (a cache, a serialized table); on load, compare for
  equality and reject the derived data on mismatch.

### Checked SU(2) representation surface

The `su2` module adds a typed, checked layer over the infallible closed-form
functions. `Su2Irrep` labels an irrep by its doubled spin (`dj = 2j`); every
`u32` is valid, so `Su2Irrep::new` is infallible and `dj` / `dim` / `dual`
cannot fail. `Su2Irrep::fusion` returns a non-allocating `Su2Fusion` iterator
over the coupled irreps, or `Err(Su2Error::LabelOverflow)` when `dj1 + dj2`
exceeds `u32`.

The checked coefficient functions — `wigner_3j_checked`, `wigner_6j_checked`,
`clebsch_gordan_checked`, `su2_f_symbol_checked`, `su2_r_symbol_checked` —
return `Err(Su2Error::NotAdmissible(_))` for a structurally forbidden tuple
(triangle / parity / m-range violation, named by `AdmissibilityViolation`) and
`Ok(value)` otherwise. That `Ok` / `Err` split is the point: an admissible 6j
can still be *accidentally* zero, so `Ok(0)` (a real zero of an admissible
coupling) and `Err(NotAdmissible)` (a forbidden coupling) are finally
distinguishable — where the infallible functions return the same exact zero for
both. The checked layer is purely additive: the infallible `wigner_6j` &c. keep
their zero convention, and both paths share one admissibility predicate, so they
can never disagree.

### Cache resource contract

The three base coefficient tiers (3j, 6j, derived-F) are each bounded
independently by a per-tier entry and byte cap. The documented aggregate ceiling
`BASE_CACHE_MAX_BYTES` (192 MiB = 3 × 64 MiB) is their sum — a **static
partition, not a dynamic shared pool** — and holds as a corollary of the true
per-tier ceilings, tied to the per-tier cap by a `const` assertion so the two
cannot drift.

`base_cache_stats() -> BaseCacheStats` exposes per-tier `TierStats` (`entries`,
`bytes`, `hits`, `misses`, `evictions`) for `three_j`, `six_j`, `derived_f`,
plus a field-wise `total()`. Each per-tier snapshot is consistent under its tier
lock; the total is a sum of per-tier snapshots, not a global atomic snapshot, so
under concurrent fills it is only eventually consistent (no global lock spans the
tiers).

`reset()` returns every tier's entries, bytes, and hit/miss/eviction counters to
zero. It acts on process-global `static` state, so reset ownership is
**single-owner**: exactly one component in a consuming process owns reset policy;
a library must not call it.

## Status

Feature-complete for its v0 scope; all three families are implemented and
oracle-checked:

- **SU(2)** (base): exact 3j / 6j / Clebsch–Gordan / F / R in big-rational
  arithmetic.
- **SU(N)** (`cgc-gen`): the full Gelfand–Tsetlin pipeline — CGC, F, R, with
  outer-multiplicity indices.
- **SO(N) / Sp(2N)** (`cgc-gen`): the generator-bootstrap pipeline (B/C/D
  Cartan series) — CGC, F, R.

Verification (every claim below is backed by a merged test; the crate ships its
self-checks — orthogonality, F-unitarity, pentagon, hexagon — as public API):

- **SU(2)**: exhaustive agreement with `wigner-symbols` 0.5.1 over its label
  domain, plus reference fixtures beyond it.
- **SU(N)**: signed element-wise table regeneration against
  SUNRepresentations.jl v0.4.0 — a dim ≤ 8 slice on every `cargo test`, and a
  full dim ≤ 27 sweep (76,853 F blocks) run explicitly. Products and
  multiplicities are cross-checked against GroupMath 1.1.3 fixtures.
- **SO(N) / Sp(2N)**: the QSpace v4 CGC projector battery — **33** of the
  fixture's rank-2/3 B/C/D channels are projector-tested against QSpace to
  round-off (via verified factor-basis dictionaries), **0** remain
  structural-only, and **9** higher-rank rows (SO(7)/Sp(6)/SO(8)) are out of the
  rank-2/3 anchor's scope and skipped. See
  `src/bcd/qspace_oracle_tests.rs` for the full coverage note.

The base SU(2) provider's stable public surface — authority fingerprint,
checked representation layer, and cache resource contract — is described under
[Provider contract](#provider-contract).

Not published to crates.io yet (blocked on the `tenferro-rs` publish); the git
dependency above is the supported path. See [Installation](#installation) and
[Feature flags](#feature-flags).

## More

- Theory primer (the objects the API computes): [`docs/theory.md`](docs/theory.md).
- Porting provenance and bibliography: [`docs/references.md`](docs/references.md).
- Gauge conventions: [`docs/gauge.md`](docs/gauge.md) (SU(N)),
  [`docs/gauge_soN.md`](docs/gauge_soN.md) (SO(N)/Sp(2N)).
- Fixture provenance and the oracle matrix: [`tools/README.md`](tools/README.md).
- Guard discipline (every port PR carries a guard inventory): issue
  [#15](https://github.com/Ryo-wtnb11/racah/issues/15).

## License

MIT OR Apache-2.0
