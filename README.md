# racah

[![CI](https://github.com/Ryo-wtnb11/racah/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/racah/actions/workflows/ci.yml)
[![wheels](https://github.com/Ryo-wtnb11/racah/actions/workflows/wheels.yml/badge.svg)](https://github.com/Ryo-wtnb11/racah/actions/workflows/wheels.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/racah/badges/badge.json)](https://github.com/Ryo-wtnb11/racah/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/racah/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/racah)
[![crates.io](https://img.shields.io/crates/v/racah.svg)](https://crates.io/crates/racah)
[![docs.rs](https://img.shields.io/docsrs/racah)](https://docs.rs/racah)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Racah–Wigner calculus for compact Lie groups, in Rust: irreducible
representations, Clebsch–Gordan coefficients, and recoupling coefficients
(3j / 6j / F / R) for SU(2), SU(N), SO(N), and Sp(2N).

Coefficients for any admissible labels are computed on demand — exactly for
SU(2), and through a deterministically gauged, verification-gated numerical
pipeline for the generated families. There is no precomputed table and no
generation-time label cut: labels are bounded only by the machine-word ranges
of the label types, which report a typed overflow error rather than a wrong
answer.

## Supported groups

| Family | Module | Feature | Labels |
|---|---|---|---|
| SU(2) | `racah::su2` | *(default)* | doubled spin `dj = 2j` |
| SU(N), SU(N)/Z_k, PSU(N) | `racah::sun` | `cgc-gen` | `N-1` Dynkin labels |
| SO(N), Spin(N) | `racah::bcd` (series `B`, `D`) | `cgc-gen` | `r` Dynkin labels |
| Sp(2r) | `racah::bcd` (series `C`) | `cgc-gen` | `r` Dynkin labels |

## What racah computes

For any admissible irreps of a supported group:

- **Labels and structure** — dimensions, duals, Frobenius–Schur indicators,
  weight multiplicities. Exact integer arithmetic.
- **Fusion** — the tensor-product decomposition and its multiplicities
  `N^c_ab`. Exact.
- **Clebsch–Gordan coefficients** — m-basis tensors with the outer multiplicity
  on a trailing axis.
- **Recoupling coefficients** — F-symbols (rank-4 blocks over the four vertex
  multiplicity indices) and R-symbols (braiding).
- **Self-checks as public API** — F-unitarity and the pentagon/hexagon
  identities per generated family, so they double as generation gates and as
  oracle harnesses for your own labels.

## Installation

```toml
[dependencies]
racah = "0.2.0"
# generated SU(N)/SO(N)/Sp(2N) families:
# racah = { version = "0.2.0", features = ["cgc-gen"] }
```

| Feature | Adds | Pulls in |
|---|---|---|
| *(default)* | Exact SU(2): 3j / 6j / Clebsch–Gordan / F / R, closed-form big-rational | `num-bigint`, `num-rational`, `num-traits` only |
| `cgc-gen` | Runtime CGC / F / R generation for SU(N) (Gelfand–Tsetlin) and SO(N)/Sp(2N) (generator bootstrap) | `tenferro-linalg` / `-cpu` / `-runtime` (the dense factorization + contraction backend) |

The feature boundary is mathematical, not organizational: SU(2) has closed forms
and needs no matrix computation, so consumers who need only SU(2) never pull a
linear-algebra stack. No fixed MSRV — `racah` builds and is tested on current
stable Rust.

With `cgc-gen` enabled the coefficient caches may retain several hundred MiB by
default; call `racah::cache::configure_cache_budgets` once before first use to
lower that bound
([User Guide: resources](docs/user-guide/resources.md#bounding-cache-memory)).

## Quick start

One example per family. Each is a literal copy of a crate doctest, so it
compiles against the current API.

Exact SU(2) 6j (base, no features). Spins are doubled (`dj = 2j`), so `2` means
spin 1; `{1 1 1; 1 1 1} = 1/6`:

```rust
use racah::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

SU(N) (`cgc-gen`). Irreps are built from Dynkin labels (length `N-1`); this
decomposes the SU(3) product `8 ⊗ 8`, where the adjoint appears twice:

```rust
use racah::sun::{directproduct, Irrep};

let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
assert_eq!(eight.dim(), 8u32.into());
assert_eq!(directproduct(&eight, &eight).unwrap()[&eight], 2);
```

SO(N)/Sp(2N) (`cgc-gen`). Generation runs through a per-(series, rank)
`CanonicalCatalog` that caches the aligned CGC; this is an Sp(4) (`C_2`)
F-symbol block:

```rust
use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4)
let triv = Irrep::trivial(Series::C, 2).unwrap();
let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // the 5
let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap();  // the adjoint 10

let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

Spinor irreps need the simply connected cover, named explicitly:

```rust
use racah::bcd::Irrep;
use racah::group::GroupId;

let spin7 = GroupId::spin(7).unwrap();
let s = Irrep::from_dynkin_in(&spin7, &[0, 0, 1]).unwrap();
assert_eq!(s.dim(), 8u32.into());

// The same label is not a representation of SO(7).
assert!(Irrep::from_dynkin_in(&GroupId::so(7).unwrap(), &[0, 0, 1]).is_err());
```

## Documentation

| Layer | For | Where |
|---|---|---|
| **User Guide** | Learning the library, task by task | [`docs/user-guide/`](docs/user-guide/README.md) |
| **API Reference** | Exact per-item semantics, shapes, errors | [docs.rs/racah](https://docs.rs/racah) |
| **Theory** | The mathematics behind the objects | [`docs/theory.pdf`](docs/theory.pdf) ([source](docs/theory.tex)) |
| **Gauge specification** | The normative basis/sign/ordering conventions (frozen) | [`docs/gauge.md`](docs/gauge.md), [`docs/gauge_soN.md`](docs/gauge_soN.md) |
| **Developer docs** | Changing racah itself | [`docs/developer/`](docs/developer/README.md), [`AGENTS.md`](AGENTS.md), [`tools/README.md`](tools/README.md) |
| **Provenance** | What was ported from where, symbol by symbol | [`docs/references.md`](docs/references.md) |

Documentation index: [`docs/README.md`](docs/README.md). Python bindings (PyO3 +
maturin, import name `racah`): [`racah-py/README.md`](racah-py/README.md).

## Why this crate exists

No library — in Rust, and essentially nowhere as a standalone component —
computes the *full* representation-theory coefficient set for the compact Lie
groups on demand, for any admissible labels: fusion multiplicities, dimensions,
duals, Frobenius–Schur indicators, Clebsch–Gordan coefficients, and the
recoupling data (3j / 6j and the F- and R-symbols). The existing supply stops
short in two ways. **Precomputed tables** are complete only for *finite*
symmetry sets; a compact Lie group has infinitely many irreps and tensor
products only make them larger, so any table has a cut a large-enough
calculation will exceed. **Single-group packages** solve one group at a fixed
scope and do not extend to SU(N≥3), SO(N), or Sp(2N), where no closed forms
exist and the coefficients must be *constructed*.

`racah` removes both limits: coefficients are computed on demand, inside the
process, in pure Rust. It is pure representation mathematics — no
fusion-category trait vocabulary, no sector-identity types, no tensor-network
concepts, no dependency on any tensor engine — so consumers translate its
numbers into their own interfaces. To do this faithfully it consolidates the
algorithms of three production references, one per family (full provenance in
[`docs/references.md`](docs/references.md)):

| Reference | What is taken from it |
|---|---|
| WignerSymbols.jl | the exact SU(2) model: big-rational Racah sums, prime-factorized factorials as the measured-need upgrade |
| SUNRepresentations.jl (Alex–Kalus–Huckleberry–von Delft, J. Math. Phys. 52, 023507 (2011)) | the SU(N) pipeline: Gelfand–Tsetlin patterns, exact ladder matrices, highest-weight nullspace, deterministic gauge canonicalization, weight-ladder descent |
| QSpace v4 (Weichselbaum) | the SO(N)/Sp(2N) pipeline: per-family defining-representation seeds feeding one family-generic decomposition loop; and the production discipline — abort on tolerance violation, per-representation error recording, precision tiers |

### Out of scope, deliberately

- **Fusion-category trait vocabulary.** `racah` answers "what are the correct
  numbers"; a consumer's engine should not be able to tell whether an F-block
  came from this crate, a closed form, or a checked-in table.
- **Pentagon solving for finite fusion categories.** Anyon models (Fibonacci,
  Ising, …) have complete exact F/R data published; converting it is a
  consumer's data problem, not a computation problem for this crate.
- **Symbolic algebraic-number coefficients.** See
  [Exactness and gauge](#exactness-and-gauge).
- **Non-connected and non-compact groups.** `O(N)` and `Pin(N)` are not central
  quotients of a simply connected group and are out of scope.

Why each family gets a different algorithm — and why that choice is forced by
the group's branching structure rather than chosen for convenience — is argued
in [`docs/theory.pdf`](docs/theory.pdf) §8, and the classification each family
comes from is §3–§5.

## Exactness and gauge

Structural and discrete data are exact; generated coefficient *values* are
deterministically gauged, verification-gated floating point. In detail:
combinatorial structure (patterns, multiplicities, weights) and discrete data
(duals, FS phases, signs, basis ordering) are exact integer/rational arithmetic;
gauge fixing is a deterministic function of the subspace; and orthogonality,
unitarity and pentagon/hexagon checks run at generation time, so a tolerance
violation is a typed error and never a silently degraded coefficient. The user-
facing summary is in
[User Guide: exact vs generated](docs/user-guide/resources.md#exact-vs-generated-values).

The conventions that fix each coefficient's basis, sign and ordering are written
down as a **frozen normative specification** — [`docs/gauge.md`](docs/gauge.md)
(base SU(2) and SU(N)) and [`docs/gauge_soN.md`](docs/gauge_soN.md)
(SO(N)/Sp(2N)). Frozen means the documents are the authority and the code
implements them: a change that moves a returned value is a bug unless it ships
as a specification correction with a fingerprint epoch bump, a CHANGELOG
breaking-change entry, and regenerated golden values in one PR. The per-family
authority fingerprints are therefore *specification versions* — persist them
next to anything you derive, and compare by equality
([User Guide: gauge and reproducibility](docs/user-guide/resources.md#gauge-and-reproducibility)).

## Status

Feature-complete for its v0 scope; all families are implemented and
oracle-checked. The base SU(2) surface — authority fingerprint, checked
representation layer, cache resource contract — is stable; the `cgc-gen`
generated-provider surface is marked **unstable** on every item's rustdoc while
its contract is negotiated.

| Family | Pipeline | Independent verification |
|---|---|---|
| SU(2) | exact closed-form big-rational | exhaustive agreement with `wigner-symbols` 0.5.1 over its label domain, plus reference fixtures beyond it |
| SU(N), SU(N)/Z_k, PSU(N) | Gelfand–Tsetlin | signed element-wise table regeneration against SUNRepresentations.jl v0.4.0 (dim ≤ 8 every `cargo test`; a full dim ≤ 27 sweep, 76,853 F blocks, run explicitly); products cross-checked against GroupMath 1.1.3 |
| SO(N) / Sp(2N) | generator bootstrap (B/C/D) | the QSpace v4 CGC projector battery — 33 rank-2/3 channels projector-tested to round-off, 0 structural-only, 9 higher-rank rows out of the anchor's scope (see `src/bcd/qspace_oracle_tests.rs`) |
| Spin(N) | same bootstrap, Clifford seeds | `Spin(6) ≅ SU(4)` and `Spin(5) ≅ Sp(4)` on the whole weight lattice — dimensions, duals, FS indicators, every ordered product, spinor labels included (`tests/isomorphism.rs`, `tests/spin.rs`) |

Internal consistency gates run alongside: Regge/tetrahedral symmetries,
pentagon/hexagon identities, orthogonality, and `tests/gauge_golden.rs`, a
committed table of coefficient values asserted at `1e-12` that fails on any
gauge drift with no reference toolchain in the loop.

## Citation

If you use racah in academic work, please cite it
(machine-readable metadata in [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{watanabe_racah_2026,
  author  = {Watanabe, Ryo},
  title   = {{racah}: Racah--Wigner calculus for compact Lie groups},
  year    = {2026},
  url     = {https://github.com/Ryo-wtnb11/racah},
  license = {MIT OR Apache-2.0}
}
```

A DOI will be added once a release is archived on Zenodo.

## License

MIT OR Apache-2.0
