# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
value/gauge rule noted below.

## [Unreleased]

### Changed

- **`BcdError::ExcludedRank::redirect` is form-aware** (stage (b) of #87, Q3):
  the low-rank isomorphism is an isomorphism of *groups*, so `Spin(3)` now
  redirects to `"use SU(2) instead"` while `SO(3)` redirects to `"use SU(2)
  with integer j only instead"` (likewise `Spin(4)`/`SO(4)`). Diagnostic strings
  only; no coefficient value changes.
- **`bcd::Irrep::partition` returns `Option<Vec<i64>>`** and
  `bcd::Irrep::weight_multiplicities` returns `Option<…>`, both `None` on a
  spinor irrep, whose ε-basis weights are half-integers. The always-exact
  doubled forms are `two_partition` / `two_weight_multiplicities`.

### Added

- **`Spin(N)` — the spinor irreps of the `B`/`D` covers**
  ([#54](https://github.com/Ryo-wtnb11/racah/issues/54), stage (b) of
  [#87](https://github.com/Ryo-wtnb11/racah/issues/87)). `Irrep::from_dynkin_in`
  takes a `GroupId`, so `GroupId::spin(N)` admits the spinor labels that
  `SO(N)` rejects; their dimensions, duals, Frobenius–Schur indicators, fusion,
  CGC and F/R are produced by the same bootstrap. Three parts:
  - `bcd::Irrep` now stores the **doubled** ε-basis weight `2λ`, so a spinor's
    half-integer highest weight is exact (`Irrep::two_partition`,
    `Irrep::is_spinor`; `Irrep::partition` returns `Option`, `None` on a
    spinor, and `Irrep::weight_multiplicities` likewise, with
    `two_weight_multiplicities` always available).
  - `bcd::spinor_seeds` — the second base case: the Clifford/Fock generator
    seeds for `ω_r` (`B_r`) and `ω_{r-1}`, `ω_r` (`D_r`), gated by the same
    exact `check_commutators`, specified in
    [`docs/gauge_soN.md`](docs/gauge_soN.md) §16 and pinned by new
    `tests/gauge_golden.rs` rows (`Spin(5)`, `Spin(6)`, `Spin(7)`, `Spin(10)`).
  - The canonical-parent candidate set is **class-indexed** (§14.2): a tensor
    irrep's parents are searched in the tensor sublattice only. **Every shipped
    `SO(N)`/`Sp(2N)` coefficient is byte-identical** and the `bcd` `epoch` stays
    at `1`; `bcd_authority_fingerprint()` gains no tag (§16.4 states why).

  Oracles in `tests/isomorphism.rs` and `tests/spin.rs`: `Spin(6) ≅ SU(4)` and
  `Spin(5) ≅ Sp(4)` now agree on the **whole** weight lattice (dimensions,
  duals, every ordered product, Frobenius–Schur), and `Spin(3) ≅ SU(2)` is the
  form-aware low-rank redirect.

- **`racah::group` — root datum plus global form** (stage (a) of
  [#87](https://github.com/Ryo-wtnb11/racah/issues/87)). `RootSystem`,
  `GlobalForm`, `CenterSubgroup`, `GroupId` with fallible named constructors
  (`GroupId::su/su_quotient/psu/sp/psp/spin/so/pso/half_spin_plus/
  half_spin_minus`), and `GroupId::admits(&[i64])` — the central-character
  predicate that says which dominant weights are representations of which
  connected compact group. Ungated (pure integer arithmetic). `bcd`'s spinor
  rejection is now the one call site of that predicate, and
  `Series::root_system(r)` is the single bridge to the label-lattice type.
  Conventions pinned in [`docs/references.md`](docs/references.md): the
  `D_r`-odd `Z4` generator `[ω_r]`, and the `D_r`-even half-spin forms named
  by the class they retain. **No coefficient value changes**; no `epoch` moves.

- **Frozen gauge specification.** [`docs/gauge.md`](docs/gauge.md) and
  [`docs/gauge_soN.md`](docs/gauge_soN.md) are declared **normative**: the
  documents are the authority and the code implements them. `docs/gauge.md` also
  now specifies the base SU(2) conventions (§12) and marks the rules the
  implementation fixes only implicitly. **No coefficient value changes**; all
  three `epoch` tags stay at `1`. What changes is the meaning of the authority
  fingerprints: they are now **specification versions**, moving only on a
  specification correction (spec edit + `epoch` bump + CHANGELOG breaking-change
  entry + regenerated goldens, in one PR), never because a refactor moved a
  value. A refactor that moves a value is now a bug by definition.
  ([#84](https://github.com/Ryo-wtnb11/racah/issues/84))
- **`tests/gauge_golden.rs`**: the in-repo gauge tripwire — a small committed
  table of SU(3) CGC (including the OM = 2 adjoint vertex), SU(3) F, and signed
  SO(5)/Sp(4) CGC and F values, asserted at `1e-12` in the default `cgc-gen` test
  run. It covers the drift the external oracles miss by default: SU(3) F symbols
  are otherwise pinned only by an `#[ignore]`d heavy table oracle, and the QSpace
  B/C/D anchor compares an isotypic projector that is blind to the coupled-side
  gauge. It is a drift detector, not an oracle.
- **`racah-py`**: PyO3/maturin Python bindings for the SU(N) surface, built as
  a workspace member with `cgc-gen` always on. Import name `racah`,
  distribution `racah-py`; abi3-py312 wheels are built by the `wheels`
  workflow. See [`racah-py/README.md`](racah-py/README.md).

### Changed

- Docs: replaced the "no label ceiling" absolute with the machine-word label
  bounds that report a typed overflow, and corrected "selectable dense
  backend" to the single Tenferro seam with no public backend-selection API.

## [0.1.1] - 2026-08-12

This release publishes the generated-provider dependency closure against the
published Tenferro 0.3.0 registry line.

### Changed

- Use registry `tenferro-* = "0.3.0"` dependencies for `cgc-gen`.
- Document crates.io installation and the published feature configuration.

Generated-provider (`cgc-gen`) observability and convention-identity surface
(issue [#47](https://github.com/Ryo-wtnb11/racah/issues/47)). This whole surface
is **unstable: shape may change while the generated-provider contract is
negotiated** — Cargo features cannot express instability tiers, so the rustdoc
labels plus issue #47 are the ledger.

### Added

- **Per-tier coefficient-cache trim**: `trim_to(CoefficientCacheTier, bytes)`
  releases the oldest FIFO prefix of exactly one tier and reports its
  linearization-point charged-entry accounting in `CacheTrimReport`. It is a
  process-global single-owner lifecycle operation; it preserves the one-shot
  cache budget and makes no allocator/RSS release claim.

- **One-shot coefficient-cache budgets**: `CoefficientCacheBudgets`,
  `CoefficientCacheTier`, `configure_cache_budgets`, and `cache_budgets` let
  applications shrink independent process-local tier caps before first use.
  Zero retains no entry; reset preserves the selected policy.

- **Shared SU(N) product tier** (`cgc-gen`, issue #59): exact
  `directproduct` decompositions are retained once as sorted shared channels
  under an order-normalized irrep pair. The unchanged public API reconstructs
  its `BTreeMap`; Racah-private multiplicity and channel consumers use the
  shared value directly. The tier is bounded at 256 entries and a 128 KiB
  retained-charge backstop, sized from the checked-in collector and a
  downstream SU(3)+SU(4) Generic HomSpace/topology probe.
- **Generated-tier cache stats** (`cgc-gen`): `generated_cache_stats() ->
  GeneratedCacheStats` (`#[non_exhaustive]`, reusing `TierStats`) reports the
  five generated tiers (SU(N) product / CGC / F, B/C/D CGC / F) per-tier plus a
  field-wise `total()`. `GeneratedCacheStats` `bytes` fields are conservative
  retained charges of cache-owned entries, with container scaffolding,
  allocator/RSS costs, external clones, and returned values excluded.
  `GENERATED_CACHE_MAX_BYTES` (640 MiB + 128 KiB) is the documented aggregate
  retained-charge cap, tied to the per-tier caps by a `const` assertion.
  Two-layer cache story: base = `BASE_CACHE_MAX_BYTES`, generated =
  `GENERATED_CACHE_MAX_BYTES`, whole = the documented sum; no cross-feature
  constant. `reset()` clears the generated tiers alongside the base ones.
- **Generated authority fingerprints** (`cgc-gen`):
  `sun::sun_authority_fingerprint()` and `bcd::bcd_authority_fingerprint()`
  (`&'static [u8]`). Their contract is weaker than the exact SU(2) fingerprint —
  equal fingerprints identify the same convention, generation pipeline, and
  tolerance policy, but do not imply byte-identical values or independently prove
  numerical agreement (verification gates and oracles own that). Epochs are
  per-family and independent. Backend identity is excluded by design.
- **Backend structural-identity gate** (`cgc-gen`, D2): a test asserting the
  discrete/structural generation outputs are a function of the convention alone
  (stable across independent in-process runs), the single-backend reduction of
  the cross-backend gate.

## [0.1.0] - 2026-07-24

First tagged release of the v0 scope: the full representation-theory
coefficient set for SU(2), SU(N), SO(N), and Sp(2N), computed on demand with no
label ceiling.

### Added

- **Exact SU(2)** (default build, no features): 3j, 6j, Clebsch–Gordan, and
  F / R / Frobenius–Schur symbols in closed-form big-rational arithmetic with a
  single final rounding. Dependency-light (`num-bigint` / `num-rational` /
  `num-traits` only).
- **Generated SU(N)** (`cgc-gen` feature): the Gelfand–Tsetlin pipeline — CGC,
  F, and R with outer-multiplicity indices.
- **Generated SO(N) / Sp(2N)** (`cgc-gen` feature): the generator-bootstrap
  pipeline over the B/C/D Cartan series — CGC, F, and R.
- **Base SU(2) provider contract:**
  - `su2_authority_fingerprint()` — opaque bytes identifying the value-fixing
    convention set; compared by equality, changed only on a value-affecting
    breaking release.
  - Checked representation surface — `Su2Irrep` (with `dj` / `dim` / `dual` /
    `fusion`), `Su2Fusion`, `Su2Error` / `AdmissibilityViolation`, and the
    `wigner_3j_checked` / `wigner_6j_checked` / `clebsch_gordan_checked` /
    `su2_f_symbol_checked` / `su2_r_symbol_checked` functions. Additive over the
    infallible zero-convention functions; distinguishes `Ok(0)` (an admissible
    accidental zero) from `Err(NotAdmissible)` (a forbidden coupling).
  - Cache resource contract — `BASE_CACHE_MAX_BYTES` static partition over the
    three base tiers, `base_cache_stats()` / `BaseCacheStats` / `TierStats`
    per-tier statistics (entries, bytes, hits, misses, evictions), and
    single-owner `reset()`.
- **Self-check / oracle batteries**, shipped as public API and used as
  generation gates: CGC orthogonality, F-unitarity, R-orthogonality, and the
  pentagon / hexagon identities.

### Notes

- Not published to crates.io: the `cgc-gen` feature depends on the unpublished
  `tenferro-rs`, so a crates.io release is blocked upstream. The git dependency
  is the supported path.

### Versioning policy

Coefficient *values* are floating point, but the *computation* is exact:
combinatorial structure, discrete data (duals, signs, Frobenius–Schur phases),
and gauge fixing are deterministic. Any change that can alter a coefficient
value, its normalization, or its canonical gauge is a **breaking** change, so
consumers may key caches and persisted data on the crate version. For the base
SU(2) provider this rule is mechanized by `su2_authority_fingerprint()`: its
epoch is bumped only on such a value-affecting release, so a fingerprint change
and a breaking release are one reviewable event.

[Unreleased]: https://github.com/Ryo-wtnb11/racah/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Ryo-wtnb11/racah/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ryo-wtnb11/racah/releases/tag/v0.1.0
