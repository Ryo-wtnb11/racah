# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
value/gauge rule noted below.

## [Unreleased]

Generated-provider (`cgc-gen`) observability and convention-identity surface
(issue [#47](https://github.com/Ryo-wtnb11/racah/issues/47)). This whole surface
is **unstable: shape may change while the generated-provider contract is
negotiated** — Cargo features cannot express instability tiers, so the rustdoc
labels plus issue #47 are the ledger.

### Added

- **Generated-tier cache stats** (`cgc-gen`): `generated_cache_stats() ->
  GeneratedCacheStats` (`#[non_exhaustive]`, reusing `TierStats`) reports the
  four generated value tiers (SU(N) CGC / F, B/C/D CGC / F) per-tier plus a
  field-wise `total()`. `GENERATED_CACHE_MAX_BYTES` (640 MiB) is the documented
  aggregate ceiling, tied to the per-tier caps by a `const` assertion. Two-layer
  cache story: base = `BASE_CACHE_MAX_BYTES`, generated =
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

[Unreleased]: https://github.com/Ryo-wtnb11/racah/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ryo-wtnb11/racah/releases/tag/v0.1.0
