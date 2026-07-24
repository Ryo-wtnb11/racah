//! Racah–Wigner calculus for compact Lie groups.
//!
//! Pure representation mathematics: irrep labels, dimensions, duals, product
//! decomposition (fusion multiplicities), Clebsch–Gordan coefficients, and
//! recoupling coefficients (6j / F / R), together with the self-check
//! functions (orthogonality, pentagon/hexagon) that gate them.
//!
//! This crate deliberately contains no fusion-category trait vocabulary, no
//! sector identity types, and no tensor-network concepts. Consumers translate
//! its output into their own categorical interfaces.
//!
//! Layers:
//! - base (no feature): exact SU(2) — closed-form 3j/6j/CGC in big-rational
//!   arithmetic with a single final rounding to floating point.
//! - `cgc-gen` feature: runtime coefficient generation for SU(N) (Gelfand–
//!   Tsetlin construction), and SO(N)/Sp(2N) (defining-representation seeds
//!   plus a family-generic decomposition loop). Dense factorizations and the
//!   CGC contractions producing F/R route through the selected backend; no
//!   hand-rolled kernels.
//!
//! Exactness contract: combinatorial structure and discrete data are exact;
//! gauge fixing is a deterministic function of the subspace; floating-point
//! stages are verification-gated and versioned.
//!
//! # Quick start
//!
//! Exact SU(2) recoupling needs no features. Spins are doubled (`dj = 2j`), so
//! `2` means spin 1; a non-admissible label set returns exact zero, never an
//! error. Here `{1 1 1; 1 1 1} = 1/6`:
//!
//! ```
//! use racah::wigner_6j;
//!
//! let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
//! assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
//! ```
//!
//! The generated families (SU(N), SO(N)/Sp(2N)) live behind the `cgc-gen`
//! feature; see the `sun` and `bcd` module docs for runnable examples of
//! their Clebsch–Gordan and F/R surfaces.
//!
//! # Further reading
//!
//! - [`docs/theory.md`] — a light primer on the objects this API computes
//!   (irreps, fusion multiplicities, CGC and gauge, recoupling, the two
//!   constructions, the exactness contract).
//! - [`docs/references.md`] — porting provenance (`file:symbol`-level) and the
//!   verified bibliography.
//! - [`docs/gauge.md`] / [`docs/gauge_soN.md`] — the gauge specifications.
//!
//! [`docs/theory.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/theory.md
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/references.md
//! [`docs/gauge.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/gauge.md
//! [`docs/gauge_soN.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/gauge_soN.md
#![warn(missing_docs)]

pub mod cache;
mod exact;
mod primefactor;

/// Exact SU(2) recoupling: doubled-spin labels, the infallible closed-form
/// Wigner 3j/6j, Clebsch–Gordan, F/R/Frobenius–Schur functions (zero
/// convention for inadmissible tuples), and an additive *checked* surface
/// (`Su2Irrep`, `wigner_6j_checked`, …) that returns a typed error instead of
/// requiring consumers to infer validity from a zero coefficient.
pub mod su2;

/// Exact SU(N) representation combinatorics (Layer 1 of the `cgc-gen` track):
/// GT patterns, Weyl dimension, duality, Littlewood–Richardson products, and
/// exact ladder matrices. Compilation-gated behind `cgc-gen`.
#[cfg(feature = "cgc-gen")]
pub mod sun;

/// Exact SO(N)/Sp(2N) representation combinatorics for the B/C/D Cartan series
/// (Layer S3.0 of the `cgc-gen` track): integer Dynkin labels, exact Weyl
/// dimensions, duals, Frobenius–Schur indicators, Freudenthal weight
/// multiplicities, and the exact Brauer–Klimyk/Racah–Speiser tensor-product
/// decomposition `N^c_ab`. Compilation-gated behind `cgc-gen`.
#[cfg(feature = "cgc-gen")]
pub mod bcd;

// Family-generic F/R contraction + gates core, shared by `sun::fr` and
// `bcd::fr` (Stage 3 S3.4, issue #27). Private: the public F/R surfaces stay
// per-family; only the block types (`FBlock`/`RBlock`) are re-exported.
#[cfg(feature = "cgc-gen")]
mod frcore;

pub use exact::SignedSqrtRational;
pub use su2::{
    canonical_regge_3j, canonical_regge_6j, clebsch_gordan, clebsch_gordan_checked,
    su2_authority_fingerprint, su2_f_symbol, su2_f_symbol_checked, su2_frobenius_schur,
    su2_r_symbol, su2_r_symbol_checked, wigner_3j, wigner_3j_checked, wigner_6j, wigner_6j_checked,
    AdmissibilityViolation, Regge3j, Regge6j, ReggeError, ReggePhase, Su2Error, Su2Fusion,
    Su2Irrep,
};
