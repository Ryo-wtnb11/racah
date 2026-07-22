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
#![warn(missing_docs)]

pub mod cache;
mod exact;
mod primefactor;
mod su2;

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

pub use exact::SignedSqrtRational;
pub use su2::{
    canonical_regge_3j, canonical_regge_6j, clebsch_gordan, su2_f_symbol, su2_frobenius_schur,
    su2_r_symbol, wigner_3j, wigner_6j, Regge3j, Regge6j, ReggeError, ReggePhase,
};
