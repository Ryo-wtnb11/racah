//! Racah–Wigner calculus for compact Lie groups: irreducible representations,
//! Clebsch–Gordan coefficients, and recoupling coefficients (3j / 6j / F / R)
//! for SU(2), SU(N), SO(N), Spin(N) and Sp(2N).
//!
//! Coefficients for any admissible labels are computed on demand — there is no
//! precomputed table and no generation-time label cut. Pure representation
//! mathematics: no fusion-category trait vocabulary, no sector identity types,
//! no tensor-network concepts. Consumers translate the numbers into their own
//! categorical interfaces.
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
//! # Where to look
//!
//! | You want | Go to |
//! |---|---|
//! | SU(2): 3j, 6j, CG, F, R, Frobenius–Schur — exact, no feature flag | [`su2`] (re-exported at the crate root) |
//! | SU(N), SU(N)/Z_k, PSU(N) — Gelfand–Tsetlin construction | [`sun`] (`cgc-gen`) |
//! | SO(N), Spin(N), Sp(2r) — generator bootstrap, B/C/D series | [`bcd`] (`cgc-gen`) |
//! | Which highest weights a global form admits | [`group`] |
//! | Cache ceilings, budgets, statistics | [`cache`] |
//!
//! # Layers
//!
//! - **base** (no feature): exact SU(2) — closed-form 3j/6j/CGC in
//!   big-rational arithmetic with a single final rounding to floating point.
//! - **`cgc-gen`**: runtime coefficient generation for SU(N) (Gelfand–Tsetlin
//!   construction), and SO(N)/Sp(2N) (defining-representation seeds plus a
//!   family-generic decomposition loop). Dense factorizations and the CGC
//!   contractions producing F/R route through the Tenferro traced surface at a
//!   single seam, currently executed on the CPU faer backend; no hand-rolled
//!   kernels, and no public backend-selection API yet.
//!
//! The boundary is mathematical, not organizational: SU(2) has closed forms and
//! needs no matrix computation, so a consumer needing only SU(2) never pulls a
//! linear-algebra stack.
//!
//! # Exactness contract
//!
//! Combinatorial structure and discrete data are exact; gauge fixing is a
//! deterministic function of the subspace; floating-point stages are
//! verification-gated and versioned. Concretely: labels, dimensions, duals,
//! Frobenius–Schur indicators and fusion multiplicities are exact integer or
//! rational arithmetic, while generated CGC / F / R *values* are `f64` that
//! passed orthogonality, unitarity and pentagon/hexagon gates at generation
//! time. A gate violation is a typed error, never a silently degraded number.
//!
//! # Provider contract
//!
//! Each family publishes an opaque **authority fingerprint** naming the
//! convention set its coefficients are computed in
//! ([`su2_authority_fingerprint`], [`sun::sun_authority_fingerprint`],
//! [`bcd::bcd_authority_fingerprint`]). Persist the bytes next to anything you
//! derive from these coefficients and compare by equality on load; never parse
//! them. The base SU(2) provider adds a checked representation surface
//! ([`su2::Su2Irrep`] and the `*_checked` coefficient functions) and a cache
//! resource contract ([`cache::base_cache_stats`],
//! [`cache::BASE_CACHE_MAX_BYTES`], [`cache::reset`]). The
//! [User Guide][guide-resources] carries the prose.
//!
//! # Documentation
//!
//! - [User Guide] — task-oriented: pick a group, build irreps, fuse, get CGC,
//!   get F/R, bound the caches. Start here if you are new.
//! - [`docs/theory.pdf`] — a self-contained note on the objects this API
//!   computes (irreps, fusion multiplicities, CGC and gauge, recoupling, the
//!   two constructions, the exactness contract).
//! - [`docs/gauge.md`] / [`docs/gauge_soN.md`] — the frozen normative gauge
//!   specifications. Reference documents, not tutorials.
//! - [`docs/references.md`] — porting provenance (`file:symbol`-level) and the
//!   verified bibliography.
//!
//! [User Guide]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/user-guide/README.md
//! [guide-resources]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/user-guide/resources.md
//! [`docs/theory.pdf`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/theory.pdf
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/references.md
//! [`docs/gauge.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/gauge.md
//! [`docs/gauge_soN.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/gauge_soN.md
#![warn(missing_docs)]

// The audit harness measures only allocations routed through Rust's System
// allocator. Keeping it test-only avoids changing the library allocator
// contract or attributing C/backend allocations to Racah.
#[cfg(all(test, feature = "cgc-gen"))]
#[global_allocator]
static TEST_ALLOCATOR: audit_alloc::TrackingAllocator = audit_alloc::TrackingAllocator;

#[cfg(all(test, feature = "cgc-gen"))]
mod audit_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub(crate) struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let new = unsafe { System.realloc(ptr, layout, size) };
            if !new.is_null() {
                if size >= layout.size() {
                    let delta = size - layout.size();
                    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                    ALLOC_BYTES.fetch_add(delta, Ordering::Relaxed);
                    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
                    PEAK.fetch_max(live, Ordering::Relaxed);
                } else {
                    LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
                }
            }
            new
        }
    }

    pub(crate) fn snapshot() -> (usize, usize) {
        (LIVE.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
    }

    pub(crate) fn reset_peak_to_live() {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Cumulative successful allocation requests. `realloc` contributes only
    /// its growth delta; deallocations never subtract from these counters.
    pub(crate) fn allocation_totals() -> (usize, usize) {
        (
            ALLOC_CALLS.load(Ordering::Relaxed),
            ALLOC_BYTES.load(Ordering::Relaxed),
        )
    }
}

pub mod cache;

/// Which highest weights are genuine representations of *which* group:
/// [`group::RootSystem`], [`group::GlobalForm`], [`group::CenterSubgroup`],
/// [`group::GroupId`] and the central-character admissibility predicate
/// [`group::GroupId::admits`].
///
/// Naming the Lie algebra does not name the group — `so(N)` belongs to both
/// `Spin(N)` and `SO(N)`, `su(N)` to `SU(N)`, every `SU(N)/Z_k` and `PSU(N)` —
/// and the groups differ in exactly one respect: which dominant weights they
/// admit. A global form deletes irreps; it never changes a coefficient value.
/// Pure integer arithmetic, no feature gate.
pub mod group;

mod exact;
mod primefactor;

#[cfg(all(test, feature = "cgc-gen"))]
mod cache_audit;
#[cfg(all(test, feature = "cgc-gen"))]
mod cache_budget_pressure;
#[cfg(all(test, feature = "cgc-gen"))]
mod cache_trim_pressure;

/// Exact SU(2) recoupling: doubled-spin labels (`dj = 2j`), the infallible
/// closed-form Wigner 3j/6j, Clebsch–Gordan, F/R/Frobenius–Schur functions
/// (exact zero for an inadmissible tuple), and an additive *checked* surface
/// ([`su2::Su2Irrep`], [`su2::wigner_6j_checked`], …) that returns a typed
/// error instead of requiring consumers to infer validity from a zero
/// coefficient. Its items are re-exported at the crate root.
pub mod su2;

/// SU(N), `SU(N)/Z_k` and `PSU(N)`: irreps from Dynkin labels, exact Weyl
/// dimensions, duals, Littlewood–Richardson products, and the Clebsch–Gordan /
/// F / R coefficients built by the Gelfand–Tsetlin construction.
/// Compilation-gated behind `cgc-gen`.
#[cfg(feature = "cgc-gen")]
pub mod sun;

/// SO(N), Spin(N) and Sp(2r) — the B, C, D Cartan series: irreps from Dynkin
/// labels, exact Weyl dimensions, duals, Frobenius–Schur indicators,
/// Freudenthal weight multiplicities, the exact Brauer–Klimyk/Racah–Speiser
/// tensor-product decomposition $N^c_{ab}$, and the Clebsch–Gordan / F / R
/// coefficients built by the generator bootstrap. Compilation-gated behind
/// `cgc-gen`.
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
