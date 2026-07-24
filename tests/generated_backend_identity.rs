//! D2 backend structural-identity gate (issue #47 leaf L2).
//!
//! Design record 2 ruling D2 (accepted with an acceptance gate): backend
//! identity stays OUT of the authority fingerprints — per-backend ULP
//! differences are not semantic identity — but the *discrete/structural* outputs
//! of generation (channel set, multiplicities, block dims, OM indices, and the
//! sign/gauge-discrete data) must be verified identical across supported
//! backends, with values agreeing within the verification tolerances.
//!
//! ## Single-backend reduction (what this test actually asserts)
//!
//! Per the L2 design confirmation: with a single supported backend at the pinned
//! tenferro revision, the cross-backend gate reduces to asserting that the
//! discrete layer is a function of the convention ALONE. Concretely, two
//! independent in-process generation runs — caches reset between them, and a
//! fresh `CanonicalCatalog` for the B/C/D run — must produce byte-identical
//! discrete/structural outputs. If the discrete layer depended on anything but
//! the convention (a race, an accumulation-order-sensitive branch, an
//! uninitialised read), the two runs would diverge here.
//!
//! ## Cross-backend activation condition
//!
//! The full cross-backend form of the gate — run the same generation on backend
//! A and backend B and compare the discrete outputs — activates when a second
//! backend becomes selectable. It is recorded here as the activation condition
//! rather than fabricating a second backend now (L2 design confirmation, "the
//! cross-backend form of the gate activates when a second backend becomes
//! selectable"). The structural signatures compared below are exactly the ones
//! that cross-backend comparison would use, so enabling it later is a matter of
//! swapping the second run's backend, not rewriting the assertions.
//!
//! The representative set is deliberately small but covers an outer-multiplicity
//! `>= 2` channel in each family (the case where a divergent OM index order or
//! sign would matter): SU(3) adjoint `(1,1) ⊗ (1,1)` (the `(1,1)` channel has
//! `N = 2`) and D3 `(0,1,1) ⊗ (0,1,1)` (`docs/gauge_soN.md` §13, `N = 2`).
#![cfg(feature = "cgc-gen")]

use racah::bcd::{
    directproduct as bcd_directproduct, CanonicalCatalog, CatalogCgc, Irrep as BcdIrrep, Series,
};
use racah::cache;
use racah::sun::{cgc, directproduct as sun_directproduct, Cgc, Irrep as SunIrrep};

/// The sign of the first significant element, or 0 if the column is all-zero.
fn sign_of(v: f64) -> i8 {
    // 1e-9 is well above the f64 round-off floor (~1e-14) and far below any
    // coefficient of interest (mirrors the "first significant" threshold both
    // gauge docs use to define the sign convention).
    const SIG: f64 = 1e-9;
    if v > SIG {
        1
    } else if v < -SIG {
        -1
    } else {
        0
    }
}

/// A comparable discrete/structural signature of one generation run.
#[derive(Debug, PartialEq, Eq)]
struct StructuralSig {
    /// Channel set + multiplicities of the whole product (sorted): each coupled
    /// irrep's Dynkin label and its fusion multiplicity.
    channels: Vec<(Vec<i64>, u32)>,
    /// Block dims of the representative channel: `(rows, d3, outer_multiplicity)`.
    block_dims: (usize, usize, usize),
    /// OM indices of the representative channel (`0..multiplicity`).
    om_indices: Vec<usize>,
    /// Sign pattern: the sign of the first significant element per CGC column,
    /// one entry per `(om_index, coupled-state)` column.
    sign_pattern: Vec<i8>,
}

/// Sign of the first significant SU(N) entry per `(mu, m3)` column. `entries()`
/// is sorted by `(m1, m2, m3, mu)`, so the first entry seen for a column is its
/// smallest `(m1, m2)` — a deterministic per-column choice.
fn sun_sign_pattern(cg: &Cgc) -> Vec<i8> {
    let [_, _, d3, n] = cg.dims();
    let mut signs = vec![0i8; d3 * n];
    for e in cg.entries() {
        let col = e.mu as usize * d3 + e.m3 as usize;
        if signs[col] == 0 {
            signs[col] = sign_of(e.value);
        }
    }
    signs
}

/// Sign of the first significant B/C/D entry per `(mu, column)`. Each copy is a
/// column-major `rows x d3` isometry; scanning a column in row order is the
/// product-basis (`m_a`-fast) storage order of `docs/gauge_soN.md` §1/§8.
fn bcd_sign_pattern(cg: &CatalogCgc) -> Vec<i8> {
    let (rows, d3) = cg.copy_shape();
    let mut signs = Vec::with_capacity(d3 * cg.multiplicity());
    for mu in 0..cg.multiplicity() {
        let copy = cg.copy(mu);
        for col in 0..d3 {
            let column = &copy[col * rows..(col + 1) * rows];
            let sign = column
                .iter()
                .map(|&v| sign_of(v))
                .find(|&s| s != 0)
                .unwrap_or(0);
            signs.push(sign);
        }
    }
    signs
}

fn channel_set_sun(a: &SunIrrep, b: &SunIrrep) -> Vec<(Vec<i64>, u32)> {
    sun_directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, m)| (c.dynkin(), m))
        .collect()
}

fn channel_set_bcd(a: &BcdIrrep, b: &BcdIrrep) -> Vec<(Vec<i64>, u32)> {
    bcd_directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, m)| (c.dynkin(), m))
        .collect()
}

/// One SU(3) generation run: the whole `(1,1) ⊗ (1,1)` channel set plus the
/// structural signature of the `N = 2` `(1,1)` channel.
fn sun_run() -> StructuralSig {
    let adj = SunIrrep::from_dynkin(&[1, 1]).unwrap();
    let channels = channel_set_sun(&adj, &adj);
    let cg = cgc(&adj, &adj, &adj).unwrap();
    let [_, _, d3, n] = cg.dims();
    assert_eq!(n, 2, "picked channel must have outer multiplicity >= 2");
    StructuralSig {
        channels,
        block_dims: (cg.dims()[0] * cg.dims()[1], d3, n),
        om_indices: (0..cg.multiplicity()).collect(),
        sign_pattern: sun_sign_pattern(&cg),
    }
}

/// One D3 generation run from a fresh catalog: the whole `(0,1,1) ⊗ (0,1,1)`
/// channel set plus the structural signature of the `N = 2` `(0,1,1)` channel.
fn bcd_run() -> StructuralSig {
    let adj = BcdIrrep::from_dynkin(Series::D, &[0, 1, 1]).unwrap();
    let channels = channel_set_bcd(&adj, &adj);
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let cg = cat.cgc(&adj, &adj, &adj).unwrap();
    let (rows, d3) = cg.copy_shape();
    assert_eq!(
        cg.multiplicity(),
        2,
        "picked channel must have outer multiplicity >= 2"
    );
    StructuralSig {
        channels,
        block_dims: (rows, d3, cg.multiplicity()),
        om_indices: (0..cg.multiplicity()).collect(),
        sign_pattern: bcd_sign_pattern(&cg),
    }
}

/// Both families in one `#[test]`: `cache::reset()` is process-global, so
/// splitting into parallel tests would let one family's reset regenerate the
/// other's caches mid-run (same shared-global-state reasoning as
/// `tests/generated_cache_contract.rs`). The discrete outputs are
/// reset-order-independent, but keeping them sequential is the cleaner contract.
#[test]
fn discrete_layer_is_convention_only_across_independent_runs() {
    // SU(N): two independent runs with a cache reset between them.
    let sun_first = sun_run();
    cache::reset(); // force the second run to regenerate, not read the cache
    let sun_second = sun_run();
    assert_eq!(
        sun_first, sun_second,
        "SU(N) discrete/structural outputs must be identical across independent \
         generation runs (single-backend reduction of the D2 gate)"
    );

    // B/C/D: two independent runs, each from a fresh CanonicalCatalog, with a
    // cache reset between them.
    let bcd_first = bcd_run();
    cache::reset();
    let bcd_second = bcd_run();
    assert_eq!(
        bcd_first, bcd_second,
        "B/C/D discrete/structural outputs must be identical across independent \
         generation runs (single-backend reduction of the D2 gate)"
    );
}
