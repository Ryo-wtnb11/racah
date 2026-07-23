//! Guard-inventory (red-first) and shape tests for the B/C/D F/R surface.
//! Heavier self-consistency oracles (unitarity/pentagon/hexagon, the OM>=2
//! family, and the Sp(4)/SO(5) isomorphism spot check) live in
//! `tests/bcd_fr.rs`.

use super::*;
use crate::bcd::{CanonicalCatalog, Series};
use crate::frcore::Family;
use std::sync::atomic::Ordering;

fn irr(s: Series, d: &[i64]) -> Irrep {
    Irrep::from_dynkin(s, d).unwrap()
}

// ---- performance contract: the CGC value tier dedups product sweeps ----

#[test]
fn warm_state_does_not_resweep() {
    // The value tier caches EVERY coupled channel from ONE decomposition sweep.
    // C2 vector v=(0,1), v⊗v = 1 ⊕ (2,0) ⊕ (0,2). After a single cgc_entries for
    // channel (2,0), the *other* channel (0,2) must already be in the tier — i.e.
    // one sweep populated both, so a later (0,2) request is a hit, not a re-sweep.
    // Red-first: before the P1 fix each cgc() re-decomposed v⊗v per channel, so
    // (0,2) would be absent after the (2,0) request.
    //
    // Asserted via tier CONTENTS (specific keys), not a shared sweep-counter
    // delta, so it is robust to other tests touching the process-global tier
    // concurrently. (The counter `cgc_sweeps()` remains for the bench/PR arith.)
    crate::cache::reset();
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let c1 = irr(Series::C, &[2, 0]);
    let c2 = irr(Series::C, &[0, 2]);

    let sweeps_before = CGC_SWEEPS.load(Ordering::Relaxed);
    {
        let mut fam = BcdFamily { cat: &mut cat };
        fam.cgc_entries(&v, &v, &c1).unwrap(); // one sweep of v⊗v
    }
    let tier = crate::cache::cache_bcd_cgc();
    assert!(
        tier.get(&(v.clone(), v.clone(), c1.clone())).is_some(),
        "requested channel must be cached"
    );
    assert!(
        tier.get(&(v.clone(), v.clone(), c2.clone())).is_some(),
        "the OTHER channel of the same product must be cached by the same sweep \
         (no per-channel re-sweep)"
    );
    // At least the one sweep ran; the point is (0,2) needed no separate one.
    assert!(CGC_SWEEPS.load(Ordering::Relaxed) > sweeps_before);
}

// ---- guard inventory: red-first ill-posed inputs ----

#[test]
fn f_symbol_zero_vertex_is_typed_error() {
    // C2 (Sp4): vector (0,1) dim5; 5⊗5 = 1 ⊕ (2,0) ⊕ (0,2) has no (0,1), so the
    // vertex a⊗b→e with e = vector is empty.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let adj = irr(Series::C, &[2, 0]);
    let err = f_symbol(&mut cat, &v, &v, &v, &v, &v, &adj).unwrap_err();
    assert!(
        matches!(
            err,
            FrError::Catalog(crate::bcd::CatalogError::ZeroFusionChannel { .. })
        ),
        "got {err:?}"
    );
}

#[test]
fn f_symbol_foreign_group_is_typed_error() {
    // A B-series label passed to a C-series catalog.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let foreign = irr(Series::B, &[0, 2]);
    let v = irr(Series::C, &[0, 1]);
    let err = f_symbol(&mut cat, &foreign, &v, &v, &v, &v, &v).unwrap_err();
    assert!(
        matches!(
            err,
            FrError::Catalog(crate::bcd::CatalogError::WrongGroup { .. })
        ),
        "got {err:?}"
    );
}

#[test]
fn f_symbol_wrong_rank_is_typed_error() {
    // Same series, wrong rank for the catalog.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let wrong_rank = irr(Series::C, &[0, 1, 0]);
    let v = irr(Series::C, &[0, 1]);
    let err = f_symbol(&mut cat, &wrong_rank, &v, &v, &v, &v, &v).unwrap_err();
    assert!(
        matches!(
            err,
            FrError::Catalog(crate::bcd::CatalogError::WrongGroup { .. })
        ),
        "got {err:?}"
    );
}

#[test]
fn r_symbol_zero_vertex_is_typed_error() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let err = r_symbol(&mut cat, &v, &v, &v).unwrap_err();
    assert!(
        matches!(
            err,
            FrError::Catalog(crate::bcd::CatalogError::ZeroFusionChannel { .. })
        ),
        "got {err:?}"
    );
}

#[test]
fn r_symbol_foreign_group_is_typed_error() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let foreign = irr(Series::C, &[0, 1]);
    let err = r_symbol(&mut cat, &foreign, &foreign, &foreign).unwrap_err();
    assert!(
        matches!(
            err,
            FrError::Catalog(crate::bcd::CatalogError::WrongGroup { .. })
        ),
        "got {err:?}"
    );
}

// ---- shapes ----

#[test]
fn c2_vector_cubed_f_is_multiplicity_free_scalar() {
    // C2 vector v=(0,1) dim5. v⊗v = 1 ⊕ (2,0) ⊕ (0,2), all multiplicity-free.
    // Pick e=(2,0), f=(2,0), d ∈ e⊗v with d also in v⊗f: take d=v is empty;
    // use a fully multiplicity-free admissible sextet with a = trivial to force
    // a 1×1×1×1 identity block.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let triv = Irrep::trivial(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let adj = irr(Series::C, &[2, 0]); // (2,0) ∈ v⊗v
                                       // a=triv: e=b=v forced; f=d ∈ v⊗v, take adj.
    let block = f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
}

#[test]
fn d3_adjoint_square_84_channel_cgc_is_coherent_or_bricks() {
    // The restored QSpace coherence guard (issue #15 instance 5) on the CGC path:
    // a coupled channel is either coherent (Ok) or refused with a typed
    // BasisIncoherent naming irrep/product/residual — never a silent wrong CGC.
    //
    // WHICH happens for a near-rank-deficient channel (the D3 84 = (0,2,2), PR
    // #24) is platform-dependent (a deterministic near-tie resolution: rotated on
    // dev macOS ARM at residual 3.65, may be coherent on CI Linux x86), so we
    // assert the disjunction. The guard's firing is pinned deterministically by
    // `sweep::tests::coherence_residual_detects_degenerate_rotation`.
    use crate::bcd::{directproduct, CatalogError};
    let g = irr(Series::D, &[0, 1, 1]);
    assert_eq!(
        directproduct(&g, &g).unwrap().get(&g).copied().unwrap(),
        2,
        "exact layer must predict OM=2 for the D3 adjoint square"
    );

    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let eightyfour = irr(Series::D, &[0, 2, 2]);
    match cat.cgc(&g, &g, &eightyfour) {
        Ok(_) => {}
        Err(CatalogError::BasisIncoherent {
            irrep,
            product,
            residual,
        }) => {
            // If it bricks, it must name THIS channel and be O(1), not noise.
            assert_eq!(irrep, vec![0, 2, 2]);
            assert_eq!(product, (vec![0, 1, 1], vec![0, 1, 1]));
            assert!(
                residual > 1e-3,
                "residual {residual} must be O(1), not noise"
            );
        }
        other => panic!("expected Ok or BasisIncoherent, got {other:?}"),
    }
}

#[test]
fn d3_adjoint_f_symbol_closes_or_bricks() {
    // The D3-adjoint OM>=2 F-move: closes (Ok) or fail-loud with BasisIncoherent
    // (via an ill-conditioned channel of g⊗g) — never a non-unitary block.
    use crate::bcd::CatalogError;
    let g = irr(Series::D, &[0, 1, 1]);
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    match f_symbol(&mut cat, &g, &g, &g, &g, &g, &g) {
        Ok(_) => {}
        Err(FrError::Catalog(CatalogError::BasisIncoherent { .. })) => {}
        other => panic!("expected Ok or BasisIncoherent, got {other:?}"),
    }
}

// ---- cache: a warm hit returns the stored block ----

#[test]
fn f_symbol_second_call_is_cache_hit() {
    crate::cache::reset();
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let triv = Irrep::trivial(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let adj = irr(Series::C, &[2, 0]);
    let first = f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
    let before = crate::cache::stats().hits;
    let second = f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
    let after = crate::cache::stats().hits;
    assert_eq!(first, second);
    assert_eq!(
        after,
        before + 1,
        "second call must be served from the cache"
    );
}
