//! Guard-inventory (red-first) and shape tests for the B/C/D F/R surface.
//! Heavier self-consistency oracles (unitarity/pentagon/hexagon, the OM>=2
//! family, and the Sp(4)/SO(5) isomorphism spot check) live in
//! `tests/bcd_fr.rs`.

use super::*;
use crate::bcd::{CanonicalCatalog, Series};

fn irr(s: Series, d: &[i64]) -> Irrep {
    Irrep::from_dynkin(s, d).unwrap()
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
fn d3_adjoint_cubed_f_block_has_om_axis() {
    // D3 adjoint g=(0,1,1) dim15; g⊗g → g has multiplicity 2 (exact S3.0), so the
    // F(g,g,g,g,g,g) block must carry a length-2 outer-multiplicity axis.
    use crate::bcd::directproduct;
    let g = irr(Series::D, &[0, 1, 1]);
    let n = directproduct(&g, &g).unwrap().get(&g).copied().unwrap();
    assert_eq!(
        n, 2,
        "exact layer must predict OM=2 for the D3 adjoint square"
    );

    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let block = f_symbol(&mut cat, &g, &g, &g, &g, &g, &g).unwrap();
    assert!(
        block.dims().contains(&2),
        "F block dims {:?} must contain an OM=2 axis",
        block.dims()
    );
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
