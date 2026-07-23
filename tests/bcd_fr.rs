//! B/C/D F/R self-consistency oracles (Stage 3 S3.4, issue #27).
//!
//! Two tiers, mirroring the SU(N) F/R suites:
//! - CI-fast (`#[test]`): F-move unitarity on seeded small families, and the
//!   Sp(4) ≅ SO(5) F-magnitude spot check.
//! - Heavy (`#[ignore]`, run with `cargo test --release --features cgc-gen --
//!   --ignored`): the pentagon and hexagon spot checks, including the OM ≥ 2
//!   D3-adjoint family. These materialize CGC (SVD sweeps) for many intermediate
//!   irreps and cost seconds-to-minutes each — the same treatment as the heavy
//!   SU(3) F/R table oracle.
//!
//! All values are racah-generated on both sides; this is an intra-stage
//! self-consistency check. The SU(N)-anchored isomorphism battery (SO(6) ≅ SU(4)
//! etc., QSpace fixtures) belongs to S3.5 and is not duplicated here.

#![cfg(feature = "cgc-gen")]

use racah::bcd::{
    check_f_unitarity, check_hexagon, check_pentagon, directproduct, f_symbol, CanonicalCatalog,
    Irrep, Series,
};

fn irr(s: Series, d: &[i64]) -> Irrep {
    Irrep::from_dynkin(s, d).unwrap()
}

// ---------------------------------------------------------------------------
// F-move unitarity (CI-fast): each family's F-move is real-orthogonal.
// ---------------------------------------------------------------------------

#[test]
fn c2_vector_f_unitarity() {
    // C2 = Sp(4), vector (0,1) dim 5.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let adj = irr(Series::C, &[2, 0]);
    check_f_unitarity(&mut cat, &v, &v, &v, &adj).unwrap();
}

#[test]
fn b2_vector_f_unitarity() {
    // B2 = SO(5), vector (1,0) dim 5.
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let v = irr(Series::B, &[1, 0]);
    let adj = irr(Series::B, &[0, 2]);
    check_f_unitarity(&mut cat, &v, &v, &v, &adj).unwrap();
}

#[test]
fn d3_vector_f_unitarity() {
    // D3 = SO(6), vector (1,0,0) dim 6.
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let v = irr(Series::D, &[1, 0, 0]);
    // vector^2 -> 1 + adjoint(0,1,1) + (2,0,0); take d = (2,0,0).
    let d = irr(Series::D, &[2, 0, 0]);
    check_f_unitarity(&mut cat, &v, &v, &v, &d).unwrap();
}

// ---------------------------------------------------------------------------
// Sp(4) ≅ SO(5) F-level spot check (CI-fast).
//
// Label dictionary (derived from the S3.0 Dynkin↔partition maps and confirmed by
// matching Weyl dims across the whole (p, q even) grid): the SO(5)/Sp(4) diagram
// isomorphism swaps the two nodes, so a B2 label (a1, a2) corresponds to the C2
// label (a2, a1). Under it, B2 vector (1,0) ↔ C2 vector (0,1), B2 adjoint (0,2) ↔
// C2 adjoint (2,0), etc.
//
// The two catalogs generate the same fusion category from independent defining
// seeds, so their CGC gauges agree only up to a per-vertex sign (a coboundary).
// For a multiplicity-free vertex that coboundary is exactly a sign, so the
// F-symbol MAGNITUDE is the gauge-invariant content. The check therefore compares
// |F_B2| to |F_C2| element-wise over every multiplicity-free vector sextet; they
// agree to machine precision (empirically ~2e-16). (Full sign/basis alignment for
// OM ≥ 2 needs the fitted-unitary harness of S3.5 and is out of scope here.)
// ---------------------------------------------------------------------------

#[test]
fn sp4_so5_vector_f_magnitudes_match() {
    let mut cb = CanonicalCatalog::new(Series::B, 2).unwrap();
    let mut cc = CanonicalCatalog::new(Series::C, 2).unwrap();
    let vb = irr(Series::B, &[1, 0]);
    let vc = irr(Series::C, &[0, 1]);
    // B2 label (a1,a2) -> C2 label (a2,a1).
    let swap = |x: &Irrep| {
        irr(Series::C, &{
            let d = x.dynkin();
            vec![d[1], d[0]]
        })
    };

    let vv = directproduct(&vb, &vb).unwrap(); // {1, (0,2), (2,0)}
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for e in vv.keys() {
        for f in vv.keys() {
            let ev = directproduct(e, &vb).unwrap();
            let vf = directproduct(&vb, f).unwrap();
            for d in ev.keys() {
                if !vf.contains_key(d) {
                    continue; // d must lie in both e⊗v and v⊗f
                }
                let fb = f_symbol(&mut cb, &vb, &vb, &vb, d, e, f).unwrap();
                if fb.dims() != [1, 1, 1, 1] {
                    continue; // magnitude check is for multiplicity-free vertices
                }
                let fc = f_symbol(&mut cc, &vc, &vc, &vc, &swap(d), &swap(e), &swap(f)).unwrap();
                let diff = (fb.at(0, 0, 0, 0).abs() - fc.at(0, 0, 0, 0).abs()).abs();
                worst = worst.max(diff);
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 10,
        "expected many matched sextets, got {compared}"
    );
    assert!(
        worst < 1e-9,
        "Sp(4)/SO(5) |F| mismatch: worst ||F_B|-|F_C|| = {worst:e} over {compared} sextets"
    );
}

// ---------------------------------------------------------------------------
// Pentagon / hexagon (heavy — run with --release --ignored).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn c2_vector_pentagon() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    check_pentagon(&mut cat, &v, &v, &v, &v).unwrap();
}

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn c2_vector_hexagon() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    check_hexagon(&mut cat, &v, &v, &v).unwrap();
}

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn b2_vector_hexagon() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let v = irr(Series::B, &[1, 0]);
    check_hexagon(&mut cat, &v, &v, &v).unwrap();
}

/// OM ≥ 2 pentagon+hexagon on the D3 adjoint g = (0,1,1): its self-fusion
/// g ⊗ g → g has multiplicity 2 (exact S3.0), so the associativity/braiding
/// consistency genuinely exercises the outer-multiplicity mixing.
///
/// `#[ignore]` (release-only): both gates materialize the adjoint CGC chain plus
/// the F-move's intermediate irreps, ~5s in release but minutes under the
/// unoptimized debug SVD, so it is not a default-CI test. There is no smaller
/// OM ≥ 2 family in B/C/D (the adjoint is the minimal one, dim 15). The default
/// suite's cheap OM ≥ 2 coverage is the exact-layer OM=2 assertion plus the
/// single-block `d3_adjoint_cubed_f_block_has_om_axis` unit test (which
/// materializes one F block over the OM=2 multiplicity structure).
#[test]
#[ignore = "heavy OM>=2 family (minutes in debug SVD): run with --release --ignored"]
fn d3_adjoint_om2_pentagon_hexagon() {
    let g = irr(Series::D, &[0, 1, 1]);
    let n = directproduct(&g, &g).unwrap().get(&g).copied().unwrap();
    assert_eq!(
        n, 2,
        "exact layer must predict OM=2 for the D3 adjoint square"
    );

    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    check_pentagon(&mut cat, &g, &g, &g, &g).unwrap();
    check_hexagon(&mut cat, &g, &g, &g).unwrap();
}
