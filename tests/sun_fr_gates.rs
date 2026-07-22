//! Pentagon, hexagon, and F-move-unitarity self-consistency gates for SU(3)
//! (oracle 3 of issue #16). These are internal consistency identities — the
//! F/R data satisfy the fusion-category axioms — and are independent of the
//! closed-form / SUNRepresentations oracles.
//!
//! Coverage includes at least one **outer-multiplicity ≥ 2** family: the SU(3)
//! octet `8 = (1,1)`, where `8 ⊗ 8 → 8` has multiplicity 2, so the F blocks are
//! genuine `2×2×2×2` arrays and the identities mix multiplicity indices.

#![cfg(feature = "cgc-gen")]

use racah::sun::{check_f_unitarity, check_hexagon, check_pentagon, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// A deterministic sample of small SU(3) irreps to sweep the gates over.
fn sample() -> Vec<Irrep> {
    [
        &[0, 0][..], // 1
        &[1, 0][..], // 3
        &[0, 1][..], // 3̄
        &[2, 0][..], // 6
        &[1, 1][..], // 8
    ]
    .iter()
    .map(|d| irr(d))
    .collect()
}

#[test]
fn su3_f_unitarity_sweep() {
    let s = sample();
    for a in &s {
        for b in &s {
            for c in &s {
                for d in &s {
                    check_f_unitarity(a, b, c, d).unwrap();
                }
            }
        }
    }
}

#[test]
fn su3_pentagon_multiplicity_free_sweep() {
    // Sweep the small multiplicity-free families (fast).
    let s = [irr(&[1, 0]), irr(&[0, 1]), irr(&[2, 0])];
    for a in &s {
        for b in &s {
            for c in &s {
                for d in &s {
                    check_pentagon(a, b, c, d).unwrap();
                }
            }
        }
    }
}

#[test]
fn su3_hexagon_multiplicity_free_sweep() {
    let s = [irr(&[1, 0]), irr(&[0, 1]), irr(&[2, 0])];
    for a in &s {
        for b in &s {
            for c in &s {
                check_hexagon(a, b, c).unwrap();
            }
        }
    }
}

// ---- outer-multiplicity ≥ 2 (the SU(3) octet) ----

#[test]
fn su3_octet_f_unitarity_om2() {
    // 8⊗8⊗8 → 8: the F-move matrix mixes the OM=2 multiplicity indices.
    let e8 = irr(&[1, 1]);
    check_f_unitarity(&e8, &e8, &e8, &e8).unwrap();
}

#[test]
fn su3_octet_hexagon_om2() {
    // Full hexagon on the octet triple (drags in the 8⊗8→8 OM=2 vertices and
    // the 10/10̄/27 intermediates). ~0.4s release.
    let e8 = irr(&[1, 1]);
    check_hexagon(&e8, &e8, &e8).unwrap();
}

#[test]
fn su3_octet_pentagon_om2() {
    // Full pentagon on the octet quadruple: the OM≥2 gate the issue names
    // explicitly. ~2s release.
    let e8 = irr(&[1, 1]);
    check_pentagon(&e8, &e8, &e8, &e8).unwrap();
}
