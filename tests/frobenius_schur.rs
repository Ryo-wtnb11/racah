//! SU(N) Frobenius–Schur phase and indicator (issue #118).
//!
//! Oracles: SUNRepresentations 0.4.0 / TensorKitSectors 0.3.9 values, the sign
//! of racah's own `F[a, ā, a, a, 1, 1]` (the reference definition,
//! `sectors.jl:frobenius_schur_phase_from_Fsymbol`), and `su2_frobenius_schur`.

#![cfg(feature = "cgc-gen")]

use racah::sun::{f_symbol, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// Every Dynkin label of rank `n - 1` with label sum `<= max`.
fn labels(n: usize, max: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 1..n {
        out = out
            .into_iter()
            .flat_map(|v: Vec<i64>| {
                (0..=max).map(move |x| {
                    let mut w = v.clone();
                    w.push(x);
                    w
                })
            })
            .collect();
    }
    out.retain(|v| v.iter().sum::<i64>() <= max);
    out
}

/// `frobenius_schur_phase(SUNIrrep{N}(d))` from SUNRepresentations 0.4.0 with
/// TensorKitSectors 0.3.9 (Julia 1.11.6, TeNeT `tools/sun-table-gen`
/// environment), each equal to `dim(a) * Fsymbol(a, dual(a), a, a, one, one)`.
#[test]
fn phase_matches_sunrepresentations() {
    let reference: &[(&[i64], f64)] = &[
        (&[1], -1.0),
        (&[2], 1.0),
        (&[3], -1.0),
        (&[1, 0], 1.0),
        (&[1, 1], 1.0),
        (&[1, 0, 0], -1.0),
        (&[0, 0, 1], -1.0),
        (&[0, 1, 0], 1.0),
        (&[1, 1, 0], -1.0),
        (&[0, 1, 1], -1.0),
        (&[1, 0, 1], 1.0),
        (&[2, 0, 0], 1.0),
        (&[1, 0, 0, 0], 1.0),
        (&[1, 0, 0, 0, 0], -1.0),
        (&[0, 0, 1, 0, 0], -1.0),
        (&[0, 1, 0, 0, 0], 1.0),
        (&[0, 0, 0, 0, 1], -1.0),
    ];
    for &(d, want) in reference {
        assert_eq!(irr(d).frobenius_schur_phase(), want, "{d:?}");
    }
}

#[test]
fn indicator_is_phase_on_self_dual_irreps_and_zero_otherwise() {
    for n in 2..=7 {
        for d in labels(n, 3) {
            let a = irr(&d);
            let want = if a == a.dual() {
                a.frobenius_schur_phase() as i32
            } else {
                0
            };
            assert_eq!(a.frobenius_schur(), want, "{d:?}");
        }
    }
    // SU(2) through the SU(N) path agrees with the SU(2) family.
    for dj in 0..12u32 {
        let a = irr(&[i64::from(dj)]);
        assert_eq!(a.frobenius_schur_phase(), racah::su2_frobenius_schur(dj));
    }
    // Representation-theoretic anchors: SU(4) 6 = SO(6) vector is real, the
    // SU(6) 20 = Λ³C⁶ is quaternionic, the adjoint is always real.
    assert_eq!(irr(&[0, 1, 0]).frobenius_schur(), 1);
    assert_eq!(irr(&[0, 0, 1, 0, 0]).frobenius_schur(), -1);
    assert_eq!(irr(&[1, 0, 0, 0, 1]).frobenius_schur(), 1);
}

/// The closed form's premise: every GT lowering matrix element is
/// non-negative.
#[test]
fn lowering_matrix_elements_are_nonnegative() {
    for n in 2..=5 {
        for d in labels(n, 3) {
            for op in irr(&d).annihilation() {
                for e in op {
                    assert!(e.value.to_f64() >= 0.0, "{d:?} {e:?}");
                }
            }
        }
    }
}

/// The reference definition, evaluated on racah's own F-symbols:
/// `phase = sign(F[a, ā, a, a, 1, 1])` with `|F| = 1 / dim(a)`.
#[test]
fn phase_is_sign_of_f_symbol() {
    for (n, max) in [(2usize, 6i64), (3, 3), (4, 2), (5, 2), (6, 1), (7, 1)] {
        let one = Irrep::trivial(n).unwrap();
        for d in labels(n, max) {
            let a = irr(&d);
            let f = f_symbol(&a, &a.dual(), &a, &a, &one, &one).unwrap();
            let dim: f64 = a.dim().to_string().parse().unwrap();
            let scaled = f.at(0, 0, 0, 0) * dim;
            assert!(
                (scaled - a.frobenius_schur_phase()).abs() < 1e-9,
                "{d:?}: dim * F = {scaled}"
            );
        }
    }
}
