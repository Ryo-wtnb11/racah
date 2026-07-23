//! Tests for the S3.1 defining-rep seeds. Anchors are independent of the code
//! under test: entry-for-entry against QSpace `Setup_*` (@ `dd2cc7e`) read by
//! hand, dimensions against Fulton–Harris / the merged S3.0 `Irrep::dim`, and
//! weights against the classical vector/fundamental weight sets.

use super::*;
use crate::bcd::Irrep;
use num_rational::Ratio;

fn ivec(entries: &[(usize, usize, i64)]) -> Vec<(usize, usize, i64)> {
    let mut v = entries.to_vec();
    v.sort_unstable();
    v
}

fn sorted_sp(seed: &Seed, i: usize) -> Vec<(usize, usize, i64)> {
    let mut v = seed.raising()[i].clone();
    v.sort_unstable();
    v
}

// ---- rank guard (inherited from S3.0) ------------------------------------

#[test]
fn excluded_low_ranks_rejected() {
    assert!(matches!(
        defining_seed(Series::B, 1),
        Err(BcdError::ExcludedRank { .. })
    ));
    assert!(matches!(
        defining_seed(Series::C, 1),
        Err(BcdError::ExcludedRank { .. })
    ));
    assert!(matches!(
        defining_seed(Series::D, 2),
        Err(BcdError::ExcludedRank { .. })
    ));
    // First admissible rank of each series is accepted.
    assert!(defining_seed(Series::B, 2).is_ok());
    assert!(defining_seed(Series::C, 2).is_ok());
    assert!(defining_seed(Series::D, 3).is_ok());
}

// ---- entry-for-entry ports (QSpace Setup_* @ dd2cc7e) --------------------

#[test]
fn spn_c2_entries() {
    // Setup_SpN, r=2, D=4.  Sp[0]:(0,1),(2,3); Sp[1]:(1,2).
    // Sz[0]=diag(1,-1,1,-1); Sz[1]=diag(1,1,-1,-1).
    let s = defining_seed(Series::C, 2).unwrap();
    assert_eq!(s.dim(), 4);
    assert_eq!(sorted_sp(&s, 0), ivec(&[(0, 1, 1), (2, 3, 1)]));
    assert_eq!(sorted_sp(&s, 1), ivec(&[(1, 2, 1)]));
    assert_eq!(s.cartan()[0], vec![1, -1, 1, -1]);
    assert_eq!(s.cartan()[1], vec![1, 1, -1, -1]);
}

#[test]
fn spn_c3_entries() {
    // Setup_SpN, r=3, D=6. Long-root ladder Sp[2] is a single entry (2,3).
    let s = defining_seed(Series::C, 3).unwrap();
    assert_eq!(s.dim(), 6);
    assert_eq!(sorted_sp(&s, 0), ivec(&[(0, 1, 1), (4, 5, 1)]));
    assert_eq!(sorted_sp(&s, 1), ivec(&[(1, 2, 1), (3, 4, 1)]));
    assert_eq!(sorted_sp(&s, 2), ivec(&[(2, 3, 1)]));
    assert_eq!(s.cartan()[0], vec![1, -1, 0, 0, 1, -1]);
    assert_eq!(s.cartan()[1], vec![1, 1, -2, 2, -1, -1]);
    assert_eq!(s.cartan()[2], vec![1, 1, 1, -1, -1, -1]);
}

#[test]
fn son_b2_entries() {
    // Setup_SON, r=2, D=5. Short-root ladder Sp[1] touches zero-weight state 4.
    let s = defining_seed(Series::B, 2).unwrap();
    assert_eq!(s.dim(), 5);
    assert_eq!(sorted_sp(&s, 0), ivec(&[(1, 3, 1), (2, 0, 1)]));
    assert_eq!(sorted_sp(&s, 1), ivec(&[(0, 4, 1), (4, 1, 1)]));
    assert_eq!(s.cartan()[0], vec![1, -1, 0, 0, 0]);
    assert_eq!(s.cartan()[1], vec![0, 0, 1, -1, 0]);
}

#[test]
fn sen_d3_entries() {
    // Setup_SEN, r=3, D=6. Fork node Sp[2] = fixed entries (2,1),(0,3).
    let s = defining_seed(Series::D, 3).unwrap();
    assert_eq!(s.dim(), 6);
    assert_eq!(sorted_sp(&s, 0), ivec(&[(1, 3, 1), (2, 0, 1)]));
    assert_eq!(sorted_sp(&s, 1), ivec(&[(3, 5, 1), (4, 2, 1)]));
    assert_eq!(sorted_sp(&s, 2), ivec(&[(0, 3, 1), (2, 1, 1)]));
    assert_eq!(s.cartan()[0], vec![1, -1, 0, 0, 0, 0]);
    assert_eq!(s.cartan()[1], vec![0, 0, 1, -1, 0, 0]);
    assert_eq!(s.cartan()[2], vec![0, 0, 0, 0, 1, -1]);
}

// ---- dimension anchors (== S3.0 wdim of the defining label) --------------

#[test]
fn dims_match_s30_defining_wdim() {
    // Sanity anchors: B2=5, C2=4, D3=6, plus more ranks vs Irrep::dim.
    for (series, r, want) in [
        (Series::B, 2usize, 5u32),
        (Series::C, 2, 4),
        (Series::D, 3, 6),
        (Series::B, 3, 7),
        (Series::C, 3, 6),
        (Series::D, 4, 8),
        (Series::B, 4, 9),
        (Series::C, 4, 8),
    ] {
        let s = defining_seed(series, r).unwrap();
        assert_eq!(s.dim() as u32, want, "{series:?}_{r} dim");
        // Defining label (1,0,...,0): wdim from the merged S3.0.
        let mut dynkin = vec![0i64; r];
        dynkin[0] = 1;
        let irr = Irrep::from_dynkin(series, &dynkin).unwrap();
        assert_eq!(
            irr.dim(),
            num_bigint::BigInt::from(s.dim()),
            "{series:?}_{r} vs S3.0 wdim"
        );
    }
}

// ---- commutator self-check passes for all admissible seeds ---------------

#[test]
fn commutators_pass_all_series_and_ranks() {
    for series in [Series::B, Series::C, Series::D] {
        for r in series_ranks(series) {
            let s = defining_seed(series, r).unwrap();
            check_commutators(&s).unwrap_or_else(|e| panic!("{series:?}_{r}: {e}"));
        }
    }
}

fn series_ranks(series: Series) -> Vec<usize> {
    // Cover through QSpace's own build ceilings (Sp D<=10 ⇒ r<=5; SO D<=12).
    match series {
        Series::B => vec![2, 3, 4, 5],
        Series::C => vec![2, 3, 4, 5],
        Series::D => vec![3, 4, 5, 6],
    }
}

// ---- structure constants documented in CommReport ------------------------

#[test]
fn report_cartan_and_roots_are_exact() {
    // C2 (hand-derived from the ported entries):
    //   short root Sp_0 = E01+E23  ⇒ [Sp_0,Sp_0^†] = diag(1,-1,1,-1) = Sz_0.
    //   long  root Sp_1 = E12      ⇒ [Sp_1,Sp_1^†] = diag(0,1,-1,0)
    //                              = -½ Sz_0 + ½ Sz_1  (a genuinely FRACTIONAL
    //     projection in QSpace's non-Chevalley basis — the long root does not
    //     land on a single Cartan generator).
    let rep = check_commutators(&defining_seed(Series::C, 2).unwrap()).unwrap();
    let one = Ratio::from_integer(1);
    let zero = Ratio::from_integer(0);
    let half = Ratio::new(1, 2);
    assert_eq!(rep.cartan_coeffs[0], vec![one, zero]);
    assert_eq!(rep.cartan_coeffs[1], vec![-half, half]);
    // Roots in the Sz basis (hand-derived from [Sz_j,Sp_i]).
    let two = Ratio::from_integer(2);
    let mtwo = Ratio::from_integer(-2);
    assert_eq!(rep.root_weights[0], vec![two, zero]); // α_0=(2,0)
    assert_eq!(rep.root_weights[1], vec![mtwo, two]); // α_1=(-2,2)

    // B2 short root: [Sp_1,Sp_1^†]=Sz_0 (projects onto a DIFFERENT Cartan).
    let repb = check_commutators(&defining_seed(Series::B, 2).unwrap()).unwrap();
    assert_eq!(repb.cartan_coeffs[1], vec![one, zero]);
}

// ---- weight consistency vs the classical defining-rep weights ------------

/// For each state, its weight vector in the Sz basis: `(Sz_0[k],…,Sz_{r-1}[k])`.
fn weight_vectors(seed: &Seed) -> Vec<Vec<i64>> {
    (0..seed.dim())
        .map(|k| seed.cartan().iter().map(|z| z[k]).collect())
        .collect()
}

#[test]
fn son_sen_weights_are_the_vector_rep_weight_set() {
    // For B and D, QSpace's Sz basis IS the orthonormal ε-basis, so the state
    // weights are exactly {±e_i} (plus 0 for the odd B vector rep) — the
    // classical vector-representation weights (Fulton–Harris §19).
    for (series, r) in [(Series::B, 3usize), (Series::D, 4usize), (Series::B, 2)] {
        let s = defining_seed(series, r).unwrap();
        let mut got = weight_vectors(&s);
        got.sort();
        let mut want: Vec<Vec<i64>> = Vec::new();
        for i in 0..r {
            let mut p = vec![0i64; r];
            p[i] = 1;
            want.push(p.clone());
            p[i] = -1;
            want.push(p);
        }
        if series == Series::B {
            want.push(vec![0i64; r]); // the zero weight of the odd vector rep
        }
        want.sort();
        assert_eq!(got, want, "{series:?}_{r} weight set");
    }
}

#[test]
fn cartan_traceless_and_weights_negation_closed() {
    // Defining reps are self-dual: the weight multiset is closed under
    // negation, and every Cartan generator is traceless. True in ANY basis, so
    // this is the cross-series (incl. Sp's non-ε basis) weight sanity.
    for series in [Series::B, Series::C, Series::D] {
        for r in series_ranks(series) {
            let s = defining_seed(series, r).unwrap();
            for z in s.cartan() {
                assert_eq!(z.iter().sum::<i64>(), 0, "{series:?}_{r} traceless");
            }
            let mut w = weight_vectors(&s);
            let mut neg: Vec<Vec<i64>> = w.iter().map(|v| v.iter().map(|x| -x).collect()).collect();
            w.sort();
            neg.sort();
            assert_eq!(w, neg, "{series:?}_{r} weight set ±-closed");
        }
    }
}

// ---- mutation sanity: the self-check is load-bearing ---------------------
//
// Each mutation perturbs one seed entry and asserts the commutator check now
// FAILS — proving the gate actually constrains the ported matrices.

/// Build a seed then hand it to a mutator that corrupts one entry.
fn mutated(series: Series, r: usize, f: impl FnOnce(&mut Seed)) -> Seed {
    let mut s = defining_seed(series, r).unwrap();
    f(&mut s);
    s
}

#[test]
fn mutation_drop_paired_ladder_entry_fails() {
    // Drop one of the two short-root entries of Sp[0] in C3: [Sz,Sp] and the
    // ladder–Cartan relation break.
    let s = mutated(Series::C, 3, |s| {
        s.raising_mut()[0].truncate(1);
    });
    assert!(matches!(
        check_commutators(&s),
        Err(BcdError::CommutatorViolation { .. })
    ));
}

#[test]
fn mutation_flip_fork_entry_fails() {
    // Move the D-series fork entry (2,1) to (2,0): breaks the root relation.
    let s = mutated(Series::D, 3, |s| {
        let fork = &mut s.raising_mut()[2];
        for e in fork.iter_mut() {
            if e.0 == 2 && e.1 == 1 {
                *e = (2, 0, 1);
            }
        }
    });
    assert!(matches!(
        check_commutators(&s),
        Err(BcdError::CommutatorViolation { .. })
    ));
}

#[test]
fn mutation_scale_ladder_entry_fails() {
    // Rescale a B2 ladder entry by 2 (the "drop the normalization" mutation):
    // [Sp,Sp^†] leaves span(Sz) integrally / root ratio splits.
    let s = mutated(Series::B, 2, |s| {
        s.raising_mut()[0][0].2 = 2;
    });
    assert!(matches!(
        check_commutators(&s),
        Err(BcdError::CommutatorViolation { .. })
    ));
}

#[test]
fn mutation_perturb_cartan_diagonal_fails() {
    // Perturb one Sz entry: breaks Cartan orthogonality and/or the root eigen-
    // relation.
    let s = mutated(Series::C, 2, |s| {
        s.cartan_mut()[0][0] = 3;
    });
    assert!(matches!(
        check_commutators(&s),
        Err(BcdError::CommutatorViolation { .. })
    ));
}
