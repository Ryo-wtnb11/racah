//! Global form (issue #87, stage (a)): the per-family congruence tables, the
//! equivalence of the predicate with the rejection `bcd` already ships, and
//! closure of admissibility under fusion.
//!
//! Nothing here computes a coefficient; stage (a) changes no value.

use racah::group::{
    d_odd_central_class, CenterSubgroup, GlobalForm, GroupError, GroupId, RootSystem,
};

// ---------------------------------------------------------------------------
// A_r — N-ality
// ---------------------------------------------------------------------------

#[test]
fn a_series_nality_congruences() {
    // SU(4): κ = a₁ + 2a₂ + 3a₃.
    let su4 = GroupId::su(4).unwrap();
    let su4_z2 = GroupId::su_quotient(4, 2).unwrap();
    let psu4 = GroupId::psu(4).unwrap();

    // (label, κ mod 4)
    let table = [
        ([0, 0, 0], 0), // trivial
        ([1, 0, 0], 1), // 4
        ([0, 0, 1], 3), // 4bar
        ([0, 1, 0], 2), // 6
        ([1, 0, 1], 0), // 15, the adjoint
        ([2, 0, 0], 2), // 10
        ([0, 2, 0], 0), // 20'
    ];
    for (label, kappa) in table {
        assert!(su4.admits(&label), "SU(4) admits everything: {label:?}");
        assert_eq!(
            su4_z2.admits(&label),
            kappa % 2 == 0,
            "SU(4)/Z2 on {label:?}"
        );
        assert_eq!(psu4.admits(&label), kappa % 4 == 0, "PSU(4) on {label:?}");
    }

    // SU(3): only k ∈ {1, 3}; PSU(3) is triality-0, i.e. adjoint and up.
    let psu3 = GroupId::psu(3).unwrap();
    assert!(psu3.admits(&[0, 0])); // 1
    assert!(!psu3.admits(&[1, 0])); // 3
    assert!(!psu3.admits(&[0, 1])); // 3bar
    assert!(psu3.admits(&[1, 1])); // 8
    assert!(psu3.admits(&[3, 0])); // 10

    // k must divide N.
    assert_eq!(
        GroupId::su_quotient(6, 4),
        Err(GroupError::NotACenterSubgroup { n: 6, k: 4 })
    );
    // `su_quotient(n, 1)` is SU(n) as a group, but a distinct `GroupId` — the
    // quotient by the trivial subgroup. Both admit everything.
    assert_eq!(
        GroupId::su_quotient(4, 1).unwrap().form,
        GlobalForm::Quotient(CenterSubgroup::Zk(1))
    );
}

#[test]
fn a_series_trivial_quotient_admits_everything() {
    let g = GroupId::su_quotient(5, 1).unwrap();
    for a in 0..3i64 {
        for b in 0..3i64 {
            assert!(g.admits(&[a, b, 0, 0]));
        }
    }
}

// ---------------------------------------------------------------------------
// B_r — a_r even
// ---------------------------------------------------------------------------

#[test]
fn b_series_last_label_even() {
    for r in 2..=5usize {
        let spin = GroupId::spin(2 * r + 1).unwrap();
        let so = GroupId::so(2 * r + 1).unwrap();
        assert_eq!(so.root_system, RootSystem::B(r));
        // PSO(2r+1) == SO(2r+1): the B_r center is already quotiented out.
        assert_eq!(GroupId::pso(2 * r + 1).unwrap(), so);
        for label in sweep(r, 3) {
            assert!(spin.admits(&label), "Spin admits {label:?}");
            assert_eq!(so.admits(&label), label[r - 1] % 2 == 0, "SO on {label:?}");
        }
    }
    // Spin(5) spinor ω₂ = (0,1) is not an SO(5) rep; the vector ω₁ = (1,0) is.
    assert!(!GroupId::so(5).unwrap().admits(&[0, 1]));
    assert!(GroupId::so(5).unwrap().admits(&[1, 0]));
    assert!(GroupId::spin(5).unwrap().admits(&[0, 1]));
}

// ---------------------------------------------------------------------------
// C_r — odd-index sum even
// ---------------------------------------------------------------------------

#[test]
fn c_series_odd_index_sum_even() {
    for r in 2..=5usize {
        let sp = GroupId::sp(2 * r).unwrap();
        let psp = GroupId::psp(2 * r).unwrap();
        assert_eq!(sp.root_system, RootSystem::C(r));
        for label in sweep(r, 3) {
            assert!(sp.admits(&label), "Sp is simply connected: {label:?}");
            let odd: i64 = label.iter().step_by(2).sum();
            assert_eq!(psp.admits(&label), odd % 2 == 0, "PSp on {label:?}");
        }
    }
    // PSp(4): the defining 4 = (1,0) is not a PSp rep; 5 = (0,1) and the
    // adjoint 10 = (2,0) are.
    let psp4 = GroupId::psp(4).unwrap();
    assert!(!psp4.admits(&[1, 0]));
    assert!(psp4.admits(&[0, 1]));
    assert!(psp4.admits(&[2, 0]));
    assert_eq!(
        GroupId::sp(3),
        Err(GroupError::UnsupportedRank {
            family: "sp",
            value: 3
        })
    );
}

// ---------------------------------------------------------------------------
// D_r, r even — the three index-2 subgroups, and the r mod 4 naming hazard
// ---------------------------------------------------------------------------

/// The half-spin congruences are stated by the class they RETAIN, so they are
/// identical formulas at `r ≡ 0` and `r ≡ 2 (mod 4)`. Were they named by the
/// central element killed, the two would swap between `r = 4` and `r = 6`.
#[test]
fn d_even_three_quotients_at_both_r_mod_4() {
    for r in [4usize, 6] {
        let n = 2 * r;
        let so = GroupId::so(n).unwrap();
        let pso = GroupId::pso(n).unwrap();
        let hs_plus = GroupId::half_spin_plus(n).unwrap();
        let hs_minus = GroupId::half_spin_minus(n).unwrap();
        assert_eq!(so.form, GlobalForm::Quotient(CenterSubgroup::DVector));
        assert_eq!(pso.form, GlobalForm::Quotient(CenterSubgroup::DFull));

        for label in sweep(r, 3) {
            let t: i64 = label[..r - 2].iter().step_by(2).sum();
            let p = (label[r - 2] + t) % 2;
            let q = (label[r - 1] + t) % 2;
            assert_eq!(
                so.admits(&label),
                (label[r - 2] + label[r - 1]) % 2 == 0,
                "SO({n}) on {label:?}"
            );
            assert_eq!(
                hs_plus.admits(&label),
                p == 0,
                "half_spin_plus({n}) on {label:?}"
            );
            assert_eq!(
                hs_minus.admits(&label),
                q == 0,
                "half_spin_minus({n}) on {label:?}"
            );
            assert_eq!(
                pso.admits(&label),
                p == 0 && q == 0,
                "PSO({n}) on {label:?}"
            );
        }

        // ω_r (the retained chirality of `plus`) and ω_{r-1} (of `minus`).
        let mut omega_last = vec![0i64; r];
        omega_last[r - 1] = 1;
        let mut omega_prev = vec![0i64; r];
        omega_prev[r - 2] = 1;
        assert!(hs_plus.admits(&omega_last), "r={r}: plus retains ω_r");
        assert!(!hs_plus.admits(&omega_prev));
        assert!(
            hs_minus.admits(&omega_prev),
            "r={r}: minus retains ω_{{r-1}}"
        );
        assert!(!hs_minus.admits(&omega_last));

        // A half-spin group has no vector representation.
        let mut vector = vec![0i64; r];
        vector[0] = 1;
        assert!(!hs_plus.admits(&vector));
        assert!(!hs_minus.admits(&vector));
        assert!(so.admits(&vector));

        // The adjoint is in every form.
        let mut adjoint = vec![0i64; r];
        adjoint[1] = 1;
        for g in [so, pso, hs_plus, hs_minus] {
            assert!(g.admits(&adjoint), "{g:?} must admit the adjoint");
        }
    }

    // Half-spin needs r even, i.e. n ≡ 0 (mod 4).
    assert_eq!(
        GroupId::half_spin_plus(10),
        Err(GroupError::UnsupportedRank {
            family: "half_spin",
            value: 10
        })
    );
}

// ---------------------------------------------------------------------------
// D_r, r odd — the Z4 class, anchored on Slansky Table 41 (SO(10))
// ---------------------------------------------------------------------------

#[test]
fn d_odd_z4_class_slansky_so10_anchors() {
    // Slansky Table 41: "the adjoint is in the 0 class, the spinor in 1, the
    // conjugate spinors in -1, and the vector … in the 2 class."
    assert_eq!(d_odd_central_class(&[0, 1, 0, 0, 0]), 0); // 45, adjoint
    assert_eq!(d_odd_central_class(&[0, 0, 0, 0, 1]), 1); // 16 = ω₅
    assert_eq!(d_odd_central_class(&[0, 0, 0, 1, 0]), 3); // 16bar = ω₄, i.e. -1
    assert_eq!(d_odd_central_class(&[1, 0, 0, 0, 0]), 2); // 10, vector
    assert_eq!(d_odd_central_class(&[0, 0, 1, 0, 0]), 2); // 120 = ω₃, class v

    // The generator convention: κ agrees with `dual` (λ_r ↦ -λ_r), so
    // conjugation negates the class.
    for label in sweep(5, 3) {
        let mut conj = label.clone();
        conj.swap(3, 4);
        assert_eq!(
            (d_odd_central_class(&label) + d_odd_central_class(&conj)) % 4,
            0,
            "κ(λ*) = -κ(λ) on {label:?}"
        );
    }

    let so10 = GroupId::so(10).unwrap();
    let pso10 = GroupId::pso(10).unwrap();
    let spin10 = GroupId::spin(10).unwrap();
    assert_eq!(pso10.form, GlobalForm::Quotient(CenterSubgroup::Z4));
    for label in sweep(5, 3) {
        let kappa = d_odd_central_class(&label);
        assert!(spin10.admits(&label));
        // κ mod 2 reproduces the shipped SO condition a_{r-1}+a_r even.
        assert_eq!(so10.admits(&label), kappa % 2 == 0, "SO(10) on {label:?}");
        assert_eq!(so10.admits(&label), (label[3] + label[4]) % 2 == 0);
        assert_eq!(pso10.admits(&label), kappa == 0, "PSO(10) on {label:?}");
    }

    // There is no half-spin group for odd r: Z4 has one proper subgroup.
    assert!(!GroupId {
        root_system: RootSystem::D(5),
        form: GlobalForm::Quotient(CenterSubgroup::DHalfSpinPlus),
    }
    .admits(&[0, 0, 0, 0, 1]));
}

// ---------------------------------------------------------------------------
// Ill-formed input
// ---------------------------------------------------------------------------

#[test]
fn admits_rejects_wrong_length_and_negative_labels() {
    let su4 = GroupId::su(4).unwrap();
    assert!(!su4.admits(&[0, 0]));
    assert!(!su4.admits(&[0, 0, 0, 0]));
    assert!(!su4.admits(&[-1, 0, 0]));
    assert!(!GroupId::so(9).unwrap().admits(&[0, 0, -2, 0]));
}

// ---------------------------------------------------------------------------
// Equivalence with the rejection `bcd` already ships, and fusion closure
// ---------------------------------------------------------------------------

#[cfg(feature = "cgc-gen")]
mod against_bcd {
    use super::*;
    use racah::bcd::{directproduct, BcdError, Irrep, Series};

    /// The group `bcd` publishes for `series` at rank `r` (issue #18 Ruling 3).
    fn published(series: Series, r: usize) -> GroupId {
        match series {
            Series::B => GroupId::so(2 * r + 1).unwrap(),
            Series::C => GroupId::sp(2 * r).unwrap(),
            Series::D => GroupId::so(2 * r).unwrap(),
        }
    }

    #[test]
    fn series_root_system_bridge() {
        assert_eq!(Series::B.root_system(3), RootSystem::B(3));
        assert_eq!(Series::C.root_system(3), RootSystem::C(3));
        assert_eq!(Series::D.root_system(4), RootSystem::D(4));
        for (series, r) in [(Series::B, 2), (Series::C, 2), (Series::D, 3)] {
            assert_eq!(series.root_system(r), published(series, r).root_system);
        }
    }

    /// The whole point of stage (a): `from_dynkin` rejects exactly what the
    /// published form fails to admit, over a sector sweep.
    #[test]
    fn predicate_matches_shipped_rejection() {
        let mut checked = 0usize;
        for (series, min_r) in [(Series::B, 2), (Series::C, 2), (Series::D, 3)] {
            for r in min_r..=min_r + 2 {
                let group = published(series, r);
                for label in sweep(r, 3) {
                    let built = Irrep::from_dynkin(series, &label);
                    let admitted = group.admits(&label);
                    match (&built, admitted) {
                        (Ok(_), true) => {}
                        (Err(BcdError::NotAdmissible { .. }), false) => {}
                        other => panic!(
                            "series {series:?} rank {r} label {label:?}: {other:?} \
                             disagrees with admits = {admitted}"
                        ),
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 500, "sweep too small: {checked}");
    }

    /// Admissibility is closed under fusion (the class map is a homomorphism),
    /// which is why enforcement is construction-only.
    #[test]
    fn admissibility_is_closed_under_fusion() {
        for (series, r) in [(Series::B, 3), (Series::C, 3), (Series::D, 4)] {
            let group = published(series, r);
            let labels: Vec<_> = sweep(r, 2)
                .into_iter()
                .filter(|l| group.admits(l))
                .collect();
            for a in &labels {
                for b in &labels {
                    let ia = Irrep::from_dynkin(series, a).unwrap();
                    let ib = Irrep::from_dynkin(series, b).unwrap();
                    for c in directproduct(&ia, &ib).unwrap().keys() {
                        assert!(
                            group.admits(&c.dynkin()),
                            "{series:?} r={r}: {a:?} ⊗ {b:?} produced {:?}",
                            c.dynkin()
                        );
                    }
                }
            }
        }
    }

    /// The same closure for a *quotient* form the engine does not itself know
    /// about: PSp(6) and PSO(8) label subsets of the shipped catalogs.
    #[test]
    fn quotient_forms_are_closed_under_fusion() {
        for (series, r, group) in [
            (Series::C, 3, GroupId::psp(6).unwrap()),
            (Series::D, 4, GroupId::pso(8).unwrap()),
        ] {
            let labels: Vec<_> = sweep(r, 2)
                .into_iter()
                .filter(|l| group.admits(l) && Irrep::from_dynkin(series, l).is_ok())
                .collect();
            assert!(labels.len() > 2, "{group:?}: nothing to fuse");
            for a in &labels {
                for b in &labels {
                    let ia = Irrep::from_dynkin(series, a).unwrap();
                    let ib = Irrep::from_dynkin(series, b).unwrap();
                    for c in directproduct(&ia, &ib).unwrap().keys() {
                        assert!(
                            group.admits(&c.dynkin()),
                            "{group:?}: {a:?} ⊗ {b:?} produced {:?}",
                            c.dynkin()
                        );
                    }
                }
            }
        }
    }

    /// `sun` is the simply connected form of `A`: it admits every non-negative
    /// Dynkin label, and `A(n-1) + SimplyConnected` says exactly that.
    #[test]
    fn sun_is_the_simply_connected_a_form() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                assert!(su.admits(&label));
                assert!(racah::sun::Irrep::from_dynkin(&label).is_ok());
            }
        }
    }

    /// **Diagnostic, not an assertion of policy** (issue #87 Q4, the flagged
    /// verification gap). Run with `--ignored`.
    ///
    /// The hand example: for `B₂ = so(5)` the adjoint `10` has both the
    /// spinor `4` and the vector `5` strictly below it in `≺` (box counts
    /// 1, 1 versus 2), and both `4⊗4` and `5⊗5` contain it. Under a
    /// cover-wide parent search (option (A)) the pair key `dim_a+dim_b`
    /// would pick `(4,4) = 8` over `(5,5) = 10`, moving the canonical parent
    /// of the SO(5) adjoint and hence its frame.
    ///
    /// racah cannot build Spin(5) spinors today, so this is measured through
    /// the isomorphism `Spin(5) ≅ Sp(4)`, where the `B₂` spinor is the `Sp(4)`
    /// defining rep `(1,0)` and the `B₂` vector is `(0,1)`.
    #[test]
    #[ignore = "diagnostic for issue #87 Q4; run with --ignored"]
    fn so5_canonical_parent_hand_example() {
        use num_bigint::BigInt;
        let irrep = |a: i64, b: i64| Irrep::from_dynkin(Series::C, &[a, b]).unwrap();
        let spinor = irrep(1, 0); // B₂ ω₂, the 4
        let vector = irrep(0, 1); // B₂ ω₁, the 5
        let adjoint = irrep(2, 0); // B₂ 2ω₂, the 10

        assert_eq!(spinor.dim(), BigInt::from(4));
        assert_eq!(vector.dim(), BigInt::from(5));
        assert_eq!(adjoint.dim(), BigInt::from(10));

        let ss = directproduct(&spinor, &spinor).unwrap();
        let vv = directproduct(&vector, &vector).unwrap();
        let dims = |m: &std::collections::BTreeMap<Irrep, u32>| {
            let mut d: Vec<i64> = m
                .keys()
                .map(|k| k.dim().to_string().parse().unwrap())
                .collect();
            d.sort_unstable();
            d
        };
        assert_eq!(dims(&ss), vec![1, 5, 10], "4⊗4 = 1+5+10");
        assert_eq!(dims(&vv), vec![1, 10, 14], "5⊗5 = 1+10+14");
        assert!(ss.contains_key(&adjoint) && vv.contains_key(&adjoint));

        let key_ss = 4 + 4;
        let key_vv = 5 + 5;
        println!(
            "SO(5) adjoint (10): (spinor,spinor) pair key {key_ss}, (V,V) pair key {key_vv} \
             — spinor pair {} under the shipped `dim_a+dim_b` rule; box counts are 1, 1, 2 \
             so both parents are ≺ the adjoint.",
            if key_ss < key_vv { "WINS" } else { "loses" }
        );
        assert!(key_ss < key_vv, "the spinor pair wins: {key_ss} < {key_vv}");
    }
}

/// Every non-negative Dynkin label of rank `r` with each entry `< cap`.
fn sweep(r: usize, cap: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 0..r {
        out = out
            .into_iter()
            .flat_map(|p| {
                (0..cap).map(move |a| {
                    let mut p = p.clone();
                    p.push(a);
                    p
                })
            })
            .collect();
    }
    out
}

// ---------------------------------------------------------------------------
// `sun` in the global-form model (issue #87 §1)
// ---------------------------------------------------------------------------

/// The A-series form-aware constructor, mirroring `bcd::Irrep::from_dynkin_in`.
#[cfg(feature = "cgc-gen")]
mod sun_forms {
    use super::*;
    use racah::sun::{Irrep, SunError};

    /// `N`-ality `Σ i·aᵢ`, computed here independently of the library so the
    /// tests are not checking `group::admits` against itself.
    fn nality(dynkin: &[i64]) -> i64 {
        dynkin
            .iter()
            .enumerate()
            .map(|(i, &a)| (i as i64 + 1) * a)
            .sum()
    }

    /// (1) `SU(N)` is simply connected: every dominant integral weight is a
    /// representation, through either constructor.
    #[test]
    fn su_n_accepts_every_dominant_integral_weight() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                let via_form = Irrep::from_dynkin_in(&su, &label).expect("SU(N) admits everything");
                assert_eq!(via_form, Irrep::from_dynkin(&label).unwrap());
            }
        }
    }

    /// (2) `PSU(N)` rejects every non-zero `N`-ality, and (3) accepts the
    /// adjoint. The rejection is the global-form error, not a label error.
    #[test]
    fn psu_n_rejects_nonzero_nality_and_accepts_the_adjoint() {
        for n in 2..=5usize {
            let psu = GroupId::psu(n).unwrap();
            for label in sweep(n - 1, 4) {
                let built = Irrep::from_dynkin_in(&psu, &label);
                let zero_nality = nality(&label).rem_euclid(n as i64) == 0;
                match (&built, zero_nality) {
                    (Ok(_), true) => {}
                    (Err(SunError::NotAdmissible { group, dynkin }), false) => {
                        assert_eq!(*group, psu);
                        assert_eq!(dynkin, &label);
                    }
                    other => panic!("PSU({n}) on {label:?}: {other:?}"),
                }
            }
            // The adjoint is (1,0,…,0,1) for N ≥ 3, (2) for N = 2 — always
            // zero N-ality, and every group with this algebra has it.
            let mut adjoint = vec![0i64; n - 1];
            adjoint[0] += 1;
            adjoint[n - 2] += 1;
            assert_eq!(nality(&adjoint).rem_euclid(n as i64), 0);
            assert!(Irrep::from_dynkin_in(&psu, &adjoint).is_ok());
        }
        // The stated example: PSU(3) takes the adjoint 8, not the fundamental 3.
        let psu3 = GroupId::psu(3).unwrap();
        assert!(Irrep::from_dynkin_in(&psu3, &[1, 1]).is_ok());
        assert!(matches!(
            Irrep::from_dynkin_in(&psu3, &[1, 0]),
            Err(SunError::NotAdmissible { .. })
        ));
        assert!(matches!(
            Irrep::from_dynkin_in(&psu3, &[0, 1]),
            Err(SunError::NotAdmissible { .. })
        ));
    }

    /// (4) `SU(N)/Z_k` needs `k | N`, and then admits exactly `N`-ality
    /// `≡ 0 (mod k)`.
    #[test]
    fn su_quotient_divisibility_condition() {
        assert!(matches!(
            GroupId::su_quotient(6, 4),
            Err(GroupError::NotACenterSubgroup { n: 6, k: 4 })
        ));
        for (n, k) in [(4usize, 2usize), (6, 2), (6, 3), (6, 6), (4, 1)] {
            let g = GroupId::su_quotient(n, k).unwrap();
            for label in sweep(n - 1, 3) {
                let ok = nality(&label).rem_euclid(k as i64) == 0;
                assert_eq!(
                    Irrep::from_dynkin_in(&g, &label).is_ok(),
                    ok,
                    "SU({n})/Z{k} on {label:?}"
                );
            }
        }
    }

    /// (5) `from_dynkin` is untouched: still SU(N), still the same object, and
    /// still the same errors.
    #[test]
    fn from_dynkin_is_unchanged_and_equals_the_simply_connected_form() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                let a = Irrep::from_dynkin(&label).unwrap();
                let b = Irrep::from_dynkin_in(&su, &label).unwrap();
                assert_eq!(a, b);
                assert_eq!(a.weight(), b.weight());
                assert_eq!(a.dynkin(), b.dynkin());
                assert_eq!(a.rank(), n);
            }
        }
        assert!(matches!(
            Irrep::from_dynkin(&[-1, 0]),
            Err(SunError::NegativeDynkin { .. })
        ));
        assert_eq!(Irrep::from_dynkin(&[]).unwrap(), Irrep::trivial(1).unwrap());
    }

    /// (6) Global-form validation is a gate, not a transform: the GT data of a
    /// weight admissible in a quotient is bit-identical to the cover's.
    #[test]
    fn global_form_validation_does_not_alter_gt_data() {
        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        for label in sweep(2, 4).into_iter().filter(|l| psu3.admits(l)) {
            let cover = Irrep::from_dynkin_in(&su3, &label).unwrap();
            let quotient = Irrep::from_dynkin_in(&psu3, &label).unwrap();
            assert_eq!(cover, quotient);
            assert_eq!(cover.dim(), quotient.dim());
            assert_eq!(cover.dual(), quotient.dual());
            assert_eq!(cover.patterns(), quotient.patterns());
            assert_eq!(cover.creation(), quotient.creation());
            assert_eq!(cover.annihilation(), quotient.annihilation());
        }
    }

    /// The constructor's own guards, distinct from a form violation.
    #[test]
    fn from_dynkin_in_rejects_a_foreign_root_system_and_a_wrong_length_label() {
        assert!(matches!(
            Irrep::from_dynkin_in(&GroupId::so(7).unwrap(), &[1, 0, 0]),
            Err(SunError::UnsupportedRootSystem {
                root_system: RootSystem::B(3)
            })
        ));
        assert!(matches!(
            Irrep::from_dynkin_in(&GroupId::su(3).unwrap(), &[1, 0, 0]),
            Err(SunError::LabelRankMismatch {
                expected: 2,
                got: 3
            })
        ));
        assert!(matches!(
            Irrep::from_dynkin_in(&GroupId::su(3).unwrap(), &[-1, 0]),
            Err(SunError::NegativeDynkin { .. })
        ));
    }

    /// A1 stays consistent: `SU(2)` is `A_1` simply connected, `PSU(2) = SO(3)`
    /// is its even-Dynkin (integer-spin) sublattice — the same statement the
    /// `B_1` low-rank redirect makes.
    #[test]
    fn a1_semantics_agree_with_the_su2_layer() {
        let psu2 = GroupId::psu(2).unwrap();
        for dj in 0..8i64 {
            assert_eq!(
                Irrep::from_dynkin_in(&psu2, &[dj]).is_ok(),
                dj % 2 == 0,
                "PSU(2) = SO(3) admits integer spin only: dj={dj}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The admissibility error is generic, not spinor-specific (issue #87 §2)
// ---------------------------------------------------------------------------

/// (9) `PSp(2N)` has no spinors at all — `C_r` is simply connected — yet it
/// rejects weights. The error must be the generic global-form violation, and
/// must name the group that did the rejecting.
#[cfg(feature = "cgc-gen")]
#[test]
fn psp_rejection_uses_the_generic_not_admissible_error() {
    use racah::bcd::{BcdError, Irrep};
    let psp6 = GroupId::psp(6).unwrap();
    match Irrep::from_dynkin_in(&psp6, &[1, 0, 0]) {
        Err(BcdError::NotAdmissible { group, dynkin }) => {
            assert_eq!(group, psp6);
            assert_eq!(dynkin, vec![1, 0, 0]);
            let msg = BcdError::NotAdmissible { group, dynkin }.to_string();
            assert!(
                !msg.to_lowercase().contains("spinor"),
                "PSp(6) rejection must not talk about spinors: {msg}"
            );
        }
        other => panic!("expected NotAdmissible, got {other:?}"),
    }
    // The half-spin and PSO forms reject tensor classes for the same reason.
    for (group, label) in [
        (GroupId::pso(8).unwrap(), vec![1, 0, 0, 0]), // the vector of SO(8)
        (GroupId::half_spin_plus(8).unwrap(), vec![1, 0, 0, 0]),
    ] {
        assert!(matches!(
            Irrep::from_dynkin_in(&group, &label),
            Err(BcdError::NotAdmissible { .. })
        ));
    }
    // And the spinor case still lands in the same variant.
    assert!(matches!(
        Irrep::from_dynkin_in(&GroupId::so(7).unwrap(), &[0, 0, 1]),
        Err(BcdError::NotAdmissible { .. })
    ));
}

// ---------------------------------------------------------------------------
// GroupId construction invariants (issue #87 §4)
// ---------------------------------------------------------------------------

/// The public fields let a caller build a root datum whose quotient subgroup is
/// not in the root system's center. That is not a hazard: such a group admits
/// nothing, so both form-aware constructors reject every label with the
/// ordinary not-admissible error. This is why `GroupId` stays a plain record
/// (no `try_new`, no private fields).
#[test]
fn nonsensical_root_datum_admits_nothing() {
    let nonsense = [
        (RootSystem::A(3), CenterSubgroup::DVector),
        (RootSystem::A(3), CenterSubgroup::Z2),
        (RootSystem::B(3), CenterSubgroup::Zk(2)),
        (RootSystem::C(3), CenterSubgroup::DFull),
        (RootSystem::D(4), CenterSubgroup::Z4), // Z4 is the r-odd center
        (RootSystem::D(5), CenterSubgroup::DHalfSpinPlus), // r odd has no half-spin form
        (RootSystem::E6, CenterSubgroup::Zk(3)),
        (RootSystem::A(3), CenterSubgroup::Zk(3)), // 3 ∤ 4
    ];
    for (root_system, sub) in nonsense {
        let g = GroupId {
            root_system,
            form: GlobalForm::Quotient(sub),
        };
        for label in sweep(root_system.rank(), 3) {
            assert!(
                !g.admits(&label),
                "{g:?} must admit nothing, but took {label:?}"
            );
        }
        // Both form-aware constructors inherit the rejection.
        #[cfg(feature = "cgc-gen")]
        {
            let zero = vec![0i64; root_system.rank()];
            assert!(racah::sun::Irrep::from_dynkin_in(&g, &zero).is_err());
            assert!(racah::bcd::Irrep::from_dynkin_in(&g, &zero).is_err());
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-global-form coefficient identity (issue #87 §3)
// ---------------------------------------------------------------------------

/// The invariant this whole design rests on: for a `λ` admissible in **both**
/// the cover and a quotient, the published coefficient data are *identical* —
/// same values, same basis ordering, same multiplicity-axis ordering, same
/// cache entry. A global form gates the request; it never forks the engine.
///
/// Equality here is **literal** (`==` on `f64` / on the entry vectors), not a
/// tolerance: the code path is shared, so any difference at all would mean a
/// second engine had appeared.
#[cfg(feature = "cgc-gen")]
mod cross_form_coefficients {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    /// (11) `Spin(N)` and `SO(N)` on their common tensor representations.
    #[test]
    fn spin_and_so_share_tensor_coefficient_data() {
        use racah::bcd::{CanonicalCatalog, Irrep, Series};

        // (matrix dim, series, rank, vector label, a label in vector ⊗ vector)
        let cases = [
            (5usize, Series::B, 2usize, vec![1i64, 0], vec![0i64, 2]), // SO(5) 5, 10
            (6, Series::D, 3, vec![1, 0, 0], vec![0, 1, 1]),           // SO(6) 6, 15
        ];
        for (n, series, r, v_label, c_label) in cases {
            let spin = GroupId::spin(n).unwrap();
            let so = GroupId::so(n).unwrap();
            // Both labels are tensor classes: admissible in cover and quotient.
            for l in [&v_label, &c_label] {
                assert!(spin.admits(l) && so.admits(l), "{n}: {l:?}");
            }

            let cover_v = Irrep::from_dynkin_in(&spin, &v_label).unwrap();
            let quot_v = Irrep::from_dynkin_in(&so, &v_label).unwrap();
            let cover_c = Irrep::from_dynkin_in(&spin, &c_label).unwrap();
            let quot_c = Irrep::from_dynkin_in(&so, &c_label).unwrap();

            // Representation data, hence the cache key, is form-free.
            assert_eq!(cover_v, quot_v);
            assert_eq!(cover_c, quot_c);
            assert_eq!(hash_of(&cover_v), hash_of(&quot_v));
            assert_eq!(
                hash_of(&(cover_v.clone(), cover_v.clone(), cover_c.clone())),
                hash_of(&(quot_v.clone(), quot_v.clone(), quot_c.clone())),
                "the CGC cache key is (Irrep, Irrep, Irrep) — no GlobalForm in it"
            );

            let mut cat = CanonicalCatalog::new(series, r).unwrap();
            let cover = cat.cgc(&cover_v, &cover_v, &cover_c).unwrap();
            let quot = cat.cgc(&quot_v, &quot_v, &quot_c).unwrap();
            assert_eq!(cover.multiplicity(), quot.multiplicity());
            assert_eq!(cover.copy_shape(), quot.copy_shape());
            assert_eq!(cover.data(), quot.data(), "SO({n}) vs Spin({n}) CGC data");
            for mu in 0..cover.multiplicity() {
                assert_eq!(cover.copy(mu), quot.copy(mu), "multiplicity axis {mu}");
            }

            // The F-symbol built on top of them agrees bit for bit too.
            let triv = Irrep::trivial(series, r).unwrap();
            let f_cover = racah::bcd::f_symbol(
                &mut cat, &triv, &cover_v, &cover_v, &cover_c, &cover_v, &cover_c,
            )
            .unwrap();
            let f_quot =
                racah::bcd::f_symbol(&mut cat, &triv, &quot_v, &quot_v, &quot_c, &quot_v, &quot_c)
                    .unwrap();
            assert_eq!(f_cover.dims(), f_quot.dims());
            let [d0, d1, d2, d3] = f_cover.dims();
            for i in 0..d0 {
                for j in 0..d1 {
                    for k in 0..d2 {
                        for l in 0..d3 {
                            assert_eq!(f_cover.at(i, j, k, l), f_quot.at(i, j, k, l));
                        }
                    }
                }
            }
        }
    }

    /// (12) `SU(3)` and `PSU(3)` on their common representations: the adjoint
    /// (a multiplicity-2 channel, so the multiplicity-axis ordering is tested)
    /// and a second nontrivial one, the decuplet.
    #[test]
    fn su_and_psu_share_coefficient_data() {
        use racah::sun::{cgc, f_symbol, Irrep};

        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        let adj = vec![1i64, 1];
        let ten = vec![3i64, 0];
        let tenbar = vec![0i64, 3];

        // 8 ⊗ 8 → 8 (multiplicity 2), and 10 ⊗ 10bar → 8.
        let channels = [
            (adj.clone(), adj.clone(), adj.clone()),
            (ten.clone(), tenbar.clone(), adj.clone()),
        ];
        for (a, b, c) in channels {
            for l in [&a, &b, &c] {
                assert!(psu3.admits(l), "{l:?} must be a PSU(3) rep");
            }
            let cov = |l: &[i64]| Irrep::from_dynkin_in(&su3, l).unwrap();
            let quo = |l: &[i64]| Irrep::from_dynkin_in(&psu3, l).unwrap();
            assert_eq!(cov(&a), quo(&a));
            assert_eq!(
                hash_of(&(cov(&a), cov(&b), cov(&c))),
                hash_of(&(quo(&a), quo(&b), quo(&c))),
                "the SU(N) CGC cache key carries no GlobalForm"
            );

            let cover = cgc(&cov(&a), &cov(&b), &cov(&c)).unwrap();
            let quot = cgc(&quo(&a), &quo(&b), &quo(&c)).unwrap();
            assert_eq!(cover.multiplicity(), quot.multiplicity());
            assert_eq!(cover.dims(), quot.dims());
            assert_eq!(cover.nnz(), quot.nnz());
            assert_eq!(
                cover.entries(),
                quot.entries(),
                "SU(3) vs PSU(3) CGC entries for {a:?} ⊗ {b:?} → {c:?}"
            );
        }
        assert!(
            cgc(
                &Irrep::from_dynkin(&adj).unwrap(),
                &Irrep::from_dynkin(&adj).unwrap(),
                &Irrep::from_dynkin(&adj).unwrap()
            )
            .unwrap()
            .multiplicity()
                == 2,
            "8 ⊗ 8 → 8 is the multiplicity-2 channel this test wants"
        );

        // An F-symbol over the common reps agrees bit for bit.
        let triv = Irrep::trivial(3).unwrap();
        let f = |g: &GroupId| {
            let i = |l: &[i64]| Irrep::from_dynkin_in(g, l).unwrap();
            f_symbol(&triv, &i(&adj), &i(&adj), &i(&adj), &i(&adj), &i(&adj)).unwrap()
        };
        let (fc, fq) = (f(&su3), f(&psu3));
        assert_eq!(fc.dims(), fq.dims());
        let [d0, d1, d2, d3] = fc.dims();
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    for l in 0..d3 {
                        assert_eq!(fc.at(i, j, k, l), fq.at(i, j, k, l));
                    }
                }
            }
        }
    }

    /// (13)/(14) The cache is shared where it should be and cannot be a bypass
    /// where it should not be.
    ///
    /// Warming the cover's entry for a weight the quotient does not admit gives
    /// the quotient no way to reach it: there is no `Irrep` for an inadmissible
    /// weight, so there is no key with which to read the entry. Validation is at
    /// construction, strictly before any cache is touched.
    #[test]
    fn an_inadmissible_rep_cannot_reach_the_cache_through_the_quotient_api() {
        use racah::sun::{cgc, Irrep, SunError};

        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        let three = Irrep::from_dynkin_in(&su3, &[1, 0]).unwrap();
        let threebar = Irrep::from_dynkin_in(&su3, &[0, 1]).unwrap();
        let adj = Irrep::from_dynkin_in(&su3, &[1, 1]).unwrap();

        // Populate the cover's entry for 3 ⊗ 3bar → 8.
        let warm = cgc(&three, &threebar, &adj).unwrap();
        assert!(warm.nnz() > 0);

        // The quotient still cannot name the fundamental — before, and after.
        for label in [[1i64, 0], [0, 1], [2, 0], [0, 2]] {
            assert!(matches!(
                Irrep::from_dynkin_in(&psu3, &label),
                Err(SunError::NotAdmissible { .. }),
            ));
        }
        // And a warm entry for a *shared* weight is genuinely shared: the
        // quotient's request returns the identical data.
        let again = cgc(
            &Irrep::from_dynkin_in(&psu3, &[1, 1]).unwrap(),
            &Irrep::from_dynkin_in(&psu3, &[1, 1]).unwrap(),
            &Irrep::from_dynkin_in(&psu3, &[1, 1]).unwrap(),
        )
        .unwrap();
        let cover_again = cgc(&adj, &adj, &adj).unwrap();
        assert_eq!(again.entries(), cover_again.entries());
    }

    /// The same bypass argument on the B/D side: `SO(N)` cannot reach a spinor
    /// entry warmed by `Spin(N)`.
    #[test]
    fn so_cannot_reach_a_spin_only_entry() {
        use racah::bcd::{BcdError, CanonicalCatalog, Irrep, Series};

        let spin5 = GroupId::spin(5).unwrap();
        let so5 = GroupId::so(5).unwrap();
        let spinor = Irrep::from_dynkin_in(&spin5, &[0, 1]).unwrap();
        let vector = Irrep::from_dynkin_in(&spin5, &[1, 0]).unwrap();
        let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
        // Warm a channel involving the spinor.
        cat.cgc(&spinor, &spinor, &vector).unwrap();
        // SO(5) still has no spinor label to key it with.
        assert!(matches!(
            Irrep::from_dynkin_in(&so5, &[0, 1]),
            Err(BcdError::NotAdmissible { .. })
        ));
        assert!(matches!(
            Irrep::from_dynkin(Series::B, &[0, 1]),
            Err(BcdError::NotAdmissible { .. })
        ));
    }
}
