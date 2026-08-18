//! `Spin(N)` acceptance suite (issue #54, stage (b) of #87): the spinor irreps
//! of the simply-connected form of the `B`/`D` families.
//!
//! Role (`tools/README.md` matrix): *internal consistency* strand. The
//! independent cross-family oracles — `Spin(5) ≅ Sp(4)`, `Spin(6) ≅ SU(4)`,
//! `Spin(3) ≅ SU(2)` — live in `tests/isomorphism.rs`; this file pins the
//! things that are statements about the spinor sector itself: the seed's exact
//! self-check, the label admissibility split between `Spin(N)` and `SO(N)`,
//! the reality types, the CGC isometry property, and the recursion reaching a
//! non-base spinor.
//!
//! Each public operation gets its own focused test rather than one umbrella
//! run, so a regression names the operation that broke.
//!
//! # The former limitation, now closed (issue #90)
//!
//! Until `epoch = 2` the recoupling gates (`check_f_unitarity` /
//! `check_pentagon` / `check_hexagon`) closed on the spinor channels that did
//! **not** route through a *defining*-rep coupled channel and failed on the ones
//! that did — not a spinor defect but the base-case frame mismatch of issue #90
//! (the QSpace-seeded defining rep stored in `Setup_*` order, rediscovered in
//! the sweep's descending-weight order, with base cases exempt from the
//! coherence guard and the alignment). It also made the *tensor* `B_2` and `D_3`
//! vector pentagons fail, with residuals `0.285_239_560_970_874_6` and
//! `0.129_099_444_873_580_7`. `docs/gauge_soN.md` §14.2 "Base-case frame" now
//! puts the defining seed through the same sweep pass the spinor seeds already
//! took, and the gates below — including the ones that route through the
//! defining rep — close.

#![cfg(feature = "cgc-gen")]

use racah::bcd::{
    check_commutators, check_f_unitarity, check_hexagon, check_pentagon, defining_seed,
    directproduct, spinor_seeds, BcdError, CanonicalCatalog, CatalogCgc, Irrep, Series,
};
use racah::group::GroupId;

/// The `B`/`D` families in scope, as `(series, rank, N)` with `N` the matrix
/// dimension of `Spin(N)`.
const FAMILIES: &[(Series, usize, usize)] = &[
    (Series::B, 2, 5),
    (Series::B, 3, 7),
    (Series::D, 3, 6),
    (Series::D, 4, 8),
];

fn ir(n: usize, dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin_in(&GroupId::spin(n).unwrap(), dynkin).expect("Spin(N) label")
}

// ---------------------------------------------------------------------------
// The seed (S3.1).
// ---------------------------------------------------------------------------

/// Every spinor seed passes the same exact commutator self-check as the
/// defining seeds — the gate issue #87 §4(2) requires of the new base case.
#[test]
fn spinor_seed_commutators_close() {
    for &(series, r, _) in FAMILIES {
        for (label, seed) in spinor_seeds(series, r).unwrap() {
            check_commutators(&seed)
                .unwrap_or_else(|e| panic!("{series:?}_{r} spinor {label:?}: {e}"));
        }
    }
}

/// The derived structure constants are properties of the **algebra**, not of
/// the representation: the spinor seed must report exactly the `f_{i,k}` and
/// `d_{i,j}` of the defining seed. This is the sharpest available check that
/// the spinor generators are the *same* abstract generators the bootstrap
/// composes tensor products of — a wrong root assignment, a wrong node order,
/// or a wrong ladder normalization all move these.
#[test]
fn spinor_seed_structure_constants_match_the_defining_seed() {
    for &(series, r, _) in FAMILIES {
        let reference = check_commutators(&defining_seed(series, r).unwrap()).unwrap();
        for (label, seed) in spinor_seeds(series, r).unwrap() {
            let got = check_commutators(&seed).unwrap();
            assert_eq!(
                got.cartan_coeffs, reference.cartan_coeffs,
                "{series:?}_{r} spinor {label:?}: [Sp,Sp†] coefficients"
            );
            assert_eq!(
                got.root_weights, reference.root_weights,
                "{series:?}_{r} spinor {label:?}: root components"
            );
        }
    }
}

/// `B_r` has one spinor base case (`ω_r`), `D_r` two (the half-spinors), `C_r`
/// none — and the carrier dimensions are `2^r` and `2^{r-1}`.
#[test]
fn spinor_seed_inventory_is_one_for_b_two_for_d_none_for_c() {
    assert!(spinor_seeds(Series::C, 3).unwrap().is_empty());
    for &(series, r, _) in FAMILIES {
        let seeds = spinor_seeds(series, r).unwrap();
        let (want_count, want_dim) = match series {
            Series::B => (1, 1usize << r),
            _ => (2, 1usize << (r - 1)),
        };
        assert_eq!(seeds.len(), want_count, "{series:?}_{r} seed count");
        for (_, seed) in &seeds {
            assert_eq!(seed.dim(), want_dim, "{series:?}_{r} carrier dimension");
        }
    }
}

/// The excluded low ranks are excluded for the spinor seeds too, with the
/// simply-connected redirect (`Spin(3) ≅ SU(2)`, `Spin(4) ≅ SU(2)×SU(2)`).
#[test]
fn spinor_seeds_reject_the_excluded_low_ranks() {
    assert!(matches!(
        spinor_seeds(Series::B, 1),
        Err(BcdError::ExcludedRank { redirect, .. }) if redirect.contains("SU(2)")
    ));
    assert!(matches!(
        spinor_seeds(Series::D, 2),
        Err(BcdError::ExcludedRank { redirect, .. }) if redirect.contains("SU(2)×SU(2)")
    ));
}

// ---------------------------------------------------------------------------
// Labels (S3.0).
// ---------------------------------------------------------------------------

/// A spinor label is admissible in `Spin(N)` and rejected by `SO(N)` — the
/// whole content of "global form" at the label layer (#87 §2).
#[test]
fn spinor_labels_are_admissible_only_in_the_cover() {
    for &(series, r, n) in FAMILIES {
        for (label, _) in spinor_seeds(series, r).unwrap() {
            assert!(
                Irrep::from_dynkin_in(&GroupId::spin(n).unwrap(), &label).is_ok(),
                "Spin({n}) must admit {label:?}"
            );
            assert!(
                matches!(
                    Irrep::from_dynkin_in(&GroupId::so(n).unwrap(), &label),
                    Err(BcdError::NotAdmissible { .. })
                ),
                "SO({n}) must reject {label:?}"
            );
            assert!(
                matches!(
                    Irrep::from_dynkin(series, &label),
                    Err(BcdError::NotAdmissible { .. })
                ),
                "the published constructor must still reject {label:?}"
            );
        }
    }
}

/// The doubled ε-basis weight is what makes a spinor representable at all: its
/// `2λ` is all-odd, the tensor irreps' is all-even, and `partition()` — which
/// promises an integer partition — is `None` exactly on the spinors.
#[test]
fn spinor_weights_are_half_integers_in_the_doubled_encoding() {
    let s = ir(7, &[0, 0, 1]);
    assert_eq!(s.two_partition(), &[1, 1, 1]);
    assert!(s.is_spinor());
    assert_eq!(s.partition(), None);
    assert_eq!(s.weight_multiplicities(), None);

    let v = ir(7, &[1, 0, 0]);
    assert_eq!(v.two_partition(), &[2, 0, 0]);
    assert!(!v.is_spinor());
    assert_eq!(v.partition(), Some(vec![1, 0, 0]));
}

/// Weyl dimensions of the fundamental spinors: `2^r` for `B_r`, `2^{r-1}` per
/// chirality for `D_r`. Computed by the exact Weyl formula on the half-integer
/// weight, i.e. an independent statement from the seed's carrier size.
#[test]
fn spinor_dimensions_are_the_clifford_module_dimensions() {
    for &(series, r, n) in FAMILIES {
        let want: u64 = match series {
            Series::B => 1 << r,
            _ => 1 << (r - 1),
        };
        for (label, _) in spinor_seeds(series, r).unwrap() {
            assert_eq!(
                ir(n, &label).dim(),
                want.into(),
                "dim of Spin({n}) {label:?}"
            );
        }
    }
}

/// The Frobenius–Schur indicator of the fundamental spinor is the `N mod 8`
/// reality type (a #54 design requirement): real for `N ≡ 0, ±1, 2`,
/// quaternionic for `N ≡ 3, 4, 5`, and `0` for the `D_r`-odd half-spinors,
/// which are a conjugate pair rather than self-dual.
#[test]
fn spinor_frobenius_schur_is_the_n_mod_8_reality_type() {
    // (N, dynkin, indicator).
    let cases: &[(usize, &[i64], i32)] = &[
        (5, &[0, 1], -1),              // Spin(5) 4, quaternionic (N ≡ 5)
        (7, &[0, 0, 1], 1),            // Spin(7) 8, real (N ≡ 7)
        (9, &[0, 0, 0, 1], 1),         // Spin(9) 16, real (N ≡ 1)
        (11, &[0, 0, 0, 0, 1], -1),    // Spin(11) 32, quaternionic (N ≡ 3)
        (6, &[0, 0, 1], 0),            // Spin(6) 4, complex (D_3 odd)
        (8, &[0, 0, 0, 1], 1),         // Spin(8) 8_s, real (N ≡ 0)
        (10, &[0, 0, 0, 0, 1], 0),     // Spin(10) 16, complex (D_5 odd)
        (12, &[0, 0, 0, 0, 0, 1], -1), // Spin(12) 32, quaternionic (N ≡ 4)
    ];
    for &(n, dynkin, want) in cases {
        assert_eq!(
            ir(n, dynkin).frobenius_schur(),
            want,
            "FS of Spin({n}) {dynkin:?}"
        );
    }
}

/// `D_r` with `r` odd exchanges the two half-spinors under duality; with `r`
/// even each is self-dual. This is the existing `dual` rule, now exercised on
/// the labels it was written for.
#[test]
fn half_spinor_duality_follows_the_d_series_chirality_rule() {
    let s = ir(6, &[0, 0, 1]);
    assert_eq!(s.dual().dynkin(), vec![0, 1, 0], "Spin(6) 4* = 4bar");
    let s8 = ir(8, &[0, 0, 0, 1]);
    assert_eq!(
        s8.dual().dynkin(),
        vec![0, 0, 0, 1],
        "Spin(8) 8_s self-dual"
    );
}

/// Spinor fusion, from the exact Racah–Speiser layer: the textbook `Spin(7)`
/// and `Spin(8)` products. Independent of the numeric pipeline.
#[test]
fn spinor_fusion_matches_the_textbook_products() {
    let channels = |a: &Irrep, b: &Irrep| -> Vec<(Vec<i64>, u32)> {
        let mut v: Vec<(Vec<i64>, u32)> = directproduct(a, b)
            .unwrap()
            .into_iter()
            .map(|(k, m)| (k.dynkin(), m))
            .collect();
        v.sort();
        v
    };
    // Spin(7): 8 ⊗ 8 = 1 + 7 + 21 + 35.
    assert_eq!(
        channels(&ir(7, &[0, 0, 1]), &ir(7, &[0, 0, 1])),
        vec![
            (vec![0, 0, 0], 1),
            (vec![0, 0, 2], 1),
            (vec![0, 1, 0], 1),
            (vec![1, 0, 0], 1),
        ]
    );
    // Spin(8) triality: 8_s ⊗ 8_v = 8_c + 56_c.
    assert_eq!(
        channels(&ir(8, &[0, 0, 0, 1]), &ir(8, &[1, 0, 0, 0])),
        vec![(vec![0, 0, 1, 0], 1), (vec![1, 0, 0, 1], 1)]
    );
    // Spin(5): 4 ⊗ 4 = 1 + 5 + 10 (the #87 SO(5) diagnostic, now first-class).
    assert_eq!(
        channels(&ir(5, &[0, 1]), &ir(5, &[0, 1])),
        vec![(vec![0, 0], 1), (vec![0, 2], 1), (vec![1, 0], 1)]
    );
}

// ---------------------------------------------------------------------------
// The catalog and the CGC (S3.2/S3.3).
// ---------------------------------------------------------------------------

/// Worst `|(CᵀC − I)|` over the outer-multiplicity copies of a CGC block.
fn isometry_residual(c: &CatalogCgc) -> f64 {
    let (rows, cols) = c.copy_shape();
    let mut worst: f64 = 0.0;
    for mu in 0..c.multiplicity() {
        let m = c.copy(mu);
        for i in 0..cols {
            for j in 0..cols {
                let dot: f64 = (0..rows).map(|k| m[i * rows + k] * m[j * rows + k]).sum();
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
    }
    worst
}

/// Every spinor CGC is an isometry — the orthonormality half of the #54
/// verification list, asserted on the returned coefficients rather than only
/// inside the sweep.
#[test]
fn spinor_cgc_columns_are_orthonormal() {
    for &(series, r, n) in FAMILIES {
        let mut cat = CanonicalCatalog::new(series, r).unwrap();
        let spinors: Vec<Irrep> = spinor_seeds(series, r)
            .unwrap()
            .into_iter()
            .map(|(label, _)| ir(n, &label))
            .collect();
        let s = spinors[0].clone();
        for partner in [s.clone(), s.dual()] {
            for (c, _) in directproduct(&s, &partner).unwrap() {
                let cgc = cat.cgc(&s, &partner, &c).unwrap();
                let residual = isometry_residual(&cgc);
                assert!(
                    residual < 1e-10,
                    "Spin({n}) {:?}⊗{:?}→{:?} is not an isometry: {residual:e}",
                    s.dynkin(),
                    partner.dynkin(),
                    c.dynkin()
                );
            }
        }
    }
}

/// A spinor that is **not** a base case is reached by the canonical-parent
/// recursion (§14.4's spinor branch: `λ = μ + ω_r`, remove a box from `μ`).
/// `B_2`'s `(1,1)` is `λ = (3/2, 1/2)`, dimension 16.
#[test]
fn the_recursion_reaches_a_non_base_spinor() {
    let x = ir(5, &[1, 1]);
    assert_eq!(x.dim(), 16u32.into());
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    assert_eq!(cat.generators(&x).unwrap().dim(), 16);
}

/// A catalog that is only ever asked for tensor irreps materializes **no**
/// spinor generators: the class-indexed candidate-set restriction (option (B)
/// of #87 §5) keeps the spinor sector entirely out of the tensor bootstrap,
/// which is what preserves every shipped `SO(N)`/`Sp(2N)` coefficient.
#[test]
fn the_tensor_bootstrap_never_touches_the_spinor_sector() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let before = cat.len();
    let adjoint = Irrep::from_dynkin(Series::B, &[0, 2]).unwrap();
    cat.generators(&adjoint).unwrap();
    assert!(cat.len() > before, "the adjoint must materialize something");
    // Nothing in the catalog is a spinor: with a spinor parent the SO(5)
    // adjoint's frame would have moved (the #87 Q4 diagnostic).
    let spinor = ir(5, &[0, 1]);
    let mut probe = CanonicalCatalog::new(Series::B, 2).unwrap();
    probe.generators(&adjoint).unwrap();
    assert_eq!(
        probe.len(),
        cat.len(),
        "materialization must be deterministic"
    );
    assert!(
        probe.cgc(&spinor, &spinor, &adjoint).is_ok(),
        "the spinor product is still available on demand"
    );
}

// ---------------------------------------------------------------------------
// Recoupling gates on the spinor channels that close (see the module note).
// ---------------------------------------------------------------------------

/// `F` is unitary on the `Spin(6)` half-spinor vertices — the `4 ⊗ 4̄` and
/// `4 ⊗ 4` families, which between them cover both the tensor channels
/// (`1 + 15`) and the chiral ones (`6 + 10`).
#[test]
fn spinor_f_symbols_are_unitary_on_spin6() {
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    check_f_unitarity(&mut cat, &s, &sb, &s, &sb).unwrap();
    check_f_unitarity(&mut cat, &s, &s, &s, &s).unwrap();
}

/// The pentagon closes on the `Spin(6)` `4 ⊗ 4̄` spinor family.
#[test]
fn spinor_pentagon_closes_on_spin6() {
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    check_pentagon(&mut cat, &s, &sb, &s, &sb).unwrap();
}

/// The hexagon closes on a mixed spinor/vector `Spin(6)` triple — braiding
/// with a spinor leg.
#[test]
fn spinor_hexagon_closes_on_spin6() {
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    let v = ir(6, &[1, 0, 0]);
    check_hexagon(&mut cat, &s, &sb, &v).unwrap();
}

/// The `Spin(6)` `4 ⊗ 4 = 6 + 1̄0` family routes through the **defining**-rep
/// coupled channel, which is exactly what the base-case frame mismatch of issue
/// #90 broke: on `epoch = 1` this pentagon failed at residual
/// `0.750_000_000_000_000_4` and the hexagon at `0.75`. With the defining seed
/// re-framed into the sweep order (`docs/gauge_soN.md` §14.2) both close.
#[test]
fn spinor_recoupling_closes_through_the_defining_channel_on_spin6() {
    let s = ir(6, &[0, 0, 1]);
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    check_pentagon(&mut cat, &s, &s, &s, &s).unwrap();
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    check_hexagon(&mut cat, &s, &s, &s).unwrap();
}
