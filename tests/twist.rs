//! Twist oracle (issue #118): every family's `twist()` must equal the
//! TensorKitSectors generic definition
//! `src/sectors.jl:twist_from_Rsymbol`,
//! θ_a = Σ_{c ∈ a⊗a} (d_c / d_a) tr R^{aa}_c,
//! evaluated here from racah's own R-symbols and fusion rules. The R-symbol
//! sum is independent of the `twist()` implementations, which return a
//! constant without generating any R-symbol.

use racah::{su2_r_symbol, su2_twist, Su2Irrep};

#[cfg(feature = "cgc-gen")]
const TOL: f64 = 1e-10;

#[test]
fn su2_twist_matches_r_symbol_formula_up_to_j5() {
    for dj in 0..=10u32 {
        let a = Su2Irrep::new(dj);
        let sum: f64 = a
            .fusion(a)
            .unwrap()
            .map(|c| c.dim() as f64 * su2_r_symbol(dj, dj, c.dj()))
            .sum();
        let theta = sum / a.dim() as f64;
        // Integer-valued sum: exact, no tolerance needed.
        assert_eq!(theta, 1.0, "R-formula twist for dj = {dj}");
        assert_eq!(su2_twist(dj), theta, "su2_twist({dj})");
    }
}

#[cfg(feature = "cgc-gen")]
fn to_f64(d: num_bigint::BigInt) -> f64 {
    use num_traits::ToPrimitive;
    d.to_f64().unwrap()
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_twist_matches_r_symbol_formula() {
    use racah::sun::{directproduct, r_symbol, Irrep};

    // 3, 3̄, 6, 8 (outer multiplicity 2 in 8⊗8→8), 15 = (2,1); SU(4) 4 and 6.
    for d in [
        &[1, 0][..],
        &[0, 1],
        &[2, 0],
        &[1, 1],
        &[2, 1],
        &[1, 0, 0],
        &[0, 1, 0],
    ] {
        let a = Irrep::from_dynkin(d).unwrap();
        let mut sum = 0.0;
        for (c, n) in directproduct(&a, &a).unwrap() {
            let r = r_symbol(&a, &a, &c).unwrap();
            assert_eq!(r.dim(), n as usize);
            let tr: f64 = (0..r.dim()).map(|mu| r.at(mu, mu)).sum();
            sum += to_f64(c.dim()) * tr;
        }
        let theta = sum / to_f64(a.dim());
        assert!((theta - 1.0).abs() < TOL, "{d:?}: θ = {theta}");
        assert_eq!(a.twist(), 1.0, "{d:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn bcd_twist_matches_r_symbol_formula() {
    use racah::bcd::{directproduct, r_symbol, CanonicalCatalog, Irrep, Series};
    use racah::group::GroupId;

    // Spin(5): vector 5 and spinor 4 (quaternionic); Sp(4): defining 4
    // (quaternionic) and 5; Spin(6): vector 6 and the non-self-dual spinor 4.
    for (series, rank, n, d, dim, spinor) in [
        (Series::B, 2, 5, &[1, 0][..], 5, false),
        (Series::B, 2, 5, &[0, 1], 4, true),
        (Series::C, 2, 0, &[1, 0], 4, false),
        (Series::C, 2, 0, &[0, 1], 5, false),
        (Series::D, 3, 6, &[1, 0, 0], 6, false),
        (Series::D, 3, 6, &[0, 1, 0], 4, true),
    ] {
        let mut cat = CanonicalCatalog::new(series, rank).unwrap();
        let a = match series {
            Series::C => Irrep::from_dynkin(series, d).unwrap(),
            _ => Irrep::from_dynkin_in(&GroupId::spin(n).unwrap(), d).unwrap(),
        };
        assert_eq!((to_f64(a.dim()), a.is_spinor()), (dim as f64, spinor));
        let mut sum = 0.0;
        for (c, n) in directproduct(&a, &a).unwrap() {
            let r = r_symbol(&mut cat, &a, &a, &c).unwrap();
            assert_eq!(r.dim(), n as usize);
            let tr: f64 = (0..r.dim()).map(|mu| r.at(mu, mu)).sum();
            sum += to_f64(c.dim()) * tr;
        }
        let theta = sum / to_f64(a.dim());
        assert!(
            (theta - 1.0).abs() < TOL,
            "{series:?}{rank} {d:?}: θ = {theta}"
        );
        assert_eq!(a.twist(), 1.0, "{series:?}{rank} {d:?}");
    }
}
