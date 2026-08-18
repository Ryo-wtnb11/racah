//! Compile-and-run guard for the code blocks in the prose documentation.
//!
//! Every Rust code block in `README.md` and `docs/user-guide/*.md` appears here
//! verbatim, one test per block, so a change to the public API breaks the build
//! instead of silently rotting the guide. Rustdoc examples are covered by
//! `cargo test --doc`; this file covers the prose that lives outside rustdoc.
//!
//! Regenerate by copying the block; there is no codegen step to run.
#![cfg(feature = "cgc-gen")]
#![allow(unused)]

/// `README.md`, code block 1.
#[test]
fn readme_1() {
    use racah::wigner_6j;

    let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
}

/// `README.md`, code block 2.
#[test]
fn readme_2() {
    use racah::sun::{directproduct, Irrep};

    let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
    assert_eq!(eight.dim(), 8u32.into());
    assert_eq!(directproduct(&eight, &eight).unwrap()[&eight], 2);
}

/// `README.md`, code block 3.
#[test]
fn readme_3() {
    use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4)
    let triv = Irrep::trivial(Series::C, 2).unwrap();
    let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // the 5
    let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap(); // the adjoint 10

    let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
}

/// `README.md`, code block 4.
#[test]
fn readme_4() {
    use racah::bcd::Irrep;
    use racah::group::GroupId;

    let spin7 = GroupId::spin(7).unwrap();
    let s = Irrep::from_dynkin_in(&spin7, &[0, 0, 1]).unwrap();
    assert_eq!(s.dim(), 8u32.into());

    // The same label is not a representation of SO(7).
    assert!(Irrep::from_dynkin_in(&GroupId::so(7).unwrap(), &[0, 0, 1]).is_err());
}

/// `docs/user-guide/clebsch-gordan.md`, code block 1.
#[test]
fn clebsch_gordan_1() {
    use racah::su2::clebsch_gordan;

    // ⟨½ +½, ½ −½ | 0 0⟩ = 1/√2, in doubled labels.
    let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
    assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
}

/// `docs/user-guide/clebsch-gordan.md`, code block 2.
#[test]
fn clebsch_gordan_2() {
    use racah::sun::{cgc, Irrep};

    let three = Irrep::from_dynkin(&[1, 0]).unwrap();
    let anti = three.dual();
    let eight = Irrep::from_dynkin(&[1, 1]).unwrap();

    let c = cgc(&three, &anti, &eight).unwrap();
    assert_eq!(c.dims(), [3, 3, 8, 1]); // [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]
    assert_eq!(c.multiplicity(), 1);
    // Only nonzero entries are stored, sorted by (m1, m2, m3, mu).
    assert!(c.entries().iter().all(|e| e.value != 0.0));
}

/// `docs/user-guide/clebsch-gordan.md`, code block 3.
#[test]
fn clebsch_gordan_3() {
    use racah::sun::{cgc, Irrep};

    let eight = Irrep::from_dynkin(&[1, 1]).unwrap();
    let c = cgc(&eight, &eight, &eight).unwrap();
    assert_eq!(c.multiplicity(), 2); // the 8 occurs twice in 8 ⊗ 8
    assert_eq!(c.dims(), [8, 8, 8, 2]);
    assert!(c.entries().iter().any(|e| e.mu == 1));
}

/// `docs/user-guide/clebsch-gordan.md`, code block 4.
#[test]
fn clebsch_gordan_4() {
    use racah::bcd::{CanonicalCatalog, Irrep, Series};

    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap(); // SO(5) / Spin(5)
    let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap(); // the 5
    let adj = Irrep::from_dynkin(Series::B, &[0, 2]).unwrap(); // the 10

    let c = cat.cgc(&v, &v, &adj).unwrap();
    assert_eq!(c.multiplicity(), 1);
    assert_eq!(c.copy_shape(), (25, 10)); // (dim(s1)·dim(s2), dim(s3))
}

/// `docs/user-guide/fusion.md`, code block 1.
#[test]
fn fusion_1() {
    use racah::su2::Su2Irrep;

    let half = Su2Irrep::new(1);
    let channels: Vec<u32> = half.fusion(half).unwrap().map(|s| s.dj()).collect();
    assert_eq!(channels, vec![0, 2]); // ½ ⊗ ½ = 0 ⊕ 1
}

/// `docs/user-guide/fusion.md`, code block 2.
#[test]
fn fusion_2() {
    use racah::sun::{directproduct, Irrep};

    let three = Irrep::from_dynkin(&[1, 0]).unwrap();
    let anti = three.dual();
    let out = directproduct(&three, &anti).unwrap();

    // 3 ⊗ 3-bar = 1 ⊕ 8, each once.
    assert_eq!(out.len(), 2);
    assert_eq!(out[&Irrep::trivial(3).unwrap()], 1);
    assert_eq!(out[&Irrep::from_dynkin(&[1, 1]).unwrap()], 1);
}

/// `docs/user-guide/fusion.md`, code block 3.
#[test]
fn fusion_3() {
    use racah::sun::{directproduct, Irrep};

    let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
    let out = directproduct(&eight, &eight).unwrap();
    assert_eq!(out[&eight], 2); // the 8 appears twice in 8 ⊗ 8
}

/// `docs/user-guide/fusion.md`, code block 4.
#[test]
fn fusion_4() {
    use racah::bcd::{directproduct, Irrep, Series};

    // SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
    let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
    let out = directproduct(&v, &v).unwrap();
    assert_eq!(out.len(), 3);
    for (irrep, mult) in &out {
        assert_eq!(*mult, 1);
        let _ = irrep.dim();
    }
}

/// `docs/user-guide/getting-started.md`, code block 1.
#[test]
fn getting_started_1() {
    use racah::wigner_6j;

    let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
}

/// `docs/user-guide/getting-started.md`, code block 2.
#[test]
fn getting_started_2() {
    use racah::sun::Irrep;

    let fund = Irrep::from_dynkin(&[1, 0]).unwrap(); // the 3 of SU(3)
    assert_eq!(fund.dim(), 3u32.into());
    assert_eq!(fund.dual().dynkin(), vec![0, 1]); // the 3-bar
}

/// `docs/user-guide/getting-started.md`, code block 3.
#[test]
fn getting_started_3() {
    use racah::su2::{wigner_6j_checked, Su2Error};

    // Admissible, and genuinely nonzero.
    assert!(wigner_6j_checked(2, 2, 2, 2, 2, 2).is_ok());

    // Triangle violation: not a real zero, a forbidden label set.
    assert!(matches!(
        wigner_6j_checked(2, 2, 20, 2, 2, 2),
        Err(Su2Error::NotAdmissible(_))
    ));
}

/// `docs/user-guide/groups.md`, code block 1.
#[test]
fn groups_1() {
    use racah::su2::Su2Irrep;

    let one = Su2Irrep::new(2); // spin 1
    let channels: Vec<u32> = one.fusion(one).unwrap().map(|s| s.dj()).collect();
    assert_eq!(channels, vec![0, 2, 4]); // 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
}

/// `docs/user-guide/groups.md`, code block 2.
#[test]
fn groups_2() {
    use racah::sun::{directproduct, Irrep};

    let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
    let decomposition = directproduct(&eight, &eight).unwrap();
    // 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10-bar ⊕ 27 — note the 8 appears twice.
    assert_eq!(decomposition[&eight], 2);
}

/// `docs/user-guide/groups.md`, code block 3.
#[test]
fn groups_3() {
    use racah::bcd::{directproduct, Irrep, Series};

    // SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
    let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
    let out = directproduct(&v, &v).unwrap();
    let mut dims: Vec<String> = out.keys().map(|s| s.dim().to_string()).collect();
    dims.sort();
    assert_eq!(dims, vec!["1", "10", "14"]);
}

/// `docs/user-guide/groups.md`, code block 4.
#[test]
fn groups_4() {
    use racah::bcd::Irrep;
    use racah::group::GroupId;

    let spin5 = GroupId::spin(5).unwrap();
    let s = Irrep::from_dynkin_in(&spin5, &[0, 1]).unwrap();
    assert_eq!(s.dim(), 4u32.into()); // the Spin(5) Dirac spinor
    assert!(s.is_spinor());
    assert_eq!(s.two_partition(), &[1, 1]); // λ = (½, ½)

    // The same label is not a representation of SO(5).
    let so5 = GroupId::so(5).unwrap();
    assert!(Irrep::from_dynkin_in(&so5, &[0, 1]).is_err());
}

/// `docs/user-guide/groups.md`, code block 5.
#[test]
fn groups_5() {
    use racah::bcd::{directproduct, Irrep, Series};

    // Sp(4) = C_2. The defining 4, and 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
    let four = Irrep::from_dynkin(Series::C, &[1, 0]).unwrap();
    assert_eq!(four.dim(), 4u32.into());
    let out = directproduct(&four, &four).unwrap();
    let mut dims: Vec<String> = out.keys().map(|s| s.dim().to_string()).collect();
    dims.sort();
    assert_eq!(dims, vec!["1", "10", "5"]); // string sort
}

/// `docs/user-guide/groups.md`, code block 6.
#[test]
fn groups_6() {
    use racah::group::GroupId;
    use racah::sun::Irrep;

    let psu3 = GroupId::psu(3).unwrap(); // SU(3)/Z₃

    // The adjoint 8 has zero triality, so it is a genuine PSU(3) representation.
    assert!(Irrep::from_dynkin_in(&psu3, &[1, 1]).is_ok());

    // The fundamental 3 has triality 1: PSU(3) has no such representation.
    assert!(Irrep::from_dynkin_in(&psu3, &[1, 0]).is_err());
}

/// `docs/user-guide/recoupling.md`, code block 1.
#[test]
fn recoupling_1() {
    use racah::{su2_f_symbol, su2_r_symbol, wigner_6j};

    // {1 1 1; 1 1 1} = 1/6 in doubled labels (all dj = 2, i.e. spin 1).
    assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);

    // F and R are multiplicity-free here: plain f64 scalars.
    let f = su2_f_symbol(1, 1, 1, 1, 0, 0); // F^{½½½}_{½}[0, 0]
    assert!((f + 0.5).abs() < 1e-14);

    let r = su2_r_symbol(1, 1, 0); // (-1)^(j1+j2-j3) = -1 for ½ ⊗ ½ → 0
    assert_eq!(r, -1.0);
}

/// `docs/user-guide/recoupling.md`, code block 2.
#[test]
fn recoupling_2() {
    use racah::sun::{f_symbol, r_symbol, Irrep};

    let three = Irrep::from_dynkin(&[1, 0]).unwrap();
    let anti = three.dual();
    let eight = Irrep::from_dynkin(&[1, 1]).unwrap();

    // F^{3 3bar 3}_{3}[8, 8]
    let block = f_symbol(&three, &anti, &three, &three, &eight, &eight).unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-12);

    // R^{3 3bar}_{8}
    let r = r_symbol(&three, &anti, &eight).unwrap();
    assert_eq!(r.dim(), 1);
    assert!((r.at(0, 0).abs() - 1.0).abs() < 1e-12);
}

/// `docs/user-guide/recoupling.md`, code block 3.
#[test]
fn recoupling_3() {
    use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4) = C_2
    let triv = Irrep::trivial(Series::C, 2).unwrap();
    let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // the 5
    let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap(); // the adjoint 10

    let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
}

/// `docs/user-guide/recoupling.md`, code block 4.
#[test]
fn recoupling_4() {
    use racah::sun::{check_f_unitarity, check_hexagon, check_pentagon, Irrep};

    let three = Irrep::from_dynkin(&[1, 0]).unwrap();
    let anti = three.dual();
    check_f_unitarity(&three, &anti, &three, &three).unwrap();
    check_pentagon(&three, &anti, &three, &anti).unwrap();
    check_hexagon(&three, &anti, &three).unwrap();
}

/// `docs/user-guide/representations.md`, code block 1.
#[test]
fn representations_1() {
    use racah::su2::{su2_frobenius_schur, Su2Irrep};

    let s = Su2Irrep::new(3); // dj = 3, i.e. spin 3/2
    assert_eq!(s.dj(), 3);
    assert_eq!(s.dim(), 4); // 2j + 1
    assert_eq!(s.dual(), s); // every SU(2) irrep is self-dual
    assert_eq!(su2_frobenius_schur(3), -1.0); // half-integer j: symplectic self-duality
}

/// `docs/user-guide/representations.md`, code block 2.
#[test]
fn representations_2() {
    use racah::sun::Irrep;

    let three = Irrep::from_dynkin(&[1, 0]).unwrap(); // SU(3) fundamental
    let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
    let singlet = Irrep::trivial(3).unwrap(); // == from_dynkin(&[0, 0])

    assert_eq!(three.dim(), 3u32.into());
    assert_eq!(eight.dim(), 8u32.into());
    assert_eq!(three.dual().dynkin(), vec![0, 1]); // 3-bar
    assert_eq!(eight.dual(), eight); // the adjoint is self-dual
}

/// `docs/user-guide/representations.md`, code block 3.
#[test]
fn representations_3() {
    use racah::bcd::{Irrep, Series};

    // SO(5) = B_2. Dynkin [1, 0] is the 5-dimensional vector representation.
    let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
    assert_eq!(v.dim(), 5u32.into());
    assert_eq!(v.dual(), v);
    assert_eq!(v.frobenius_schur(), 1); // real (orthogonal) self-duality
    assert_eq!(v.partition(), Some(vec![1, 0])); // λ = (1, 0)
}

/// `docs/user-guide/representations.md`, code block 4.
#[test]
fn representations_4() {
    use racah::bcd::{Irrep, Series};

    // Sp(4) = C_2. Its defining 4 is pseudo-real; the 5 is real.
    let four = Irrep::from_dynkin(Series::C, &[1, 0]).unwrap();
    let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap();
    assert_eq!(four.dim(), 4u32.into());
    assert_eq!(four.frobenius_schur(), -1);
    assert_eq!(five.dim(), 5u32.into());
    assert_eq!(five.frobenius_schur(), 1);
}

/// `docs/user-guide/resources.md`, code block 1.
#[test]
fn resources_1() {
    use racah::cache::{configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier};

    let budgets =
        CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::SixJ, 1 << 20); // 1 MiB of 6j
    let _ = configure_cache_budgets(budgets);
}

/// `docs/user-guide/resources.md`, code block 2.
#[test]
fn resources_2() {
    use racah::cache::{base_cache_stats, reset, trim_to, CoefficientCacheTier};

    let stats = base_cache_stats();
    let _ = stats.six_j.entries; // per-tier entries / bytes / hits / misses / evictions
    let _ = stats.total(); // field-wise sum across the base tiers

    // Release one tier down to a target charge, oldest entries first.
    let report = trim_to(CoefficientCacheTier::SixJ, 0);
    let _ = report.removed_entries;

    reset(); // all tiers: entries, bytes, counters to zero
}

/// `docs/user-guide/resources.md`, code block 3.
#[test]
fn resources_3() {
    let fingerprint = racah::su2_authority_fingerprint();
    assert!(!fingerprint.is_empty());
    // Persist `fingerprint` alongside any table you derive from racah's SU(2) values.
}
