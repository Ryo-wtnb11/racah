//! Tests for the exact B/C/D combinatorics. All anchors are independent of the
//! code under test: dimension/product values are hand-verified against the
//! representation theory of the classical groups (Fulton–Harris tables), never
//! read back from `racah`'s own output.

use super::*;

fn irr(series: Series, dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin(series, dynkin).unwrap()
}

// ---- labels / normalization ----------------------------------------------

#[test]
fn dynkin_partition_round_trip() {
    for (series, d) in [
        (Series::B, vec![1, 0]),
        (Series::B, vec![2, 2, 4]),
        (Series::C, vec![1, 0, 1]),
        (Series::C, vec![3, 1]),
        (Series::D, vec![1, 0, 0, 0]),
        (Series::D, vec![0, 1, 1]), // D_3, a2+a3 even
        (Series::D, vec![2, 1, 1, 1]),
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dynkin(), d, "round trip {series:?} {d:?}");
        assert_eq!(s.rank(), d.len());
    }
}

#[test]
fn partition_is_nonincreasing_and_dominant() {
    // B_3 vector rep (1,0,0) → partition (1,0,0).
    assert_eq!(irr(Series::B, &[1, 0, 0]).partition(), &[1, 0, 0]);
    // C_2 fundamental (1,0) → (1,0); adjoint (2,0) → (2,0).
    assert_eq!(irr(Series::C, &[1, 0]).partition(), &[1, 0]);
    assert_eq!(irr(Series::C, &[2, 0]).partition(), &[2, 0]);
    // D_3 chiral (0,2,0) vs (0,0,2): opposite last-coordinate signs (chirality).
    let p = irr(Series::D, &[0, 2, 0]).partition().to_vec();
    let q = irr(Series::D, &[0, 0, 2]).partition().to_vec();
    assert_eq!(p, vec![1, 1, -1]);
    assert_eq!(q, vec![1, 1, 1]);
}

// ---- guards (issue #15 inventory) ----------------------------------------

#[test]
fn excluded_low_ranks_are_typed_errors_with_redirection() {
    // B_1 = SO(3) ≅ SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::B, &[2]),
        Err(BcdError::ExcludedRank {
            series: Series::B,
            rank: 1,
            redirect
        }) if redirect.contains("SU(2)")
    ));
    // C_1 = Sp(2) ≅ SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::C, &[1]),
        Err(BcdError::ExcludedRank {
            series: Series::C,
            rank: 1,
            redirect
        }) if redirect.contains("SU(2)")
    ));
    // D_2 = SO(4) ≅ SU(2)×SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::D, &[0, 0]),
        Err(BcdError::ExcludedRank {
            series: Series::D,
            rank: 2,
            redirect
        }) if redirect.contains("SU(2)×SU(2)")
    ));
}

#[test]
fn malformed_and_spinor_labels_are_typed_errors() {
    assert_eq!(
        Irrep::from_dynkin(Series::B, &[]),
        Err(BcdError::EmptyLabel)
    );
    assert!(matches!(
        Irrep::from_dynkin(Series::C, &[1, -1]),
        Err(BcdError::NegativeDynkin { .. })
    ));
    // B_2 spinor: a_2 odd.
    assert!(matches!(
        Irrep::from_dynkin(Series::B, &[0, 1]),
        Err(BcdError::SpinorLabel { .. })
    ));
    // D_3 tensor vs spinor by parity of a_2 + a_3.
    assert!(Irrep::from_dynkin(Series::D, &[1, 0, 0]).is_ok()); // 0+0 even (vector)
    assert!(Irrep::from_dynkin(Series::D, &[0, 1, 1]).is_ok()); // 1+1 even
    assert!(matches!(
        Irrep::from_dynkin(Series::D, &[0, 1, 2]),
        Err(BcdError::SpinorLabel { .. }) // 1+2 = 3 odd
    ));
    // C accepts every non-negative integer label (Sp is simply connected).
    assert!(Irrep::from_dynkin(Series::C, &[1, 1]).is_ok());
}

// ---- Weyl dimension anchors ----------------------------------------------

#[test]
fn dim_anchors_b2_so5() {
    // SO(5) = B_2. Known dims: 1, 5 (vector), 10 (adjoint), 14 (sym traceless),
    // 35, 16.
    let d = |a: &[i64]| irr(Series::B, a).dim();
    assert_eq!(d(&[0, 0]), BigInt::from(1));
    assert_eq!(d(&[1, 0]), BigInt::from(5)); // vector
    assert_eq!(d(&[0, 2]), BigInt::from(10)); // adjoint (a_2 even)
    assert_eq!(d(&[2, 0]), BigInt::from(14)); // sym traceless
    assert_eq!(d(&[1, 2]), BigInt::from(35)); // partition (2,1)
    assert_eq!(d(&[2, 2]), BigInt::from(81)); // partition (3,1)
}

#[test]
fn dim_anchors_c2_sp4() {
    // Sp(4) = C_2. Known dims: 1, 4 (fundamental), 5, 10 (adjoint), 16.
    let d = |a: &[i64]| irr(Series::C, a).dim();
    assert_eq!(d(&[0, 0]), BigInt::from(1));
    assert_eq!(d(&[1, 0]), BigInt::from(4)); // fundamental
    assert_eq!(d(&[0, 1]), BigInt::from(5));
    assert_eq!(d(&[2, 0]), BigInt::from(10)); // adjoint
    assert_eq!(d(&[1, 1]), BigInt::from(16));
}

#[test]
fn dim_anchors_d_series() {
    // SO(8) = D_4 vector 8v dim 8; adjoint 28; sym traceless 35v.
    assert_eq!(irr(Series::D, &[1, 0, 0, 0]).dim(), BigInt::from(8));
    assert_eq!(irr(Series::D, &[0, 1, 0, 0]).dim(), BigInt::from(28)); // adjoint
    assert_eq!(irr(Series::D, &[2, 0, 0, 0]).dim(), BigInt::from(35)); // 35v
                                                                       // SO(6) = D_3 ≅ SU(4): vector 6, adjoint 15, 20'.
    assert_eq!(irr(Series::D, &[1, 0, 0]).dim(), BigInt::from(6));
    assert_eq!(irr(Series::D, &[0, 1, 1]).dim(), BigInt::from(15)); // adjoint
    assert_eq!(irr(Series::D, &[2, 0, 0]).dim(), BigInt::from(20));
}

// ---- dual --------------------------------------------------------------

#[test]
fn dual_is_involution_and_preserves_dim() {
    for (series, d) in [
        (Series::B, vec![1, 2]),
        (Series::B, vec![2, 0, 2]),
        (Series::C, vec![1, 1]),
        (Series::C, vec![2, 0, 1]),
        (Series::D, vec![1, 0, 0]),
        (Series::D, vec![1, 1, 3]),    // D_3 (a2+a3 even)
        (Series::D, vec![1, 0, 1, 1]), // D_4 (a3+a4 even)
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dual().dual(), s, "involution {series:?} {d:?}");
        assert_eq!(s.dim(), s.dual().dim(), "dim {series:?} {d:?}");
    }
}

#[test]
fn dual_bc_and_d_even_self_dual() {
    // B, C, and D_even are self-dual.
    for (series, d) in [
        (Series::B, vec![2, 4]),
        (Series::C, vec![1, 3]),
        (Series::D, vec![1, 1, 0, 2]), // D_4 (even)
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dual(), s, "self-dual {series:?} {d:?}");
    }
}

#[test]
fn dual_d_odd_swaps_last_two_dynkin() {
    // D_3 (r odd): duality swaps a_2 ↔ a_3.
    let s = irr(Series::D, &[0, 1, 3]); // 1+3 even
    assert_eq!(s.dual().dynkin(), vec![0, 3, 1]);
    // Vector (1,0,0) is self-dual (swap of the two zeros).
    assert_eq!(irr(Series::D, &[1, 0, 0]).dual().dynkin(), vec![1, 0, 0]);
    // Non-self-dual chiral pair.
    assert_ne!(irr(Series::D, &[0, 2, 0]), irr(Series::D, &[0, 0, 2]));
    assert_eq!(
        irr(Series::D, &[0, 2, 0]).dual(),
        irr(Series::D, &[0, 0, 2])
    );
}

// ---- Frobenius–Schur -----------------------------------------------------

#[test]
fn frobenius_schur_by_series() {
    // B: always +1 (SO(odd) tensor irreps are real).
    for d in [vec![0, 0], vec![1, 0], vec![2, 2], vec![1, 2]] {
        assert_eq!(irr(Series::B, &d).frobenius_schur(), 1, "B {d:?}");
    }
    // C = Sp: quaternionic iff sum of odd-position Dynkin labels is odd.
    // Vector (1,0): a_1 = 1 odd → -1 (quaternionic).
    assert_eq!(irr(Series::C, &[1, 0]).frobenius_schur(), -1);
    // Adjoint (2,0): a_1 = 2 even → +1 (real).
    assert_eq!(irr(Series::C, &[2, 0]).frobenius_schur(), 1);
    // (0,1): odd positions {a_1} = 0 even → +1 (the 5 of Sp(4), real).
    assert_eq!(irr(Series::C, &[0, 1]).frobenius_schur(), 1);
    // C_3 (1,0,0): odd positions a_1 + a_3 = 1 → -1.
    assert_eq!(irr(Series::C, &[1, 0, 0]).frobenius_schur(), -1);
    // C_3 (0,0,1): odd positions a_1 + a_3 = 1 → -1.
    assert_eq!(irr(Series::C, &[0, 0, 1]).frobenius_schur(), -1);
    // D_even: self-dual → +1.
    assert_eq!(irr(Series::D, &[1, 0, 0, 0]).frobenius_schur(), 1);
    // D_odd non-self-dual chiral → 0.
    assert_eq!(irr(Series::D, &[0, 2, 0]).frobenius_schur(), 0);
    // D_odd self-dual (vector) → +1.
    assert_eq!(irr(Series::D, &[1, 0, 0]).frobenius_schur(), 1);
}

// ---- weight multiplicities (Freudenthal) ---------------------------------

#[test]
fn weight_multiplicities_sum_to_dim() {
    // Σ_μ m(μ) · |Weyl orbit of μ| == dim.
    for (series, d) in [
        (Series::B, vec![1, 0]),
        (Series::B, vec![2, 2]), // adjoint of SO(5) has a zero weight of mult 2
        (Series::C, vec![1, 0]),
        (Series::C, vec![2, 0]),
        (Series::D, vec![1, 0, 0]),
        (Series::D, vec![0, 1, 1]),
        (Series::D, vec![1, 0, 0, 0]),
    ] {
        let s = irr(series, &d);
        let total: u64 = s
            .weight_multiplicities()
            .iter()
            .map(|(mu, &m)| m * weyl_orbit(series, mu).len() as u64)
            .sum();
        assert_eq!(
            BigInt::from(total),
            s.dim(),
            "weight count {series:?} {d:?}"
        );
    }
}

#[test]
fn adjoint_has_rank_zero_weight() {
    // The adjoint of SO(5) (B_2 (0,2)) has the zero weight with multiplicity =
    // rank = 2.
    let s = irr(Series::B, &[0, 2]);
    assert_eq!(s.weight_multiplicities().get(&vec![0, 0]), Some(&2));
}

// ---- fundamental products (independent anchors) --------------------------

fn decomp(a: &Irrep, b: &Irrep) -> Vec<(Vec<i64>, u32)> {
    let mut v: Vec<(Vec<i64>, u32)> = directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, n)| (c.dynkin(), n))
        .collect();
    v.sort();
    v
}

#[test]
fn so5_vector_squared() {
    // SO(5): 5 ⊗ 5 = 1 + 10 + 14.
    let v = irr(Series::B, &[1, 0]);
    let got = decomp(&v, &v);
    // dynkin: 1=(0,0), 14=(2,0), 10=(0,2).
    assert_eq!(got, vec![(vec![0, 0], 1), (vec![0, 2], 1), (vec![2, 0], 1)]);
}

#[test]
fn sp4_fundamental_squared() {
    // Sp(4): 4 ⊗ 4 = 1 + 5 + 10.
    let v = irr(Series::C, &[1, 0]);
    let got = decomp(&v, &v);
    // dynkin: 1=(0,0), 5=(0,1), 10=(2,0).
    assert_eq!(got, vec![(vec![0, 0], 1), (vec![0, 1], 1), (vec![2, 0], 1)]);
}

#[test]
fn so8_vector_squared() {
    // SO(8): 8v ⊗ 8v = 1 + 28 + 35v.
    let v = irr(Series::D, &[1, 0, 0, 0]);
    let got = decomp(&v, &v);
    // 1=(0,0,0,0), 28=(0,1,0,0), 35v=(2,0,0,0).
    assert_eq!(
        got,
        vec![
            (vec![0, 0, 0, 0], 1),
            (vec![0, 1, 0, 0], 1),
            (vec![2, 0, 0, 0], 1),
        ]
    );
}

#[test]
fn so6_vector_squared() {
    // SO(6): 6 ⊗ 6 = 1 + 15 + 20'.
    let v = irr(Series::D, &[1, 0, 0]);
    let got = decomp(&v, &v);
    assert_eq!(
        got,
        vec![(vec![0, 0, 0], 1), (vec![0, 1, 1], 1), (vec![2, 0, 0], 1)]
    );
}

// ---- exact property tests (tool-independent) -----------------------------

fn dim_sum_rule(a: &Irrep, b: &Irrep) {
    let lhs = a.dim() * b.dim();
    let rhs: BigInt = directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, n)| c.dim() * BigInt::from(n))
        .sum();
    assert_eq!(lhs, rhs, "dim-sum rule {:?} ⊗ {:?}", a.dynkin(), b.dynkin());
}

fn product_symmetry(a: &Irrep, b: &Irrep) {
    assert_eq!(
        directproduct(a, b).unwrap(),
        directproduct(b, a).unwrap(),
        "N^c_ab == N^c_ba for {:?}, {:?}",
        a.dynkin(),
        b.dynkin()
    );
}

fn dual_twist(a: &Irrep, b: &Irrep) {
    // N^c_ab == N^{c̄}_{ā b̄}.
    let dp = directproduct(a, b).unwrap();
    let dpd = directproduct(&a.dual(), &b.dual()).unwrap();
    let twisted: BTreeMap<Irrep, u32> = dp.into_iter().map(|(c, n)| (c.dual(), n)).collect();
    assert_eq!(
        dpd,
        twisted,
        "dual twist {:?}, {:?}",
        a.dynkin(),
        b.dynkin()
    );
}

#[test]
fn group_mismatch_is_typed_error() {
    let b2 = irr(Series::B, &[1, 0]);
    let c2 = irr(Series::C, &[1, 0]);
    let b3 = irr(Series::B, &[1, 0, 0]);
    assert!(matches!(
        directproduct(&b2, &c2),
        Err(BcdError::GroupMismatch { .. })
    ));
    assert!(matches!(
        directproduct(&b2, &b3),
        Err(BcdError::GroupMismatch { .. })
    ));
}

#[test]
fn randomized_property_sweep() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x0BC0_DE19_5150_4E37);
    // Keep labels small so weight systems (hence Freudenthal + orbits) stay
    // cheap — this is a property check, not a large-irrep stress test.
    let max_label = |r: usize| -> i64 {
        match r {
            2 => 3,
            3 => 2,
            _ => 1,
        }
    };
    for series in [Series::B, Series::C, Series::D] {
        let min_r = series.min_rank();
        for r in min_r..=4usize {
            let hi = max_label(r);
            let rand_irrep = |rng: &mut rand_chacha::ChaCha8Rng| -> Irrep {
                loop {
                    let d: Vec<i64> = (0..r).map(|_| rng.gen_range(0..=hi)).collect();
                    if let Ok(s) = Irrep::from_dynkin(series, &d) {
                        return s; // reject spinor labels by resampling
                    }
                }
            };
            for _ in 0..30 {
                let a = rand_irrep(&mut rng);
                let b = rand_irrep(&mut rng);
                dim_sum_rule(&a, &b);
                product_symmetry(&a, &b);
                dual_twist(&a, &b);
                assert_eq!(a.dual().dual(), a);
            }
        }
    }
}

// ---- external-oracle fixtures (Sage/OSCAR) -------------------------------

/// Independent-implementation oracle (issue #19 item 5). Neither Sage nor
/// OSCAR is installed in the development environment, so this test is
/// `#[ignore]`d and the fixture is absent; the maintainer runs
/// `tools/gen_bcd_fixtures.py` (Sage) or `tools/gen_bcd_fixtures.jl` (OSCAR)
/// once to produce `tests/fixtures/bcd_fixtures.json`, then removes the
/// `#[ignore]`. Values are never fabricated here.
#[test]
#[ignore = "requires tests/fixtures/bcd_fixtures.json from Sage/OSCAR (not installed locally)"]
fn external_oracle_fixtures() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bcd_fixtures.json"
    );
    let raw = std::fs::read_to_string(path)
        .expect("fixture file present (run tools/gen_bcd_fixtures.{py,jl} first)");
    check_fixtures(&raw);
}

/// Parse and check the fixture JSON. Kept as a separate fn (not gated on the
/// file's presence) so its logic is compiled and type-checked every build.
///
/// Schema (minimal, hand-parsed to avoid a serde dependency in this base-ish
/// crate): a flat list of records, one per line after the `---` provenance
/// header, each `SERIES RANK | dynkin_a | dynkin_b | dim_a dim_b | c:n c:n ...`
/// with `dynkin`/`c` as comma-separated ints.
#[allow(dead_code)]
fn check_fixtures(raw: &str) {
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("---") {
            continue;
        }
        let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        assert!(parts.len() >= 5, "malformed fixture line: {line}");
        let head: Vec<&str> = parts[0].split_whitespace().collect();
        let series = match head[0] {
            "B" => Series::B,
            "C" => Series::C,
            "D" => Series::D,
            other => panic!("unknown series {other}"),
        };
        let ints = |s: &str| -> Vec<i64> {
            s.split(',')
                .filter(|x| !x.is_empty())
                .map(|x| x.parse().unwrap())
                .collect()
        };
        let a = Irrep::from_dynkin(series, &ints(parts[1])).unwrap();
        let b = Irrep::from_dynkin(series, &ints(parts[2])).unwrap();
        let dims: Vec<&str> = parts[3].split_whitespace().collect();
        assert_eq!(a.dim().to_string(), dims[0], "dim_a for {line}");
        assert_eq!(b.dim().to_string(), dims[1], "dim_b for {line}");
        let mut want: BTreeMap<Vec<i64>, u32> = BTreeMap::new();
        for tok in parts[4].split_whitespace() {
            let (c, n) = tok.split_once(':').expect("c:n token");
            want.insert(ints(c), n.parse().unwrap());
        }
        let got: BTreeMap<Vec<i64>, u32> = directproduct(&a, &b)
            .unwrap()
            .into_iter()
            .map(|(c, n)| (c.dynkin(), n))
            .collect();
        assert_eq!(got, want, "decomposition mismatch for {line}");
    }
}
