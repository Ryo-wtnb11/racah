//! Self-consistency and robustness gates that do not depend on the oracle:
//! Regge orbit invariance, exact orthogonality identities, the zero convention,
//! and a no-panic property over arbitrary inputs.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use racah::{canonical_regge_6j, clebsch_gordan, wigner_3j, wigner_6j, SignedSqrtRational};

fn tri(a: u32, b: u32, c: u32) -> bool {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
}

fn admissible_6j(dj: [u32; 6]) -> bool {
    tri(dj[0], dj[1], dj[2])
        && tri(dj[0], dj[4], dj[5])
        && tri(dj[3], dj[1], dj[5])
        && tri(dj[3], dj[4], dj[2])
}

/// The 24 tetrahedral symmetries of a 6j: permute the three columns
/// `(j1,j4),(j2,j5),(j3,j6)` and flip top/bottom in an even number of columns.
fn tetrahedral_images(dj: [u32; 6]) -> Vec<[u32; 6]> {
    let cols = [[dj[0], dj[3]], [dj[1], dj[4]], [dj[2], dj[5]]];
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    // Row-flip patterns that flip an even number of columns (0 or 2): these are
    // the value-preserving classical symmetries.
    let flips = [
        [false, false, false],
        [true, true, false],
        [true, false, true],
        [false, true, true],
    ];
    let mut out = Vec::with_capacity(24);
    for p in perms {
        for f in flips {
            let mut c = [[0u32; 2]; 3];
            for slot in 0..3 {
                let src = cols[p[slot]];
                c[slot] = if f[slot] { [src[1], src[0]] } else { src };
            }
            out.push([c[0][0], c[1][0], c[2][0], c[0][1], c[1][1], c[2][1]]);
        }
    }
    out
}

/// The non-classical Regge symmetry generator: fix `(j1,j4)`, map each of
/// `(j2,j3,j5,j6)` to `rho - j_i` with `rho = (j2+j3+j5+j6)/2`. Together with
/// the 24 classical images this generates the full 144-element symmetry group.
/// Returns `None` if an image spin would be negative (never for admissible
/// input, but guard against underflow rather than panic).
fn regge_map(dj: [u32; 6]) -> Option<[u32; 6]> {
    let s = (dj[1] as i64 + dj[2] as i64 + dj[4] as i64 + dj[5] as i64) / 2;
    let m = |x: u32| -> Option<u32> {
        let v = s - x as i64;
        (v >= 0).then_some(v as u32)
    };
    Some([dj[0], m(dj[1])?, m(dj[2])?, dj[3], m(dj[4])?, m(dj[5])?])
}

/// The full symmetry orbit of a 6j label set: closure under the 24 classical
/// images and the Regge generator.
fn full_orbit(dj: [u32; 6]) -> Vec<[u32; 6]> {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![dj];
    seen.insert(dj);
    while let Some(x) = stack.pop() {
        let mut neighbours = tetrahedral_images(x);
        if let Some(r) = regge_map(x) {
            neighbours.push(r);
        }
        for n in neighbours {
            if seen.insert(n) {
                stack.push(n);
            }
        }
    }
    seen.into_iter().collect()
}

#[test]
fn regge_orbit_key_and_value_invariant() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x8E66_1234);
    let mut tested = 0;
    let mut attempts = 0;
    let mut max_orbit = 0usize;
    while tested < 200 && attempts < 200_000 {
        attempts += 1;
        let dj = [
            rng.gen_range(0..=20),
            rng.gen_range(0..=20),
            rng.gen_range(0..=20),
            rng.gen_range(0..=20),
            rng.gen_range(0..=20),
            rng.gen_range(0..=20),
        ];
        if !admissible_6j(dj) {
            continue;
        }
        let key = canonical_regge_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]).unwrap();
        let val = wigner_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]);
        let orbit = full_orbit(dj);
        max_orbit = max_orbit.max(orbit.len());
        for img in &orbit {
            // Every image is itself an admissible 6j, so it has a key.
            let k2 = canonical_regge_6j(img[0], img[1], img[2], img[3], img[4], img[5]).unwrap();
            let v2 = wigner_6j(img[0], img[1], img[2], img[3], img[4], img[5]);
            assert_eq!(
                key, k2,
                "Regge key differs on orbit image {img:?} of {dj:?}"
            );
            assert_eq!(
                val.signed_square(),
                v2.signed_square(),
                "6j value differs on orbit image {img:?} of {dj:?}"
            );
        }
        tested += 1;
    }
    assert!(tested >= 50, "only {tested} orbits tested");
    // The Regge generator reaches non-classical images: a generic orbit exceeds
    // the 24 classical symmetries, confirming the extension is exercised.
    assert!(
        max_orbit > 24,
        "orbit never exceeded 24 images; Regge generator not exercised"
    );
}

/// Exact rational square root of a nonnegative rational that is a perfect
/// square; `None` otherwise. Used to divide out the common irrational factor.
fn exact_rational_sqrt(r: &Ratio<BigInt>) -> Option<Ratio<BigInt>> {
    if r.is_zero() {
        return Some(Ratio::zero());
    }
    let n = r.numer().magnitude();
    let d = r.denom().magnitude();
    let ns = n.sqrt();
    let ds = d.sqrt();
    if &(&ns * &ns) == n && &(&ds * &ds) == d {
        Some(Ratio::new(BigInt::from(ns), BigInt::from(ds)))
    } else {
        None
    }
}

/// Sum a list of `SignedSqrtRational` that share a common irrational factor
/// `sqrt(K)` (every pairwise radicand ratio is a perfect square), returning the
/// rational coefficient `R` such that the sum equals `R * sqrt(rho0)` where
/// `rho0` is the first nonzero term's radicand. Also returns `rho0`.
fn factor_common_sqrt(terms: &[SignedSqrtRational]) -> (Ratio<BigInt>, Ratio<BigInt>) {
    let mut rho0: Option<Ratio<BigInt>> = None;
    let mut acc = Ratio::zero();
    for t in terms {
        if t.sign() == 0 {
            continue;
        }
        let rho = t.radicand().clone();
        match &rho0 {
            None => {
                rho0 = Some(rho.clone());
                acc += Ratio::from(BigInt::from(t.sign() as i64));
            }
            Some(r0) => {
                // g = sqrt(rho / rho0), guaranteed rational for a shared factor.
                let g = exact_rational_sqrt(&(rho / r0)).expect("shared sqrt factor");
                acc += Ratio::from(BigInt::from(t.sign() as i64)) * g;
            }
        }
    }
    (acc, rho0.unwrap_or_else(Ratio::zero))
}

#[test]
fn six_j_orthogonality_exact() {
    // sum_x (2x+1)(2j3+1) {j1 j2 x; j4 j5 j3}{j1 j2 x; j4 j5 j6} = delta(j3,j6).
    // All terms share the same irrational factor sqrt(K) (independent of x), so
    // the sum is (rational) * sqrt(K); we assert the exact rational identity.
    let mut rng = ChaCha8Rng::seed_from_u64(0x0447_0007);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 40 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=12);
        let dj2 = rng.gen_range(0..=12);
        let dj4 = rng.gen_range(0..=12);
        let dj5 = rng.gen_range(0..=12);
        let dj3 = rng.gen_range(0..=12);
        let dj6 = rng.gen_range(0..=12);
        // j3 must be admissible with (j4,j5) and (j1,j2)-reachable via some x;
        // require both dj3 and dj6 valid outer couplings.
        if !tri(dj1, dj5, dj6) || !tri(dj4, dj2, dj6) {
            continue;
        }
        if !tri(dj1, dj5, dj3) || !tri(dj4, dj2, dj3) {
            continue;
        }

        // x ranges over admissible couplings of (j1,j2) and (j4,j5).
        let xmin = (dj1 as i64 - dj2 as i64)
            .abs()
            .max((dj4 as i64 - dj5 as i64).abs()) as u32;
        let xmax = (dj1 + dj2).min(dj4 + dj5);
        let mut terms = Vec::new();
        let mut x = xmin;
        while x <= xmax {
            let a = wigner_6j(dj1, dj2, x, dj4, dj5, dj3);
            let b = wigner_6j(dj1, dj2, x, dj4, dj5, dj6);
            let prod = (a * b).scale_int((x as i64 + 1) * (dj3 as i64 + 1));
            terms.push(prod);
            x += 2;
        }
        if terms.iter().all(|t| t.sign() == 0) {
            continue;
        }
        let (r, rho0) = factor_common_sqrt(&terms);
        if dj3 == dj6 {
            // sqrt(rho0) must be rational and R*sqrt(rho0) == 1.
            let s = exact_rational_sqrt(&rho0).expect("diagonal radicand is a square");
            assert_eq!(r * s, Ratio::one(), "6j orthogonality diagonal != 1");
        } else {
            assert!(r.is_zero(), "6j orthogonality off-diagonal != 0");
        }
        cases += 1;
    }
    assert!(cases >= 10, "only {cases} orthogonality cases");
}

#[test]
fn cg_orthogonality_sum_of_squares_is_one() {
    // For fixed admissible (j1,j2,j3,m3): sum_{m1} CG(j1 m1, j2 (m3-m1) | j3 m3)^2 = 1.
    // Each square is exactly the radicand (a rational), so the sum is exact.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC6_u64 * 99991);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 60 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=14);
        let dj2 = rng.gen_range(0..=14);
        let dj3 = rng.gen_range(0..=14);
        if !tri(dj1, dj2, dj3) {
            continue;
        }
        // pick m3 with valid parity/range
        let m3choices: Vec<i32> = (-(dj3 as i32)..=dj3 as i32).step_by(2).collect();
        let dm3 = m3choices[rng.gen_range(0..m3choices.len())];

        let mut sum: Ratio<BigInt> = Ratio::zero();
        let mut dm1 = -(dj1 as i32);
        while dm1 <= dj1 as i32 {
            let dm2 = dm3 - dm1;
            if dm2.unsigned_abs() <= dj2 && (dj2 as i32 + dm2) % 2 == 0 {
                let cg = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3);
                // CG^2 = radicand (nonneg rational).
                sum += cg.radicand();
            }
            dm1 += 2;
        }
        assert_eq!(
            sum,
            Ratio::one(),
            "CG sum of squares != 1 for j=({dj1},{dj2},{dj3}) m3={dm3}"
        );
        cases += 1;
    }
    assert!(cases >= 20, "only {cases} CG cases");
}

#[test]
fn cg_cross_j3_orthogonality_zero() {
    // sum_{m1} CG(j1 m1, j2 (M-m1) | j3 M) CG(j1 m1, j2 (M-m1) | j3' M) = 0 for
    // j3 != j3'. Terms share a common sqrt factor across m1; assert the exact
    // rational coefficient is zero.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC7_u64 * 40009);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 30 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=12);
        let dj2 = rng.gen_range(0..=12);
        let dj3 = rng.gen_range(0..=12);
        let dj3p = rng.gen_range(0..=12);
        if dj3 == dj3p || !tri(dj1, dj2, dj3) || !tri(dj1, dj2, dj3p) {
            continue;
        }
        // Total M must be valid for both j3 and j3'.
        let mmax = (dj3.min(dj3p)) as i32;
        let mchoices: Vec<i32> = (-mmax..=mmax).step_by(2).collect();
        if mchoices.is_empty() {
            continue;
        }
        let big_m = mchoices[rng.gen_range(0..mchoices.len())];

        let mut terms = Vec::new();
        let mut dm1 = -(dj1 as i32);
        while dm1 <= dj1 as i32 {
            let dm2 = big_m - dm1;
            if dm2.unsigned_abs() <= dj2 && (dj2 as i32 + dm2) % 2 == 0 {
                let a = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, big_m);
                let b = clebsch_gordan(dj1, dm1, dj2, dm2, dj3p, big_m);
                terms.push(a * b);
            }
            dm1 += 2;
        }
        if terms.iter().all(|t| t.sign() == 0) {
            continue;
        }
        let (r, _rho0) = factor_common_sqrt(&terms);
        assert!(r.is_zero(), "CG cross-j3 orthogonality != 0");
        cases += 1;
    }
    assert!(cases >= 10, "only {cases} CG cross cases");
}

#[test]
fn zero_convention() {
    // Each non-admissible class returns exact zero.
    // 6j: a broken triangle.
    assert_eq!(wigner_6j(1, 1, 1, 1, 1, 1), SignedSqrtRational::zero()); // parity
    assert_eq!(wigner_6j(2, 2, 10, 2, 2, 2), SignedSqrtRational::zero()); // range

    // 3j: triangle violation, |m|>j, parity mismatch, m-sum != 0.
    assert_eq!(wigner_3j(2, 2, 10, 0, 0, 0), SignedSqrtRational::zero());
    assert_eq!(wigner_3j(2, 2, 2, 4, -4, 0), SignedSqrtRational::zero()); // |m1|>j1
    assert_eq!(wigner_3j(2, 2, 2, 1, -1, 0), SignedSqrtRational::zero()); // parity: dj+dm odd
    assert_eq!(wigner_3j(2, 2, 2, 2, 2, 0), SignedSqrtRational::zero()); // m-sum != 0

    // CG inherits: total m mismatch and triangle violation.
    assert_eq!(clebsch_gordan(1, 1, 1, 1, 2, 0), SignedSqrtRational::zero()); // m3 != m1+m2
    assert_eq!(
        clebsch_gordan(2, 0, 2, 0, 10, 0),
        SignedSqrtRational::zero()
    ); // triangle
}

#[test]
fn no_panic_arbitrary_inputs() {
    // Arbitrary (bounded) doubled labels, including negative and out-of-range
    // projections, must never panic and must produce a well-formed value.
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
    for _ in 0..20_000 {
        let dj = |r: &mut ChaCha8Rng| r.gen_range(0u32..=40);
        let dm = |r: &mut ChaCha8Rng| r.gen_range(-50i32..=50);
        let (a, b, c, d, e, f) = (
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
        );
        let _ = wigner_6j(a, b, c, d, e, f).to_f64();
        let (m1, m2, m3) = (dm(&mut rng), dm(&mut rng), dm(&mut rng));
        let _ = wigner_3j(a, b, c, m1, m2, m3).to_f64();
        let _ = clebsch_gordan(a, m1, b, m2, c, m3).to_f64();
        let _ = canonical_regge_6j(a, b, c, d, e, f);
    }
}
