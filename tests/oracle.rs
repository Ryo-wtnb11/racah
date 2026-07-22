//! Exact agreement with `wigner-symbols 0.5.1` on the overlap domain.
//!
//! The oracle is independent (a separate crate backed by rug/GMP). We compare
//! on the *exact* representation: sign and squared rational (`signed_sq`), never
//! on rounded floats, plus a separate <= 1 ulp check of `to_f64`.

use num_bigint::BigInt;
use num_rational::Ratio;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use racah::{clebsch_gordan, wigner_3j, wigner_6j, SignedSqrtRational};

use wigner_symbols::{ClebschGordan, Wigner3jm, Wigner6j};

/// Canonical `(numerator, denominator)` decimal strings of a num rational.
fn num_key(r: &Ratio<BigInt>) -> (String, String) {
    (r.numer().to_string(), r.denom().to_string())
}

/// Canonical `(numerator, denominator)` decimal strings of a rug rational
/// (`signed_sq` of the oracle value).
fn rug_key(r: &rug::Rational) -> (String, String) {
    (r.numer().to_string(), r.denom().to_string())
}

/// Assert our exact value equals the oracle's exact value (signed square).
fn assert_exact(ours: &SignedSqrtRational, theirs: wigner_symbols::SignedSqrt, ctx: &str) {
    let our = num_key(&ours.signed_square());
    let their = rug_key(&theirs.signed_sq());
    assert_eq!(our, their, "signed-square mismatch at {ctx}");
}

/// Assert `to_f64` agrees with the oracle f64 to <= 1 ulp.
fn assert_f64_close(ours: f64, theirs: f64, ctx: &str) {
    if ours == theirs {
        return;
    }
    let ulp = theirs.abs() * f64::EPSILON;
    assert!(
        (ours - theirs).abs() <= ulp + f64::MIN_POSITIVE,
        "f64 mismatch at {ctx}: ours={ours} theirs={theirs}"
    );
}

fn oracle_6j(dj: [u32; 6]) -> wigner_symbols::SignedSqrt {
    Wigner6j {
        tj1: dj[0] as i32,
        tj2: dj[1] as i32,
        tj3: dj[2] as i32,
        tj4: dj[3] as i32,
        tj5: dj[4] as i32,
        tj6: dj[5] as i32,
    }
    .value()
}

fn oracle_3j(dj: [u32; 3], dm: [i32; 3]) -> wigner_symbols::SignedSqrt {
    Wigner3jm {
        tj1: dj[0] as i32,
        tm1: dm[0],
        tj2: dj[1] as i32,
        tm2: dm[1],
        tj3: dj[2] as i32,
        tm3: dm[2],
    }
    .value()
}

fn oracle_cg(dj: [u32; 3], dm: [i32; 3]) -> wigner_symbols::SignedSqrt {
    ClebschGordan {
        tj1: dj[0] as i32,
        tm1: dm[0],
        tj2: dj[1] as i32,
        tm2: dm[1],
        tj12: dj[2] as i32,
        tm12: dm[2],
    }
    .value()
}

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

#[test]
fn six_j_exhaustive_small() {
    // All twice-spins <= 12, compared exactly. Non-admissible symbols only get
    // a cheap zero check (both sides zero); the expensive exact/float
    // comparison runs on admissible symbols, keeping the sextuple loop fast.
    const MAX: u32 = 12;
    let mut checked = 0u64;
    for dj1 in 0..=MAX {
        for dj2 in 0..=MAX {
            for dj3 in 0..=MAX {
                for dj4 in 0..=MAX {
                    for dj5 in 0..=MAX {
                        for dj6 in 0..=MAX {
                            let dj = [dj1, dj2, dj3, dj4, dj5, dj6];
                            let ours = wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6);
                            if !admissible_6j(dj) {
                                assert_eq!(ours.sign(), 0, "non-admissible 6j {dj:?} not zero");
                                continue;
                            }
                            let theirs = oracle_6j(dj);
                            assert_exact(&ours, theirs.clone(), &format!("6j {dj:?}"));
                            if ours.sign() != 0 {
                                assert_f64_close(
                                    ours.to_f64(),
                                    f64::from(theirs),
                                    &format!("6j {dj:?}"),
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

fn rand_dj(rng: &mut ChaCha8Rng, max: u32) -> u32 {
    rng.gen_range(0..=max)
}

/// A random admissible 6j label set (all four triangles) with twice-spins <= max.
fn rand_admissible_6j(rng: &mut ChaCha8Rng, max: u32) -> Option<[u32; 6]> {
    let dj1 = rand_dj(rng, max);
    let dj2 = rand_dj(rng, max);
    let dj3 = rand_dj(rng, max);
    let dj4 = rand_dj(rng, max);
    let dj5 = rand_dj(rng, max);
    let dj6 = rand_dj(rng, max);
    let tri = |a: u32, b: u32, c: u32| {
        let (a, b, c) = (a as i64, b as i64, c as i64);
        (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
    };
    if tri(dj1, dj2, dj3) && tri(dj1, dj5, dj6) && tri(dj4, dj2, dj6) && tri(dj4, dj5, dj3) {
        Some([dj1, dj2, dj3, dj4, dj5, dj6])
    } else {
        None
    }
}

#[test]
fn six_j_randomized_large() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA55A_1234);
    let mut samples = 0;
    let mut attempts = 0;
    while samples < 1500 && attempts < 5_000_000 {
        attempts += 1;
        let Some(dj) = rand_admissible_6j(&mut rng, 254) else {
            continue;
        };
        let ours = wigner_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]);
        let theirs = oracle_6j(dj);
        assert_exact(&ours, theirs.clone(), &format!("6j {dj:?}"));
        if ours.sign() != 0 {
            assert_f64_close(ours.to_f64(), f64::from(theirs), &format!("6j {dj:?}"));
        }
        samples += 1;
    }
    assert!(samples >= 1000, "only {samples} admissible samples");
}

/// All valid doubled projections `dm` for a doubled spin `dj`.
fn projections(dj: u32) -> Vec<i32> {
    let dj = dj as i32;
    (-dj..=dj).step_by(2).collect()
}

#[test]
fn three_j_and_cg_exhaustive_small() {
    // Exhaustive over twice-spins <= 8 and every valid projection.
    const MAX: u32 = 8;
    let mut checked = 0u64;
    for dj1 in 0..=MAX {
        for dj2 in 0..=MAX {
            for dj3 in 0..=MAX {
                for &dm1 in &projections(dj1) {
                    for &dm2 in &projections(dj2) {
                        for &dm3 in &projections(dj3) {
                            let dj = [dj1, dj2, dj3];
                            let dm = [dm1, dm2, dm3];
                            let o3 = wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3);
                            assert_exact(&o3, oracle_3j(dj, dm), &format!("3j {dj:?}{dm:?}"));

                            // CG uses m12 = -m3 convention in the oracle struct.
                            let oc = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3);
                            assert_exact(
                                &oc,
                                oracle_cg(dj, [dm1, dm2, dm3]),
                                &format!("cg {dj:?}{dm:?}"),
                            );
                            if o3.sign() != 0 {
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn three_j_and_cg_randomized_large() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x1357_9BDF);
    let mut samples = 0;
    let mut attempts = 0;
    while samples < 1500 && attempts < 5_000_000 {
        attempts += 1;
        let dj1 = rand_dj(&mut rng, 254);
        let dj2 = rand_dj(&mut rng, 254);
        let dj3 = rand_dj(&mut rng, 254);
        let tri = {
            let (a, b, c) = (dj1 as i64, dj2 as i64, dj3 as i64);
            (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
        };
        if !tri {
            continue;
        }
        // Choose m1, m2 within range and matching parity; m3 = -(m1+m2).
        let p1 = projections(dj1);
        let p2 = projections(dj2);
        let dm1 = p1[rng.gen_range(0..p1.len())];
        let dm2 = p2[rng.gen_range(0..p2.len())];
        let dm3 = -(dm1 + dm2);
        if dm3.unsigned_abs() > dj3 {
            continue;
        }
        let dj = [dj1, dj2, dj3];
        let dm = [dm1, dm2, dm3];
        let o3 = wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3);
        assert_exact(&o3, oracle_3j(dj, dm), &format!("3j {dj:?}{dm:?}"));
        let theirs3 = oracle_3j(dj, dm);
        if o3.sign() != 0 {
            assert_f64_close(o3.to_f64(), f64::from(theirs3), &format!("3j {dj:?}{dm:?}"));
        }
        let oc = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3);
        assert_exact(&oc, oracle_cg(dj, dm), &format!("cg {dj:?}{dm:?}"));
        samples += 1;
    }
    assert!(samples >= 1000, "only {samples} admissible samples");
}
