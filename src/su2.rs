//! Exact SU(2) recoupling coefficients: Wigner 3j, 6j, Clebsch-Gordan, and the
//! canonical Regge key for 6j symbols.
//!
//! All spins are in the doubled ("twice") convention: `dj = 2j` as `u32`,
//! `dm = 2m` as `i32`. Non-admissible label combinations return the exact zero
//! value (never an error and never a panic), matching the reference-crate
//! semantics. Values are computed as big-rational Racah sums and carried as
//! [`SignedSqrtRational`] until a single final rounding to `f64`.

use num_rational::Ratio;

use crate::exact::SignedSqrtRational;
use crate::primefactor::{factorial as pf_factorial, mul_factorial, sum_series, Pf};

/// Prime-factorized squared triangle coefficient
/// `Delta^2(a,b,c) = num/den` with `num = t1! t2! t3!` and `den = t4!`,
/// `t4 = (a+b+c)/2 + 1`. Callers guarantee admissibility, so every factorial
/// argument is a nonnegative integer. (`WignerSymbols.jl::Delta^2`.)
fn delta_sq_pf(a: u32, b: u32, c: u32) -> (Pf, Pf) {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    let t1 = ((a + b - c) / 2) as u64;
    let t2 = ((a - b + c) / 2) as u64;
    let t3 = ((-a + b + c) / 2) as u64;
    let t4 = ((a + b + c) / 2 + 1) as u64;
    let mut num = pf_factorial(t1);
    mul_factorial(&mut num, t2);
    mul_factorial(&mut num, t3);
    (num, pf_factorial(t4))
}

/// Triangle admissibility for a doubled-spin triple `(a, b, c)`:
/// `|a-b| <= c <= a+b` and `a+b+c` even.
fn triangle_ok(a: u32, b: u32, c: u32) -> bool {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
}

/// Wigner 6j symbol `{dj1 dj2 dj3; dj4 dj5 dj6}` (doubled spins).
///
/// Returns exact zero unless all four triangles
/// `(1,2,3), (1,5,6), (4,2,6), (4,5,3)` are admissible. Uses the Racah
/// single-sum closed form in big-rational arithmetic.
pub fn wigner_6j(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> SignedSqrtRational {
    if !(triangle_ok(dj1, dj2, dj3)
        && triangle_ok(dj1, dj5, dj6)
        && triangle_ok(dj4, dj2, dj6)
        && triangle_ok(dj4, dj5, dj3))
    {
        return SignedSqrtRational::zero();
    }

    // Radical part: the product of the four squared triangle coefficients,
    // assembled in exponent space. splitsquare pulls the perfect-square part
    // out into the rational prefactor `s`, leaving a square-free radicand `r`,
    // so no huge intermediate factorial is ever formed as a big integer.
    // (`WignerSymbols.jl::_wigner6j`.)
    let (n1, d1) = delta_sq_pf(dj1, dj2, dj3);
    let (n2, d2) = delta_sq_pf(dj1, dj5, dj6);
    let (n3, d3) = delta_sq_pf(dj4, dj2, dj6);
    let (n4, d4) = delta_sq_pf(dj4, dj5, dj3);
    let mut num = n1;
    num.mul_assign(&n2);
    num.mul_assign(&n3);
    num.mul_assign(&n4);
    let mut den = d1;
    den.mul_assign(&d2);
    den.mul_assign(&d3);
    den.mul_assign(&d4);
    let (mut snum, mut rnum) = num.splitsquare();
    let (mut sden, mut rden) = den.splitsquare();
    Pf::divgcd(&mut snum, &mut sden);
    Pf::divgcd(&mut rnum, &mut rden);
    let s = Ratio::new(snum.to_bigint(), sden.to_bigint());
    let r = Ratio::new(rnum.to_bigint(), rden.to_bigint());

    // Racah alternating sum over k (in halved units). Widen to i64 before
    // summing (as delta_sq_pf does): u32 addition of four doubled spins could
    // wrap for absurd labels.
    let (j1, j2, j3, j4, j5, j6) = (
        dj1 as i64, dj2 as i64, dj3 as i64, dj4 as i64, dj5 as i64, dj6 as i64,
    );
    // t1..t4 are the triangle sums; t5..t7 the "square" sums.
    let t1 = (j1 + j2 + j3) / 2;
    let t2 = (j1 + j5 + j6) / 2;
    let t3 = (j4 + j2 + j6) / 2;
    let t4 = (j4 + j5 + j3) / 2;
    let t5 = (j1 + j2 + j4 + j5) / 2;
    let t6 = (j2 + j3 + j5 + j6) / 2;
    let t7 = (j3 + j1 + j6 + j4) / 2;

    let kmin = t1.max(t2).max(t3).max(t4);
    let kmax = t5.min(t6).min(t7);

    // Each term is (-1)^k (k+1)! / [(k-t1)!...(k-t4)! (t5-k)!(t6-k)!(t7-k)!]
    // built as a prime-factorized numerator/denominator pair; sum_series
    // combines them over a common denominator. (`compute6jseries`.)
    let mut terms = Vec::with_capacity((kmax - kmin + 1).max(0) as usize);
    for k in kmin..=kmax {
        let mut nump = pf_factorial((k + 1) as u64);
        if k % 2 != 0 {
            nump = nump.neg();
        }
        let mut denp = pf_factorial((k - t1) as u64);
        mul_factorial(&mut denp, (k - t2) as u64);
        mul_factorial(&mut denp, (k - t3) as u64);
        mul_factorial(&mut denp, (k - t4) as u64);
        mul_factorial(&mut denp, (t5 - k) as u64);
        mul_factorial(&mut denp, (t6 - k) as u64);
        mul_factorial(&mut denp, (t7 - k) as u64);
        Pf::divgcd(&mut nump, &mut denp);
        terms.push((nump, denp));
    }
    let series = sum_series(terms);

    // value = (s * series) * sqrt(r); `r` is nonnegative (square-free part of a
    // factorial product), so the clamp in from_prefactor_radical never fires.
    SignedSqrtRational::from_prefactor_radical(s * series, r)
}

/// Wigner 3j symbol `(dj1 dj2 dj3; dm1 dm2 dm3)` (doubled spins/projections).
///
/// Returns exact zero unless the labels are admissible: triangle `(1,2,3)`,
/// `|dm_i| <= dj_i`, `dj_i + dm_i` even for each `i`, and `dm1+dm2+dm3 == 0`.
/// Condon-Shortley phase, matching the standard closed form.
pub fn wigner_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> SignedSqrtRational {
    if !admissible_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        return SignedSqrtRational::zero();
    }

    let (j1, j2, j3) = (dj1 as i64, dj2 as i64, dj3 as i64);
    let (m1, m2, m3) = (dm1 as i64, dm2 as i64, dm3 as i64);

    // Radical: Delta^2(j1,j2,j3) * prod_i (j_i+m_i)! (j_i-m_i)!, all halved,
    // assembled in exponent space. splitsquare separates the perfect-square
    // prefactor `s` from the square-free radicand `r`, so the big factorials
    // are never multiplied out as big integers. (`WignerSymbols.jl::_wigner3j`.)
    let (mut num, den) = delta_sq_pf(dj1, dj2, dj3);
    for (dj, dm) in [(j1, m1), (j2, m2), (j3, m3)] {
        mul_factorial(&mut num, ((dj + dm) / 2) as u64);
        mul_factorial(&mut num, ((dj - dm) / 2) as u64);
    }
    let (mut snum, mut rnum) = num.splitsquare();
    let (mut sden, mut rden) = den.splitsquare();
    Pf::divgcd(&mut snum, &mut sden);
    Pf::divgcd(&mut rnum, &mut rden);
    let s = Ratio::new(snum.to_bigint(), sden.to_bigint());
    let r = Ratio::new(rnum.to_bigint(), rden.to_bigint());

    // Alternating k-sum (halved units). Arguments (all >= 0 within range):
    //   k, (j1+j2-j3)/2 - k, (j1-m1)/2 - k, (j2+m2)/2 - k,
    //   k + (j3-j2+m1)/2, k + (j3-j1-m2)/2
    let a = (j1 + j2 - j3) / 2;
    let b = (j1 - m1) / 2;
    let c = (j2 + m2) / 2;
    let add1 = (j3 - j2 + m1) / 2;
    let add2 = (j3 - j1 - m2) / 2;

    let kmin = 0i64.max(-add1).max(-add2);
    let kmax = a.min(b).min(c);

    // Each term is (-1)^k / [k! (a-k)! (b-k)! (c-k)! (k+add1)! (k+add2)!]:
    // a prime-factorized numerator (+-1) / denominator, combined by sum_series.
    // (`compute3jseries`.)
    let mut terms = Vec::with_capacity((kmax - kmin + 1).max(0) as usize);
    for k in kmin..=kmax {
        let nump = if k % 2 == 0 {
            Pf::one()
        } else {
            Pf::one().neg()
        };
        let mut denp = pf_factorial(k as u64);
        mul_factorial(&mut denp, (a - k) as u64);
        mul_factorial(&mut denp, (b - k) as u64);
        mul_factorial(&mut denp, (c - k) as u64);
        mul_factorial(&mut denp, (k + add1) as u64);
        mul_factorial(&mut denp, (k + add2) as u64);
        terms.push((nump, denp));
    }
    let mut value = s * sum_series(terms);

    // Overall Condon-Shortley phase (-1)^((j1-j2-m3)/2) folds into the sign.
    if phase_is_negative((j1 - j2 - m3) / 2) {
        value = -value;
    }

    // `r` is the square-free part of a factorial product, hence nonnegative, so
    // the clamp in from_prefactor_radical is never exercised here.
    SignedSqrtRational::from_prefactor_radical(value, r)
}

/// Clebsch-Gordan coefficient `<dj1 dm1, dj2 dm2 | dj3 dm3>` (doubled spins).
///
/// Composed exactly from [`wigner_3j`] via the standard relation
/// `CG = (-1)^((-j1+j2-m3)) sqrt(2 j3 + 1) (j1 j2 j3; m1 m2 -m3)`
/// (multiply the radicand by `dj3+1`, adjust the sign) — no recomputation.
pub fn clebsch_gordan(
    dj1: u32,
    dm1: i32,
    dj2: u32,
    dm2: i32,
    dj3: u32,
    dm3: i32,
) -> SignedSqrtRational {
    let w3 = wigner_3j(dj1, dj2, dj3, dm1, dm2, -dm3);
    if w3.sign() == 0 {
        return SignedSqrtRational::zero();
    }
    // sqrt(2 j3 + 1) folds into the radicand; the CS phase folds into the sign.
    let cg = w3.times_sqrt_int((dj3 + 1) as u64);
    if phase_is_negative(((dj2 as i64) - (dj1 as i64) - (dm3 as i64)) / 2) {
        cg.neg_value()
    } else {
        cg
    }
}

fn admissible_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> bool {
    if dm1 + dm2 + dm3 != 0 {
        return false;
    }
    for (dj, dm) in [(dj1, dm1), (dj2, dm2), (dj3, dm3)] {
        let dj = dj as i64;
        let dm = dm as i64;
        if dm.abs() > dj || (dj + dm) % 2 != 0 {
            return false;
        }
    }
    triangle_ok(dj1, dj2, dj3)
}

/// `(-1)^p < 0`, i.e. `p` odd.
#[inline]
fn phase_is_negative(p: i64) -> bool {
    p.rem_euclid(2) == 1
}

/// Canonical Regge key for a 6j symbol.
///
/// Six nonnegative doubled-integer components (Rasch-Yu canonical form). Every
/// element of a 6j symmetry orbit maps to the same key. Stored losslessly as
/// `u16`; a component exceeding `u16::MAX` yields [`ReggeError::Overflow`]
/// rather than a silent truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Regge6j([u16; 6]);

impl Regge6j {
    /// The six canonical components `(e, l, x, t, b, s)`.
    pub fn components(&self) -> [u16; 6] {
        self.0
    }
}

/// Why a 6j label set has no canonical Regge key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReggeError {
    /// The label set is not an admissible 6j (a triangle parity or triangle
    /// inequality is violated), so the symbol is exactly zero and has no
    /// canonical representative. Keying it anyway would collide with a distinct
    /// admissible symbol and hand its nonzero value to a zero symbol.
    NonAdmissible,
    /// A canonical component exceeds `u16::MAX` (doubled spin beyond the
    /// supported range). Reported rather than silently truncated.
    Overflow,
}

impl std::fmt::Display for ReggeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReggeError::NonAdmissible => write!(f, "non-admissible 6j label set has no Regge key"),
            ReggeError::Overflow => write!(f, "Regge key component exceeds u16::MAX"),
        }
    }
}

impl std::error::Error for ReggeError {}

/// Canonical Regge key for `{dj1 dj2 dj3; dj4 dj5 dj6}`.
///
/// Ports the alpha/beta construction of the reference crate (Rasch-Yu 2003):
/// three "column" sums `alpha` and four triangle sums `beta`, then the six
/// nonnegative differences `alpha - beta` in a fixed order. Widened to `u16`
/// with a checked conversion.
///
/// Only admissible 6j symbols have a key: the four triangles are checked first,
/// so a non-admissible (zero-valued) set returns [`ReggeError::NonAdmissible`]
/// instead of a floored-halving collision with a distinct admissible symbol.
pub fn canonical_regge_6j(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<Regge6j, ReggeError> {
    // Admissibility gate before any halving: the same four triangles as the 6j.
    // This rejects both odd (parity) sums and triangle-inequality violations,
    // so the alpha/beta differences below are all nonnegative integers.
    if !(triangle_ok(dj1, dj2, dj3)
        && triangle_ok(dj1, dj5, dj6)
        && triangle_ok(dj4, dj2, dj6)
        && triangle_ok(dj4, dj5, dj3))
    {
        return Err(ReggeError::NonAdmissible);
    }

    let (dj1, dj2, dj3, dj4, dj5, dj6) = (
        dj1 as i64, dj2 as i64, dj3 as i64, dj4 as i64, dj5 as i64, dj6 as i64,
    );

    // alpha1 <= alpha2 <= alpha3: the three four-term "square" sums (halved).
    let mut alpha = [
        (dj1 + dj2 + dj4 + dj5) / 2,
        (dj1 + dj3 + dj4 + dj6) / 2,
        (dj2 + dj3 + dj5 + dj6) / 2,
    ];
    alpha.sort_unstable();

    // beta1 >= beta2 >= beta3 >= beta4: the four triangle sums (halved),
    // sorted descending.
    let mut beta = [
        (dj1 + dj2 + dj3) / 2,
        (dj1 + dj5 + dj6) / 2,
        (dj2 + dj4 + dj6) / 2,
        (dj3 + dj4 + dj5) / 2,
    ];
    beta.sort_unstable_by(|a, b| b.cmp(a));

    // s = a1-b1, b = a1-b2, t = a1-b3, x = a1-b4, l = a2-b4, e = a3-b4.
    let raw = [
        alpha[2] - beta[3], // e
        alpha[1] - beta[3], // l
        alpha[0] - beta[3], // x
        alpha[0] - beta[2], // t
        alpha[0] - beta[1], // b
        alpha[0] - beta[0], // s
    ];

    let mut out = [0u16; 6];
    for (slot, &v) in out.iter_mut().zip(raw.iter()) {
        // v >= 0 is guaranteed by the admissibility gate above (Rasch-Yu); the
        // only remaining failure is exceeding the u16 storage width.
        debug_assert!(v >= 0, "admissible 6j produced a negative Regge component");
        if v > u16::MAX as i64 {
            return Err(ReggeError::Overflow);
        }
        *slot = v as u16;
    }
    Ok(Regge6j(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn sq(v: &SignedSqrtRational) -> Ratio<BigInt> {
        v.signed_square()
    }

    #[test]
    fn triangle_admissibility() {
        assert!(triangle_ok(1, 1, 2)); // 1/2,1/2,1
        assert!(triangle_ok(2, 2, 2)); // 1,1,1
        assert!(!triangle_ok(1, 1, 1)); // parity fails
        assert!(!triangle_ok(2, 2, 6)); // out of range
    }

    #[test]
    fn six_j_known_value() {
        // {1/2 1/2 1; 1/2 1/2 1} = 1/6, so signed_square = 1/36.
        let v = wigner_6j(1, 1, 2, 1, 1, 2);
        assert_eq!(sq(&v), Ratio::new(BigInt::from(1), BigInt::from(36)));
        assert!((v.to_f64() - 1.0 / 6.0).abs() < 1e-14);
    }

    #[test]
    fn six_j_all_ones() {
        // {1 1 1; 1 1 1} = 1/6.
        let v = wigner_6j(2, 2, 2, 2, 2, 2);
        assert_eq!(sq(&v), Ratio::new(BigInt::from(1), BigInt::from(36)));
    }

    #[test]
    fn six_j_nonadmissible_is_zero() {
        // triangle (1,1,1) parity violation -> zero.
        let v = wigner_6j(1, 1, 1, 1, 1, 1);
        assert_eq!(v, SignedSqrtRational::zero());
    }

    #[test]
    fn three_j_known_value() {
        // (1/2 1/2 1; 1/2 -1/2 0) = +1/sqrt(6), signed_square = +1/6
        // (verified against wigner-symbols 0.5.1 / WignerSymbols.jl).
        let v = wigner_3j(1, 1, 2, 1, -1, 0);
        assert_eq!(sq(&v), Ratio::new(BigInt::from(1), BigInt::from(6)));
    }

    #[test]
    fn three_j_m_sum_nonzero_is_zero() {
        let v = wigner_3j(1, 1, 2, 1, 1, 0);
        assert_eq!(v, SignedSqrtRational::zero());
    }

    #[test]
    fn cg_known_value() {
        // <1/2 1/2, 1/2 -1/2 | 1 0> = 1/sqrt(2), signed_square = 1/2.
        let v = clebsch_gordan(1, 1, 1, -1, 2, 0);
        assert_eq!(sq(&v), Ratio::new(BigInt::from(1), BigInt::from(2)));
        assert!((v.to_f64() - (0.5f64).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn cg_stretched_is_one() {
        // <1/2 1/2, 1/2 1/2 | 1 1> = 1.
        let v = clebsch_gordan(1, 1, 1, 1, 2, 2);
        assert_eq!(sq(&v), Ratio::from(BigInt::from(1)));
        assert!((v.to_f64() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn regge_key_orbit_invariance_small() {
        // An asymmetric admissible 6j whose three "square" sums are NOT all
        // equal, so the alpha ordering genuinely matters. Doubled labels
        // {1 2 2; 3 2 1}. Swapping columns 1<->2 (j2,j5)<->(j3,j6) permutes the
        // alpha entries; the canonical key must be invariant.
        let base = canonical_regge_6j(2, 4, 4, 6, 4, 2).unwrap();
        let swap12 = canonical_regge_6j(2, 4, 4, 6, 2, 4).unwrap();
        assert_eq!(base, swap12);
        // Sanity: this case is not degenerate (the two label sets differ).
        assert_ne!([2u32, 4, 4, 6, 4, 2], [2u32, 4, 4, 6, 2, 4]);
    }

    #[test]
    fn regge_overflow_reported() {
        // A doubled spin large enough to overflow a component -> typed error.
        let big = 200_000u32;
        assert_eq!(
            canonical_regge_6j(big, big, big, big, big, big),
            Err(ReggeError::Overflow)
        );
    }

    #[test]
    fn regge_nonadmissible_is_error_not_a_key() {
        // Parity-violating {1/2 1/2 1/2; ...}: the exact 6j is zero, while the
        // admissible {1 1 1; 1 1 1} is 1/6. Keying the former (floored halving)
        // would collide with the latter and hand a nonzero value to a zero
        // symbol via the publication cache -- exactly the silent-wrong-answer
        // class the crate excludes. So a non-admissible set has no key.
        assert_eq!(
            canonical_regge_6j(1, 1, 1, 1, 1, 1),
            Err(ReggeError::NonAdmissible)
        );
        let admissible = canonical_regge_6j(2, 2, 2, 2, 2, 2);
        assert!(admissible.is_ok());
        assert_ne!(
            canonical_regge_6j(1, 1, 1, 1, 1, 1).ok(),
            admissible.ok(),
            "non-admissible input must not share a key with an admissible one"
        );
    }

    #[test]
    fn regge_triangle_inequality_is_nonadmissible_not_overflow() {
        // (2,2,20) violates the triangle inequality: report non-admissibility,
        // not the u16-overflow variant.
        assert_eq!(
            canonical_regge_6j(2, 2, 20, 2, 2, 2),
            Err(ReggeError::NonAdmissible)
        );
    }
}
