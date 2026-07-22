//! Exact value type for SU(2) recoupling coefficients.
//!
//! Every Wigner 3j/6j/Clebsch-Gordan value equals `sign * sqrt(radicand)` with
//! `radicand` a nonnegative rational over big integers. That is the closed
//! form of the Racah expressions: a triangle/dimension product under a square
//! root times a rational alternating sum. We keep the value in this exact form
//! all the way through and round to `f64` exactly once, in [`SignedSqrtRational::to_f64`].

use std::cell::RefCell;

use num_bigint::{BigInt, BigUint};
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

/// Signed square root of a nonnegative rational: the value `sign * sqrt(radicand)`.
///
/// `sign` is `-1`, `0`, or `+1`; `radicand` is a reduced nonnegative
/// `Ratio<BigInt>`. The zero value has `sign == 0` and `radicand == 0`.
///
/// Exact equality (`Eq`) compares `sign` and `radicand`, so two values are
/// equal iff they denote the same real number (the representation is canonical
/// because `radicand` is reduced and `sign == 0` iff `radicand == 0`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedSqrtRational {
    sign: i8,
    radicand: Ratio<BigInt>,
}

impl SignedSqrtRational {
    /// The exact zero value.
    pub fn zero() -> Self {
        SignedSqrtRational {
            sign: 0,
            radicand: Ratio::zero(),
        }
    }

    /// Construct `s * sqrt(d)` from a rational prefactor `s` (which carries the
    /// sign) and a nonnegative rational `d` under the root.
    ///
    /// The stored form is `sign = sign(s)`, `radicand = s^2 * d`. `d` must be
    /// nonnegative (all triangle/dimension products are); a negative `d` is
    /// clamped to its absolute value rather than panicking, but callers never
    /// pass one.
    pub fn from_prefactor_radical(s: Ratio<BigInt>, d: Ratio<BigInt>) -> Self {
        if s.is_zero() || d.is_zero() {
            return Self::zero();
        }
        let sign: i8 = if s.is_negative() { -1 } else { 1 };
        let d = if d.is_negative() { -d } else { d };
        let radicand = (&s * &s) * d;
        SignedSqrtRational { sign, radicand }
    }

    /// The sign: `-1`, `0`, or `+1`.
    pub fn sign(&self) -> i8 {
        self.sign
    }

    /// The nonnegative radicand `r` such that the value is `sign * sqrt(r)`.
    pub fn radicand(&self) -> &Ratio<BigInt> {
        &self.radicand
    }

    /// The signed square `sign * radicand`. This is the natural exact
    /// comparison key: it equals the reference crate's `signed_sq`, so two
    /// coefficients agree exactly iff their signed squares are equal.
    pub fn signed_square(&self) -> Ratio<BigInt> {
        match self.sign {
            0 => Ratio::zero(),
            1 => self.radicand.clone(),
            _ => -self.radicand.clone(),
        }
    }

    /// Multiply by a real integer `k`: `(sign*sqrt(r)) * k = sign(k)*sign * sqrt(k^2 r)`.
    pub fn scale_int(mut self, k: i64) -> Self {
        if k == 0 || self.sign == 0 {
            return Self::zero();
        }
        if k < 0 {
            self.sign = -self.sign;
        }
        let k2 = BigInt::from(k) * BigInt::from(k);
        self.radicand *= Ratio::from(k2);
        self
    }

    /// Multiply the value by `sqrt(n)` for a nonnegative integer `n`
    /// (`radicand *= n`, sign unchanged). Used to fold a dimension factor
    /// `sqrt(2 j3 + 1)` into a coefficient exactly.
    pub fn times_sqrt_int(mut self, n: u64) -> Self {
        if n == 0 || self.sign == 0 {
            return Self::zero();
        }
        self.radicand *= Ratio::from(BigInt::from(n));
        self
    }

    /// Negate the value (flip the sign).
    pub fn neg_value(mut self) -> Self {
        self.sign = -self.sign;
        self
    }

    /// Correctly-rounded conversion to `f64` — the single rounding point.
    ///
    /// The value is `sign * sqrt(N/D)` with `N, D` nonnegative big integers.
    /// We return the `f64` nearest to that real number (ties to even), i.e.
    /// error <= 0.5 ulp.
    ///
    /// Method (integer square root with scaling, then an exact midpoint test):
    /// pick an even scale `2g` so that `Q = floor(sqrt(floor(N*2^(2g)/D)))`
    /// has 54..=56 bits. Then `Q ≈ sqrt(N/D) * 2^g` and the inner division
    /// floor perturbs `Q` by less than `1/(2*sqrt(scaled)) < 2^-54` of a unit,
    /// so `Q` is the exact `floor(sqrt(N/D) * 2^g)`. Let `mant = Q >> (bits-53)`
    /// be the 53-bit significand floor and `exp` its binary exponent. The
    /// correctly rounded significand is `mant` or `mant+1` depending on whether
    /// the true value lies below or above the midpoint `(mant + 1/2) * 2^exp`.
    /// We decide this by the exact big-integer comparison of `N/D` against
    /// `(2*mant+1)^2 * 2^(2*exp-2)` — no floating point enters the decision, so
    /// the only rounding is the final exactly-representable `mant * 2^exp`.
    pub fn to_f64(&self) -> f64 {
        if self.sign == 0 {
            return 0.0;
        }
        let n = self.radicand.numer();
        let d = self.radicand.denom();
        // radicand is nonnegative and reduced; numer/denom are nonnegative.
        let n = n.magnitude(); // BigUint
        let d = d.magnitude();
        let mag = sqrt_ratio_to_f64(n, d);
        if self.sign < 0 {
            -mag
        } else {
            mag
        }
    }
}

impl std::ops::Mul for SignedSqrtRational {
    type Output = SignedSqrtRational;
    /// `(s1 sqrt(r1)) * (s2 sqrt(r2)) = (s1 s2) sqrt(r1 r2)` — again a
    /// signed square root of a rational, so the product stays exact.
    fn mul(self, other: SignedSqrtRational) -> SignedSqrtRational {
        if self.sign == 0 || other.sign == 0 {
            return Self::zero();
        }
        SignedSqrtRational {
            sign: self.sign * other.sign,
            radicand: self.radicand * other.radicand,
        }
    }
}

/// Nearest `f64` to `sqrt(n/d)` for nonnegative big integers, ties to even.
fn sqrt_ratio_to_f64(n: &BigUint, d: &BigUint) -> f64 {
    if n.is_zero() {
        return 0.0;
    }
    let nb = n.bits() as i64;
    let db = d.bits() as i64;
    // floor(log2(sqrt(n/d))) ≈ (nb - db) / 2; choose g so Q ≈ sqrt(n/d)*2^g
    // lands around 54 bits.
    let mut g: i64 = 54 - (nb - db) / 2;
    let (q, exp) = loop {
        let two_g = 2 * g;
        // scaled = floor(n * 2^(2g) / d)
        let scaled = if two_g >= 0 {
            (n << (two_g as u64)) / d
        } else {
            n / (d << ((-two_g) as u64))
        };
        if scaled.is_zero() {
            // g too small (only possible if n/d rounds below 1 at this scale);
            // push the scale up and retry.
            g += 8;
            continue;
        }
        let q = scaled.sqrt();
        let bits = q.bits() as i64;
        if bits < 54 {
            g += (54 - bits + 1) / 2 + 1;
            continue;
        }
        if bits > 56 {
            g -= (bits - 56 + 1) / 2 + 1;
            continue;
        }
        break (q, g);
    };

    let bits = q.bits() as i64;
    let drop = bits - 53; // >= 1 since bits in 54..=56
    let mant: BigUint = &q >> (drop as u64);
    // value ≈ mant * 2^(drop - exp)
    let value_exp = drop - exp;

    // Exact midpoint decision: compare sqrt(n/d) against (mant + 1/2)*2^value_exp.
    // Squared: n/d  vs  (2*mant+1)^2 * 2^(2*value_exp - 2).
    // Cross-multiply by d and clear the sign of the power of two.
    let two_m1 = (&mant << 1u32) + BigUint::one();
    let mid_sq_int = &two_m1 * &two_m1; // (2*mant+1)^2
    let p = 2 * value_exp - 2;
    let (lhs, rhs) = if p >= 0 {
        (n.clone(), (&mid_sq_int * d) << (p as u64))
    } else {
        (n << ((-p) as u64), &mid_sq_int * d)
    };

    let chosen: BigUint = match lhs.cmp(&rhs) {
        std::cmp::Ordering::Less => mant, // below midpoint -> round down
        std::cmp::Ordering::Greater => mant + BigUint::one(), // above -> round up
        std::cmp::Ordering::Equal => {
            // Exact midpoint: round half to even.
            if (&mant & BigUint::one()).is_zero() {
                mant
            } else {
                mant + BigUint::one()
            }
        }
    };

    // `chosen` has at most 54 bits (53, or 2^53 on carry) so it is exact in f64.
    let mant_f = biguint_to_f64_exact(&chosen);
    scale_pow2(mant_f, value_exp)
}

/// Exact `f64` of a `BigUint` that fits in 54 bits.
fn biguint_to_f64_exact(x: &BigUint) -> f64 {
    // Fits in u64 (<= 2^54); u64->f64 is exact for values < 2^53 and exact for
    // 2^53 and 2^54 as well since they are powers of two with a single bit.
    let digits = x.to_u64_digits();
    match digits.as_slice() {
        [] => 0.0,
        [lo] => *lo as f64,
        // Unreachable for <= 54-bit inputs, but stay total.
        _ => f64::INFINITY,
    }
}

/// Multiply `x` by `2^exp` without libm, staying exact within the normal range.
fn scale_pow2(x: f64, exp: i64) -> f64 {
    // Domain: coefficients are O(1); exp stays well inside f64 range. Chunk the
    // exponent so each factor is an exact power of two.
    let mut r = x;
    let mut e = exp;
    while e > 1023 {
        r *= f64::from_bits(0x7FEu64 << 52); // 2^1023
        e -= 1023;
    }
    while e < -1022 {
        r *= f64::from_bits(1u64 << 52); // 2^-1022 (smallest normal)
        e += 1022;
    }
    r * two_pow_i(e)
}

/// `2^e` for `-1022 <= e <= 1023`, exact.
fn two_pow_i(e: i64) -> f64 {
    let biased = (e + 1023) as u64;
    f64::from_bits(biased << 52)
}

thread_local! {
    static FACT: RefCell<Vec<BigInt>> = RefCell::new(vec![BigInt::one()]);
}

/// `n!` as a big integer, memoized per thread in a growing table.
pub fn factorial(n: u64) -> BigInt {
    FACT.with(|cell| {
        let mut table = cell.borrow_mut();
        while (table.len() as u64) <= n {
            let len = table.len();
            let next = &table[len - 1] * BigInt::from(len as u64);
            table.push(next);
        }
        table[n as usize].clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ssr(s: i64, num: i64, den: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Ratio::new(BigInt::from(s), BigInt::from(1)),
            Ratio::new(BigInt::from(num), BigInt::from(den)),
        )
    }

    #[test]
    fn zero_is_canonical() {
        let z = SignedSqrtRational::zero();
        assert_eq!(z.sign(), 0);
        assert_eq!(z.to_f64(), 0.0);
        // 0 * sqrt(anything) collapses to the canonical zero.
        let also_zero =
            SignedSqrtRational::from_prefactor_radical(Ratio::zero(), Ratio::from(BigInt::from(5)));
        assert_eq!(z, also_zero);
    }

    #[test]
    fn sign_and_signed_square() {
        // value = -1 * sqrt(3/4): sign -1, radicand 3/4, signed_square -3/4.
        let v = ssr(-1, 3, 4);
        assert_eq!(v.sign(), -1);
        assert_eq!(*v.radicand(), Ratio::new(BigInt::from(3), BigInt::from(4)));
        assert_eq!(
            v.signed_square(),
            Ratio::new(BigInt::from(-3), BigInt::from(4))
        );
    }

    #[test]
    fn prefactor_folds_into_radicand() {
        // value = (2/3) * sqrt(5) has radicand (4/9)*5 = 20/9.
        let v = SignedSqrtRational::from_prefactor_radical(
            Ratio::new(BigInt::from(2), BigInt::from(3)),
            Ratio::from(BigInt::from(5)),
        );
        assert_eq!(v.sign(), 1);
        assert_eq!(*v.radicand(), Ratio::new(BigInt::from(20), BigInt::from(9)));
    }

    #[test]
    fn product_multiplies_radicands_and_signs() {
        // (-sqrt(2)) * (+sqrt(8)) = -sqrt(16) = -4.
        let a = ssr(-1, 2, 1);
        let b = ssr(1, 8, 1);
        let p = a * b;
        assert_eq!(p.sign(), -1);
        assert_eq!(*p.radicand(), Ratio::from(BigInt::from(16)));
        assert!((p.to_f64() - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn scale_int_signs_and_squares() {
        // sqrt(3) scaled by -2 = -2 sqrt(3) = -sqrt(12).
        let v = ssr(1, 3, 1).scale_int(-2);
        assert_eq!(v.sign(), -1);
        assert_eq!(*v.radicand(), Ratio::from(BigInt::from(12)));
    }

    #[test]
    fn to_f64_known_values() {
        // sqrt exact perfect squares and simple rationals.
        assert!((ssr(1, 1, 4).to_f64() - 0.5).abs() == 0.0);
        assert!((ssr(1, 9, 1).to_f64() - 3.0).abs() == 0.0);
        assert!((ssr(-1, 1, 6).to_f64() - (-(1.0f64 / 6.0).sqrt())).abs() <= f64::EPSILON);
        // 1/6 exactly: compare to correctly rounded reference.
        let got = ssr(1, 1, 6).to_f64();
        let want = (1.0f64 / 6.0).sqrt();
        assert!((got - want).abs() <= (want * f64::EPSILON));
    }

    #[test]
    fn to_f64_correctly_rounded_against_exact() {
        // For many rationals p/q, our correctly-rounded sqrt must match the
        // nearest f64 computed independently via high-precision-ish check:
        // compare with the two neighbouring doubles of the true value.
        for num in 1u64..60 {
            for den in 1u64..60 {
                let got = SignedSqrtRational::from_prefactor_radical(
                    Ratio::one(),
                    Ratio::new(BigInt::from(num), BigInt::from(den)),
                )
                .to_f64();
                let approx = (num as f64 / den as f64).sqrt();
                // got is correctly rounded; approx is within ~1 ulp. They must
                // be within 1 ulp of each other.
                let ulp = approx.abs() * f64::EPSILON;
                assert!(
                    (got - approx).abs() <= ulp + f64::MIN_POSITIVE,
                    "num={num} den={den} got={got} approx={approx}"
                );
            }
        }
    }

    #[test]
    fn to_f64_big_radicand() {
        // A radicand with hundreds of bits still rounds; sanity vs f64 path.
        let big: BigInt = BigInt::from(2u64).pow(200) * 3;
        let v = SignedSqrtRational::from_prefactor_radical(Ratio::one(), Ratio::from(big.clone()));
        let got = v.to_f64();
        let want = (3.0f64).sqrt() * 2f64.powi(100);
        assert!(
            (got - want).abs() <= want * 4.0 * f64::EPSILON,
            "got={got} want={want}"
        );
    }

    #[test]
    fn factorial_table_grows() {
        assert_eq!(factorial(0), BigInt::one());
        assert_eq!(factorial(5), BigInt::from(120));
        assert_eq!(factorial(10), BigInt::from(3_628_800));
        // reproducible under repeated / out-of-order queries
        assert_eq!(factorial(3), BigInt::from(6));
    }
}
