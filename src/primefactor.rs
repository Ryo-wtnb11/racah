//! Prime-factorized integer arithmetic for the Racah/Wigner factorial engine.
//!
//! Ported from WignerSymbols.jl v2.0.0 `src/primefactorization.jl`
//! (`PrimeFactorization`, `primefactor`, `primefactorial`, `mul!`,
//! `divexact!`, `gcd!`, `lcm!`, `divgcd!`, `splitsquare`). A nonnegative
//! integer is stored as the exponent vector of its prime factorization plus a
//! sign; products and exact quotients are integer vector add/sub, so no big
//! integer is multiplied until a single final reconstruction. Factorials and
//! their ratios (the whole cost of a 3j/6j evaluation) stay in exponent space.
//!
//! ## Thread-safety and the memory contract
//!
//! Two process-global tables grow monotonically and are shared across threads:
//! the prime list and the factorial exponent-vector table. Both are guarded by
//! an [`RwLock`]: the common case takes a read lock and returns a cached row;
//! only extending the table past the largest label seen takes the write lock.
//! WignerSymbols.jl uses the same growing-global-table design (its
//! `GrowingList`); an `RwLock` is the direct thread-safe Rust equivalent.
//!
//! The tables never shrink. Their size is bounded by the largest factorial
//! argument `N` any call has requested: the factorial table holds `N + 1` rows
//! of `~pi(N)` `u32` exponents each, and the prime list holds `~pi(N)` values.
//! For doubled spins in the thousands `N` is a few thousand, so the tables are
//! a few megabytes and are amortized across every subsequent symbol.

use std::sync::RwLock;

use num_bigint::BigInt;
use num_traits::{One, Zero};

/// A signed integer stored as `sign * prod_i prime(i)^powers[i]`.
///
/// `powers[i]` is the exponent of the `i`-th prime (`powers[0]` -> 2). The
/// vector is normalized so its last entry is nonzero (an empty vector is the
/// integer 1, or 0 when `sign == 0`). `sign` is `-1`, `0`, or `+1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pf {
    powers: Vec<u32>,
    sign: i8,
}

impl Pf {
    /// The integer 1.
    pub(crate) fn one() -> Self {
        Pf {
            powers: Vec::new(),
            sign: 1,
        }
    }

    /// Wrap a normalized (trailing-zero-trimmed) positive exponent vector.
    fn from_powers(mut powers: Vec<u32>) -> Self {
        trim(&mut powers);
        Pf { powers, sign: 1 }
    }

    /// Negate in place (flip the sign).
    pub(crate) fn neg(mut self) -> Self {
        self.sign = -self.sign;
        self
    }

    fn is_zero(&self) -> bool {
        self.sign == 0
    }

    /// `self *= other` — exponent vectors add, signs multiply
    /// (`primefactorization.jl::mul!`).
    pub(crate) fn mul_assign(&mut self, other: &Pf) {
        if self.sign == 0 || other.sign == 0 {
            self.sign = 0;
            self.powers.clear();
            return;
        }
        self.sign *= other.sign;
        if other.powers.len() > self.powers.len() {
            self.powers.resize(other.powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a += *b;
        }
    }

    /// `self *= prod_i prime(i)^powers[i]`, folding a raw positive exponent
    /// slice (a factorial table row) straight into `self` without allocating a
    /// [`Pf`] for it. This is the clone-free multiplicand path: callers
    /// accumulate a product of factorials in one buffer instead of cloning each
    /// `O(pi(N))` row only to add and drop it. Sign is unchanged (the slice is a
    /// factorial, hence positive).
    fn mul_by_powers(&mut self, powers: &[u32]) {
        if self.sign == 0 {
            return;
        }
        if powers.len() > self.powers.len() {
            self.powers.resize(powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(powers.iter()) {
            *a += *b;
        }
    }

    /// `self /= other`, exact division — exponent vectors subtract
    /// (`primefactorization.jl::divexact!`). `other` must divide `self`, which
    /// is guaranteed at every call site (a factorial ratio, or a divisor pulled
    /// out by gcd/lcm); the debug assertion documents that invariant.
    pub(crate) fn divexact_assign(&mut self, other: &Pf) {
        debug_assert!(other.sign != 0, "exact division by zero");
        if self.sign == 0 {
            return;
        }
        self.sign *= other.sign;
        debug_assert!(
            other.powers.len() <= self.powers.len(),
            "divexact: divisor has a prime the dividend lacks"
        );
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            debug_assert!(*a >= *b, "divexact: non-divisible exponent");
            *a -= *b;
        }
        trim(&mut self.powers);
    }

    /// `self = lcm(self, other)` (positive) — exponent-wise max
    /// (`primefactorization.jl::lcm!`).
    fn lcm_assign(&mut self, other: &Pf) {
        self.sign = 1;
        if other.powers.len() > self.powers.len() {
            self.powers.resize(other.powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// `self = gcd(self, other)` (positive) — exponent-wise min
    /// (`primefactorization.jl::gcd!`).
    fn gcd_assign(&mut self, other: &Pf) {
        self.sign = 1;
        let l = self.powers.len().min(other.powers.len());
        self.powers.truncate(l);
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a = (*a).min(*b);
        }
        trim(&mut self.powers);
    }

    /// Split off the common gcd of `a` and `b` in place, so afterwards
    /// `gcd(a, b) == 1` (`primefactorization.jl::divgcd!`). Keeps the two
    /// reconstructed big integers as small as possible.
    pub(crate) fn divgcd(a: &mut Pf, b: &mut Pf) {
        let l = a.powers.len().min(b.powers.len());
        for k in 0..l {
            let g = a.powers[k].min(b.powers[k]);
            a.powers[k] -= g;
            b.powers[k] -= g;
        }
        trim(&mut a.powers);
        trim(&mut b.powers);
    }

    /// Split `a = s^2 * r` with `r` square-free (all exponents 0 or 1)
    /// (`primefactorization.jl::splitsquare`). `r` carries the sign.
    pub(crate) fn splitsquare(&self) -> (Pf, Pf) {
        let s = Pf::from_powers(self.powers.iter().map(|&e| e >> 1).collect());
        let mut r = Pf::from_powers(self.powers.iter().map(|&e| e & 1).collect());
        r.sign = self.sign;
        (s, r)
    }

    /// Reconstruct the big integer (`primefactorization.jl::_convert!`). This is
    /// the only place a prime is raised to a power and multiplied out.
    pub(crate) fn to_bigint(&self) -> BigInt {
        if self.sign == 0 {
            return BigInt::zero();
        }
        let mut acc = BigInt::one();
        for (i, &e) in self.powers.iter().enumerate() {
            if e != 0 {
                acc *= BigInt::from(nth_prime(i)).pow(e);
            }
        }
        if self.sign < 0 {
            -acc
        } else {
            acc
        }
    }
}

/// Drop trailing zero exponents so the representation is canonical.
fn trim(powers: &mut Vec<u32>) {
    while matches!(powers.last(), Some(0)) {
        powers.pop();
    }
}

// ---------------------------------------------------------------------------
// Growing global tables (prime list, factorial exponent vectors).
// ---------------------------------------------------------------------------

static PRIMES: RwLock<Vec<u64>> = RwLock::new(Vec::new());
static FACT: RwLock<Vec<Vec<u32>>> = RwLock::new(Vec::new());

/// The `idx`-th prime (`idx == 0` -> 2), extending the shared list if needed
/// (`primefactorization.jl::prime`).
fn nth_prime(idx: usize) -> u64 {
    {
        let primes = PRIMES.read().unwrap();
        if idx < primes.len() {
            return primes[idx];
        }
    }
    let mut primes = PRIMES.write().unwrap();
    if primes.is_empty() {
        primes.push(2);
    }
    while primes.len() <= idx {
        // Trial-divide by the already-known primes: all primes up to
        // sqrt(candidate) are present because the candidate starts just above
        // the current largest prime.
        let mut cand = primes[primes.len() - 1] + 1;
        while !is_prime_by(cand, &primes) {
            cand += 1;
        }
        primes.push(cand);
    }
    primes[idx]
}

fn is_prime_by(n: u64, primes: &[u64]) -> bool {
    for &p in primes {
        if p * p > n {
            break;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    true
}

/// Exponent vector of the prime factorization of `n >= 1`
/// (`primefactorization.jl::primefactor`).
fn primefactor_powers(mut n: u64) -> Vec<u32> {
    let mut powers = Vec::new();
    let mut idx = 0;
    while n > 1 {
        let p = nth_prime(idx);
        let mut e = 0u32;
        while n.is_multiple_of(p) {
            n /= p;
            e += 1;
        }
        powers.push(e);
        idx += 1;
    }
    trim(&mut powers);
    powers
}

/// Extend the shared factorial table so row `n` exists
/// (`primefactorization.jl::primefactorial`, growth half). Built incrementally:
/// `factorial(m) = factorial(m-1) * primefactor(m)` in exponent space.
fn grow_factorial(n: usize) {
    let mut table = FACT.write().unwrap();
    if table.is_empty() {
        table.push(Vec::new()); // 0! = 1
    }
    while table.len() <= n {
        let m = table.len() as u64;
        let fm = primefactor_powers(m);
        let mut next = table[table.len() - 1].clone();
        if fm.len() > next.len() {
            next.resize(fm.len(), 0);
        }
        for (a, b) in next.iter_mut().zip(fm.iter()) {
            *a += *b;
        }
        table.push(next);
    }
}

/// Multiply `acc` by `n!` in place. The common path takes only a read lock and
/// folds the memoized exponent row into `acc` with no per-call allocation or
/// row clone -- the whole point of the prime-factorized engine is that a
/// product of factorials costs one buffer, not one clone per factor.
pub(crate) fn mul_factorial(acc: &mut Pf, n: u64) {
    let n = n as usize;
    {
        let table = FACT.read().unwrap();
        if n < table.len() {
            acc.mul_by_powers(&table[n]);
            return;
        }
    }
    grow_factorial(n);
    let table = FACT.read().unwrap();
    acc.mul_by_powers(&table[n]);
}

/// `n!` as a fresh [`Pf`]. A thin wrapper over [`mul_factorial`] for the base
/// factor of a product (and the tests); multiplicands should use
/// [`mul_factorial`] to stay clone-free.
pub(crate) fn factorial(n: u64) -> Pf {
    let mut p = Pf::one();
    mul_factorial(&mut p, n);
    p
}

// ---------------------------------------------------------------------------
// Series over a common denominator (compute3jseries / compute6jseries core).
// ---------------------------------------------------------------------------

/// Sum a list of exact fractions `nums[i] / dens[i]`, each already prime
/// factorized, into one reduced [`num_bigint`] rational
/// (`primefactorization.jl::commondenominator!` + `sumlist!`,
/// `WignerSymbols.jl::compute{3,6}jseries`).
///
/// Puts every term over the lcm of the denominators, factors the gcd out of the
/// numerators before reconstruction, sums the small remainders as big integers,
/// then multiplies the common factor back. Big-integer work is confined here.
pub(crate) fn sum_series(mut terms: Vec<(Pf, Pf)>) -> num_rational::Ratio<BigInt> {
    use num_rational::Ratio;
    if terms.is_empty() {
        return Ratio::zero();
    }

    // Common denominator = lcm of all denominators; rescale each numerator.
    let mut den = terms[0].1.clone();
    for (_, d) in &terms[1..] {
        den.lcm_assign(d);
    }
    for (num, d) in &mut terms {
        num.mul_assign(&den);
        num.divexact_assign(d);
    }

    // Pull the gcd of all numerators out before reconstructing big integers, so
    // the summed integers stay as small as the series allows (`sumlist!`).
    let mut g = terms[0].0.clone();
    for (num, _) in &terms[1..] {
        g.gcd_assign(num);
    }
    let mut total = BigInt::zero();
    for (num, _) in &mut terms {
        if !g.is_zero() {
            num.divexact_assign(&g);
        }
        total += num.to_bigint();
    }
    total *= g.to_bigint();

    Ratio::new(total, den.to_bigint())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    /// Direct big-integer factorial, independent of the prime-factorized path.
    fn direct_factorial(n: u64) -> BigInt {
        let mut f = BigInt::one();
        for k in 2..=n {
            f *= BigInt::from(k);
        }
        f
    }

    #[test]
    fn nth_prime_matches_known() {
        let known = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        for (i, &p) in known.iter().enumerate() {
            assert_eq!(nth_prime(i), p, "prime index {i}");
        }
    }

    #[test]
    fn primefactor_reconstructs() {
        for n in 1u64..=200 {
            let pf = Pf::from_powers(primefactor_powers(n));
            assert_eq!(pf.to_bigint(), BigInt::from(n), "factoring {n}");
        }
    }

    #[test]
    fn factorial_exponents_vs_direct() {
        // The core TDD gate: prime-factorized n! reconstructs to the directly
        // computed n! for every n we care about at small scale.
        for n in 0u64..=30 {
            assert_eq!(
                factorial(n).to_bigint(),
                direct_factorial(n),
                "factorial({n})"
            );
        }
        // Out-of-order / repeated queries hit the memoized table.
        assert_eq!(factorial(10).to_bigint(), direct_factorial(10));
        assert_eq!(factorial(0).to_bigint(), BigInt::one());
        assert_eq!(factorial(1).to_bigint(), BigInt::one());
    }

    #[test]
    fn factorial_large_still_exact() {
        assert_eq!(factorial(100).to_bigint(), direct_factorial(100));
        assert_eq!(factorial(257).to_bigint(), direct_factorial(257));
    }

    #[test]
    fn mul_factorial_accumulates_clone_free() {
        // Accumulate 3! * 5! * 7! into one buffer via mul_factorial, and check
        // it equals the same product built by materializing each factorial.
        let mut acc = Pf::one();
        mul_factorial(&mut acc, 3);
        mul_factorial(&mut acc, 5);
        mul_factorial(&mut acc, 7);
        assert_eq!(
            acc.to_bigint(),
            direct_factorial(3) * direct_factorial(5) * direct_factorial(7)
        );
        // factorial(n) is the same as mul_factorial into one().
        assert_eq!(factorial(12).to_bigint(), {
            let mut p = Pf::one();
            mul_factorial(&mut p, 12);
            p.to_bigint()
        });
    }

    #[test]
    fn mul_and_divexact_roundtrip() {
        let a = factorial(12);
        let b = factorial(7);
        let mut p = a.clone();
        p.mul_assign(&b);
        assert_eq!(
            p.to_bigint(),
            direct_factorial(12) * direct_factorial(7),
            "12! * 7!"
        );
        p.divexact_assign(&b);
        assert_eq!(p.to_bigint(), a.to_bigint(), "(12! * 7!) / 7! == 12!");
    }

    #[test]
    fn divgcd_reduces_to_coprime() {
        let mut a = factorial(9); // 9!
        let mut b = factorial(6); // 6!
        Pf::divgcd(&mut a, &mut b);
        // gcd(9!,6!) = 6!, so a = 9!/6! = 7*8*9 = 504, b = 1.
        assert_eq!(a.to_bigint(), BigInt::from(504));
        assert_eq!(b.to_bigint(), BigInt::one());
    }

    #[test]
    fn splitsquare_roundtrips() {
        // a = s^2 * r with r square-free; check for a range of integers.
        for n in 1u64..=120 {
            let a = Pf::from_powers(primefactor_powers(n));
            let (s, r) = a.splitsquare();
            let recon = s.to_bigint().pow(2) * r.to_bigint();
            assert_eq!(recon, BigInt::from(n), "splitsquare {n}");
        }
        // A factorial: (2n)! splits and reconstructs.
        let a = factorial(20);
        let (s, r) = a.splitsquare();
        assert_eq!(s.to_bigint().pow(2) * r.to_bigint(), direct_factorial(20));
    }

    #[test]
    fn sign_propagates() {
        let mut a = factorial(5);
        a = a.neg();
        assert_eq!(a.to_bigint(), -direct_factorial(5));
        let b = factorial(3).neg();
        a.mul_assign(&b); // (-5!) * (-3!) = +5!*3!
        assert_eq!(a.to_bigint(), direct_factorial(5) * direct_factorial(3));
    }

    #[test]
    fn sum_series_matches_direct_rational() {
        use num_rational::Ratio;
        // Sum 1/3! - 1/(2!*4!) + 3!/(5!) as prime-factorized terms.
        let terms = vec![
            (Pf::one(), factorial(3)),
            (
                {
                    let mut n = Pf::one();
                    n = n.neg();
                    n
                },
                {
                    let mut d = factorial(2);
                    d.mul_assign(&factorial(4));
                    d
                },
            ),
            (factorial(3), factorial(5)),
        ];
        let got = sum_series(terms);
        let want = Ratio::new(BigInt::one(), BigInt::from(6))
            - Ratio::new(BigInt::one(), BigInt::from(48))
            + Ratio::new(BigInt::from(6), BigInt::from(120));
        assert_eq!(got, want);
    }
}
