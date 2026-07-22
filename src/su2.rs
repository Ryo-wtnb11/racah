//! Exact SU(2) recoupling coefficients: Wigner 3j, 6j, Clebsch-Gordan, and the
//! canonical Regge key for 6j symbols.
//!
//! All spins are in the doubled ("twice") convention: `dj = 2j` as `u32`,
//! `dm = 2m` as `i32`. Non-admissible label combinations return the exact zero
//! value (never an error and never a panic), matching the reference-crate
//! semantics. Values are computed as big-rational Racah sums and carried as
//! [`SignedSqrtRational`] until a single final rounding to `f64`.

use num_rational::Ratio;

use crate::cache::{cache_3j, cache_6j, cache_f};
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
///
/// Transparently served from the process-local [`crate::cache`]: an admissible
/// label set is keyed by its canonical Regge class ([`canonical_regge_6j`]) and
/// the exact value is stored once per class (the 6j symmetries preserve value,
/// so no compensation is needed). A label too large to key ([`ReggeError::
/// Overflow`]) or non-admissible falls through to the uncached engine, which is
/// the single source of truth for the value.
pub fn wigner_6j(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> SignedSqrtRational {
    match canonical_regge_6j(dj1, dj2, dj3, dj4, dj5, dj6) {
        Ok(key) => {
            cache_6j().get_or_compute(key, || wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6))
        }
        Err(_) => wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6),
    }
}

/// The uncached Racah single-sum engine behind [`wigner_6j`].
fn wigner_6j_uncached(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> SignedSqrtRational {
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
///
/// Transparently served from the process-local [`crate::cache`]. Unlike 6j, the
/// classical 3j symmetries relate orbit members by a sign, so the cache stores
/// one representative per canonical class and the [`ReggePhase`] from
/// [`canonical_regge_3j`] compensates on both store and retrieval — the stored
/// value is the representative's, and `phase.apply` moves between it and this
/// input's value. A label too large to key or non-admissible falls through to
/// the uncached engine.
pub fn wigner_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> SignedSqrtRational {
    match canonical_regge_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        Ok((key, phase)) => {
            // Stored value is the representative's: value(rep) = phase.apply(value(input)).
            // Retrieval undoes it: value(input) = phase.apply(value(rep)).
            let rep = cache_3j().get_or_compute(key, || {
                phase.apply(wigner_3j_uncached(dj1, dj2, dj3, dm1, dm2, dm3))
            });
            phase.apply(rep)
        }
        Err(_) => wigner_3j_uncached(dj1, dj2, dj3, dm1, dm2, dm3),
    }
}

/// The uncached closed-form engine behind [`wigner_3j`].
fn wigner_3j_uncached(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i32,
    dm2: i32,
    dm3: i32,
) -> SignedSqrtRational {
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

/// SU(2) R-symbol `R^{dj1,dj2}_{dj3}` as a multiplicity-free scalar.
///
/// `(-1)^(j1+j2-j3)` on an admissible fusion triangle, exact `0.0` otherwise.
/// In doubled units the exponent is `(dj1+dj2-dj3)/2`, an integer whenever the
/// triangle is admissible (the parity check guarantees it). The zero on a
/// non-admissible triple mirrors `Nsymbol == 0`, so a caller never multiplies a
/// spurious sign into a forbidden fusion channel.
///
/// (TensorKitSectors tugbK `src/irreps/su2irrep.jl:Rsymbol`: `Nsymbol(...) ||
/// return 0; iseven(sa.j+sb.j-sc.j) ? 1 : -1`.)
pub fn su2_r_symbol(dj1: u32, dj2: u32, dj3: u32) -> f64 {
    if !triangle_ok(dj1, dj2, dj3) {
        return 0.0;
    }
    if phase_is_negative(((dj1 as i64) + (dj2 as i64) - (dj3 as i64)) / 2) {
        -1.0
    } else {
        1.0
    }
}

/// Frobenius-Schur phase of an SU(2) irrep: `(-1)^(2j)` as `+-1.0`.
///
/// Every SU(2) irrep is self-dual; the FS indicator is the sign that
/// distinguishes the orthogonal (integer `j`, `+1`) from the symplectic
/// (half-integer `j`, `-1`) self-duality. In doubled units `2j = dj`, so the
/// phase is simply the parity of `dj`. (TensorKitSectors self-dual convention;
/// see the generic `frobeniusschur` and `SU2Irrep` `dual(s) = s`.)
pub fn su2_frobenius_schur(dj: u32) -> f64 {
    if dj.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Exact SU(2) F-symbol as a [`SignedSqrtRational`] -- the value authority.
///
/// `F = (-1)^(j1+j2+j3+j4) * sqrt((dj5+1)(dj6+1)) * {6j: dj1 dj2 dj5 / dj3 dj4
/// dj6}`, composed exactly: the dimension factor folds into the radicand
/// (`times_sqrt_int`), the phase into the sign, with no intermediate rounding.
///
/// Convention and the exact 6j argument order / phase exponent are derived from
/// the reference chain (verified numerically against TensorKitSectors 0.3.6,
/// max abs error 4.4e-16 over the doubled-spin <= 12 grid). TensorKitSectors
/// tugbK `src/irreps/su2irrep.jl:Fsymbol` is `sqrtdim(s5)*sqrtdim(s6) *
/// racahW(T, j1,j2,j4,j3, j5,j6)`, and WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:racahW(T,j1,j2,J,j3,J12,J23)` is `wigner6j(T,
/// j1,j2,J12,j3,J,J23) * (-1)^(j1+j2+j3+J)`. Substituting racahW's arguments
/// `(j1,j2,J=j4,j3,J12=j5,J23=j6)` gives the 6j `{j1 j2 j5 / j3 j4 j6}` (whose
/// triangle set matches [`wigner_6j`]) and the phase `(-1)^(j1+j2+j3+j4)` --
/// integer-valued exactly when that 6j is admissible, i.e. `(dj1+dj2+dj3+dj4)/2`
/// here.
fn f_symbol_exact(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> SignedSqrtRational {
    let w = wigner_6j(dj1, dj2, dj5, dj3, dj4, dj6);
    if w.sign() == 0 {
        return SignedSqrtRational::zero();
    }
    let v = w
        .times_sqrt_int((dj5 as u64) + 1)
        .times_sqrt_int((dj6 as u64) + 1);
    if phase_is_negative(((dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64)) / 2) {
        v.neg_value()
    } else {
        v
    }
}

/// Multiplicity-free SU(2) F-symbol `F^{dj1 dj2 dj3}_{dj4}[dj5, dj6]` as `f64`.
///
/// The consumer-facing presentation of the exact F-symbol. Consumers need an
/// `f64` scalar per recoupling; rounding the exact value on every call would
/// re-run the big-integer `sqrt` in [`SignedSqrtRational::to_f64`] on the hot
/// path. This tier caches the rounded scalar, so a warm hit is a hash lookup
/// returning a `Copy` `f64` -- the only `to_f64` call happens inside the miss
/// closure, which the private `FifoCache::get_or_compute` provably skips on
/// a hit (see the f64-tier test in `cache.rs`).
///
/// Layering (the reference splits `@cached Fsymbol` from the coefficient
/// caches for the same reason): the exact 6j tier (#5) owns the *value*; this
/// tier owns the *presentation*. The two never disagree because the f64 here is
/// derived from that same exact value, never independently.
pub fn su2_f_symbol(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> f64 {
    // Key on the 6j class actually evaluated, {dj1 dj2 dj5 / dj3 dj4 dj6}, plus
    // the two determinants that class does NOT carry (dimension factor, phase).
    // See FKey for the key-completeness argument.
    match canonical_regge_6j(dj1, dj2, dj5, dj3, dj4, dj6) {
        Ok(regge) => {
            let key = FKey {
                regge,
                dim: ((dj5 as u64) + 1) * ((dj6 as u64) + 1),
                phase_neg: phase_is_negative(
                    ((dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64)) / 2,
                ),
            };
            cache_f().get_or_compute(key, || {
                f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64()
            })
        }
        // Non-admissible 6j (F is exactly 0) or a label too large to key: round
        // the exact value directly. Not worth a cache slot -- zeros are free and
        // overflow-scale labels do not recur in the TN hot loop. Bypassing the
        // tier here also means a truncated/ambiguous key can never be formed.
        Err(_) => f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64(),
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

/// Key for the derived-f64 F-symbol tier ([`su2_f_symbol`], #7).
///
/// # Why these three components, and why they are complete
///
/// `F = phase * sqrt((dj5+1)(dj6+1)) * {6j}`, evaluating the 6j `{dj1 dj2 dj5 /
/// dj3 dj4 dj6}`. Its `f64` value is fixed by exactly three data:
///
/// * `regge` -- the canonical Regge class of that 6j. A 6j is invariant under
///   its full symmetry group, so the class names one exact 6j value (sign
///   included). This is the *only* part shared with the exact 6j tier.
/// * `dim` -- the product `(dj5+1)(dj6+1)` under the square root. F is **not**
///   invariant under the full Regge group of its 6j: two distinct F inputs can
///   share a 6j class yet carry different `(dj5, dj6)`, hence a different
///   radicand factor. Keying on `regge` alone would collide them and hand one
///   F's value to the other. The product -- symmetric in `dj5 <-> dj6`, exactly
///   as the factor is -- restores that missing determinant.
/// * `phase_neg` -- the parity of `(dj1+dj2+dj3+dj4)/2`, the overall sign. Again
///   not carried by the 6j class, and again able to differ between two inputs
///   that share a class.
///
/// Conversely, any two inputs agreeing on all three yield the identical F (same
/// 6j value, same radicand factor, same sign), so the triple is a *complete*
/// key: no collision returns a wrong value and no determinant is omitted.
/// Overflow-scale 6j labels have no Regge class and bypass the tier entirely
/// (computed directly), so a truncated key can never be formed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct FKey {
    regge: Regge6j,
    dim: u64,
    phase_neg: bool,
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

/// Compensation phase relating a 3j symbol to its canonical Regge
/// representative.
///
/// The *classical* 3j symmetry group has 12 elements: the `3!` column
/// permutations and the simultaneous m-negation. Even column permutations and
/// the identity preserve the symbol's value; odd column permutations and
/// m-negation multiply it by `(-1)^(j1+j2+j3)`. When `j1+j2+j3` is even every
/// orbit member has the identical value, so the phase collapses to `Plus`.
/// [`canonical_regge_3j`] returns the single net sign; the cache layer applies
/// it on both store and retrieval to move between an input and its stored
/// representative. (WignerSymbols.jl v2.0.0 `WignerSymbols.jl:reorder3j`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReggePhase {
    /// The input and its canonical representative have equal value.
    Plus,
    /// The input's value is the representative's, negated.
    Minus,
}

impl ReggePhase {
    /// Apply the compensation to a representative value (`+1` or `-1`).
    pub fn apply(self, v: SignedSqrtRational) -> SignedSqrtRational {
        match self {
            ReggePhase::Plus => v,
            ReggePhase::Minus => v.neg_value(),
        }
    }
}

/// Canonical Regge key for a 3j symbol.
///
/// The six doubled labels — three doubled spins `dj` and three doubled
/// projections `dm` — of the classical-orbit representative (the
/// lexicographically maximal image, see `canonicalize3j`). Every
/// element of the 12-element classical symmetry orbit maps to this same key;
/// the value relation is carried separately as a [`ReggePhase`], so the key
/// alone is the orbit invariant.
///
/// `dj` is stored as `u16` with the same checked-widening discipline as
/// [`Regge6j`] (a doubled spin past `u16::MAX` is [`ReggeError::Overflow`]).
/// `dm` is stored signed in `i32`: once the `dj` gate passes, admissibility
/// bounds `|dm| <= dj <= u16::MAX`, so `i32` holds it losslessly.
///
/// Deviation from the reference: WignerSymbols.jl keys on the derived
/// `(β1,β2,β3,α1,α2)` reparametrization of these same canonical labels; racah
/// stores the labels directly (a 1-1 re-encoding), which the orbit test can
/// read against without reconstructing alphas. (WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:_wigner3j`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Regge3j {
    dj: [u16; 3],
    dm: [i32; 3],
}

impl Regge3j {
    /// The canonical doubled spins `(dj1, dj2, dj3)`.
    pub fn doubled_spins(&self) -> [u16; 3] {
        self.dj
    }

    /// The canonical doubled projections `(dm1, dm2, dm3)`.
    pub fn doubled_projections(&self) -> [i32; 3] {
        self.dm
    }
}

/// Canonicalize a doubled-unit 3j label set over its 12-element classical
/// symmetry group, returning the representative labels and the net sign `eps`
/// with `value(input) == eps * value(representative)`.
///
/// The group is `S3` (column permutations) times `Z2` (simultaneous
/// m-negation). Each generator multiplies the symbol by `(-1)^(j1+j2+j3)` when
/// it is an odd column permutation or an m-negation; even permutations and the
/// identity preserve it — the phase rule ported from `WignerSymbols.jl:
/// reorder3j`. The representative is the lexicographically maximal image over
/// all 12 group elements.
///
/// Deviation from the reference: `reorder3j` sorts only by `j` and leaves the
/// m-order among equal-`j` columns unbroken, so it is *not* a full orbit
/// invariant (two column-permutations of an equal-`j` symbol get distinct
/// keys). racah completes the canonicalization — breaking every tie and
/// tracking the extra permutation's sign — so the acceptance-required
/// "all 12 images share one key" holds, at the cost of enumerating 12 tuples
/// (a fixed, tiny constant) instead of the reference's bubble sort.
fn canonicalize3j(dj: [i64; 3], dm: [i64; 3]) -> ([i64; 3], [i64; 3], i8) {
    // (column index map, whether the permutation is odd).
    const PERMS: [([usize; 3], bool); 6] = [
        ([0, 1, 2], false),
        ([0, 2, 1], true),
        ([1, 0, 2], true),
        ([1, 2, 0], false),
        ([2, 0, 1], false),
        ([2, 1, 0], true),
    ];
    let j_odd = ((dj[0] + dj[1] + dj[2]) / 2) % 2 != 0;

    let mut best: Option<([i64; 3], [i64; 3])> = None;
    let mut best_eps = 1i8;
    for (p, p_odd) in PERMS {
        let pj = [dj[p[0]], dj[p[1]], dj[p[2]]];
        let pm = [dm[p[0]], dm[p[1]], dm[p[2]]];
        for neg in [false, true] {
            let m = if neg { [-pm[0], -pm[1], -pm[2]] } else { pm };
            // value(image)/value(input): (-1)^(j1+j2+j3) once per odd generator
            // (odd permutation and/or m-negation), and +1 whenever J is even.
            let odd_ops = (p_odd ^ neg) as i8;
            let eps: i8 = if j_odd && odd_ops == 1 { -1 } else { 1 };
            let cand = (pj, m);
            if best.as_ref().is_none_or(|b| cand > *b) {
                best = Some(cand);
                best_eps = eps;
            }
        }
    }
    let (bj, bm) = best.expect("the identity image always seeds `best`");
    // eps relates input and representative (±1, self-inverse).
    (bj, bm, best_eps)
}

/// Canonical Regge key and compensation phase for `(dj1 dj2 dj3; dm1 dm2 dm3)`.
///
/// Admissibility is gated first — exactly like [`canonical_regge_6j`] and for
/// the same reason (the `(1,1,1,1,1,1)`-class lesson): a non-admissible 3j is
/// exactly zero and has no representative, so it must never produce a key that
/// could collide with a distinct nonzero symbol. Only then is the label set
/// reordered to its canonical representative and widened losslessly.
///
/// Returns `(key, phase)` such that `value(input) == phase.apply(value(rep))`
/// where `rep` is the symbol named by `key`. (WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:_wigner3j` / `reorder3j`.)
pub fn canonical_regge_3j(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i32,
    dm2: i32,
    dm3: i32,
) -> Result<(Regge3j, ReggePhase), ReggeError> {
    if !admissible_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        return Err(ReggeError::NonAdmissible);
    }

    let (dj, dm, sign) = canonicalize3j(
        [dj1 as i64, dj2 as i64, dj3 as i64],
        [dm1 as i64, dm2 as i64, dm3 as i64],
    );

    // dj values only reorder under canonicalization, but widen with the same
    // checked u16 discipline as Regge6j. dm is bounded by dj (admissibility), so
    // once the dj gate passes it fits i32 with room to spare; store it signed.
    let mut dju = [0u16; 3];
    for (slot, &v) in dju.iter_mut().zip(dj.iter()) {
        if v > u16::MAX as i64 {
            return Err(ReggeError::Overflow);
        }
        *slot = v as u16;
    }
    let dmi = [dm[0] as i32, dm[1] as i32, dm[2] as i32];

    let phase = if sign < 0 {
        ReggePhase::Minus
    } else {
        ReggePhase::Plus
    };
    Ok((Regge3j { dj: dju, dm: dmi }, phase))
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

    /// The six column permutations (as index maps) times the m-negation give
    /// the 12-element classical 3j orbit.
    fn orbit_images(dj: [u32; 3], dm: [i32; 3]) -> Vec<([u32; 3], [i32; 3])> {
        const PERMS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut out = Vec::with_capacity(12);
        for p in PERMS {
            let pj = [dj[p[0]], dj[p[1]], dj[p[2]]];
            let pm = [dm[p[0]], dm[p[1]], dm[p[2]]];
            out.push((pj, pm));
            out.push((pj, [-pm[0], -pm[1], -pm[2]]));
        }
        out
    }

    #[test]
    fn regge3j_orbit_same_key_and_phase_compensated_value() {
        // {1 1 1; 1 0 -1} (doubled 2,2,2 / 2,0,-2). J = 3 is ODD, so the phase
        // is nontrivial: odd column permutations and m-negation flip the sign.
        let dj = [2u32, 2, 2];
        let dm = [2i32, 0, -2];

        let mut key0 = None;
        let mut rep_value = None;
        let mut saw_negative_phase = false;
        for (pj, pm) in orbit_images(dj, dm) {
            let (key, phase) =
                canonical_regge_3j(pj[0], pj[1], pj[2], pm[0], pm[1], pm[2]).unwrap();
            // All 12 images share the canonical key.
            match key0 {
                None => key0 = Some(key),
                Some(k) => assert_eq!(k, key, "orbit image produced a different key"),
            }
            // phase.apply(raw value) recovers the single representative value.
            let raw = wigner_3j(pj[0], pj[1], pj[2], pm[0], pm[1], pm[2]);
            let compensated = phase.apply(raw.clone());
            match &rep_value {
                None => rep_value = Some(compensated),
                Some(rv) => assert_eq!(
                    rv, &compensated,
                    "phase-compensated value differs across the orbit"
                ),
            }
            if phase == ReggePhase::Minus {
                saw_negative_phase = true;
                // The phase genuinely acts: raw and compensated differ in sign.
                assert_ne!(raw, phase.apply(raw.clone()));
            }
        }
        assert!(
            saw_negative_phase,
            "J-odd orbit must exercise the Minus phase"
        );
    }

    #[test]
    fn regge3j_even_j_phase_is_always_plus() {
        // {1 1 2; 1 -1 0} (doubled 2,2,4 / 2,-2,0). J = 4 is EVEN, so every
        // orbit member has the identical value and the phase is always Plus.
        let dj = [2u32, 2, 4];
        let dm = [2i32, -2, 0];
        for (pj, pm) in orbit_images(dj, dm) {
            let (_, phase) = canonical_regge_3j(pj[0], pj[1], pj[2], pm[0], pm[1], pm[2]).unwrap();
            assert_eq!(phase, ReggePhase::Plus, "even-J phase must be Plus");
        }
    }

    #[test]
    fn regge3j_nonadmissible_is_error_not_a_key() {
        // m-sum nonzero: not an admissible 3j, so no representative and no key.
        assert_eq!(
            canonical_regge_3j(2, 2, 2, 2, 2, 0),
            Err(ReggeError::NonAdmissible)
        );
        // |m| > j.
        assert_eq!(
            canonical_regge_3j(2, 2, 2, 4, -2, -2),
            Err(ReggeError::NonAdmissible)
        );
        // Triangle parity violation.
        assert_eq!(
            canonical_regge_3j(1, 1, 1, 1, -1, 0),
            Err(ReggeError::NonAdmissible)
        );
    }

    #[test]
    fn cached_3j_matches_uncached_over_grid() {
        // Every admissible and non-admissible small 3j: the transparently cached
        // public value must equal the uncached engine, and a repeat call (a
        // cache hit) must still match. This exercises the ReggePhase
        // compensation end-to-end across the whole orbit structure.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dm1 in -4..=4i32 {
                        for dm2 in -4..=4i32 {
                            for dm3 in -4..=4i32 {
                                let raw = wigner_3j_uncached(dj1, dj2, dj3, dm1, dm2, dm3);
                                let ctx = (dj1, dj2, dj3, dm1, dm2, dm3);
                                assert_eq!(
                                    wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3),
                                    raw,
                                    "cached != uncached at {ctx:?}"
                                );
                                assert_eq!(
                                    wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3),
                                    raw,
                                    "cache-hit != uncached at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_6j_matches_uncached_over_grid() {
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let raw = wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6);
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(
                                    wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6),
                                    raw,
                                    "cached != uncached at {ctx:?}"
                                );
                                assert_eq!(
                                    wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6),
                                    raw,
                                    "cache-hit != uncached at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn public_stats_and_reset_smoke() {
        // The global wrappers aggregate the per-kind caches (rigorously covered
        // in cache::tests); here just confirm they observe activity and reset
        // without panicking. Counts are only checked monotonically because the
        // global cache is shared with other tests running in parallel.
        let _ = wigner_6j(2, 2, 2, 2, 2, 2);
        let _ = wigner_3j(2, 2, 2, 2, 0, -2);
        let s = crate::cache::stats();
        assert!(s.hits + s.misses >= 1, "activity should register in stats");
        crate::cache::reset();
        let _ = crate::cache::stats();
    }

    #[test]
    fn regge3j_overflow_reported() {
        let big = 200_000u32;
        assert_eq!(
            canonical_regge_3j(big, big, big, 0, 0, 0),
            Err(ReggeError::Overflow)
        );
    }

    #[test]
    fn f_symbol_exact_composition_identity() {
        // Independent exact check of the closed form. Reconstruct signed_square(F)
        // separately from the 6j value, the dimension factor, and the phase --
        // NOT by re-running f_symbol_exact -- and require exact equality:
        //   signed_square(F) == phase * (dj5+1)(dj6+1) * signed_square({6j}).
        // This catches a wrong 6j argument order or a wrong phase exponent (the
        // silent-wrong-answer class) because a mistaken order lands on a
        // different 6j class whose signed_square differs on non-symmetric labels.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let f = f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6);
                                let w = wigner_6j(dj1, dj2, dj5, dj3, dj4, dj6);
                                let dim = BigInt::from(((dj5 + 1) * (dj6 + 1)) as i64);
                                let mut expected = w.signed_square() * Ratio::from(dim);
                                let sum = (dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64);
                                if (sum / 2).rem_euclid(2) == 1 {
                                    expected = -expected;
                                }
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(f.signed_square(), expected, "F^2 identity at {ctx:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f_symbol_cached_matches_exact_over_grid() {
        // The cached f64 path equals the rounded exact composition, and a repeat
        // (a cache hit) still matches.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let want = f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64();
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(
                                    su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6),
                                    want,
                                    "cached != exact at {ctx:?}"
                                );
                                assert_eq!(
                                    su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6),
                                    want,
                                    "cache-hit != exact at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f_symbol_all_trivial_is_one() {
        // All six labels trivial: {0 0 0 / 0 0 0} 6j = 1, dims = 1, phase +.
        assert_eq!(su2_f_symbol(0, 0, 0, 0, 0, 0), 1.0);
    }

    #[test]
    fn f_symbol_nonadmissible_is_zero() {
        // The evaluated 6j {1/2 1/2 1/2 / 1/2 1/2 1/2} is parity-forbidden, so F
        // is exactly zero and takes the Err (bypass) branch.
        assert_eq!(su2_f_symbol(1, 1, 1, 1, 1, 1), 0.0);
    }

    #[test]
    fn r_symbol_matches_convention() {
        // R^{ab}_c = (-1)^(j1+j2-j3) on an admissible triangle, else 0.
        // (Oracle: TensorKitSectors Rsymbol, exact over the full dj<=12 grid.)
        // {1/2 1/2 1}: (1+1-2)/2 = 0 even -> +1.
        assert_eq!(su2_r_symbol(1, 1, 2), 1.0);
        // {1/2 1/2 0}: (1+1-0)/2 = 1 odd -> -1.
        assert_eq!(su2_r_symbol(1, 1, 0), -1.0);
        // {1 1 1}: doubled (2,2,2), (2+2-2)/2 = 1 odd -> -1.
        assert_eq!(su2_r_symbol(2, 2, 2), -1.0);
        // {1 1 2}: doubled (2,2,4), (2+2-4)/2 = 0 even -> +1.
        assert_eq!(su2_r_symbol(2, 2, 4), 1.0);
        // Non-admissible (parity) -> exact zero, no phase.
        assert_eq!(su2_r_symbol(1, 1, 1), 0.0);
        // Non-admissible (triangle inequality) -> exact zero.
        assert_eq!(su2_r_symbol(2, 2, 8), 0.0);
    }

    #[test]
    fn frobenius_schur_is_sign_of_doubled_spin() {
        // (-1)^(2j) as +-1: integer spins +1, half-integer spins -1.
        assert_eq!(su2_frobenius_schur(0), 1.0); // j=0
        assert_eq!(su2_frobenius_schur(1), -1.0); // j=1/2
        assert_eq!(su2_frobenius_schur(2), 1.0); // j=1
        assert_eq!(su2_frobenius_schur(3), -1.0); // j=3/2
        assert_eq!(su2_frobenius_schur(4), 1.0); // j=2
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
