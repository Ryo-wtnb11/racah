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
///
/// This is *the* triangle predicate: every admissibility rule in the crate
/// (the infallible 3j/6j engines, the Regge canonicalizers, and the checked
/// surface via [`check_triangle`]) reduces its triangle conditions to this one
/// function, so the checked and unchecked paths cannot disagree.
fn triangle_ok(a: u32, b: u32, c: u32) -> bool {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
}

/// Typed wrapper over [`triangle_ok`] for the checked surface: `Ok(())` when the
/// triple is admissible, else the specific [`AdmissibilityViolation::Triangle`].
/// Routes through the same [`triangle_ok`] the infallible engines use, so the
/// admissibility logic is shared rather than duplicated.
fn check_triangle(a: u32, b: u32, c: u32) -> Result<(), AdmissibilityViolation> {
    if triangle_ok(a, b, c) {
        Ok(())
    } else {
        Err(AdmissibilityViolation::Triangle { a, b, c })
    }
}

/// The four triangle couplings a 6j `{dj1 dj2 dj3; dj4 dj5 dj6}` must satisfy,
/// as the typed shared predicate for the checked surface. Mirrors exactly the
/// set gated by [`wigner_6j_uncached`] and [`canonical_regge_6j`], each atom
/// delegating to [`triangle_ok`].
fn check_6j_admissible(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<(), AdmissibilityViolation> {
    check_triangle(dj1, dj2, dj3)?;
    check_triangle(dj1, dj5, dj6)?;
    check_triangle(dj4, dj2, dj6)?;
    check_triangle(dj4, dj5, dj3)?;
    Ok(())
}

/// Wigner 6j symbol $\{dj_1\, dj_2\, dj_3;\, dj_4\, dj_5\, dj_6\}$ (doubled spins).
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

/// Wigner 3j symbol $(dj_1\, dj_2\, dj_3;\, dm_1\, dm_2\, dm_3)$ (doubled spins/projections).
///
/// Returns exact zero unless the labels are admissible: triangle $(1,2,3)$,
/// $|dm_i| \le dj_i$, $dj_i + dm_i$ even for each $i$, and $dm_1+dm_2+dm_3 = 0$.
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

/// Clebsch-Gordan coefficient $\langle dj_1\, dm_1, dj_2\, dm_2 \,|\, dj_3\, dm_3 \rangle$ (doubled spins).
///
/// Composed exactly from [`wigner_3j`] via the standard relation
/// $CG = (-1)^{-j_1+j_2-m_3}\, \sqrt{2 j_3 + 1}\; (j_1\, j_2\, j_3;\, m_1\, m_2\, -m_3)$
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

/// SU(2) R-symbol $R^{dj_1,dj_2}_{dj_3}$ as a multiplicity-free scalar.
///
/// $(-1)^{j_1+j_2-j_3}$ on an admissible fusion triangle, exact `0.0` otherwise.
/// In doubled units the exponent is $(dj_1+dj_2-dj_3)/2$, an integer whenever the
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

/// Frobenius-Schur phase of an SU(2) irrep: $(-1)^{2j}$ as `+-1.0`.
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
/// $F = (-1)^{j_1+j_2+j_3+j_4}\, \sqrt{(dj_5+1)(dj_6+1)}\; \{6j\!: dj_1\, dj_2\, dj_5 / dj_3\, dj_4\, dj_6\}$,
/// composed exactly: the dimension factor folds into the radicand
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

/// Typed 3j admissibility predicate — the single source of truth shared by the
/// infallible engine ([`wigner_3j_uncached`] via [`admissible_3j`]) and the
/// checked surface ([`wigner_3j_checked`], [`clebsch_gordan_checked`]). Keeping
/// one predicate means the two paths can never drift: weakening it fails both an
/// unchecked oracle test (a forbidden tuple stops returning zero) and a checked
/// guard test (it stops returning `NotAdmissible`).
///
/// Returns the first violated rule as an [`AdmissibilityViolation`]; the order
/// (m-sum, then per-column projection, then triangle) is an implementation
/// detail callers must not depend on.
fn check_3j_admissible(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i32,
    dm2: i32,
    dm3: i32,
) -> Result<(), AdmissibilityViolation> {
    if dm1 + dm2 + dm3 != 0 {
        return Err(AdmissibilityViolation::ProjectionSum { dm1, dm2, dm3 });
    }
    for (dj, dm) in [(dj1, dm1), (dj2, dm2), (dj3, dm3)] {
        let dji = dj as i64;
        let dmi = dm as i64;
        if dmi.abs() > dji || (dji + dmi) % 2 != 0 {
            return Err(AdmissibilityViolation::Projection { dj, dm });
        }
    }
    check_triangle(dj1, dj2, dj3)
}

/// Boolean 3j admissibility, byte-identical to the pre-checked-surface behavior
/// the infallible engine relies on. Thin `.is_ok()` wrapper over the typed
/// predicate so both share one implementation.
fn admissible_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> bool {
    check_3j_admissible(dj1, dj2, dj3, dm1, dm2, dm3).is_ok()
}

/// $(-1)^p < 0$, i.e. `p` odd.
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

// ---------------------------------------------------------------------------
// Checked SU(2) representation surface (issue #43, section 2).
//
// Additive layer over the infallible functions above. The infallible functions
// keep their zero convention unchanged (an inadmissible tuple returns exact
// zero); the checked layer instead returns a typed [`Su2Error`], so a consumer
// can finally distinguish a *structurally forbidden* coupling (`NotAdmissible`)
// from an *accidental* zero of an admissible coupling (`Ok(zero)` — these exist
// for 6j). Every checked function shares its admissibility predicate with the
// infallible path (`check_triangle` / `check_3j_admissible`), never a copy.
// ---------------------------------------------------------------------------

/// An SU(2) irreducible representation, labeled by its doubled spin `dj = 2j`.
///
/// Every `u32` is a valid label, so construction is infallible (see [`new`]).
/// Fusion of two irreps can overflow the `u32` label space, which is where the
/// only fallible operation ([`fusion`]) lives.
///
/// [`new`]: Su2Irrep::new
/// [`fusion`]: Su2Irrep::fusion
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Su2Irrep(u32);

impl Su2Irrep {
    /// Construct the irrep with doubled spin `dj = 2j`.
    ///
    /// Infallible by construction: SU(2) has one irrep for every nonnegative
    /// half-integer spin, i.e. exactly one for every `u32` doubled spin, so
    /// there is no invalid label to reject.
    pub fn new(dj: u32) -> Self {
        Su2Irrep(dj)
    }

    /// The doubled spin `dj = 2j`.
    pub fn dj(self) -> u32 {
        self.0
    }

    /// The dimension `2j + 1 = dj + 1`.
    ///
    /// Returned as `u64` so it cannot overflow: the maximum `dj` is
    /// `u32::MAX`, and `u32::MAX as u64 + 1` fits `u64` with room to spare.
    pub fn dim(self) -> u64 {
        self.0 as u64 + 1
    }

    /// The dual irrep, which for SU(2) is the irrep itself (every SU(2) irrep
    /// is self-dual).
    ///
    /// The self-duality carries a Frobenius–Schur phase distinguishing the
    /// orthogonal (integer `j`) from the symplectic (half-integer `j`) case;
    /// that phase is [`su2_frobenius_schur`], not folded into this identity.
    pub fn dual(self) -> Self {
        self
    }

    /// Fusion decomposition `self ⊗ other`: the irreps `dj` in
    /// `|dj1 − dj2| ..= dj1 + dj2` in steps of 2 (each with multiplicity one).
    ///
    /// Returns [`Su2Error::LabelOverflow`] when `dj1 + dj2` exceeds `u32`; the
    /// returned [`Su2Fusion`] is an allocation-free
    /// [`ExactSizeIterator`]/[`DoubleEndedIterator`].
    ///
    /// ```
    /// use racah::su2::Su2Irrep;
    ///
    /// let half = Su2Irrep::new(1); // spin-1/2
    /// let channels: Vec<u32> = half.fusion(half).unwrap().map(|s| s.dj()).collect();
    /// assert_eq!(channels, vec![0, 2]); // 1/2 ⊗ 1/2 = 0 ⊕ 1
    /// ```
    pub fn fusion(self, other: Self) -> Result<Su2Fusion, Su2Error> {
        let hi = self.0.checked_add(other.0).ok_or(Su2Error::LabelOverflow {
            left: self.0,
            right: other.0,
        })?;
        let lo = self.0.abs_diff(other.0);
        // lo and hi always share parity (both congruent to dj1+dj2 mod 2), so
        // the half-open count is exact and the last step lands on hi.
        let remaining = ((hi - lo) / 2) as usize + 1;
        Ok(Su2Fusion {
            front: lo,
            back: hi,
            remaining,
        })
    }
}

/// Allocation-free iterator over the fusion channels of two SU(2) irreps
/// (see [`Su2Irrep::fusion`]): doubled spins from `|dj1 − dj2|` to `dj1 + dj2`
/// in steps of 2.
#[derive(Clone, Copy, Debug)]
pub struct Su2Fusion {
    /// Next doubled spin yielded from the front (ascending).
    front: u32,
    /// Next doubled spin yielded from the back (descending).
    back: u32,
    /// Channels not yet yielded from either end.
    remaining: usize,
}

impl Iterator for Su2Fusion {
    type Item = Su2Irrep;

    fn next(&mut self) -> Option<Su2Irrep> {
        if self.remaining == 0 {
            return None;
        }
        let dj = self.front;
        self.remaining -= 1;
        // Only advance while channels remain: on the last element `front` may
        // sit at u32::MAX (hi == dj1 + dj2), where `+= 2` would overflow.
        if self.remaining > 0 {
            self.front += 2;
        }
        Some(Su2Irrep(dj))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for Su2Fusion {
    fn next_back(&mut self) -> Option<Su2Irrep> {
        if self.remaining == 0 {
            return None;
        }
        let dj = self.back;
        self.remaining -= 1;
        // Only advance while channels remain: on the last element `back` may
        // sit at 0 (lo == 0, the singlet channel), where `-= 2` would underflow.
        if self.remaining > 0 {
            self.back -= 2;
        }
        Some(Su2Irrep(dj))
    }
}

impl ExactSizeIterator for Su2Fusion {}

/// The specific SU(2) admissibility rule a checked request violated.
///
/// Carried inside [`Su2Error::NotAdmissible`]. Marked `#[non_exhaustive]` so the
/// checked layer can refine how it decomposes admissibility (adding a distinct
/// reason) without that being a breaking change — which is why [`Su2Error`]
/// itself stays at exactly two variants regardless of internal rule structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissibilityViolation {
    /// The doubled-spin triple `(a, b, c)` fails the triangle condition
    /// `|a − b| ≤ c ≤ a + b` with even perimeter `a + b + c`.
    Triangle {
        /// First doubled spin of the coupling.
        a: u32,
        /// Second doubled spin of the coupling.
        b: u32,
        /// Third doubled spin of the coupling.
        c: u32,
    },
    /// A 3j column violates its projection constraint: either `|dm| > dj`, or
    /// `dj + dm` is odd (projection off the spin's `m`-ladder).
    Projection {
        /// Doubled spin of the offending column.
        dj: u32,
        /// Doubled projection of the offending column.
        dm: i32,
    },
    /// The 3j projections do not sum to zero (`dm1 + dm2 + dm3 ≠ 0`).
    ProjectionSum {
        /// First doubled projection.
        dm1: i32,
        /// Second doubled projection.
        dm2: i32,
        /// Third doubled projection.
        dm3: i32,
    },
}

impl std::fmt::Display for AdmissibilityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissibilityViolation::Triangle { a, b, c } => write!(
                f,
                "doubled-spin triple ({a}, {b}, {c}) violates the triangle condition"
            ),
            AdmissibilityViolation::Projection { dj, dm } => write!(
                f,
                "doubled projection {dm} is not on the ladder of doubled spin {dj}"
            ),
            AdmissibilityViolation::ProjectionSum { dm1, dm2, dm3 } => write!(
                f,
                "doubled projections {dm1} + {dm2} + {dm3} do not sum to zero"
            ),
        }
    }
}

/// Error from a checked SU(2) representation or coefficient request.
///
/// Exactly two variants by design (see [`AdmissibilityViolation`] for why the
/// rule detail lives in a payload rather than in more variants):
///
/// * [`LabelOverflow`](Su2Error::LabelOverflow) — a doubled-spin label would
///   exceed the `u32` label space (only [`Su2Irrep::fusion`] can reach it).
/// * [`NotAdmissible`](Su2Error::NotAdmissible) — the request is structurally
///   forbidden. Distinct from an admissible request whose coefficient is an
///   accidental zero, which the checked functions return as `Ok(zero)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Su2Error {
    /// The sum of two doubled spins overflows `u32`.
    LabelOverflow {
        /// Left doubled spin `dj1`.
        left: u32,
        /// Right doubled spin `dj2`.
        right: u32,
    },
    /// The requested coupling is not admissible; the payload names the rule.
    NotAdmissible(AdmissibilityViolation),
}

impl std::fmt::Display for Su2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Su2Error::LabelOverflow { left, right } => write!(
                f,
                "doubled spins {left} + {right} overflow the u32 label space"
            ),
            Su2Error::NotAdmissible(v) => write!(f, "inadmissible SU(2) coupling: {v}"),
        }
    }
}

impl std::error::Error for Su2Error {}

/// Checked [`wigner_6j`]: `Ok(value)` for an admissible label set (the value may
/// be an accidental zero), else [`Su2Error::NotAdmissible`].
///
/// Admissibility is the same four-triangle predicate the infallible engine
/// gates on (`check_6j_admissible`); on success the value is delegated to
/// [`wigner_6j`] unchanged.
pub fn wigner_6j_checked(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<SignedSqrtRational, Su2Error> {
    check_6j_admissible(dj1, dj2, dj3, dj4, dj5, dj6).map_err(Su2Error::NotAdmissible)?;
    Ok(wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6))
}

/// Checked [`wigner_3j`]: `Ok(value)` for an admissible label set, else
/// [`Su2Error::NotAdmissible`] naming the violated rule (m-sum, projection, or
/// triangle). Shares `check_3j_admissible` with the infallible engine.
pub fn wigner_3j_checked(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i32,
    dm2: i32,
    dm3: i32,
) -> Result<SignedSqrtRational, Su2Error> {
    check_3j_admissible(dj1, dj2, dj3, dm1, dm2, dm3).map_err(Su2Error::NotAdmissible)?;
    Ok(wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3))
}

/// Checked [`clebsch_gordan`]: `Ok(value)` for an admissible coupling, else
/// [`Su2Error::NotAdmissible`].
///
/// The CG coefficient is composed from the 3j `(dj1 dj2 dj3; dm1 dm2 -dm3)`, so
/// admissibility is exactly that 3j's — checked through the shared
/// `check_3j_admissible` with `dm3` negated.
pub fn clebsch_gordan_checked(
    dj1: u32,
    dm1: i32,
    dj2: u32,
    dm2: i32,
    dj3: u32,
    dm3: i32,
) -> Result<SignedSqrtRational, Su2Error> {
    check_3j_admissible(dj1, dj2, dj3, dm1, dm2, -dm3).map_err(Su2Error::NotAdmissible)?;
    Ok(clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3))
}

/// Checked [`su2_f_symbol`]: `Ok(value)` for an admissible F-symbol, else
/// [`Su2Error::NotAdmissible`].
///
/// The F-symbol evaluates the 6j `{dj1 dj2 dj5; dj3 dj4 dj6}`, so its
/// admissibility is that 6j's four triangles — checked through the shared
/// `check_6j_admissible` with the F-symbol's argument order.
pub fn su2_f_symbol_checked(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<f64, Su2Error> {
    check_6j_admissible(dj1, dj2, dj5, dj3, dj4, dj6).map_err(Su2Error::NotAdmissible)?;
    Ok(su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6))
}

/// Checked [`su2_r_symbol`]: `Ok(±1.0)` on an admissible fusion triangle, else
/// [`Su2Error::NotAdmissible`]. Shares `check_triangle` with the infallible
/// path.
///
/// (Frobenius–Schur has no checked counterpart: [`su2_frobenius_schur`] is
/// total — every `u32` doubled spin is a valid irrep — so it can never fail.)
pub fn su2_r_symbol_checked(dj1: u32, dj2: u32, dj3: u32) -> Result<f64, Su2Error> {
    check_triangle(dj1, dj2, dj3).map_err(Su2Error::NotAdmissible)?;
    Ok(su2_r_symbol(dj1, dj2, dj3))
}

/// Opaque authority fingerprint of the base SU(2) provider.
///
/// The bytes identify the *convention set* that every returned SU(2)
/// coefficient (3j, 6j, Clebsch–Gordan, F, R, Frobenius–Schur) is computed in.
/// Their sole use is equality comparison: two builds that return the same
/// fingerprint agree on every value-fixing convention below, so a consumer may
/// persist the bytes next to data derived from these coefficients and later
/// compare them to decide whether that derived data is still valid.
///
/// # Consumer contract
///
/// - **Opaque.** Treat the bytes as an identifier, not a document. Compare by
///   equality only; never parse the tags or split on `:` / `=` — the internal
///   shape is not a stable interface and may be reorganized without changing
///   what the fingerprint *means* (as long as the epoch rule below holds).
/// - **Stable across patch and minor releases.** The value is *not* derived
///   from the crate version, source, docs, a pointer, or any process-local
///   state, so a rebuild, a dependency bump, or an additive-API release leaves
///   it byte-identical.
/// - **Changes exactly with a value-affecting breaking release.** The trailing
///   `epoch` is bumped by hand — and only — when a change can alter a returned
///   coefficient value, its normalization, or the canonical convention it is
///   expressed in. That is the same event class the crate's semantic-versioning
///   contract already declares breaking (README, "Exactness contract", point 4
///   "Versioned values"; the analogous SU(N) statement is `docs/gauge.md`), so
///   "fingerprint changed ⇔ value-affecting breaking release" is one reviewable
///   invariant. Adding the `cg` and `fs` tags keeps `epoch=1`: the fingerprint
///   is still unreleased (no consumer has persisted these bytes), so extending
///   the tag set now is not a compatibility break for anyone.
/// - **Persist alongside derived data.** A consumer that caches or serializes
///   anything computed from these coefficients may store the fingerprint with
///   it and reject the cache on mismatch.
///
/// # Why a manual epoch, not a hash or the crate version
///
/// A hash of the source or docs is fragile (it moves on a comment edit or a
/// refactor that changes no value) and not reviewable (a human cannot look at
/// it and confirm it *should* have changed). The crate version moves on every
/// patch, which would force consumers to re-derive on releases that change no
/// value. A hand-bumped epoch makes the change an explicit, mutation-visible
/// review event: the compatibility-policy test (`tests/su2_fingerprint.rs`)
/// pins the exact bytes, so any value-affecting PR must touch that test, and
/// updating it is where the breaking-release decision is recorded. (Rationale
/// per the issue #43 design record.)
///
/// # Tags and the conventions they pin
///
/// Each tag names a convention the crate's docs already establish; nothing here
/// invents a convention. Value-encoding conventions are included; the API shape
/// (how labels are passed) is not — see the exclusions below.
///
/// - `model=bigrational-round-once` — the exact evaluation model: values are
///   big-rational sums carried as [`SignedSqrtRational`] with a single final
///   rounding to `f64` (module docs above; README "Exactness contract",
///   "compute in rationals, round once").
/// - `3j=condon-shortley` — the 3j sign convention ([`wigner_3j`] docs,
///   "Condon-Shortley phase").
/// - `cg=condon-shortley` — the Clebsch-Gordan convention: composed from the 3j
///   via the standard relation with phase $(-1)^{(dj_2-dj_1-dm_3)/2}$ and
///   $\sqrt{dj_3+1}$ normalization ([`clebsch_gordan`] docs). Tagged separately
///   because that phase-and-normalization step is its own value convention, not
///   pure 3j inheritance.
/// - `6j=racah-single-sum` — the 6j evaluation ([`wigner_6j`] docs, "Racah
///   single-sum closed form").
/// - `f=tks-su2irrep` — the F-symbol convention: the TensorKitSectors
///   `su2irrep.jl:Fsymbol` correspondence, including the exact 6j argument order
///   and the $(-1)^{j_1+j_2+j_3+j_4}$ phase. That convention is fixed by the exact F
///   evaluation (the private `f_symbol_exact`, see the module's TensorKitSectors
///   correspondence); [`su2_f_symbol`] is its cached `f64` presentation.
/// - `r=tks-su2irrep` — the R-symbol convention: TensorKitSectors
///   `su2irrep.jl:Rsymbol`, $(-1)^{j_1+j_2-j_3}$ on an admissible triangle
///   ([`su2_r_symbol`] docs).
/// - `fs=tks-su2irrep` — the Frobenius-Schur convention: $(-1)^{dj}$, the
///   TensorKitSectors `su2irrep.jl` self-dual correspondence
///   ([`su2_frobenius_schur`] docs). Tagged separately from `r` because it is a
///   distinct formula, not the R-symbol convention.
/// - `epoch=1` — the manual epoch (see above).
///
/// Deliberately **excluded**: the doubled-spin label encoding (`dj = 2j`) is
/// input addressing / API shape, not a property of a returned value — changing
/// it would be an API-compatibility matter that leaves every coefficient value
/// unchanged, so it does not belong in a value fingerprint (design record: API
/// shape is not included). The crate version and any source/doc hash are
/// excluded for the reasons above.
pub fn su2_authority_fingerprint() -> &'static [u8] {
    // Manual epoch: bump the trailing `epoch=N` (and the literal in
    // tests/su2_fingerprint.rs) only on a value-affecting breaking release.
    b"racah:su2-exact:model=bigrational-round-once:3j=condon-shortley:cg=condon-shortley:6j=racah-single-sum:f=tks-su2irrep:r=tks-su2irrep:fs=tks-su2irrep:epoch=1"
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

#[cfg(test)]
mod checked_tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // -- Su2Irrep basics ----------------------------------------------------

    #[test]
    fn irrep_accessors_and_self_dual() {
        let s = Su2Irrep::new(3); // spin 3/2
        assert_eq!(s.dj(), 3);
        assert_eq!(s.dim(), 4); // 2j+1
        assert_eq!(s.dual(), s); // SU(2) is self-dual
        assert_eq!(Su2Irrep::new(0).dim(), 1);
    }

    #[test]
    fn dim_does_not_overflow_at_max_label() {
        assert_eq!(Su2Irrep::new(u32::MAX).dim(), u32::MAX as u64 + 1);
    }

    // -- Fusion range / iterator contract -----------------------------------

    #[test]
    fn fusion_range_matches_direct_triangle_scan() {
        // The fusion channels must be exactly those doubled spins dj3 for which
        // (dj1, dj2, dj3) is an admissible triangle — an independent oracle.
        for dj1 in 0..=8u32 {
            for dj2 in 0..=8u32 {
                let got: Vec<u32> = Su2Irrep::new(dj1)
                    .fusion(Su2Irrep::new(dj2))
                    .unwrap()
                    .map(|s| s.dj())
                    .collect();
                let want: Vec<u32> = (0..=dj1 + dj2)
                    .filter(|&c| triangle_ok(dj1, dj2, c))
                    .collect();
                assert_eq!(got, want, "fusion {dj1} x {dj2}");
            }
        }
    }

    #[test]
    fn fusion_exact_size_and_double_ended() {
        // len() is exact and consistent with the yielded count; iterating from
        // the back reverses the front order.
        let mut it = Su2Irrep::new(4).fusion(Su2Irrep::new(2)).unwrap(); // 2..=6 step 2
        assert_eq!(it.len(), 3);
        assert_eq!(it.next().map(|s| s.dj()), Some(2));
        assert_eq!(it.next_back().map(|s| s.dj()), Some(6));
        assert_eq!(it.len(), 1);
        assert_eq!(it.next().map(|s| s.dj()), Some(4));
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());

        let full: Vec<u32> = Su2Irrep::new(4)
            .fusion(Su2Irrep::new(2))
            .unwrap()
            .rev()
            .map(|s| s.dj())
            .collect();
        assert_eq!(full, vec![6, 4, 2]);
    }

    #[test]
    fn fusion_overflow_at_u32_boundary() {
        // dj1 + dj2 overflows u32 -> typed LabelOverflow, no wraparound.
        let a = Su2Irrep::new(u32::MAX);
        assert!(matches!(
            a.fusion(Su2Irrep::new(1)),
            Err(Su2Error::LabelOverflow {
                left: u32::MAX,
                right: 1,
            })
        ));
        // Exactly on the boundary is fine (sum == u32::MAX).
        assert!(Su2Irrep::new(u32::MAX - 1).fusion(Su2Irrep::new(1)).is_ok());
    }

    #[test]
    fn fusion_full_back_drain_reaches_singlet_without_underflow() {
        // Draining from the back down to lo = 0 (dj1 == dj2, the singlet
        // channel) must not advance the cursor past the final element:
        // `back -= 2` at back == 0 would underflow u32 and panic in debug.
        let got: Vec<u32> = Su2Irrep::new(2)
            .fusion(Su2Irrep::new(2))
            .unwrap()
            .rev()
            .map(|s| s.dj())
            .collect();
        assert_eq!(got, vec![4, 2, 0]);
    }

    #[test]
    fn fusion_full_forward_drain_at_u32_max_without_overflow() {
        // Draining forward up to hi == u32::MAX must not advance the cursor
        // past the final element: `front += 2` at front == u32::MAX would
        // overflow and panic in debug. lo = MAX-2, hi = MAX (parity-consistent).
        let got: Vec<u32> = Su2Irrep::new(u32::MAX - 1)
            .fusion(Su2Irrep::new(1))
            .unwrap()
            .map(|s| s.dj())
            .collect();
        assert_eq!(got, vec![u32::MAX - 2, u32::MAX]);
    }

    #[test]
    fn fusion_mixed_drain_to_singlet_without_underflow() {
        // Alternating next/next_back to exhaustion, ending at lo = 0, must not
        // underflow on either cursor. 2 x 2 = {0, 2, 4}.
        let mut it = Su2Irrep::new(2).fusion(Su2Irrep::new(2)).unwrap();
        let mut got = vec![
            it.next().unwrap().dj(),      // front: 0
            it.next_back().unwrap().dj(), // back: 4
            it.next_back().unwrap().dj(), // back: 2 (lands cursor at lo = 0)
        ];
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
        got.sort_unstable();
        assert_eq!(got, vec![0, 2, 4]);
    }

    // -- Guard inventory: one red-first typed-error test per rule ------------

    #[test]
    fn guard_6j_triangle() {
        // (1,2,3) triangle inequality violated: dj3 = 20 with dj1=dj2=2.
        assert_eq!(
            wigner_6j_checked(2, 2, 20, 2, 2, 2),
            Err(Su2Error::NotAdmissible(AdmissibilityViolation::Triangle {
                a: 2,
                b: 2,
                c: 20,
            }))
        );
        // A different triangle of the four: (4,2,6) = (7,3,2) parity/inequality.
        assert!(matches!(
            wigner_6j_checked(2, 3, 3, 7, 2, 100),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::Triangle { .. }
            ))
        ));
    }

    #[test]
    fn guard_3j_projection_sum() {
        assert_eq!(
            wigner_3j_checked(2, 2, 2, 2, 2, 2),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::ProjectionSum {
                    dm1: 2,
                    dm2: 2,
                    dm3: 2,
                }
            ))
        );
    }

    #[test]
    fn guard_3j_projection_out_of_range() {
        // |dm1| = 4 > dj1 = 2 (m-sum still zero, so this rule is what fails).
        assert_eq!(
            wigner_3j_checked(2, 2, 2, 4, -2, -2),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::Projection { dj: 2, dm: 4 }
            ))
        );
    }

    #[test]
    fn guard_3j_projection_parity() {
        // dj1 + dm1 = 2 + 1 = 3 is odd: projection off the ladder (m-sum zero).
        assert_eq!(
            wigner_3j_checked(2, 2, 2, 1, 1, -2),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::Projection { dj: 2, dm: 1 }
            ))
        );
    }

    #[test]
    fn guard_3j_triangle() {
        // Projections all admissible (dm = 0 everywhere, even dj) and sum to
        // zero, but (2,2,8) fails the triangle inequality (8 > 2+2).
        assert_eq!(
            wigner_3j_checked(2, 2, 8, 0, 0, 0),
            Err(Su2Error::NotAdmissible(AdmissibilityViolation::Triangle {
                a: 2,
                b: 2,
                c: 8,
            }))
        );
    }

    #[test]
    fn guard_cg_inadmissible() {
        // CG couples via the 3j (dj1 dj2 dj3; dm1 dm2 -dm3); m1+m2-m3 != 0.
        assert!(matches!(
            clebsch_gordan_checked(2, 2, 2, 2, 2, 0),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::ProjectionSum { .. }
            ))
        ));
    }

    #[test]
    fn guard_f_symbol_triangle() {
        // F evaluates 6j {dj1 dj2 dj5 / dj3 dj4 dj6} = {1/2 1/2 1/2 / ...};
        // (1,1,1) parity-forbidden.
        assert!(matches!(
            su2_f_symbol_checked(1, 1, 1, 1, 1, 1),
            Err(Su2Error::NotAdmissible(
                AdmissibilityViolation::Triangle { .. }
            ))
        ));
    }

    #[test]
    fn guard_r_symbol_triangle() {
        assert_eq!(
            su2_r_symbol_checked(1, 1, 1),
            Err(Su2Error::NotAdmissible(AdmissibilityViolation::Triangle {
                a: 1,
                b: 1,
                c: 1,
            }))
        );
        assert_eq!(su2_r_symbol_checked(1, 1, 2), Ok(1.0));
    }

    #[test]
    fn checked_value_near_u32_max_is_not_admissible_without_overflow() {
        // A triangle-violating near-max 6j must return NotAdmissible cheaply
        // (the predicate widens to i64), never overflow or hang.
        assert!(matches!(
            wigner_6j_checked(u32::MAX, 2, 0, 2, 2, 2),
            Err(Su2Error::NotAdmissible(_))
        ));
    }

    // -- Accidental zero vs structural forbiddance --------------------------

    #[test]
    fn admissible_but_zero_6j_is_ok_not_err() {
        // {2 3 3; 7 6 6} passes all four triangles yet the 6j is exactly zero
        // (an accidental zero). The checked layer must report Ok(zero), NOT
        // NotAdmissible — the distinction that motivates the checked surface.
        let v = wigner_6j_checked(2, 3, 3, 7, 6, 6).expect("admissible tuple");
        assert_eq!(v, SignedSqrtRational::zero());
        assert_eq!(wigner_6j(2, 3, 3, 7, 6, 6), SignedSqrtRational::zero());
    }

    // -- Property: checked == unchecked on the admissible domain ------------

    #[test]
    fn property_checked_equals_unchecked_when_admissible() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xCA5C_ADE5_u64);
        let mut tested_6j = 0;
        let mut tested_3j = 0;
        for _ in 0..20_000 {
            // 6j
            let d = [(); 6].map(|_| rng.gen_range(0..=10u32));
            if let Ok(v) = wigner_6j_checked(d[0], d[1], d[2], d[3], d[4], d[5]) {
                assert_eq!(v, wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]));
                tested_6j += 1;
            }
            // 3j: bias toward the admissible domain by forcing m-sum = 0
            // (dm3 = -(dm1+dm2)); triangle/projection filters still cull most.
            let dj = [(); 3].map(|_| rng.gen_range(0..=8u32));
            let dm1 = rng.gen_range(-(dj[0] as i32)..=dj[0] as i32);
            let dm2 = rng.gen_range(-(dj[1] as i32)..=dj[1] as i32);
            let dm = [dm1, dm2, -(dm1 + dm2)];
            if let Ok(v) = wigner_3j_checked(dj[0], dj[1], dj[2], dm[0], dm[1], dm[2]) {
                assert_eq!(v, wigner_3j(dj[0], dj[1], dj[2], dm[0], dm[1], dm[2]));
                tested_3j += 1;
            }
        }
        assert!(
            tested_6j > 100,
            "too few admissible 6j samples ({tested_6j})"
        );
        assert!(
            tested_3j > 100,
            "too few admissible 3j samples ({tested_3j})"
        );
    }

    // -- Property: forbidden <=> unchecked zero AND checked NotAdmissible ----

    #[test]
    fn property_forbidden_is_zero_and_not_admissible() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
        let mut tested_6j = 0;
        let mut tested_3j = 0;
        for _ in 0..40_000 {
            let d = [(); 6].map(|_| rng.gen_range(0..=8u32));
            match wigner_6j_checked(d[0], d[1], d[2], d[3], d[4], d[5]) {
                Err(Su2Error::NotAdmissible(_)) => {
                    assert_eq!(
                        wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]),
                        SignedSqrtRational::zero(),
                        "structurally forbidden 6j must be an unchecked zero"
                    );
                    tested_6j += 1;
                }
                Err(Su2Error::LabelOverflow { .. }) => unreachable!("6j checked never overflows"),
                Ok(_) => {}
            }

            let dj = [(); 3].map(|_| rng.gen_range(0..=6u32));
            let dm = [(); 3].map(|_| rng.gen_range(-6..=6i32));
            if let Err(Su2Error::NotAdmissible(_)) =
                wigner_3j_checked(dj[0], dj[1], dj[2], dm[0], dm[1], dm[2])
            {
                assert_eq!(
                    wigner_3j(dj[0], dj[1], dj[2], dm[0], dm[1], dm[2]),
                    SignedSqrtRational::zero(),
                    "structurally forbidden 3j must be an unchecked zero"
                );
                tested_3j += 1;
            }
        }
        assert!(
            tested_6j > 100,
            "too few forbidden 6j samples ({tested_6j})"
        );
        assert!(
            tested_3j > 100,
            "too few forbidden 3j samples ({tested_3j})"
        );
    }
}
