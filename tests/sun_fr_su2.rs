//! SU(2) cross-check of the SU(N) F/R contraction against the crate's exact
//! closed-form SU(2) recoupling (`su2_f_symbol` / `su2_r_symbol`), which are
//! the independent value authority (phase-composed exact 6j; Racah big-rational
//! sums rounded once). This is oracle 2 of issue #16.
//!
//! # The per-channel sign relation (documented reconciliation)
//!
//! SUNRepresentations' deterministic SU(2) CGC gauge and the Condon–Shortley
//! convention of the exact core differ by one overall sign `ε(j1,j2,j3) = ±1`
//! per fusion channel (the highest-weight `qrpos!` sign — the same sign
//! `tests/su2_embedding.rs` aligns). The F/R symbols are gauge-*covariant*
//! under a per-vertex CGC sign flip `C(x,y,z) → ε(x,y,z) C(x,y,z)`:
//!
//! - F contracts one CGC at each of its four vertices `(a,b,e)`, `(e,c,d)`,
//!   `(b,c,f)`, `(a,f,d)`, so
//!   `F_racah[e,f] = ε(a,b,e) ε(e,c,d) ε(b,c,f) ε(a,f,d) · F_CS[e,f]`.
//! - R contracts `C(a,b,c)` and `C(b,a,c)`, so
//!   `R_racah = ε(a,b,c) ε(b,a,c) · R_CS`.
//!
//! # The sign relation, as a theorem (not just an observation)
//!
//! `ε(j1,j2,j3) = (-1)^(j1+j2-j3)`. The Layer-2 gauge fixes the highest-weight
//! coupled state by `qrpos!` (positive pivot) over the coupling pairs ordered by
//! `m1` ascending then matching `m2` — i.e. it normalizes the **`m2`-stretched**
//! first pair positive; Condon–Shortley instead fixes the **`m1`-stretched**
//! coefficient `⟨j1 j1; j2, j3−j1 | j3 j3⟩ > 0`. Those two extreme coefficients
//! differ in sign by exactly `(-1)^(j1+j2-j3)` (the CGC row-reversal symmetry),
//! so that is `ε`. Each channel is otherwise ladder-fixed identically in both
//! gauges, hence one global sign per channel. `tests/su2_embedding.rs` observes
//! this same per-channel sign; here it is pinned to the closed form.
//!
//! Consequences for F and R, both telescoping to an **even** exponent:
//! - F: `∏ε = (-1)^((a+b−e)+(e+c−d)+(b+c−f)+(a+f−d)) = (-1)^(2(a+b+c−d)) = +1`.
//! - R: `∏ε = (-1)^((a+b−c)+(b+a−c)) = (-1)^(2(a+b−c)) = +1`.
//!
//! So the F/R **blocks are sign-difference-invariant** (the per-vertex
//! coboundary cancels) even though 56 of 140 individual SU(2) channels
//! (`dj ≤ 6`) have `ε = -1`. The comparison reconciles via the exact `ε` rule —
//! each `ε` computed independently from `racah::clebsch_gordan`, and asserted
//! equal to `(-1)^(j1+j2−j3)` per channel (strictly stronger than the product
//! pin) — so a future gauge change that broke either the per-channel law or the
//! cancellation would surface here.

#![cfg(feature = "cgc-gen")]

use racah::sun::{cgc, f_symbol, r_symbol, Irrep};
use racah::{clebsch_gordan, su2_f_symbol, su2_r_symbol};

fn su2(dj: u32) -> Irrep {
    Irrep::from_dynkin(&[dj as i64]).unwrap()
}

/// GT basis index `x` (0..=dj) → doubled magnetic number `dm = 2x - dj`.
fn dm(x: usize, dj: u32) -> i32 {
    2 * x as i32 - dj as i32
}

fn admissible(dj1: u32, dj2: u32, dj3: u32) -> bool {
    dj3 >= dj1.abs_diff(dj2) && dj3 <= dj1 + dj2 && (dj1 + dj2 + dj3).is_multiple_of(2)
}

/// `ε(j1,j2,j3) = ±1`: the global sign relating the racah SU(N)-pipeline CGC
/// for the channel `j1 ⊗ j2 → j3` to the Condon–Shortley `clebsch_gordan`.
/// Determined from the largest-magnitude exact element (independent oracle).
fn eps_channel(dj1: u32, dj2: u32, dj3: u32) -> f64 {
    let c = cgc(&su2(dj1), &su2(dj2), &su2(dj3)).unwrap();
    let mut best = 0.0f64;
    let mut sign = 1.0f64;
    for e in c.entries() {
        let ex = clebsch_gordan(
            dj1,
            dm(e.m1 as usize, dj1),
            dj2,
            dm(e.m2 as usize, dj2),
            dj3,
            dm(e.m3 as usize, dj3),
        )
        .to_f64();
        if ex.abs() > best {
            best = ex.abs();
            sign = if (ex.signum() - e.value.signum()).abs() < 0.5 {
                1.0
            } else {
                -1.0
            };
        }
    }
    // Theorem: ε(j1,j2,j3) = (-1)^(j1+j2-j3); in doubled units the exponent is
    // (dj1+dj2-dj3)/2 (integral on an admissible triangle). Pin it per channel.
    let expected = if ((dj1 + dj2 - dj3) / 2).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    assert_eq!(
        sign, expected,
        "ε({dj1},{dj2},{dj3}) = {sign} != (-1)^(j1+j2-j3) = {expected}"
    );
    sign
}

#[test]
fn f_from_cgc_matches_su2_closed_form() {
    // F is O(1) here; the tolerance covers the SVD/QR/descent round-off in the
    // CGC pipeline plus the exact core's single rounding, far below any
    // structural (sign/index) error which would be O(1)-sized.
    const TOL: f64 = 1e-9;
    let mut checked = 0u64;
    let mut flipped = 0u64; // sextets where ∏ε = -1 (F NOT plainly sign-invariant)
    let mut worst = 0.0f64;

    // Exhaustive small sweep over doubled spins, all six labels ≤ 4.
    for dja in 0..=4u32 {
        for djb in 0..=4u32 {
            for djc in 0..=4u32 {
                for dje in 0..=4u32 {
                    if !admissible(dja, djb, dje) {
                        continue;
                    }
                    for djd in 0..=6u32 {
                        if !admissible(dje, djc, djd) {
                            continue;
                        }
                        for djf in 0..=6u32 {
                            if !admissible(djb, djc, djf) || !admissible(dja, djf, djd) {
                                continue;
                            }
                            let block = f_symbol(
                                &su2(dja),
                                &su2(djb),
                                &su2(djc),
                                &su2(djd),
                                &su2(dje),
                                &su2(djf),
                            )
                            .unwrap();
                            assert_eq!(block.dims(), [1, 1, 1, 1], "SU(2) F is multiplicity-free");
                            let got = block.at(0, 0, 0, 0);

                            let eps = eps_channel(dja, djb, dje)
                                * eps_channel(dje, djc, djd)
                                * eps_channel(djb, djc, djf)
                                * eps_channel(dja, djf, djd);
                            if eps < 0.0 {
                                flipped += 1;
                            }
                            let want = eps * su2_f_symbol(dja, djb, djc, djd, dje, djf);
                            let err = (got - want).abs();
                            worst = worst.max(err);
                            assert!(
                                err <= TOL,
                                "F mismatch a={dja} b={djb} c={djc} d={djd} e={dje} f={djf}: \
                                 got={got} want(εF_CS)={want} ε={eps} err={err:e}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 500, "expected a broad sweep, got {checked}");
    // Empirically ∏ε = +1 for every sextet: F blocks are sign-difference-
    // invariant. Pin that as a contract — a regression that broke the
    // coboundary cancellation would surface here rather than silently.
    assert_eq!(
        flipped, 0,
        "expected the per-vertex sign coboundary to cancel in every F block"
    );
    println!(
        "SU(2) F oracle: {checked} sextets, worst |Δ| {worst:e}, ∏ε=-1 in {flipped} \
         (F blocks sign-difference-invariant)"
    );
}

#[test]
fn r_from_cgc_matches_su2_closed_form() {
    // R is an exact ±1; after the ε reconciliation the agreement is exact to
    // round-off.
    const TOL: f64 = 1e-9;
    let mut checked = 0u64;
    let mut flipped = 0u64;
    for dja in 0..=6u32 {
        for djb in 0..=6u32 {
            for djc in dja.abs_diff(djb)..=(dja + djb) {
                if !admissible(dja, djb, djc) {
                    continue;
                }
                let got = r_symbol(&su2(dja), &su2(djb), &su2(djc)).unwrap();
                assert_eq!(got.dim(), 1, "SU(2) R is multiplicity-free");
                let eps = eps_channel(dja, djb, djc) * eps_channel(djb, dja, djc);
                if eps < 0.0 {
                    flipped += 1;
                }
                let want = eps * su2_r_symbol(dja, djb, djc);
                assert!(
                    (got.at(0, 0) - want).abs() <= TOL,
                    "R mismatch a={dja} b={djb} c={djc}: got={} want={want} ε={eps}",
                    got.at(0, 0)
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "expected many R triples, got {checked}");
    assert_eq!(
        flipped, 0,
        "expected the per-vertex sign coboundary to cancel in every R block"
    );
    println!("SU(2) R oracle: {checked} triples, ∏ε=-1 in {flipped} (R sign-difference-invariant)");
}
