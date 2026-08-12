//! Low-rank isomorphism oracle battery (issue #32, deliverable 1).
//!
//! Role (see `tools/README.md` matrix): *internal consistency* strand. Each of
//! the three exceptional low-rank Lie-algebra isomorphisms is a fact of
//! representation theory, so the two racah code paths joined by the
//! isomorphism must produce the **same** integer data (dimensions, product
//! decompositions, and the fusion multiplicities `N^c_{ab}` — the "N-symbols")
//! exactly. No external tool is involved: the oracle is racah's own SU(N)
//! pipeline ([`racah::sun`]) checking racah's B/C/D pipeline ([`racah::bcd`]),
//! and racah's two independent SU(2) implementations checking each other.
//!
//! This strand verifies only the *tensor* (linear, non-spinor) image of each
//! isomorphism: [`racah::bcd`] represents tensor irreps of `SO(N)`/`Sp(2N)`
//! only (spinors belong to the covering group and are rejected with
//! [`racah::bcd::BcdError::SpinorLabel`]). Labels outside the isomorphic
//! tensor image are out of scope here (role matrix).
//!
//! # Dynkin label maps
//!
//! Each label map is a **theorem** — the diagram-theoretic derivation is given
//! per isomorphism in the sections below (the node identification from the folded
//! / relabelled Dynkin diagram) and then cross-checked against matching dimensions
//! and product channels through both sides, never merely assumed from a numerical
//! coincidence. `A_{n}` uses [`racah::sun`]
//! Dynkin labels `(a_1,…,a_n)`; `B_r`/`C_r`/`D_r` use [`racah::bcd`] Dynkin
//! labels `(b_0,…,b_{r-1})` in the module's node order (node 0 = vector for
//! `B`/`D`, per `docs/gauge_soN.md`).
//!
//! ## `SO(6) ≅ SU(4)` (`D_3 ≅ A_3`)
//!
//! The `D_3` and `A_3` Dynkin diagrams coincide (a 3-node chain whose middle
//! node is `D_3`'s trivalent vector node). The label map is
//!
//! ```text
//! SU(4) (a_1, a_2, a_3)  ↦  SO(6) (a_2, a_1, a_3)
//! ```
//!
//! i.e. the `A_3` central node `a_2` (the `6`, antisymmetric square) is the
//! `D_3` vector node `b_0`, and the two `A_3` leaf nodes `a_1, a_3` (the
//! fundamental `4` and antifundamental `4̄`) are the two `D_3` chiral spinor
//! nodes `b_1, b_2`. Verified: `4↔spinor`, `6↔[1,0,0]` vector, `15↔[0,1,1]`
//! adjoint, and the chiral pair `10 = [2,0,0]_{A_3} ↔ [0,2,0]_{D_3}`,
//! `1̄0̄ = [0,0,2]_{A_3} ↔ [0,0,2]_{D_3}`.
//!
//! The `D_3` label `(a_2, a_1, a_3)` is a *tensor* irrep iff `a_1 + a_3` is
//! even (the `D_3` spinor test is `b_1 + b_2 ≡ a_1 + a_3` odd). `a_1 + a_3`
//! mod 2 is exactly the `SU(4)` N-ality mod 2, which is additive under tensor
//! product; so the even sublattice is closed and the cross-check stays inside
//! the tensor image for every product.
//!
//! ## `Sp(4) ≅ SO(5)` (`C_2 ≅ B_2`)
//!
//! The `B_2` and `C_2` diagrams are the same two nodes with the arrow
//! reversed, so the isomorphism swaps the long and short nodes:
//!
//! ```text
//! Sp(4) (c_0, c_1)  ↦  SO(5) (c_1, c_0)
//! ```
//!
//! Verified: `Sp(4)` adjoint `10 = [2,0] ↔ [0,2]` `SO(5)` adjoint; the `SO(5)`
//! vector `5 = [1,0]_{B_2} ↔ [0,1]_{C_2}` the `Sp(4)` `5`; `14 = [0,2]_{C_2} ↔
//! [2,0]_{B_2}`. The `B_2` label `(c_1, c_0)` is a *tensor* irrep iff `c_0` is
//! even (the `B_2` spinor test is `b_1 = c_0` odd). The `Sp(4)` fundamental
//! `4 = [1,0]` maps to the `SO(5)` spinor and is therefore out of the tensor
//! image. `c_0` mod 2 is conserved under the covered products, so the even
//! sublattice is closed.
//!
//! ## `SO(4) ≅ SU(2) × SU(2)` (`D_2 ≅ A_1 × A_1`)
//!
//! `SO(4)` is the excluded rank `D_2` in [`racah::bcd`] (redirected to the
//! SU(2) machinery). The isomorphism content — an `SO(4)` tensor irrep is a
//! pair of `SU(2)` irreps with `SO(4)` products factoring as independent
//! `SU(2)` products — is checked here between racah's **two independent
//! `SU(2)` code paths**: the general Gelfand–Tsetlin engine ([`racah::sun`]
//! at `N = 2`) and the exact closed-form base module ([`racah::su2`],
//! reached through [`racah::clebsch_gordan`]). An `SO(4)` label `(p, q)`
//! carries `SU(2)` spins `(p/2, q/2)` (doubled spins `dj = p, q`), dimension
//! `(p+1)(q+1)`, and is a tensor irrep iff `p + q` is even.

#![cfg(feature = "cgc-gen")]

use std::collections::BTreeMap;

use racah::bcd::{directproduct as bdp, Irrep as Bcd, Series};
use racah::clebsch_gordan;
use racah::sun::{directproduct as sdp, Irrep as Sun};

/// `SU(4)` Dynkin `(a_1,a_2,a_3)` → `SO(6)=D_3` Dynkin `(a_2,a_1,a_3)`.
/// Panics if the image is a spinor (`a_1 + a_3` odd) — callers stay in the
/// even (tensor) sublattice.
fn su4_to_so6(a: &[i64]) -> Bcd {
    Bcd::from_dynkin(Series::D, &[a[1], a[0], a[2]]).expect("SO(6) tensor image")
}

/// `Sp(4)=C_2` Dynkin `(c_0,c_1)` → `SO(5)=B_2` Dynkin `(c_1,c_0)`.
/// Panics if the image is a spinor (`c_0` odd) — callers stay in the tensor
/// sublattice.
fn sp4_to_so5(c: &[i64]) -> Bcd {
    Bcd::from_dynkin(Series::B, &[c[1], c[0]]).expect("SO(5) tensor image")
}

/// Product decomposition of a B/C/D product keyed by the summand's Dynkin
/// label, for structural comparison with an SU(N) decomposition.
fn bcd_channels(a: &Bcd, b: &Bcd) -> BTreeMap<Vec<i64>, u32> {
    bdp(a, b)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k.dynkin(), v))
        .collect()
}

/// Product decomposition of an SU(N) product keyed by the summand's Dynkin
/// label.
fn sun_channels(a: &Sun, b: &Sun) -> BTreeMap<Vec<i64>, u32> {
    sdp(a, b)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k.dynkin(), v))
        .collect()
}

/// `SU(4)` labels in the `SO(6)` tensor image (`a_1 + a_3` even), spanning the
/// small reps, the adjoint, both chiral `10`s, and a multiplicity-bearing pair.
const SO6_LABELS: &[[i64; 3]] = &[
    [0, 1, 0], // 6   vector
    [1, 0, 1], // 15  adjoint
    [2, 0, 0], // 10  (chiral)
    [0, 0, 2], // 10bar (chiral)
    [0, 2, 0], // 20'
    [1, 1, 1], // 64
    [2, 1, 0], // 45
];

/// `Sp(4)` labels in the `SO(5)` tensor image (`c_0` even).
const SO5_LABELS: &[[i64; 2]] = &[
    [0, 1], // 5   vector
    [2, 0], // 10  adjoint
    [0, 2], // 14
    [2, 2], // 81
    [0, 4], // 55
    [2, 1], // 35
];

#[test]
fn so6_su4_dimensions_agree() {
    for a in SO6_LABELS {
        let s = Sun::from_dynkin(a).unwrap();
        let d = su4_to_so6(a);
        assert_eq!(s.dim(), d.dim(), "dim mismatch SU(4) {a:?} vs SO(6)");
    }
}

#[test]
fn so6_su4_products_agree() {
    // Every ordered pair: the SU(4) decomposition, remapped label-by-label to
    // D_3, must equal the SO(6) decomposition exactly (channels and N-symbols).
    for a in SO6_LABELS {
        for b in SO6_LABELS {
            let s = sun_channels(&Sun::from_dynkin(a).unwrap(), &Sun::from_dynkin(b).unwrap());
            let d = bcd_channels(&su4_to_so6(a), &su4_to_so6(b));
            // Remap each SU(4) summand label to its D_3 image and compare.
            let s_mapped: BTreeMap<Vec<i64>, u32> = s
                .into_iter()
                .map(|(k, v)| (su4_to_so6(&k).dynkin(), v))
                .collect();
            assert_eq!(
                s_mapped, d,
                "product {a:?} x {b:?} disagrees SU(4) vs SO(6)"
            );
        }
    }
}

#[test]
fn sp4_so5_dimensions_agree() {
    for c in SO5_LABELS {
        let cc = Bcd::from_dynkin(Series::C, c).unwrap();
        let bb = sp4_to_so5(c);
        assert_eq!(cc.dim(), bb.dim(), "dim mismatch Sp(4) {c:?} vs SO(5)");
    }
}

#[test]
fn sp4_so5_products_agree() {
    for a in SO5_LABELS {
        for b in SO5_LABELS {
            let cc = bcd_channels(
                &Bcd::from_dynkin(Series::C, a).unwrap(),
                &Bcd::from_dynkin(Series::C, b).unwrap(),
            );
            let bb = bcd_channels(&sp4_to_so5(a), &sp4_to_so5(b));
            let c_mapped: BTreeMap<Vec<i64>, u32> = cc
                .into_iter()
                .map(|(k, v)| (sp4_to_so5(&k).dynkin(), v))
                .collect();
            assert_eq!(
                c_mapped, bb,
                "product {a:?} x {b:?} disagrees Sp(4) vs SO(5)"
            );
        }
    }
}

/// SU(2) fusion multiplicity of doubled-spin `dj3` in `dj1 ⊗ dj2`, via racah's
/// exact closed-form base module ([`racah::su2`]): the highest-weight Clebsch–
/// Gordan coefficient is nonzero iff the coupling is allowed, and SU(2)
/// multiplicities are 0/1. This is an independent code path from the GT engine.
fn su2_mult_base(dj1: i32, dj2: i32, dj3: i32) -> u32 {
    // Highest-weight state of j3: m3 = dj3, m1 = dj1, m2 = dj3 - dj1.
    let dm2 = dj3 - dj1;
    if dm2.abs() > dj2 {
        return 0;
    }
    let cg = clebsch_gordan(dj1 as u32, dj1, dj2 as u32, dm2, dj3 as u32, dj3);
    (cg.sign() != 0) as u32
}

#[test]
fn so4_su2xsu2_two_paths_agree() {
    // SO(4) tensor irreps (p + q even), spins (p/2, q/2). Check both the
    // dimension identity and that the two SU(2) code paths give the same
    // fusion multiplicities on each factor.
    let spins = [0i64, 1, 2, 3, 4]; // doubled spins dj
    for &p in &spins {
        for &q in &spins {
            if (p + q) % 2 != 0 {
                continue; // not a tensor (linear) SO(4) irrep
            }
            // dim (p+1)(q+1) via the sun N=2 dims, integer-exact.
            let dl = Sun::from_dynkin(&[p]).unwrap().dim();
            let dr = Sun::from_dynkin(&[q]).unwrap().dim();
            assert_eq!(dl.clone() * dr, num_bigint::BigInt::from((p + 1) * (q + 1)));

            // Factor products: SU(2) via GT engine (sun N=2) vs exact base. Only
            // the left factor (p, p2) is exercised — the right factor (q) obeys the
            // identical SU(2) rule, so a q2 loop would add redundant, not new,
            // coverage (the reviewer's dead-loop nit, PR #33).
            for &p2 in &spins {
                let gt_left = sun_channels(
                    &Sun::from_dynkin(&[p]).unwrap(),
                    &Sun::from_dynkin(&[p2]).unwrap(),
                );
                // Compare against base-module multiplicities for every dj3.
                for dj3 in 0..=(p + p2) {
                    let gt = *gt_left.get(&vec![dj3]).unwrap_or(&0);
                    let base = su2_mult_base(p as i32, p2 as i32, dj3 as i32);
                    assert_eq!(gt, base, "SU(2) mult {p} x {p2} -> {dj3}");
                }
            }
        }
    }
}

/// Negative control: the isomorphism tests have teeth. A deliberately wrong
/// `SO(6)` label map (dropping the node-0/node-1 swap) must break the
/// dimension agreement, proving the passing tests are not vacuous.
#[test]
fn wrong_so6_map_is_detected() {
    // Identity map (a_1,a_2,a_3) -> D_3 (a_1,a_2,a_3) is wrong for e.g. the
    // vector/6: SU(4) [0,1,0] is dim 6, but D_3 [0,1,0] is a spinor (rejected)
    // or, for [2,0,0] (dim 10), D_3 [2,0,0] is dim 20 != 10.
    let s = Sun::from_dynkin(&[2, 0, 0]).unwrap();
    let wrong = Bcd::from_dynkin(Series::D, &[2, 0, 0]).unwrap();
    assert_ne!(
        s.dim(),
        wrong.dim(),
        "wrong map must not accidentally agree"
    );
}
