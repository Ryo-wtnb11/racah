//! Golden gauge tripwire: a small committed table of coefficient values that
//! the frozen gauge specification (`docs/gauge.md`, `docs/gauge_soN.md`) fixes.
//!
//! # What this suite is, and is not
//!
//! It is **not** an oracle. Every number below was produced by this crate, so
//! it proves nothing about correctness — that is the job of the independent
//! suites (`tests/sun_cgc_fixtures.rs` against SUNRepresentations.jl,
//! `tests/sun_fr_su3_oracle.rs`, `src/bcd/qspace_oracle_tests.rs` against
//! QSpace, `tests/su2_embedding.rs`).
//!
//! It is the **drift detector** for the frozen spec: a change anywhere in the
//! pipeline that moves a coefficient value fails here, in the default `cgc-gen`
//! test run, with no reference toolchain in the loop. Under the freeze
//! (`docs/gauge.md` §Status) such a change is a defect unless it ships as a
//! spec correction with an epoch bump — and this is what makes it impossible to
//! ship silently.
//!
//! It covers the two places the oracle suites do not fire by default:
//!
//! - **SU(3) F symbols** — pinned externally only by the `#[ignore]`d heavy
//!   table oracle (`tests/sun_fr_su3_oracle.rs`), so an F-contraction gauge
//!   drift passes an ordinary `cargo test`.
//! - **SO(5) / Sp(4) CGC** — the QSpace anchor compares the isotypic projector
//!   `P = Σ_μ C_μ C_μᵀ`, which is invariant under exactly the coupled-side
//!   gauge this spec fixes (`docs/gauge_soN.md` §8 sign convention, §4 sweep
//!   order, §9 OM assignment). No test pins a signed B/C/D coefficient. This
//!   one does.
//!
//! The SU(3) CGC rows are the cheap overlap with the SUNRepresentations
//! fixtures, kept because they include the OM = 2 adjoint vertex — the entry
//! most sensitive to the multiplicity-column order (`docs/gauge.md` §8), which
//! is the rule with no independent structural check at all.
//!
//! # Tolerance
//!
//! `1e-12`, not the last ULP. The spec deliberately promises value agreement
//! within the oracle tolerance rather than cross-process bit-identity
//! (`docs/gauge.md` header, §10; `docs/gauge_soN.md` §12): the dense backend's
//! parallel reductions are not bit-reproducible, so an exact `==` here would be
//! a flake, not a tripwire. `1e-12` sits four orders below the external oracle
//! budget (`1e-8`) and far above the observed round-off floor, so it catches
//! every discrete gauge event (a sign flip, a column swap, a pivot change,
//! reordered multiplicity) while ignoring the ULP class the spec disclaims.
//!
//! # Regenerating (spec-correction path only)
//!
//! Values change only when the specification is corrected. Print the current
//! numbers with
//!
//! ```text
//! cargo test --features cgc-gen --test gauge_golden -- --ignored --nocapture print_golden
//! ```
//!
//! and update this file in the same PR as the spec edit, the `epoch` bump, and
//! the CHANGELOG breaking-change entry (`docs/gauge.md` §Status).

#![cfg(feature = "cgc-gen")]
// The golden values are literals on purpose: they are the frozen data this
// suite exists to pin. Rewriting one as `FRAC_1_SQRT_2` would hide a drift
// that lands on a nearby constant, and reintroduces a computation where the
// point is to have none.
#![allow(clippy::approx_constant)]

use racah::bcd::{f_symbol as bcd_f_symbol, CanonicalCatalog, Irrep as Bcd, Series};
use racah::group::GroupId;
use racah::sun::{cgc, f_symbol, Irrep as Sun};

/// Discrete-event tolerance; see the module docs.
const TOL: f64 = 1e-12;

#[track_caller]
fn close(what: &str, got: f64, want: f64) {
    let err = (got - want).abs();
    assert!(
        err <= TOL,
        "gauge drift in {what}: got={got:.17e} want={want:.17e} err={err:e}\n\
         If this is a deliberate specification correction, it needs a docs/gauge*.md \
         edit, an authority-fingerprint epoch bump, and a CHANGELOG breaking-change \
         entry in the same PR (docs/gauge.md, Status).",
    );
}

fn sun(d: &[i64]) -> Sun {
    Sun::from_dynkin(d).unwrap()
}

fn bcd(series: Series, d: &[i64]) -> Bcd {
    Bcd::from_dynkin(series, d).unwrap()
}

// ---------------------------------------------------------------------------
// SU(N): CGC.
// ---------------------------------------------------------------------------

/// `(m1, m2, m3, mu, value)` in the GT basis order of `docs/gauge.md` §1,
/// 0-based, with `mu` the multiplicity axis of §8.
type SunRow = (u32, u32, u32, u32, f64);

/// SU(3) `8 ⊗ 8 → 8`: the outer-multiplicity-2 vertex. The `mu` split is fixed
/// by the single block `qrpos! ∘ cref!` gauge-fix of `docs/gauge.md` §4/§8, so
/// swapping the two columns (or re-signing one) fails here.
const SU3_ADJOINT_CUBED: &[SunRow] = &[
    // The same (m1, m2, m3) on both multiplicity columns: a swapped or
    // re-signed mu axis moves these two and nothing else.
    (0, 2, 0, 0, 0.534_522_483_824_848_6),
    (0, 2, 0, 1, -0.119_522_860_933_439_46),
    (0, 4, 0, 1, 0.483_045_891_539_647_8),
    (0, 5, 1, 1, 0.683_130_051_063_973_6),
    (0, 6, 3, 0, 0.654_653_670_707_976_9),
    (0, 7, 2, 0, -0.267_261_241_912_424_4),
];

/// SU(3) `3 ⊗ 3̄ → 8`: multiplicity-free, sensitive to the GT basis order (§1)
/// and the highest-weight column order (§2).
const SU3_FUNDAMENTAL_ADJOINT: &[SunRow] = &[
    (0, 0, 0, 0, 0.999_999_999_999_999_8),
    (0, 2, 2, 0, 0.816_496_580_927_726_1),
    (1, 1, 2, 0, 0.408_248_290_463_863_1),
    (1, 1, 4, 0, 0.707_106_781_186_547_4),
];

fn sun_value(c: &racah::sun::Cgc, m1: u32, m2: u32, m3: u32, mu: u32) -> f64 {
    c.entries()
        .iter()
        .find(|e| e.m1 == m1 && e.m2 == m2 && e.m3 == m3 && e.mu == mu)
        .map_or(0.0, |e| e.value)
}

#[test]
fn su3_cgc_matches_the_frozen_gauge() {
    let adj = sun(&[1, 1]);
    let c = cgc(&adj, &adj, &adj).unwrap();
    assert_eq!(c.multiplicity(), 2, "8 x 8 -> 8 must have OM 2");
    for &(m1, m2, m3, mu, want) in SU3_ADJOINT_CUBED {
        close(
            &format!("su3 cgc 8x8->8 [{m1},{m2},{m3},{mu}]"),
            sun_value(&c, m1, m2, m3, mu),
            want,
        );
    }

    let c = cgc(&sun(&[1, 0]), &sun(&[0, 1]), &adj).unwrap();
    for &(m1, m2, m3, mu, want) in SU3_FUNDAMENTAL_ADJOINT {
        close(
            &format!("su3 cgc 3x3bar->8 [{m1},{m2},{m3},{mu}]"),
            sun_value(&c, m1, m2, m3, mu),
            want,
        );
    }
}

// ---------------------------------------------------------------------------
// SU(N): F symbols (the CGC contraction, only heavy-oracle-pinned otherwise).
// ---------------------------------------------------------------------------

/// `F^{3,3,3}_{1}` with `e = f = 3̄`: every vertex is multiplicity-free, so the
/// block is the single scalar below. It is a pure function of the CGC gauge
/// through the §11 contraction, which makes it the cheapest F-side tripwire.
const SU3_F_333_1: f64 = 0.999_999_999_999_999_9;

#[test]
fn su3_f_symbol_matches_the_frozen_gauge() {
    let three = sun(&[1, 0]);
    let bar = sun(&[0, 1]);
    let one = sun(&[0, 0]);
    let block = f_symbol(&three, &three, &three, &one, &bar, &bar).unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    close("su3 F 3,3,3->1", block.at(0, 0, 0, 0), SU3_F_333_1);
}

// ---------------------------------------------------------------------------
// SO(N)/Sp(2N): signed CGC entries (nothing else pins their sign).
// ---------------------------------------------------------------------------

/// `(mu, row, col, value)` into `CatalogCgc::copy(mu)`, a column-major
/// `d1·d2 × d3` isometry; the product row index is `i1 + d1·i2`
/// (`docs/gauge_soN.md` §1).
type BcdRow = (usize, usize, usize, f64);

/// SO(5) `[1,0] ⊗ [1,0] → [0,2]` (vector² → adjoint, 5⊗5 → 10).
///
/// Regenerated at `epoch = 2` (issue #90): the defining rep is now stored in the
/// sweep's descending-weight frame, which permutes the product rows `i1 + 5·i2`
/// of every coupling with a defining-rep factor — the old coordinates
/// `(2,0)/(10,0)/(14,1)` are exactly zero in the corrected frame. The Sp(4) rows
/// below are byte-identical across the correction (`Setup_SpN` was already in
/// that order), which is the C-family half of the claim.
const SO5_V_V_ADJ: &[BcdRow] = &[
    (0, 1, 0, 0.707_106_781_186_547_6),
    (0, 5, 0, -0.707_106_781_186_547_6),
    (0, 4, 4, 0.5),
    (0, 4, 5, -0.5),
];

/// Sp(4) `[1,0] ⊗ [1,0] → [2,0]` (defining² → adjoint, 4⊗4 → 10).
const SP4_V_V_ADJ: &[BcdRow] = &[
    (0, 0, 0, 1.0),
    (0, 1, 1, 0.707_106_781_186_547_6),
    (0, 2, 3, 0.707_106_781_186_547_6),
];

fn bcd_value(c: &racah::bcd::CatalogCgc, mu: usize, row: usize, col: usize) -> f64 {
    let (rows, _) = c.copy_shape();
    c.copy(mu)[col * rows + row]
}

#[test]
fn so5_and_sp4_cgc_match_the_frozen_gauge() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let c = cat
        .cgc(
            &bcd(Series::B, &[1, 0]),
            &bcd(Series::B, &[1, 0]),
            &bcd(Series::B, &[0, 2]),
        )
        .unwrap();
    for &(mu, row, col, want) in SO5_V_V_ADJ {
        close(
            &format!("so5 cgc [1,0]x[1,0]->[0,2] mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }

    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let c = cat
        .cgc(
            &bcd(Series::C, &[1, 0]),
            &bcd(Series::C, &[1, 0]),
            &bcd(Series::C, &[2, 0]),
        )
        .unwrap();
    for &(mu, row, col, want) in SP4_V_V_ADJ {
        close(
            &format!("sp4 cgc [1,0]x[1,0]->[2,0] mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }
}

/// Sp(4) F symbol: the B/C/D contraction path, multiplicity-free vertices.
const SP4_F: f64 = 0.790_569_415_042_094_8;

#[test]
fn sp4_f_symbol_matches_the_frozen_gauge() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = bcd(Series::C, &[1, 0]);
    let one = bcd(Series::C, &[0, 0]);
    let adj = bcd(Series::C, &[2, 0]);
    let block = bcd_f_symbol(&mut cat, &v, &v, &v, &v, &one, &adj).unwrap();
    close("sp4 F [1,0]^3", block.at(0, 0, 0, 0), SP4_F);
}

// ---------------------------------------------------------------------------
// Spin(N): signed spinor CGC and F (issue #54; docs/gauge_soN.md §16).
//
// The spinor sector is new gauge, so it needs its own pins from the day it
// ships: nothing else fixes the sign of a spinor coefficient, and the seed
// conventions of §16 (Fock index order, Jordan–Wigner signs, the descending
// weight order, the 1/√2 short-root scale) are all value-affecting. These rows
// are the tripwire for every one of them.
// ---------------------------------------------------------------------------

fn spin(n: usize, d: &[i64]) -> Bcd {
    Bcd::from_dynkin_in(&GroupId::spin(n).unwrap(), d).unwrap()
}

/// `Spin(5)`: `4 ⊗ 4 → 10` (spinor² → adjoint).
const SPIN5_S_S_ADJ: &[BcdRow] = &[
    (0, 0, 0, 1.0),
    (0, 1, 1, 0.707_106_781_186_547_6),
    (0, 4, 1, 0.707_106_781_186_547_6),
    (0, 5, 2, 1.0),
];

/// `Spin(6)`: `4 ⊗ 4̄ → 15` (half-spinor × conjugate → adjoint).
const SPIN6_S_SBAR_ADJ: &[BcdRow] = &[
    (0, 0, 0, 1.0),
    (0, 4, 1, 1.0),
    (0, 1, 2, 1.0),
    (0, 5, 3, 1.0),
];

/// `Spin(7)`: `8 ⊗ 8 → 21` (spinor² → adjoint).
const SPIN7_S_S_ADJ: &[BcdRow] = &[
    (0, 1, 0, 0.707_106_781_186_547_6),
    (0, 8, 0, -0.707_106_781_186_547_6),
    (0, 2, 1, 0.707_106_781_186_547_6),
    (0, 16, 1, -0.707_106_781_186_547_6),
];

/// `Spin(10)`: `16 ⊗ 1̄6 → 45`. The rank-5 pin — it exercises a five-mode
/// Clifford seed and a `256`-dimensional product sweep.
const SPIN10_S_SBAR_ADJ: &[BcdRow] = &[
    (0, 3, 0, 0.5),
    (0, 18, 0, -0.500_000_000_000_000_1),
    (0, 33, 0, 0.5),
    (0, 48, 0, -0.500_000_000_000_000_1),
];

#[test]
fn spinor_cgc_matches_the_frozen_gauge() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let c = cat
        .cgc(&spin(5, &[0, 1]), &spin(5, &[0, 1]), &spin(5, &[0, 2]))
        .unwrap();
    for &(mu, row, col, want) in SPIN5_S_S_ADJ {
        close(
            &format!("spin5 cgc 4x4->10 mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }

    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let c = cat
        .cgc(
            &spin(6, &[0, 0, 1]),
            &spin(6, &[0, 1, 0]),
            &spin(6, &[0, 1, 1]),
        )
        .unwrap();
    for &(mu, row, col, want) in SPIN6_S_SBAR_ADJ {
        close(
            &format!("spin6 cgc 4x4bar->15 mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }

    let mut cat = CanonicalCatalog::new(Series::B, 3).unwrap();
    let c = cat
        .cgc(
            &spin(7, &[0, 0, 1]),
            &spin(7, &[0, 0, 1]),
            &spin(7, &[0, 1, 0]),
        )
        .unwrap();
    for &(mu, row, col, want) in SPIN7_S_S_ADJ {
        close(
            &format!("spin7 cgc 8x8->21 mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }

    let mut cat = CanonicalCatalog::new(Series::D, 5).unwrap();
    let c = cat
        .cgc(
            &spin(10, &[0, 0, 0, 0, 1]),
            &spin(10, &[0, 0, 0, 1, 0]),
            &spin(10, &[0, 1, 0, 0, 0]),
        )
        .unwrap();
    for &(mu, row, col, want) in SPIN10_S_SBAR_ADJ {
        close(
            &format!("spin10 cgc 16x16bar->45 mu={mu} ({row},{col})"),
            bcd_value(&c, mu, row, col),
            want,
        );
    }
}

/// `Spin(6)` F symbol on the `4 ⊗ 4̄` spinor family, with the trivial internal
/// channel. Every vertex is multiplicity-free and no vertex couples into the
/// defining rep, so this is a clean spinor-side F pin.
const SPIN6_F_TRIVIAL_INTERNAL: f64 = 0.968_245_836_551_854_4;
/// The same family with the adjoint internal channel.
const SPIN6_F_ADJOINT_INTERNAL: f64 = 0.250_000_000_000_000_06;

#[test]
fn spinor_f_symbol_matches_the_frozen_gauge() {
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let s = spin(6, &[0, 0, 1]);
    let sb = spin(6, &[0, 1, 0]);
    let one = spin(6, &[0, 0, 0]);
    let adj = spin(6, &[0, 1, 1]);

    let block = bcd_f_symbol(&mut cat, &s, &sb, &s, &s, &one, &adj).unwrap();
    close(
        "spin6 F 4,4bar,4->4 (e=1)",
        block.at(0, 0, 0, 0),
        SPIN6_F_TRIVIAL_INTERNAL,
    );

    let block = bcd_f_symbol(&mut cat, &s, &sb, &s, &s, &adj, &adj).unwrap();
    close(
        "spin6 F 4,4bar,4->4 (e=15)",
        block.at(0, 0, 0, 0),
        SPIN6_F_ADJOINT_INTERNAL,
    );
}

// ---------------------------------------------------------------------------
// Regeneration helper (spec-correction path only; see the module docs).
// ---------------------------------------------------------------------------

/// Print every significant entry of a B/C/D CGC as a `BcdRow` literal. The
/// pinned coordinates are not stable across a specification correction — a
/// re-framed factor basis permutes the product rows (issue #90 moved the SO(5)
/// rows off their old coordinates entirely) — so the regeneration path dumps the
/// whole block and the pins are re-picked from it.
fn print_nonzeros(c: &racah::bcd::CatalogCgc) {
    let (rows, cols) = c.copy_shape();
    for mu in 0..c.multiplicity() {
        for col in 0..cols {
            for row in 0..rows {
                let v = c.copy(mu)[col * rows + row];
                if v.abs() > 1e-9 {
                    println!("    ({mu}, {row}, {col}, {v:?}),");
                }
            }
        }
    }
}

#[test]
#[ignore = "prints the current values for a spec correction; not an assertion"]
fn print_golden() {
    let adj = sun(&[1, 1]);
    let c = cgc(&adj, &adj, &adj).unwrap();
    println!("SU3_ADJOINT_CUBED:");
    for &(m1, m2, m3, mu, _) in SU3_ADJOINT_CUBED {
        println!(
            "    ({m1}, {m2}, {m3}, {mu}, {:?}),",
            sun_value(&c, m1, m2, m3, mu)
        );
    }
    let c = cgc(&sun(&[1, 0]), &sun(&[0, 1]), &adj).unwrap();
    println!("SU3_FUNDAMENTAL_ADJOINT:");
    for &(m1, m2, m3, mu, _) in SU3_FUNDAMENTAL_ADJOINT {
        println!(
            "    ({m1}, {m2}, {m3}, {mu}, {:?}),",
            sun_value(&c, m1, m2, m3, mu)
        );
    }
    let block = f_symbol(
        &sun(&[1, 0]),
        &sun(&[1, 0]),
        &sun(&[1, 0]),
        &sun(&[0, 0]),
        &sun(&[0, 1]),
        &sun(&[0, 1]),
    )
    .unwrap();
    println!("SU3_F_333_1 = {:?};", block.at(0, 0, 0, 0));

    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let c = cat
        .cgc(
            &bcd(Series::B, &[1, 0]),
            &bcd(Series::B, &[1, 0]),
            &bcd(Series::B, &[0, 2]),
        )
        .unwrap();
    println!("SO5_V_V_ADJ: shape {:?}", c.copy_shape());
    print_nonzeros(&c);

    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let c = cat
        .cgc(
            &bcd(Series::C, &[1, 0]),
            &bcd(Series::C, &[1, 0]),
            &bcd(Series::C, &[2, 0]),
        )
        .unwrap();
    println!("SP4_V_V_ADJ: shape {:?}", c.copy_shape());
    print_nonzeros(&c);
    let v = bcd(Series::C, &[1, 0]);
    let block = bcd_f_symbol(
        &mut cat,
        &v,
        &v,
        &v,
        &v,
        &bcd(Series::C, &[0, 0]),
        &bcd(Series::C, &[2, 0]),
    )
    .unwrap();
    println!("SP4_F = {:?};", block.at(0, 0, 0, 0));

    // Spin(N): the §16 spinor rows.
    /// `(series, rank, N, [a, b, c], tag)` for one spinor CGC row block.
    type SpinorBlock = (Series, usize, usize, [&'static [i64]; 3], &'static str);
    let spinor_blocks: &[SpinorBlock] = &[
        (
            Series::B,
            2,
            5,
            [&[0, 1], &[0, 1], &[0, 2]],
            "SPIN5_S_S_ADJ",
        ),
        (
            Series::D,
            3,
            6,
            [&[0, 0, 1], &[0, 1, 0], &[0, 1, 1]],
            "SPIN6_S_SBAR_ADJ",
        ),
        (
            Series::B,
            3,
            7,
            [&[0, 0, 1], &[0, 0, 1], &[0, 1, 0]],
            "SPIN7_S_S_ADJ",
        ),
        (
            Series::D,
            5,
            10,
            [&[0, 0, 0, 0, 1], &[0, 0, 0, 1, 0], &[0, 1, 0, 0, 0]],
            "SPIN10_S_SBAR_ADJ",
        ),
    ];
    for (series, r, n, labels, tag) in spinor_blocks {
        let mut cat = CanonicalCatalog::new(*series, *r).unwrap();
        let c = cat
            .cgc(
                &spin(*n, labels[0]),
                &spin(*n, labels[1]),
                &spin(*n, labels[2]),
            )
            .unwrap();
        println!("{tag}: shape {:?}", c.copy_shape());
        print_nonzeros(&c);
    }

    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let s = spin(6, &[0, 0, 1]);
    let sb = spin(6, &[0, 1, 0]);
    let one = spin(6, &[0, 0, 0]);
    let adj = spin(6, &[0, 1, 1]);
    println!(
        "SPIN6_F_TRIVIAL_INTERNAL = {:?};",
        bcd_f_symbol(&mut cat, &s, &sb, &s, &s, &one, &adj)
            .unwrap()
            .at(0, 0, 0, 0)
    );
    println!(
        "SPIN6_F_ADJOINT_INTERNAL = {:?};",
        bcd_f_symbol(&mut cat, &s, &sb, &s, &s, &adj, &adj)
            .unwrap()
            .at(0, 0, 0, 0)
    );
}
