//! SU(3) table-regeneration oracle (oracle 1 of issue #16) — the
//! gauge-continuity gate that authorizes a downstream consumer to delete its
//! precomputed SU(3) F/R table.
//!
//! For **every** admissible SU(3) sextet (F) and triple (R) in which all irreps
//! have Weyl dimension ≤ 27 (the table's cut), racah's generated F/R must agree
//! with SUNRepresentations.jl v0.4.0 **signed, element-wise**. Both compute the
//! same Gelfand–Tsetlin construction in the same gauge, so a faithful port
//! agrees to near machine precision; a divergent value (or a divergent
//! multiplicity-axis order) fails here.
//!
//! Fixtures are emitted by `tools/gen_su3_fr_fixtures.jl` directly from
//! SUNRepresentations (never hand-authored); see the provenance headers.
//!
//! Two tiers, same fixtures and same signed element-wise assertion:
//! - a **dim ≤ 8 slice** (all labels among `{1, 3, 3̄, 6, 6̄, 8}`) runs in the
//!   default `cargo test` (a few hundred blocks, ~1s) so CI always exercises
//!   the table-deletion gate;
//! - the **full dim ≤ 27 sweep** (76 853 F blocks, CGC up to `27 ⊗ 27`) is
//!   `#[ignore]`d and run explicitly
//!   (`cargo test --release --features cgc-gen -- --ignored`).

#![cfg(feature = "cgc-gen")]

use racah::sun::{f_symbol, r_symbol, Irrep};

/// Signed element-wise tolerance. Same algorithm and gauge on both sides, so a
/// faithful port agrees to the CGC pipeline's SVD/QR/descent round-off floor,
/// far below any coefficient of interest (a structural error is O(1)).
const TOL: f64 = 1e-8;

fn dynkin(s: &str) -> Irrep {
    let d: Vec<i64> = s.split(',').map(|x| x.parse().unwrap()).collect();
    Irrep::from_dynkin(&d).unwrap()
}

fn data_lines(s: &str) -> impl Iterator<Item = &str> {
    s.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// Weyl dimension of an SU(3) irrep from its `"p,q"` Dynkin string.
fn weyl_dim(dynkin_str: &str) -> i64 {
    let d: Vec<i64> = dynkin_str.split(',').map(|x| x.parse().unwrap()).collect();
    let (p, q) = (d[0], d[1]);
    (p + 1) * (q + 1) * (p + q + 2) / 2
}

/// Check every F fixture line whose six labels all pass `keep`, returning
/// `(elements checked, worst |Δ|)`. Blocks are computed once per sextet.
fn check_f(keep: impl Fn(&str) -> bool) -> (u64, f64) {
    let text = include_str!("fixtures/su3_fr_f.txt");
    let (mut checked, mut worst) = (0u64, 0.0f64);
    let mut cur_key: Option<[String; 6]> = None;
    let mut cur_block = None;
    for line in data_lines(text) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 11, "malformed F line: {line}");
        if !f[0..6].iter().all(|s| keep(s)) {
            continue;
        }
        let key: [String; 6] = std::array::from_fn(|i| f[i].to_string());
        if cur_key.as_ref() != Some(&key) {
            cur_block = Some(
                f_symbol(
                    &dynkin(f[0]),
                    &dynkin(f[1]),
                    &dynkin(f[2]),
                    &dynkin(f[3]),
                    &dynkin(f[4]),
                    &dynkin(f[5]),
                )
                .unwrap_or_else(|e| panic!("f_symbol {key:?}: {e}")),
            );
            cur_key = Some(key.clone());
        }
        let (mu, nu, ka, la): (usize, usize, usize, usize) = (
            f[6].parse().unwrap(),
            f[7].parse().unwrap(),
            f[8].parse().unwrap(),
            f[9].parse().unwrap(),
        );
        let want: f64 = f[10].parse().unwrap();
        let got = cur_block.as_ref().unwrap().at(mu, nu, ka, la);
        let err = (got - want).abs();
        worst = worst.max(err);
        assert!(
            err <= TOL,
            "F mismatch {key:?} [{mu},{nu},{ka},{la}]: got={got} want={want} err={err:e}"
        );
        checked += 1;
    }
    (checked, worst)
}

/// Check every R fixture line whose three labels all pass `keep`.
fn check_r(keep: impl Fn(&str) -> bool) -> (u64, f64) {
    let text = include_str!("fixtures/su3_fr_r.txt");
    let (mut checked, mut worst) = (0u64, 0.0f64);
    let mut cur_key: Option<[String; 3]> = None;
    let mut cur_block = None;
    for line in data_lines(text) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 6, "malformed R line: {line}");
        if !f[0..3].iter().all(|s| keep(s)) {
            continue;
        }
        let key: [String; 3] = std::array::from_fn(|i| f[i].to_string());
        if cur_key.as_ref() != Some(&key) {
            cur_block = Some(
                r_symbol(&dynkin(f[0]), &dynkin(f[1]), &dynkin(f[2]))
                    .unwrap_or_else(|e| panic!("r_symbol {key:?}: {e}")),
            );
            cur_key = Some(key.clone());
        }
        let (mu, nu): (usize, usize) = (f[3].parse().unwrap(), f[4].parse().unwrap());
        let want: f64 = f[5].parse().unwrap();
        let got = cur_block.as_ref().unwrap().at(mu, nu);
        let err = (got - want).abs();
        worst = worst.max(err);
        assert!(
            err <= TOL,
            "R mismatch {key:?} [{mu},{nu}]: got={got} want={want} err={err:e}"
        );
        checked += 1;
    }
    (checked, worst)
}

// ---- dim <= 8 slice: runs in default `cargo test` (CI's table-deletion gate) ----

#[test]
fn f_table_dim8_slice_regenerates_from_cgc() {
    let (checked, worst) = check_f(|s| weyl_dim(s) <= 8);
    assert!(checked > 200, "expected the dim<=8 slice, got {checked}");
    println!("SU(3) F table oracle (dim<=8): {checked} elements, worst |Δ| {worst:e}");
}

#[test]
fn r_table_dim8_slice_regenerates_from_cgc() {
    let (checked, worst) = check_r(|s| weyl_dim(s) <= 8);
    assert!(checked > 50, "expected the dim<=8 R slice, got {checked}");
    println!("SU(3) R table oracle (dim<=8): {checked} elements, worst |Δ| {worst:e}");
}

// ---- full dim <= 27 sweep: heavy, run explicitly with --ignored --release ----

#[test]
#[ignore = "heavy: regenerates the whole SU(3) dim<=27 F table; run with --ignored --release"]
fn f_table_regenerates_from_cgc() {
    let (checked, worst) = check_f(|_| true);
    assert!(checked > 100_000, "expected the full table, got {checked}");
    println!("SU(3) F table oracle: {checked} elements, worst |Δ| {worst:e}");
}

#[test]
#[ignore = "heavy: regenerates the whole SU(3) dim<=27 R table; run with --ignored --release"]
fn r_table_regenerates_from_cgc() {
    let (checked, worst) = check_r(|_| true);
    assert!(checked > 500, "expected the full R table, got {checked}");
    println!("SU(3) R table oracle: {checked} elements, worst |Δ| {worst:e}");
}
