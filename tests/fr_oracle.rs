//! SU(2) F-symbol and R-symbol agreement against TensorKitSectors, the crate's
//! F/R convention oracle. Fixtures are generated offline by
//! `tools/gen_fr_fixtures.jl` over all admissible doubled spins <= 12, with a
//! provenance header recording the resolved package versions.
//!
//! racah composes F exactly (big-rational 6j times the dimension factor under
//! the root, phase in the sign) and rounds once, so it is at least as accurate
//! as the reference's Float64 arithmetic; the tolerance covers the reference's
//! own last-ulp rounding, not any racah error. R is an exact +-1/0 and must
//! match to the bit.

use racah::{su2_f_symbol, su2_r_symbol};

const F_FIXTURES: &str = include_str!("fixtures/su2_f.txt");
const R_FIXTURES: &str = include_str!("fixtures/su2_r.txt");

/// Data lines (skip blank and `#` provenance/header lines).
fn data_lines(s: &str) -> impl Iterator<Item = (usize, &str)> {
    s.lines().enumerate().filter_map(|(i, l)| {
        let l = l.trim();
        (!l.is_empty() && !l.starts_with('#')).then_some((i + 1, l))
    })
}

#[test]
fn f_symbol_matches_tensorkitsectors() {
    // Absolute tolerance: F is O(1) here (|F| stays within a few units for
    // dj <= 12), so 1e-14 is several ulp of headroom over the reference's own
    // Float64 rounding while still catching any structural (phase/order) error,
    // which would be O(1)-sized, not ulp-sized.
    const TOL: f64 = 1e-14;
    let mut max_err = 0.0f64;
    let mut checked = 0u64;
    for (lineno, line) in data_lines(F_FIXTURES) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 7, "malformed F fixture line {lineno}: {line}");
        let mut dj = [0u32; 6];
        for (slot, s) in dj.iter_mut().zip(&f[0..6]) {
            *slot = s.parse().unwrap();
        }
        let want: f64 = f[6].parse().unwrap();
        let got = su2_f_symbol(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]);
        let err = (got - want).abs();
        max_err = max_err.max(err);
        assert!(
            err <= TOL,
            "F mismatch at line {lineno} dj={dj:?}: got={got} want={want} err={err:e}"
        );
        checked += 1;
    }
    assert!(
        checked > 100_000,
        "expected the full dj<=12 sweep, got {checked}"
    );
    println!("F oracle: {checked} cases, max abs err {max_err:e}");
}

#[test]
fn r_symbol_matches_tensorkitsectors() {
    // R is an exact discrete value; require the bit-exact +1.0 / -1.0.
    let mut checked = 0u64;
    for (lineno, line) in data_lines(R_FIXTURES) {
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 4, "malformed R fixture line {lineno}: {line}");
        let dj: [u32; 3] = [
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        ];
        let want: f64 = f[3].parse().unwrap();
        let got = su2_r_symbol(dj[0], dj[1], dj[2]);
        assert_eq!(
            got, want,
            "R mismatch at line {lineno} dj={dj:?}: got={got} want={want}"
        );
        checked += 1;
    }
    assert!(
        checked > 100,
        "expected many admissible R triples, got {checked}"
    );
}
