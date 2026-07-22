//! Exact 6j agreement against checked-in fixtures generated offline by
//! WignerSymbols.jl, for doubled spins beyond the `wigner-symbols 0.5.1` domain
//! (255..600 and a few in the thousands). See `tools/gen_fixtures.jl`.
//!
//! Split into two tiers by spin magnitude. Both run by default: the
//! prime-factorized engine (issue #3) evaluates the thousands tier in a
//! fraction of a second, where the earlier direct big-rational sum took
//! minutes, so the no-ceiling property is now a default-suite gate.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;

use racah::wigner_6j;

const FIXTURES: &str = include_str!("fixtures/su2_6j_large.txt");

/// A parsed fixture row: doubled spins and the expected signed square.
struct Row {
    dj: [u32; 6],
    expected: Ratio<BigInt>,
    lineno: usize,
}

fn rows() -> Vec<Row> {
    let mut out = Vec::new();
    for (i, line) in FIXTURES.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 9, "malformed fixture line {}: {line}", i + 1);
        let mut dj = [0u32; 6];
        for (slot, s) in dj.iter_mut().zip(&f[0..6]) {
            *slot = s.parse().unwrap();
        }
        let sign: i8 = f[6].parse().unwrap();
        let num: BigInt = f[7].parse().unwrap();
        let den: BigInt = f[8].parse().unwrap();
        let expected = match sign {
            0 => Ratio::zero(),
            1 => Ratio::new(num, den),
            -1 => Ratio::new(-num, den),
            _ => panic!("bad sign in fixture line {}", i + 1),
        };
        out.push(Row {
            dj,
            expected,
            lineno: i + 1,
        });
    }
    out
}

fn check(row: &Row) {
    let d = row.dj;
    let got = wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]);
    assert_eq!(
        got.signed_square(),
        row.expected,
        "6j mismatch at fixture line {}: {:?}",
        row.lineno,
        row.dj
    );
}

#[test]
fn six_j_large_fixtures_exact() {
    // Doubled spins in 255..600: beyond the reference-crate u8 ceiling, cheap
    // enough to run every build.
    let mut checked = 0u64;
    for row in rows().iter().filter(|r| r.dj.iter().all(|&x| x <= 600)) {
        check(row);
        checked += 1;
    }
    assert!(
        checked >= 20,
        "expected many small-tier fixtures, got {checked}"
    );
}

#[test]
fn six_j_huge_fixtures_exact() {
    // Doubled spins in the thousands: proves the no-ceiling property. With the
    // prime-factorized engine this runs in well under a second.
    let mut checked = 0u64;
    for row in rows().iter().filter(|r| r.dj.iter().any(|&x| x > 600)) {
        check(row);
        checked += 1;
    }
    assert!(
        checked >= 3,
        "expected a few thousands-tier fixtures, got {checked}"
    );
}
