//! Exact 6j agreement against checked-in fixtures generated offline by
//! WignerSymbols.jl, for doubled spins beyond the `wigner-symbols 0.5.1` domain
//! (255..600 and a few in the thousands). See `tools/gen_fixtures.jl`.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::Zero;

use racah::wigner_6j;

const FIXTURES: &str = include_str!("fixtures/su2_6j_large.txt");

#[test]
fn six_j_large_fixtures_exact() {
    let mut checked = 0u64;
    for (lineno, line) in FIXTURES.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(f.len(), 9, "malformed fixture line {}: {line}", lineno + 1);
        let dj: Vec<u32> = f[0..6].iter().map(|s| s.parse().unwrap()).collect();
        let sign: i8 = f[6].parse().unwrap();
        let num: BigInt = f[7].parse().unwrap();
        let den: BigInt = f[8].parse().unwrap();

        let expected = match sign {
            0 => Ratio::zero(),
            1 => Ratio::new(num, den),
            -1 => Ratio::new(-num, den),
            _ => panic!("bad sign in fixture line {}", lineno + 1),
        };

        let got = wigner_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]);
        assert_eq!(
            got.signed_square(),
            expected,
            "6j mismatch at fixture line {}: {dj:?}",
            lineno + 1
        );
        checked += 1;
    }
    assert!(checked >= 20, "expected many fixtures, got {checked}");
}
