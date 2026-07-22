//! Exact agreement with SUNRepresentations.jl v0.4.0 for the cgc-gen Layer 1
//! SU(N) combinatorics (issue #10).
//!
//! The oracle is independent: values come from the Julia reference package
//! (`tools/gen_sun_fixtures.jl`), checked in with a provenance header, never
//! from the code under test. We compare dimensions, duals, the load-bearing GT
//! basis order (index-for-index), Littlewood-Richardson multiplicities, and the
//! exact ladder-matrix entries (via `signed_square`, no floats).
#![cfg(feature = "cgc-gen")]

use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

use num_bigint::BigInt;
use num_rational::Ratio;

use racah::sun::{directproduct, Irrep};

const FIXTURES: &str = include_str!("fixtures/sun/sun_fixtures.txt");

/// Fusion decomposition keyed by the product irrep's Dynkin label.
type Decomp = BTreeMap<Vec<i64>, u32>;
/// One ladder entry: level `l`, 1-based `(i, j)`, and `signed_square`.
type LadderRow = (usize, usize, usize, Ratio<BigInt>);

fn csv_i64(s: &str) -> Vec<i64> {
    s.split(',').map(|x| x.parse::<i64>().unwrap()).collect()
}

/// One accumulated fixture corpus, parsed once.
#[derive(Default)]
struct Corpus {
    dims: Vec<(Vec<i64>, BigInt, Vec<i64>)>, // (dynkin, dim, dual_dynkin)
    patterns: HashMap<Vec<i64>, Vec<Vec<i64>>>, // dynkin -> ordered pattern data
    pattern_keys: Vec<Vec<i64>>,             // dynkin, in first-seen order
    products: Vec<(Vec<i64>, Vec<i64>, Decomp)>, // a, b, decomp
    ladders: HashMap<Vec<i64>, Vec<LadderRow>>, // dynkin -> entries
    ladder_keys: Vec<Vec<i64>>,
    ranks_seen: std::collections::BTreeSet<usize>,
}

fn parse() -> Corpus {
    let mut c = Corpus::default();
    for line in FIXTURES.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(';').collect();
        let n: usize = f[1].parse().unwrap();
        c.ranks_seen.insert(n);
        match f[0] {
            "DIM" => {
                let dynkin = csv_i64(f[2]);
                let dim = BigInt::from_str(f[3]).unwrap();
                let dual = csv_i64(f[4]);
                c.dims.push((dynkin, dim, dual));
            }
            "PAT" => {
                let dynkin = csv_i64(f[2]);
                let data = csv_i64(f[3]);
                if !c.patterns.contains_key(&dynkin) {
                    c.pattern_keys.push(dynkin.clone());
                }
                c.patterns.entry(dynkin).or_default().push(data);
            }
            "DP" => {
                let a = csv_i64(f[2]);
                let b = csv_i64(f[3]);
                let mut decomp = BTreeMap::new();
                for part in f[4].split('|') {
                    let (cd, m) = part.split_once(':').unwrap();
                    decomp.insert(csv_i64(cd), m.parse::<u32>().unwrap());
                }
                c.products.push((a, b, decomp));
            }
            "LAD" => {
                let dynkin = csv_i64(f[2]);
                let e = csv_i64(f[3]); // l,i,j,num,den
                let sq = Ratio::new(BigInt::from(e[3]), BigInt::from(e[4]));
                if !c.ladders.contains_key(&dynkin) {
                    c.ladder_keys.push(dynkin.clone());
                }
                c.ladders.entry(dynkin).or_default().push((
                    e[0] as usize,
                    e[1] as usize,
                    e[2] as usize,
                    sq,
                ));
            }
            other => panic!("unknown fixture tag {other}"),
        }
    }
    c
}

#[test]
fn covers_all_ranks_and_enough_cases() {
    let c = parse();
    assert!(
        c.ranks_seen
            .is_superset(&[2, 3, 4, 5].into_iter().collect()),
        "fixtures must span N in 2..=5, saw {:?}",
        c.ranks_seen
    );
    assert!(
        c.products.len() >= 200,
        "expected >= 200 directproduct cases, got {}",
        c.products.len()
    );
    assert!(!c.dims.is_empty() && !c.pattern_keys.is_empty() && !c.ladder_keys.is_empty());
}

#[test]
fn dims_and_duals_match_reference() {
    let c = parse();
    for (dynkin, dim, dual) in &c.dims {
        let s = Irrep::from_dynkin(dynkin).unwrap();
        assert_eq!(&s.dim(), dim, "dim mismatch for dynkin {dynkin:?}");
        assert_eq!(
            &s.dual().dynkin(),
            dual,
            "dual mismatch for dynkin {dynkin:?}"
        );
    }
}

#[test]
fn gt_basis_order_matches_reference_index_for_index() {
    let c = parse();
    for dynkin in &c.pattern_keys {
        let s = Irrep::from_dynkin(dynkin).unwrap();
        let got: Vec<Vec<i64>> = s.patterns().iter().map(|p| p.data().to_vec()).collect();
        let want = &c.patterns[dynkin];
        assert_eq!(&got, want, "GT basis order mismatch for dynkin {dynkin:?}");
    }
}

#[test]
fn directproduct_multiplicities_match_reference() {
    let c = parse();
    for (a, b, decomp) in &c.products {
        let ia = Irrep::from_dynkin(a).unwrap();
        let ib = Irrep::from_dynkin(b).unwrap();
        let got: BTreeMap<Vec<i64>, u32> = directproduct(&ia, &ib)
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k.dynkin(), v))
            .collect();
        assert_eq!(&got, decomp, "directproduct mismatch for {a:?} (x) {b:?}");
    }
}

#[test]
fn ladder_entries_match_reference() {
    let c = parse();
    for dynkin in &c.ladder_keys {
        let s = Irrep::from_dynkin(dynkin).unwrap();
        let cr = s.creation();
        // Our entries -> (l, i, j, signed_square) with 1-based i,j, sorted.
        let mut got: Vec<(usize, usize, usize, Ratio<BigInt>)> = Vec::new();
        for (l0, mat) in cr.iter().enumerate() {
            for e in mat {
                got.push((l0 + 1, e.row + 1, e.col + 1, e.value.signed_square()));
            }
        }
        got.sort_by_key(|x| (x.0, x.1, x.2));
        let mut want = c.ladders[dynkin].clone();
        want.sort_by_key(|x| (x.0, x.1, x.2));
        assert_eq!(got, want, "ladder-entry mismatch for dynkin {dynkin:?}");
    }
}
