//! Gauge-continuity oracle: the generated SU(N) CGC must reproduce
//! SUNRepresentations.jl v0.4.0's coefficients *signed and element-wise*.
//!
//! The claim under test is gauge continuity -- that this crate's port
//! reproduces the reference's deterministic gauge, not merely the same subspace
//! -- so the comparison is on signed values (no per-channel sign or column
//! freedom), including the outer-multiplicity axis order for OM >= 2 channels.
//!
//! Fixtures (`tests/fixtures/sun_cgc.txt`) are emitted by
//! `tools/gen_sun_cgc_fixtures.jl` directly from SUNRepresentations; they are
//! never hand-authored. See the file's provenance header.
//!
//! Tolerance: `1e-8`. Both implementations run the *same* algorithm, so a
//! faithful port agrees to near machine precision; the budget is the reference
//! tolerances (`TOL_GAUGE = 1e-11`) amplified by f64 round-off through the SVD,
//! QR, and multi-step descent, and stays far below any coefficient magnitude of
//! interest.

#![cfg(feature = "cgc-gen")]

use std::collections::HashMap;

use racah::sun::{cgc, Irrep};

const TOL: f64 = 1e-8;

struct Row {
    s1: Vec<i64>,
    s2: Vec<i64>,
    s3: Vec<i64>,
    m1: usize,
    m2: usize,
    m3: usize,
    mu: usize,
    value: f64,
}

fn dynkin(s: &str) -> Vec<i64> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn load() -> Vec<Row> {
    let text = include_str!("fixtures/sun_cgc.txt");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            assert_eq!(f.len(), 9, "bad fixture line: {l}");
            // f[0] = N (implied by dynkin length); indices are 1-based in Julia.
            Row {
                s1: dynkin(f[1]),
                s2: dynkin(f[2]),
                s3: dynkin(f[3]),
                m1: f[4].parse::<usize>().unwrap() - 1,
                m2: f[5].parse::<usize>().unwrap() - 1,
                m3: f[6].parse::<usize>().unwrap() - 1,
                mu: f[7].parse::<usize>().unwrap() - 1,
                value: f[8].parse().unwrap(),
            }
        })
        .collect()
}

#[test]
fn generated_cgc_matches_sunrepresentations_signed_elementwise() {
    let rows = load();
    assert!(rows.len() > 1000, "fixture too small: {}", rows.len());

    // Group fixture rows by channel.
    type Chan = (Vec<i64>, Vec<i64>, Vec<i64>);
    let mut by_channel: HashMap<Chan, Vec<&Row>> = HashMap::new();
    for r in &rows {
        by_channel
            .entry((r.s1.clone(), r.s2.clone(), r.s3.clone()))
            .or_default()
            .push(r);
    }

    let mut channels = 0usize;
    let mut om_channels = 0usize;
    let mut worst = 0.0f64;

    for ((d1, d2, d3), frows) in &by_channel {
        let s1 = Irrep::from_dynkin(d1).unwrap();
        let s2 = Irrep::from_dynkin(d2).unwrap();
        let s3 = Irrep::from_dynkin(d3).unwrap();
        let c = cgc(&s1, &s2, &s3).unwrap();
        channels += 1;
        if c.multiplicity() >= 2 {
            om_channels += 1;
        }

        // Index generated entries for lookup.
        let mut mine: HashMap<(u32, u32, u32, u32), f64> = HashMap::new();
        for e in c.entries() {
            mine.insert((e.m1, e.m2, e.m3, e.mu), e.value);
        }

        // Every reference nonzero must be reproduced signed.
        for r in frows {
            let key = (r.m1 as u32, r.m2 as u32, r.m3 as u32, r.mu as u32);
            let got = mine.get(&key).copied().unwrap_or(0.0);
            let d = (got - r.value).abs();
            worst = worst.max(d);
            assert!(
                d < TOL,
                "gauge mismatch {d1:?}⊗{d2:?}→{d3:?} at (m1={},m2={},m3={},mu={}): \
                 got {got}, reference {} (|Δ|={d:e})",
                r.m1,
                r.m2,
                r.m3,
                r.mu,
                r.value
            );
        }

        // Conversely: no generated entry above tolerance is absent from the
        // reference (would be a spurious coefficient / wrong support).
        let ref_keys: std::collections::HashSet<(u32, u32, u32, u32)> = frows
            .iter()
            .map(|r| (r.m1 as u32, r.m2 as u32, r.m3 as u32, r.mu as u32))
            .collect();
        for e in c.entries() {
            if e.value.abs() > TOL {
                assert!(
                    ref_keys.contains(&(e.m1, e.m2, e.m3, e.mu)),
                    "spurious coefficient {d1:?}⊗{d2:?}→{d3:?} at \
                     (m1={},m2={},m3={},mu={}) = {} not in reference",
                    e.m1,
                    e.m2,
                    e.m3,
                    e.mu,
                    e.value
                );
            }
        }
    }

    assert!(channels >= 15, "expected many channels, got {channels}");
    assert!(
        om_channels >= 2,
        "gauge continuity must be verified on >= 2 outer-multiplicity channels, got {om_channels}"
    );
    eprintln!("matched {channels} channels ({om_channels} with OM>=2), worst |Δ| = {worst:e}");
}
