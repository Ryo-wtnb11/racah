//! CGC cache: facade reset/stats wiring and concurrency equivalence.
//!
//! Kept in one `#[test]` because the cache facade is process-global and
//! `cache::stats()` aggregates every tier; splitting into several parallel
//! tests would race the shared counters.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;

use racah::cache;
use racah::sun::{cgc, Cgc, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

// A spread of SU(3) channels used by both the sequential reference and the
// concurrent threads.
fn channels() -> Vec<(Irrep, Irrep, Irrep)> {
    vec![
        (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1])), // 3⊗3̄→8
        (irr(&[1, 0]), irr(&[1, 0]), irr(&[2, 0])), // 3⊗3→6
        (irr(&[1, 1]), irr(&[1, 1]), irr(&[1, 1])), // 8⊗8→8 (OM=2)
        (irr(&[1, 1]), irr(&[1, 1]), irr(&[2, 2])), // 8⊗8→27
        (irr(&[2, 0]), irr(&[0, 2]), irr(&[1, 1])), // 6⊗6̄→8
    ]
}

#[test]
fn cache_reset_stats_and_concurrency() {
    // ---- reset + stats wiring ----
    cache::reset();
    let s0 = cache::stats();
    assert_eq!(
        (s0.hits, s0.misses, s0.entries, s0.bytes),
        (0, 0, 0, 0),
        "reset must zero all tiers"
    );

    let (s1, s2, s3) = (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1]));
    let first = cgc(&s1, &s2, &s3).unwrap(); // miss
    let after_miss = cache::stats();
    assert!(after_miss.misses >= 1, "first call must be a miss");
    assert!(after_miss.entries >= 1, "entry must be retained");
    assert!(after_miss.bytes > 0, "byte charge must be nonzero");

    let again = cgc(&s1, &s2, &s3).unwrap(); // hit
    let after_hit = cache::stats();
    assert!(after_hit.hits >= 1, "second call must be a hit");
    assert_eq!(after_hit.entries, after_miss.entries, "hit adds no entry");
    assert_eq!(first, again, "cached value must be byte-identical");

    // ---- concurrency: N threads over mixed pairs == sequential ----
    cache::reset();
    let chans = channels();
    // Sequential reference (also warms the cache; that is fine -- values are
    // deterministic).
    let seq: Vec<Arc<Cgc>> = chans
        .iter()
        .map(|(a, b, c)| Arc::new(cgc(a, b, c).unwrap()))
        .collect();

    let mut handles = Vec::new();
    for t in 0..8usize {
        let chans = chans.clone();
        let seq = seq.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..chans.len() {
                // Offset so threads interleave the pairs differently.
                let k = (i + t) % chans.len();
                let (a, b, c) = &chans[k];
                let got = cgc(a, b, c).unwrap();
                assert_eq!(
                    got, *seq[k],
                    "thread {t} diverged on channel {k}: concurrent != sequential"
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Every distinct channel is retained exactly once (no duplication under the
    // concurrent-insert race).
    let stats = cache::stats();
    assert!(
        stats.entries >= chans.len(),
        "expected >= {} entries, got {}",
        chans.len(),
        stats.entries
    );

    // ---- COLD race: no warm-up, threads hit an empty cache together ----
    // Every thread requests the same cold channels at once, so each channel is
    // generated concurrently by several racers (the insert() race-loser path).
    // The cache's contract is that it serializes to ONE winner value per key:
    // every racer -- winner or loser -- returns that same stored value.
    cache::reset();
    let mut handles = Vec::new();
    for t in 0..8usize {
        let chans = chans.clone();
        handles.push(std::thread::spawn(move || {
            let mut by_k: std::collections::HashMap<usize, Cgc> = std::collections::HashMap::new();
            for i in 0..chans.len() {
                let k = (i + t) % chans.len();
                let (a, b, c) = &chans[k];
                by_k.insert(k, cgc(a, b, c).unwrap());
            }
            by_k
        }));
    }
    let observed: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // (a) Cache consistency: all racers observe the byte-identical winner value
    // for each key. This is exact -- the cache hands back one stored Arc.
    for t in 1..observed.len() {
        for k in 0..chans.len() {
            assert_eq!(
                observed[t][&k], observed[0][&k],
                "cold-race: thread {t} saw a different value than thread 0 for channel {k} \
                 (cache did not serialize to a single winner)"
            );
        }
    }

    // (b) The winner value is a valid CGC: it matches a from-scratch
    // recomputation within tolerance. NOT exact -- the faer backend's parallel
    // reductions are not bit-reproducible across runs, so two independent
    // generations of the same channel can differ by a few ULPs. (Determinism of
    // the STORED value under one cache is (a); reproducibility of the gauge
    // *values* is the fixture oracle, at 2.4e-15.)
    cache::reset();
    for (k, (a, b, c)) in chans.iter().enumerate() {
        let fresh = cgc(a, b, c).unwrap();
        let raced = &observed[0][&k];
        assert_eq!(raced.nnz(), fresh.nnz(), "channel {k}: support differs");
        for (re, fe) in raced.entries().iter().zip(fresh.entries()) {
            assert_eq!(
                (re.m1, re.m2, re.m3, re.mu),
                (fe.m1, fe.m2, fe.m3, fe.mu),
                "channel {k}: entry index differs"
            );
            assert!(
                (re.value - fe.value).abs() < 1e-9,
                "channel {k}: value {} vs {} (|Δ|={:e})",
                re.value,
                fe.value,
                (re.value - fe.value).abs()
            );
        }
    }
}
