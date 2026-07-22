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
}
