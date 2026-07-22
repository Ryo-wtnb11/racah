//! Derived-f64 SU(N) F-symbol cache behaviour through the public API
//! (oracle 4 of issue #16): a warm hit returns the identical block, `reset`
//! clears it, `stats` accounts it, and concurrent generation is race-free.
//!
//! The FIFO bound / byte-eviction / eviction-never-changes-values machinery is
//! covered generically in `src/cache.rs` (and specialised for the `Arc<FBlock>`
//! tier there); this file exercises the end-to-end public path.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use std::thread;

use racah::cache;
use racah::sun::{f_symbol, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// One test (not two): both phases call the process-global `cache::reset`, so
/// they must not run concurrently with each other.
#[test]
fn f_symbol_cache_hit_reset_stats_and_concurrency() {
    let e8 = irr(&[1, 1]);

    // --- hit / stats / reset ---
    cache::reset();
    let first = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let after_miss = cache::stats();
    assert!(
        after_miss.entries >= 1 && after_miss.misses >= 1,
        "first call should record a miss + entry"
    );

    let again = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    assert_eq!(first, again, "warm hit must return the identical block");
    let after_hit = cache::stats();
    assert!(after_hit.hits > after_miss.hits, "second call is a hit");
    assert_eq!(after_hit.entries, after_miss.entries, "a hit adds no entry");

    cache::reset();
    let cleared = cache::stats();
    assert_eq!(
        (cleared.hits, cleared.misses, cleared.entries, cleared.bytes),
        (0, 0, 0, 0),
        "reset clears entries and counters"
    );

    // --- concurrency: many threads race on the same key ---
    let e8 = Arc::new(e8);
    let want = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let mut handles = Vec::new();
    for _ in 0..8 {
        let e8 = Arc::clone(&e8);
        let want = want.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..16 {
                let got = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
                assert_eq!(got, want, "concurrent F block diverged");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(cache::stats().entries >= 1, "one entry survives the race");
}
