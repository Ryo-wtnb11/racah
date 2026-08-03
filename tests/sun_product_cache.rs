//! SU(N) product-cache API and concurrency contract (issue #59).
//!
//! One sequential test owns the process-global reset policy for this binary.

#![cfg(feature = "cgc-gen")]

use std::sync::{Arc, Barrier};

use racah::cache::{self, TierStats};
use racah::sun::{directproduct, Irrep, SunError};
use racah::Su2Irrep;

fn irr(dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin(dynkin).unwrap()
}

#[test]
fn sun_product_cache_contract() {
    cache::reset();

    // Failure admission is transactional: distinct ranks never query or
    // publish the cache.
    let su3 = irr(&[1, 0]);
    let su4 = irr(&[1, 0, 0]);
    assert_eq!(
        directproduct(&su3, &su4),
        Err(SunError::RankMismatch { a: 3, b: 4 })
    );
    assert_eq!(
        cache::generated_cache_stats().sun_product,
        TierStats::default()
    );

    // A cold public call sweeps once. Reversing the factors must hit the same
    // unordered entry and reconstruct the same owned map without re-sweeping.
    let a = irr(&[2, 1]);
    let b = irr(&[1, 2]);
    let cold = directproduct(&a, &b).unwrap();
    let after_cold = cache::generated_cache_stats().sun_product;
    assert_eq!(after_cold.hits, 0);
    assert_eq!(after_cold.misses, 1);
    assert_eq!(after_cold.entries, 1);
    assert!(after_cold.bytes > 0);

    let warm_reversed = directproduct(&b, &a).unwrap();
    assert_eq!(warm_reversed, cold);
    let after_warm = cache::generated_cache_stats().sun_product;
    assert_eq!(after_warm.hits, 1);
    assert_eq!(after_warm.misses, 1, "warm public map must not re-sweep");
    assert_eq!(after_warm.entries, 1);

    // Compute-outside-lock permits concurrent duplicate misses, but the write
    // recheck publishes exactly one identical entry.
    cache::reset();
    let a = Arc::new(irr(&[3, 1]));
    let b = Arc::new(irr(&[1, 2]));
    let barrier = Arc::new(Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                directproduct(&a, &b).unwrap()
            })
        })
        .collect::<Vec<_>>();
    let products = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert!(products.iter().all(|product| product == &products[0]));
    let concurrent = cache::generated_cache_stats().sun_product;
    assert_eq!(concurrent.entries, 1);
    assert!((1..=8).contains(&concurrent.misses));

    let misses = concurrent.misses;
    let hits = concurrent.hits;
    assert_eq!(directproduct(&a, &b).unwrap(), products[0]);
    let warm = cache::generated_cache_stats().sun_product;
    assert_eq!(warm.misses, misses, "warm call must not re-sweep");
    assert_eq!(warm.hits, hits + 1);

    // Cheap one-channel products isolate the production entry bound: all 256
    // retained entries together remain below the 128 KiB charge backstop.
    cache::reset();
    let trivial = Irrep::trivial(3).unwrap();
    for label in 1..=257 {
        let _ = directproduct(&trivial, &irr(&[label, 0])).unwrap();
    }
    let entry_bound = cache::generated_cache_stats().sun_product;
    assert_eq!(entry_bound.entries, 256);
    assert_eq!(entry_bound.evictions, 1);
    assert!(entry_bound.bytes < (128 << 10), "byte bound fired first");

    // The dedicated allocation-free SU(2) decomposition remains independent
    // and uncached by this generated SU(N) tier.
    cache::reset();
    let channels = Su2Irrep::new(3)
        .fusion(Su2Irrep::new(2))
        .unwrap()
        .map(Su2Irrep::dj)
        .collect::<Vec<_>>();
    assert_eq!(channels, [1, 3, 5]);
    assert_eq!(
        cache::generated_cache_stats().sun_product,
        TierStats::default()
    );
}
