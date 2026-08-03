#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use std::sync::Mutex;

use racah::{
    cache,
    sun::{directproduct, Irrep, SunError},
};

fn irrep(labels: &[i64]) -> Irrep {
    Irrep::from_dynkin(labels).unwrap()
}

static LOCK: Mutex<()> = Mutex::new(());

#[test]
fn adjoint_products_share_one_canonical_warm_entry() {
    let _guard = LOCK.lock().unwrap();
    cache::reset();
    for (a, b) in [
        (irrep(&[1, 1]), irrep(&[1, 1])),
        (irrep(&[1, 0, 1]), irrep(&[1, 0, 1])),
    ] {
        let cold = directproduct(&a, &b).unwrap();
        let after_cold = cache::generated_cache_stats().sun_product;
        assert_eq!(directproduct(&b, &a).unwrap(), cold);
        let warm = cache::generated_cache_stats().sun_product;
        assert_eq!(warm.entries, after_cold.entries);
        assert!(warm.hits > after_cold.hits);
        assert!(warm.bytes > 0);
    }
}

#[test]
fn rank_error_does_not_publish() {
    let _guard = LOCK.lock().unwrap();
    cache::reset();
    let before = cache::generated_cache_stats().sun_product;
    assert!(matches!(
        directproduct(&irrep(&[1, 1]), &irrep(&[1, 0, 1])),
        Err(SunError::RankMismatch { .. })
    ));
    assert_eq!(cache::generated_cache_stats().sun_product, before);
}

#[test]
fn concurrent_cold_callers_return_equal_products() {
    let _guard = LOCK.lock().unwrap();
    cache::reset();
    let a = Arc::new(irrep(&[1, 1]));
    let outputs = std::thread::scope(|scope| {
        (0..8)
            .map(|_| {
                let a = Arc::clone(&a);
                scope.spawn(move || directproduct(&a, &a).unwrap())
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(outputs.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(cache::generated_cache_stats().sun_product.entries, 1);
}
