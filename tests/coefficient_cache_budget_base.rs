//! Base-only one-shot coefficient-cache budget contract.

use racah::cache::{
    base_cache_stats, cache_budgets, configure_cache_budgets, CacheBudgetError,
    CoefficientCacheBudgets,
};
use racah::wigner_6j;

#[test]
fn zero_budget_is_retryable_then_compute_without_retention() {
    let invalid = CoefficientCacheBudgets {
        three_j_bytes: usize::MAX,
        ..Default::default()
    };
    assert!(matches!(
        configure_cache_budgets(invalid),
        Err(CacheBudgetError::ExceedsMaximum {
            tier: "three_j",
            ..
        })
    ));

    let budgets = CoefficientCacheBudgets {
        three_j_bytes: 0,
        six_j_bytes: 0,
        derived_f_bytes: 0,
        ..Default::default()
    };
    configure_cache_budgets(budgets).unwrap();
    assert_eq!(cache_budgets(), budgets);
    let value = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((value.to_f64() - 1.0 / 6.0).abs() < 1e-14);
    let stats = base_cache_stats();
    assert_eq!(stats.six_j.entries, 0);
    assert_eq!(stats.six_j.bytes, 0);
    assert_eq!(stats.six_j.evictions, 1);
    assert_eq!(
        configure_cache_budgets(CoefficientCacheBudgets::default()),
        Err(CacheBudgetError::AlreadyInitialized)
    );
}
