//! Cache policy observation freezes the compiled default in a fresh process.

use racah::cache::{
    base_cache_stats, cache_budgets, configure_cache_budgets, CacheBudgetError,
    CoefficientCacheBudgets,
};

#[test]
fn observing_cache_stats_freezes_the_default_policy() {
    let default = CoefficientCacheBudgets::default();
    let _ = base_cache_stats();
    assert_eq!(cache_budgets(), default);
    assert_eq!(
        configure_cache_budgets(CoefficientCacheBudgets::disabled()),
        Err(CacheBudgetError::AlreadyInitialized)
    );
}
