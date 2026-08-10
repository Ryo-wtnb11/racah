//! Generated-family zero-budget coefficient-cache contract.

#![cfg(feature = "cgc-gen")]

use racah::cache::{
    cache_budgets, configure_cache_budgets, generated_cache_stats, CoefficientCacheBudgets,
};
use racah::sun::{directproduct, Irrep};

#[test]
fn disabled_policy_preserves_sun_product_value_without_retention() {
    let budgets = CoefficientCacheBudgets::disabled();
    configure_cache_budgets(budgets).unwrap();
    let three = Irrep::from_dynkin(&[1, 0]).unwrap();
    let three_bar = Irrep::from_dynkin(&[0, 1]).unwrap();
    let product = directproduct(&three, &three_bar).unwrap();
    assert_eq!(product.len(), 2);
    assert_eq!(cache_budgets(), budgets);
    let stats = generated_cache_stats().sun_product;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, 1);
}
