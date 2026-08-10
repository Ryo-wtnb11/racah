//! Public base-tier cache trim contract.

use racah::cache::{base_cache_stats, cache_budgets, reset, trim_to, CoefficientCacheTier};
use racah::wigner_6j;

#[test]
fn trim_is_tier_local_preserves_budget_and_allows_refill() {
    reset();
    let budget = cache_budgets();
    for labels in [[2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4]] {
        let _ = wigner_6j(
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
        );
    }
    let before = base_cache_stats();
    assert!(before.six_j.entries >= 2);
    let report = trim_to(CoefficientCacheTier::SixJ, 0);
    assert_eq!(report.tier, CoefficientCacheTier::SixJ);
    assert_eq!(report.remaining_entries, 0);
    assert_eq!(report.remaining_charged_bytes, 0);
    assert_eq!(report.removed_entries, before.six_j.entries);
    assert_eq!(report.removed_charged_bytes, before.six_j.bytes);
    let after = base_cache_stats();
    assert_eq!(after.three_j, before.three_j);
    assert_eq!(after.derived_f, before.derived_f);
    assert_eq!(after.six_j.hits, before.six_j.hits);
    assert_eq!(after.six_j.misses, before.six_j.misses);
    assert_eq!(
        after.six_j.evictions,
        before.six_j.evictions + before.six_j.entries as u64
    );
    assert_eq!(cache_budgets(), budget);
    assert_eq!(trim_to(CoefficientCacheTier::SixJ, 0).removed_entries, 0);

    let _ = wigner_6j(2, 2, 2, 2, 2, 2);
    assert_eq!(base_cache_stats().six_j.entries, 1);
    assert_eq!(cache_budgets(), budget);
}
