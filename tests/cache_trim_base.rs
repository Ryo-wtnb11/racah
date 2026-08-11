//! Public base-tier cache trim contract.

use racah::cache::{base_cache_stats, cache_budgets, reset, trim_to, CoefficientCacheTier};
use racah::{su2_f_symbol, wigner_3j, wigner_6j};

#[test]
fn trim_is_tier_local_preserves_budget_and_allows_refill() {
    reset();
    let budget = cache_budgets();
    let _ = wigner_3j(2, 2, 2, 0, 0, 0);
    let _ = wigner_6j(2, 2, 2, 2, 2, 2);
    let _ = su2_f_symbol(2, 2, 2, 2, 2, 2);
    let before = base_cache_stats();
    for tier in [
        CoefficientCacheTier::ThreeJ,
        CoefficientCacheTier::SixJ,
        CoefficientCacheTier::DerivedF,
    ] {
        let snapshot = base_cache_stats();
        let selected = match tier {
            CoefficientCacheTier::ThreeJ => snapshot.three_j,
            CoefficientCacheTier::SixJ => snapshot.six_j,
            CoefficientCacheTier::DerivedF => snapshot.derived_f,
            _ => unreachable!(),
        };
        assert!(selected.entries > 0);
        let report = trim_to(tier, 0);
        assert_eq!(report.removed_entries, selected.entries);
        let after = base_cache_stats();
        let now = match tier {
            CoefficientCacheTier::ThreeJ => after.three_j,
            CoefficientCacheTier::SixJ => after.six_j,
            CoefficientCacheTier::DerivedF => after.derived_f,
            _ => unreachable!(),
        };
        assert_eq!(now.entries, 0);
    }
    assert_eq!(cache_budgets(), budget);
    let _ = wigner_6j(2, 2, 2, 2, 2, 2);
    assert_eq!(base_cache_stats().six_j.entries, 1);
    assert_eq!(cache_budgets(), budget);
}
