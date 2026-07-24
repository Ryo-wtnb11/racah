//! Base SU(2) coefficient-cache resource contract (issue #43, PR-A).
//!
//! Kept in one `#[test]`: the 3j/6j/derived-F tiers are process-global and
//! `cache::reset()`/`base_cache_stats()` act on shared state, so splitting into
//! several parallel tests would race the shared counters (same reasoning as
//! `tests/sun_cgc_cache.rs`).
//!
//! This exercises the *base* tiers on the real global caches. Forcing eviction
//! here is intentionally NOT attempted: the production caps are 1M entries /
//! 64 MiB per tier, and reaching them needs millions of big-rational Wigner
//! computations (measured: 1.2M `wigner_3j` calls collapse to ~35k distinct
//! Regge classes). The eviction counter itself is covered deterministically on
//! local caches in `src/cache.rs`; here we prove the aggregate byte bound, the
//! per-tier/total statistic coherence, hit/miss accounting, and reset zeroing.

use racah::cache::{self, BASE_CACHE_MAX_BYTES};
use racah::{su2_f_symbol, wigner_3j, wigner_6j};

#[test]
fn base_cache_resource_contract() {
    let per_tier_cap = BASE_CACHE_MAX_BYTES / 3;

    // ---- hit/miss accounting on a deterministic per-tier pattern ----
    cache::reset();

    // 3j: only `wigner_3j` touches this tier, so its counters are exact.
    let _ = wigner_3j(2, 2, 2, 0, 0, 0); // miss
    let _ = wigner_3j(2, 2, 2, 0, 0, 0); // hit
    let s = cache::base_cache_stats();
    assert_eq!(
        (s.three_j.misses, s.three_j.hits),
        (1, 1),
        "one miss then one hit on the 3j tier"
    );

    // 6j: `wigner_6j` (and, below, `su2_f_symbol`) can populate this tier, so
    // assert monotone accounting rather than exact deltas.
    let _ = wigner_6j(2, 2, 2, 2, 2, 2); // miss
    let _ = wigner_6j(2, 2, 2, 2, 2, 2); // hit
    let s = cache::base_cache_stats();
    assert!(s.six_j.misses >= 1 && s.six_j.hits >= 1, "6j miss+hit");

    // derived-F: `su2_f_symbol` owns this tier.
    let _ = su2_f_symbol(2, 2, 2, 2, 2, 2); // miss
    let _ = su2_f_symbol(2, 2, 2, 2, 2, 2); // hit
    let s = cache::base_cache_stats();
    assert!(
        s.derived_f.misses >= 1 && s.derived_f.hits >= 1,
        "F miss+hit"
    );

    // ---- fill all three tiers with a spread of small labels ----
    cache::reset();
    for a in 0..12u32 {
        for b in 0..12u32 {
            for c in 0..12u32 {
                // 3j and 6j direct fills.
                let _ = wigner_3j(a, b, c, 0, 0, 0);
                let _ = wigner_6j(a, b, c, a, b, c);
                // derived-F sweep.
                let _ = su2_f_symbol(a, b, c, a, b, c);
            }
        }
    }

    let s = cache::base_cache_stats();

    // Per-tier byte bound (each tier's own true ceiling).
    assert!(
        s.three_j.bytes <= per_tier_cap,
        "3j over cap: {}",
        s.three_j.bytes
    );
    assert!(
        s.six_j.bytes <= per_tier_cap,
        "6j over cap: {}",
        s.six_j.bytes
    );
    assert!(
        s.derived_f.bytes <= per_tier_cap,
        "F over cap: {}",
        s.derived_f.bytes
    );

    // Aggregate bound — the documented corollary of the per-tier ceilings.
    let total = s.total();
    assert!(
        total.bytes <= BASE_CACHE_MAX_BYTES,
        "aggregate over BASE_CACHE_MAX_BYTES: {} > {}",
        total.bytes,
        BASE_CACHE_MAX_BYTES
    );

    // `total()` is the field-wise sum of the three tiers (statistics agree with
    // retained state).
    assert_eq!(
        total.entries,
        s.three_j.entries + s.six_j.entries + s.derived_f.entries
    );
    assert_eq!(
        total.bytes,
        s.three_j.bytes + s.six_j.bytes + s.derived_f.bytes
    );
    assert!(total.entries > 0, "the fill retained entries");

    // Base totals agree with the aggregate `stats()`. Under the default feature
    // set the base tiers are the only tiers, so the two are equal; `cgc-gen`
    // adds generated tiers, so there `stats()` is a superset (>=).
    let agg = cache::stats();
    #[cfg(not(feature = "cgc-gen"))]
    {
        assert_eq!(total.entries, agg.entries, "base == aggregate (no cgc-gen)");
        assert_eq!(total.bytes, agg.bytes);
    }
    #[cfg(feature = "cgc-gen")]
    {
        assert!(
            total.entries <= agg.entries,
            "base is a subset of aggregate"
        );
        assert!(total.bytes <= agg.bytes);
    }

    // ---- reset zeroing: every field of every tier and of total() ----
    cache::reset();
    let s = cache::base_cache_stats();
    let zero = racah::cache::TierStats::default();
    assert_eq!(s.three_j, zero, "3j not cleared");
    assert_eq!(s.six_j, zero, "6j not cleared");
    assert_eq!(s.derived_f, zero, "F not cleared");
    assert_eq!(s.total(), zero, "total not cleared");
}
