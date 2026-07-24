//! Generated-tier coefficient-cache resource contract (issue #47, leaf L1).
//!
//! The `cgc-gen` analogue of `tests/base_cache_contract.rs`: the four generated
//! tiers (SU(N) CGC, SU(N) F, B/C/D CGC, B/C/D F) are process-global and
//! `cache::reset()`/`generated_cache_stats()` act on shared state, so this is
//! kept in one `#[test]` (splitting into parallel tests would race the shared
//! counters — same reasoning as `tests/sun_cgc_cache.rs`).
//!
//! Forcing eviction on the real global caches is intentionally NOT attempted:
//! the production caps are 64 MiB / 256 MiB per tier. The FIFO bound /
//! byte-eviction / oversize-eviction machinery is covered deterministically on
//! local caches in `src/cache.rs`; here we prove per-tier hit/miss accounting on
//! deterministic accesses, the aggregate byte bound, the const/cap tie, `total()`
//! coherence, that the generated stats are a subset of the aggregate `stats()`,
//! and reset zeroing of every field of every generated tier and of `total()`.

#![cfg(feature = "cgc-gen")]

use racah::bcd::{self, CanonicalCatalog, Series};
use racah::cache::{self, GeneratedCacheStats, GENERATED_CACHE_MAX_BYTES};
use racah::sun;

fn sun_irr(d: &[i64]) -> sun::Irrep {
    sun::Irrep::from_dynkin(d).unwrap()
}

fn bcd_irr(s: Series, d: &[i64]) -> bcd::Irrep {
    bcd::Irrep::from_dynkin(s, d).unwrap()
}

#[test]
fn generated_cache_resource_contract() {
    // ---- the const is the documented sum of the four tier caps ----
    // (The compile-time assertion in src/cache.rs is the real drift guard; this
    // pins the human-facing value.)
    assert_eq!(
        GENERATED_CACHE_MAX_BYTES,
        (256 + 64 + 256 + 64) << 20,
        "640 MiB = SU(N) CGC + SU(N) F + BCD CGC + BCD F"
    );

    // ---- per-tier hit/miss accounting on deterministic accesses ----
    cache::reset();
    let g = cache::generated_cache_stats();
    assert_eq!(
        g,
        GeneratedCacheStats::default(),
        "reset must zero every tier"
    );

    // SU(N) CGC tier: `sun::cgc` owns it. Miss then hit on the same channel.
    let (s1, s2, s3) = (sun_irr(&[1, 0]), sun_irr(&[0, 1]), sun_irr(&[1, 1])); // 3⊗3̄→8
    let first = sun::cgc(&s1, &s2, &s3).unwrap();
    let g = cache::generated_cache_stats();
    assert!(g.sun_cgc.misses >= 1, "first sun cgc call is a miss");
    assert!(
        g.sun_cgc.entries >= 1 && g.sun_cgc.bytes > 0,
        "entry retained"
    );
    let again = sun::cgc(&s1, &s2, &s3).unwrap();
    assert_eq!(first, again, "warm hit returns the identical CGC");
    let g = cache::generated_cache_stats();
    assert!(g.sun_cgc.hits >= 1, "second sun cgc call is a hit");

    // SU(N) F tier: `sun::f_symbol` owns it.
    let e8 = sun_irr(&[1, 1]);
    let _ = sun::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // miss
    let _ = sun::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // hit
    let g = cache::generated_cache_stats();
    assert!(g.sun_f.misses >= 1 && g.sun_f.hits >= 1, "sun F miss + hit");

    // B/C/D CGC + F tiers: the public `bcd::f_symbol` owns the BCD F tier and
    // its CGC materialization owns the BCD CGC tier. Sextet from the crate's own
    // doctest (Sp(4), F = 1×1×1×1 identity). Miss then hit.
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let triv = bcd::Irrep::trivial(Series::C, 2).unwrap();
    let v = bcd_irr(Series::C, &[0, 1]);
    let adj = bcd_irr(Series::C, &[2, 0]);
    let _ = bcd::f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // miss
    let g = cache::generated_cache_stats();
    assert!(
        g.bcd_cgc.entries >= 1 && g.bcd_f.entries >= 1,
        "an F block populates both BCD tiers"
    );
    let _ = bcd::f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // hit
    let g = cache::generated_cache_stats();
    assert!(g.bcd_f.hits >= 1, "second F block hits the BCD F tier");

    // ---- aggregate byte bound and total() coherence ----
    let g = cache::generated_cache_stats();
    let total = g.total();
    // Each tier's own true ceiling, and the documented aggregate corollary.
    assert!(
        g.sun_cgc.bytes <= (256 << 20) && g.bcd_cgc.bytes <= (256 << 20),
        "a CGC tier is over its 256 MiB cap"
    );
    assert!(
        g.sun_f.bytes <= (64 << 20) && g.bcd_f.bytes <= (64 << 20),
        "an F tier is over its 64 MiB cap"
    );
    assert!(
        total.bytes <= GENERATED_CACHE_MAX_BYTES,
        "aggregate over GENERATED_CACHE_MAX_BYTES: {} > {}",
        total.bytes,
        GENERATED_CACHE_MAX_BYTES
    );
    // total() is the field-wise sum of the four tiers.
    assert_eq!(
        total.entries,
        g.sun_cgc.entries + g.sun_f.entries + g.bcd_cgc.entries + g.bcd_f.entries
    );
    assert_eq!(
        total.bytes,
        g.sun_cgc.bytes + g.sun_f.bytes + g.bcd_cgc.bytes + g.bcd_f.bytes
    );
    assert!(total.entries > 0, "the accesses retained entries");

    // Generated tiers are a subset of the whole-process aggregate `stats()`.
    let agg = cache::stats();
    assert!(total.entries <= agg.entries, "generated ⊆ aggregate");
    assert!(total.bytes <= agg.bytes);

    // ---- reset zeroing: every field of every generated tier and of total() ----
    cache::reset();
    let g = cache::generated_cache_stats();
    let zero = racah::cache::TierStats::default();
    assert_eq!(g.sun_cgc, zero, "sun_cgc not cleared");
    assert_eq!(g.sun_f, zero, "sun_f not cleared");
    assert_eq!(g.bcd_cgc, zero, "bcd_cgc not cleared");
    assert_eq!(g.bcd_f, zero, "bcd_f not cleared");
    assert_eq!(g.total(), zero, "total not cleared");
}
