//! Ignored release-only coefficient-cache budget pressure collector.

use std::hint::black_box;
use std::time::Instant;

use crate::cache::{self, CoefficientCacheBudgets, CoefficientCacheTier};
use crate::sun::{directproduct, Irrep};
use crate::{wigner_3j, wigner_6j};

fn rss_bytes() -> Option<usize> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .map(|kib| kib.saturating_mul(1024))
}

fn workload() {
    for labels in [
        [20, 20, 20, 20, 20, 20],
        [60, 60, 60, 60, 60, 60],
        [120, 120, 120, 120, 120, 120],
    ] {
        black_box(wigner_6j(
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
        ));
        black_box(wigner_3j(labels[0], labels[1], labels[2], 0, 0, 0));
    }
    for (a, b) in [
        (&[1, 0][..], &[0, 1][..]),
        (&[1, 1], &[1, 1]),
        (&[1, 0, 0], &[0, 0, 1]),
    ] {
        let a = Irrep::from_dynkin(a).unwrap();
        let b = Irrep::from_dynkin(b).unwrap();
        black_box(directproduct(&a, &b).unwrap());
    }
}

fn run(mode: &str, budgets: CoefficientCacheBudgets, require_eviction: bool) {
    cache::configure_cache_budgets(budgets).unwrap();
    cache::reset();
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    workload();
    let elapsed_ns = start.elapsed().as_nanos();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    let base = cache::base_cache_stats();
    let generated = cache::generated_cache_stats();
    let evictions = base.total().evictions + generated.total().evictions;
    assert_eq!(evictions > 0, require_eviction);
    eprintln!(
        "COEFFICIENT_CACHE_BUDGET_PRESSURE {{\"mode\":\"{mode}\",\"elapsed_ns\":{elapsed_ns},\"base\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],\"generated\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],\"requested_live_start_bytes\":{start_live},\"requested_live_end_bytes\":{end_live},\"requested_live_peak_bytes\":{peak_live},\"rss_bytes\":{} }}",
        base.three_j.entries, base.three_j.bytes, base.three_j.hits, base.three_j.misses, base.three_j.evictions,
        base.six_j.entries, base.six_j.bytes, base.six_j.hits, base.six_j.misses, base.six_j.evictions,
        base.derived_f.entries, base.derived_f.bytes, base.derived_f.hits, base.derived_f.misses, base.derived_f.evictions,
        generated.sun_product.entries, generated.sun_product.bytes, generated.sun_product.hits, generated.sun_product.misses, generated.sun_product.evictions,
        generated.sun_cgc.entries, generated.sun_cgc.bytes, generated.sun_cgc.hits, generated.sun_cgc.misses, generated.sun_cgc.evictions,
        generated.sun_f.entries, generated.sun_f.bytes, generated.sun_f.hits, generated.sun_f.misses, generated.sun_f.evictions,
        generated.bcd_cgc.entries, generated.bcd_cgc.bytes, generated.bcd_cgc.hits, generated.bcd_cgc.misses, generated.bcd_cgc.evictions,
        generated.bcd_f.entries, generated.bcd_f.bytes, generated.bcd_f.hits, generated.bcd_f.misses, generated.bcd_f.evictions,
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_default() {
    run("default", CoefficientCacheBudgets::default(), false);
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_constrained() {
    let budgets = CoefficientCacheBudgets::default()
        .with_limit(CoefficientCacheTier::ThreeJ, 1)
        .with_limit(CoefficientCacheTier::SixJ, 1)
        .with_limit(CoefficientCacheTier::SunProduct, 1);
    run("constrained", budgets, true);
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_disabled() {
    run("disabled", CoefficientCacheBudgets::disabled(), true);
}
