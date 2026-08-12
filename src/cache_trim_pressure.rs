//! Ignored release-only semantic collector for per-tier cache trim.

use std::hint::black_box;
use std::time::Instant;

use crate::cache::{self, CoefficientCacheTier};
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
        black_box(
            directproduct(
                &Irrep::from_dynkin(a).unwrap(),
                &Irrep::from_dynkin(b).unwrap(),
            )
            .unwrap(),
        );
    }
}

fn run(mode: &str) {
    cache::reset();
    workload();
    let before = cache::base_cache_stats();
    let before_product = cache::generated_cache_stats().sun_product;
    let targets = match mode {
        "retained" => [
            before.three_j.bytes,
            before.six_j.bytes,
            before_product.bytes,
        ],
        "partial" => [
            before.three_j.bytes / 2,
            before.six_j.bytes / 2,
            before_product.bytes / 2,
        ],
        "zero" => [0, 0, 0],
        _ => unreachable!(),
    };
    let reports = [
        cache::trim_to(CoefficientCacheTier::ThreeJ, targets[0]),
        cache::trim_to(CoefficientCacheTier::SixJ, targets[1]),
        cache::trim_to(CoefficientCacheTier::SunProduct, targets[2]),
    ];
    match mode {
        "retained" => assert!(reports.iter().all(|report| report.removed_entries == 0)),
        "partial" => assert!(reports
            .iter()
            .all(|report| report.removed_entries > 0 && report.remaining_entries > 0)),
        "zero" => assert!(reports.iter().all(|report| report.remaining_entries == 0)),
        _ => unreachable!(),
    }
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    workload();
    let elapsed_ns = start.elapsed().as_nanos();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    let after = cache::base_cache_stats();
    let after_product = cache::generated_cache_stats().sun_product;
    assert!(after.three_j.bytes <= cache::cache_budgets().limit(CoefficientCacheTier::ThreeJ));
    assert!(after.six_j.bytes <= cache::cache_budgets().limit(CoefficientCacheTier::SixJ));
    assert!(after_product.bytes <= cache::cache_budgets().limit(CoefficientCacheTier::SunProduct));
    let sample = std::env::var("COEFFICIENT_CACHE_TRIM_SAMPLE").unwrap_or_else(|_| "0".into());
    let revision = option_env!("RACAH_TRIM_REVISION").unwrap_or("not-embedded");
    eprintln!(
        "COEFFICIENT_CACHE_TRIM {{\"schema\":1,\"revision\":\"{revision}\",\"mode\":\"{mode}\",\"sample\":{sample},\"targets\":[{},{},{}],\"reports\":[[{},{},{},{}],[{},{},{},{}],[{},{},{},{}]],\"elapsed_ns\":{elapsed_ns},\"after\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],\"requested_live_start_bytes\":{start_live},\"requested_live_end_bytes\":{end_live},\"requested_live_peak_bytes\":{peak_live},\"rss_bytes\":{} }}",
        targets[0], targets[1], targets[2],
        reports[0].removed_entries, reports[0].removed_charged_bytes, reports[0].remaining_entries, reports[0].remaining_charged_bytes,
        reports[1].removed_entries, reports[1].removed_charged_bytes, reports[1].remaining_entries, reports[1].remaining_charged_bytes,
        reports[2].removed_entries, reports[2].removed_charged_bytes, reports[2].remaining_entries, reports[2].remaining_charged_bytes,
        after.three_j.entries, after.three_j.bytes, after.three_j.hits, after.three_j.misses, after.three_j.evictions,
        after.six_j.entries, after.six_j.bytes, after.six_j.hits, after.six_j.misses, after.six_j.evictions,
        after_product.entries, after_product.bytes, after_product.hits, after_product.misses, after_product.evictions,
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

#[test]
#[ignore = "manual release-only coefficient-cache trim measurement"]
fn coefficient_cache_trim_retained() {
    run("retained");
}
#[test]
#[ignore = "manual release-only coefficient-cache trim measurement"]
fn coefficient_cache_trim_partial() {
    run("partial");
}
#[test]
#[ignore = "manual release-only coefficient-cache trim measurement"]
fn coefficient_cache_trim_zero() {
    run("zero");
}
