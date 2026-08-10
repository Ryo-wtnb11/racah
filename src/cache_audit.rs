//! Ignored local evidence collector for racah #65; not a production benchmark.

use std::hint::black_box;
use std::time::{Duration, Instant};

use crate::cache;
use crate::primefactor;
use crate::sun::{cgc, directproduct, f_symbol, shared_directproduct, Irrep};
use crate::{su2_f_symbol, wigner_3j, wigner_6j};

fn irr(dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin(dynkin).unwrap()
}

fn rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages = statm.split_whitespace().nth(1)?.parse::<usize>().ok()?;
        return Some(pages.saturating_mul(4096));
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let kib = std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()?;
        return Some(kib.saturating_mul(1024));
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

struct Measurement {
    elapsed: Duration,
    start_live: usize,
    end_live: usize,
    peak_live: usize,
}

fn emit(phase: &str, measurement: Option<Measurement>) {
    let base = cache::base_cache_stats();
    let generated = cache::generated_cache_stats();
    let table = primefactor::table_stats();
    let (live, _) = crate::audit_alloc::snapshot();
    let (elapsed, start_live, end_live, peak_live) = measurement
        .map_or((0, live, live, live), |m| {
            (m.elapsed.as_nanos(), m.start_live, m.end_live, m.peak_live)
        });
    eprintln!(
        concat!(
            "ISSUE65_AUDIT {{\"kind\":\"phase\",\"phase\":\"{}\",\"elapsed_ns\":{},",
            "\"base\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],",
            "\"generated\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],",
            "\"factorial_rows\":{},\"primes\":{},\"table_capacity_bytes\":{},",
            "\"system_requested_live_start_bytes\":{},\"system_requested_live_end_bytes\":{},\"system_requested_live_peak_bytes\":{},\"system_requested_live_transient_lower_bound_bytes\":{},\"rss_bytes\":{} }}"
        ),
        phase,
        elapsed,
        base.three_j.entries, base.three_j.bytes, base.three_j.hits, base.three_j.misses, base.three_j.evictions,
        base.six_j.entries, base.six_j.bytes, base.six_j.hits, base.six_j.misses, base.six_j.evictions,
        base.derived_f.entries, base.derived_f.bytes, base.derived_f.hits, base.derived_f.misses, base.derived_f.evictions,
        generated.sun_product.entries, generated.sun_product.bytes, generated.sun_product.hits, generated.sun_product.misses, generated.sun_product.evictions,
        generated.sun_cgc.entries, generated.sun_cgc.bytes, generated.sun_cgc.hits, generated.sun_cgc.misses, generated.sun_cgc.evictions,
        generated.sun_f.entries, generated.sun_f.bytes, generated.sun_f.hits, generated.sun_f.misses, generated.sun_f.evictions,
        generated.bcd_cgc.entries, generated.bcd_cgc.bytes, generated.bcd_cgc.hits, generated.bcd_cgc.misses, generated.bcd_cgc.evictions,
        generated.bcd_f.entries, generated.bcd_f.bytes, generated.bcd_f.hits, generated.bcd_f.misses, generated.bcd_f.evictions,
        table.factorial_rows, table.primes, table.retained_capacity_bytes,
        start_live,
        end_live,
        peak_live,
        peak_live.saturating_sub(start_live.max(end_live)),
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

fn timed(work: impl FnOnce()) -> Measurement {
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    work();
    let elapsed = start.elapsed();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    Measurement {
        elapsed,
        start_live,
        end_live,
        peak_live,
    }
}

fn clone_slope<T>(label: &str, make: impl Fn() -> T) {
    let before = crate::audit_alloc::allocation_totals();
    let one = make();
    black_box(&one);
    let one_totals = crate::audit_alloc::allocation_totals();
    let mut eight = Vec::with_capacity(8);
    for _ in 0..8 {
        eight.push(make());
    }
    black_box(&eight);
    let nine_totals = crate::audit_alloc::allocation_totals();
    drop(eight);
    drop(one);
    eprintln!(
        "ISSUE65_AUDIT {{\"kind\":\"clone\",\"label\":\"{label}\",\"one_successful_alloc_calls\":{},\"one_requested_alloc_bytes\":{},\"nine_successful_alloc_calls\":{},\"nine_requested_alloc_bytes\":{}}}",
        one_totals.0.saturating_sub(before.0),
        one_totals.1.saturating_sub(before.1),
        nine_totals.0.saturating_sub(before.0),
        nine_totals.1.saturating_sub(before.1),
    );
}

fn su2() {
    let _ = wigner_3j(20, 20, 20, 0, 0, 0);
    let _ = wigner_6j(60, 60, 60, 60, 60, 60);
    let _ = su2_f_symbol(2, 2, 2, 2, 2, 2);
}

fn su2_no_reuse() {
    cache::reset();
    let _ = wigner_3j(20, 20, 20, 0, 0, 0);
    cache::reset();
    let _ = wigner_6j(60, 60, 60, 60, 60, 60);
    cache::reset();
    let _ = su2_f_symbol(2, 2, 2, 2, 2, 2);
}

fn su3() {
    let three = irr(&[1, 0]);
    let three_bar = irr(&[0, 1]);
    let eight = irr(&[1, 1]);
    let _ = shared_directproduct(&three, &three_bar).unwrap();
    let warmed_cgc = cgc(&three, &three_bar, &eight).unwrap();
    assert!(!warmed_cgc.entries().is_empty());
    let _ = f_symbol(&three, &three_bar, &three, &three, &eight, &eight).unwrap();
}

fn su3_no_reuse() {
    let three = irr(&[1, 0]);
    let three_bar = irr(&[0, 1]);
    let eight = irr(&[1, 1]);
    cache::reset();
    let _ = shared_directproduct(&three, &three_bar).unwrap();
    cache::reset();
    let _ = cgc(&three, &three_bar, &eight).unwrap();
    cache::reset();
    let _ = f_symbol(&three, &three_bar, &three, &three, &eight, &eight).unwrap();
}

fn su4() {
    let four = irr(&[1, 0, 0]);
    let four_bar = irr(&[0, 0, 1]);
    let fifteen = irr(&[1, 0, 1]);
    let _ = shared_directproduct(&four, &four_bar).unwrap();
    let _ = cgc(&four, &four_bar, &fifteen).unwrap();
    let _ = f_symbol(&four, &four_bar, &four, &four, &fifteen, &fifteen).unwrap();
}

fn su4_no_reuse() {
    let four = irr(&[1, 0, 0]);
    let four_bar = irr(&[0, 0, 1]);
    let fifteen = irr(&[1, 0, 1]);
    cache::reset();
    let _ = shared_directproduct(&four, &four_bar).unwrap();
    cache::reset();
    let _ = cgc(&four, &four_bar, &fifteen).unwrap();
    cache::reset();
    let _ = f_symbol(&four, &four_bar, &four, &four, &fifteen, &fifteen).unwrap();
}

fn run(label: &str, work: fn(), no_reuse: fn()) {
    cache::reset();
    emit(&format!("{label}:reset"), None);
    emit(&format!("{label}:cold"), Some(timed(work)));
    emit(&format!("{label}:warm"), Some(timed(work)));
    emit(
        &format!("{label}:reset_before_each_query"),
        Some(timed(no_reuse)),
    );
}

fn sequential_trace(label: &str, work: &[(&str, fn())]) {
    cache::reset();
    emit(&format!("{label}:reset"), None);
    for (family, run) in work {
        emit(&format!("{label}:after_{family}"), Some(timed(*run)));
    }
}

#[test]
#[ignore = "manual release-only cache measurement; run --release --features cgc-gen -- --ignored --nocapture"]
fn issue_65_cache_audit() {
    eprintln!(
        "ISSUE65_AUDIT {{\"kind\":\"metadata\",\"revision\":\"{}\",\"features\":\"cgc-gen\",\"consumer_revision\":\"N/A\",\"allocator\":\"System tracking test wrapper\",\"platform\":\"{}/{}\",\"rustc\":\"recorded by command\"}}",
        option_env!("RACAH_AUDIT_REVISION").unwrap_or("not-embedded; record git rev-parse HEAD with command"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    run("su2", su2, su2_no_reuse);
    run("su3", su3, su3_no_reuse);
    run("su4", su4, su4_no_reuse);
    sequential_trace(
        "sequential_su2_su3_su4",
        &[("su2", su2), ("su3", su3), ("su4", su4)],
    );
    sequential_trace(
        "sequential_su4_su3_su2",
        &[("su4", su4), ("su3", su3), ("su2", su2)],
    );

    cache::reset();
    su2();
    clone_slope("exact_su2_6j", || wigner_6j(60, 60, 60, 60, 60, 60));
    let three = irr(&[1, 0]);
    let three_bar = irr(&[0, 1]);
    let eight = irr(&[1, 1]);
    let _ = shared_directproduct(&three, &three_bar).unwrap();
    clone_slope("owned_sun_product", || {
        directproduct(&three, &three_bar).unwrap()
    });
    let _ = cgc(&three, &three_bar, &eight).unwrap();
    clone_slope("public_cgc", || cgc(&three, &three_bar, &eight).unwrap());
    let _ = f_symbol(&three, &three_bar, &three, &three, &eight, &eight).unwrap();
    clone_slope("public_f", || {
        f_symbol(&three, &three_bar, &three, &three, &eight, &eight).unwrap()
    });

    cache::reset();
    let a = irr(&[1, 0]);
    let b = irr(&[0, 1]);
    let retained = shared_directproduct(&a, &b).unwrap();
    let cache_shared = shared_directproduct(&a, &b).unwrap();
    assert!(retained.ptr_eq(&cache_shared));
    let owners_with_two_public_handles = retained.strong_count();
    drop(cache_shared);
    let owners_before_reset = retained.strong_count();
    cache::reset();
    assert_eq!(retained.strong_count() + 1, owners_before_reset);
    assert!(!retained.iter().collect::<Vec<_>>().is_empty());
    let before_drop = crate::audit_alloc::snapshot().0;
    drop(retained);
    let after_drop = crate::audit_alloc::snapshot().0;
    eprintln!(
        "ISSUE65_AUDIT {{\"kind\":\"retention\",\"owners_with_two_public_handles\":{},\"owners_before_reset\":{},\"freed_bytes_after_final_drop\":{}}}",
        owners_with_two_public_handles,
        owners_before_reset,
        before_drop.saturating_sub(after_drop),
    );
    emit("public_sun_product_retained_after_reset", None);
}
