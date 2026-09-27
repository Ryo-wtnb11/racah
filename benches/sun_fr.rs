//! SU(N) F-symbol generation cost across representative families, cold (full
//! four-CGC contraction, CGC caches cleared) and warm (derived-f64 F cache
//! hit). Not a CI gate; run with `cargo bench --features cgc-gen`.
//!
//! The cached-CGC cases isolate F recomputation (by trimming only its F cache)
//! and uncached R contraction from CGC generation.

use criterion::{criterion_group, criterion_main, Criterion};
use racah::cache::{self, CoefficientCacheTier};
use racah::sun::{f_symbol, r_symbol, Irrep};
use std::hint::black_box;

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// (label, a, b, c, d, e, f) representative admissible sextets spanning N, dim,
/// and outer multiplicity.
#[allow(clippy::type_complexity)]
fn cases() -> Vec<(&'static str, [Irrep; 6])> {
    vec![
        (
            "su2_half_cubed_d1",
            [
                irr(&[1]),
                irr(&[1]),
                irr(&[1]),
                irr(&[1]),
                irr(&[0]),
                irr(&[0]),
            ],
        ),
        (
            "su3_3x3bx3_d_mfree",
            [
                irr(&[1, 0]),
                irr(&[0, 1]),
                irr(&[1, 0]),
                irr(&[1, 0]),
                irr(&[1, 1]),
                irr(&[1, 1]),
            ],
        ),
        (
            "su3_octet_cubed_om2_2x2x2x2",
            [
                irr(&[1, 1]),
                irr(&[1, 1]),
                irr(&[1, 1]),
                irr(&[1, 1]),
                irr(&[1, 1]),
                irr(&[1, 1]),
            ],
        ),
        (
            "su4_adjoint_d15",
            [
                irr(&[1, 0, 1]),
                irr(&[1, 0, 1]),
                irr(&[1, 0, 1]),
                irr(&[1, 0, 1]),
                irr(&[1, 0, 1]),
                irr(&[1, 0, 1]),
            ],
        ),
    ]
}

fn bench_cold(c: &mut Criterion) {
    let mut g = c.benchmark_group("f_symbol_cold");
    for (label, s) in cases() {
        g.bench_function(label, |b| {
            b.iter(|| {
                // Clear all caches so each iteration pays the full CGC
                // generation + four-CGC contraction.
                cache::reset();
                black_box(
                    f_symbol(
                        black_box(&s[0]),
                        black_box(&s[1]),
                        black_box(&s[2]),
                        black_box(&s[3]),
                        black_box(&s[4]),
                        black_box(&s[5]),
                    )
                    .unwrap(),
                )
            })
        });
    }
    g.finish();
}

fn bench_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("f_symbol_cache_hit");
    for (label, s) in cases() {
        let _ = f_symbol(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]).unwrap(); // warm
        g.bench_function(label, |b| {
            b.iter(|| {
                black_box(
                    f_symbol(
                        black_box(&s[0]),
                        black_box(&s[1]),
                        black_box(&s[2]),
                        black_box(&s[3]),
                        black_box(&s[4]),
                        black_box(&s[5]),
                    )
                    .unwrap(),
                )
            })
        });
    }
    g.finish();
}

fn bench_cached_cgc(c: &mut Criterion) {
    let mut f_group = c.benchmark_group("f_symbol_cached_cgc");
    for (label, s) in cases().into_iter().skip(1) {
        let _ = f_symbol(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]).unwrap();
        f_group.bench_function(label, |b| {
            b.iter(|| {
                cache::trim_to(CoefficientCacheTier::SunF, 0);
                black_box(f_symbol(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]).unwrap())
            })
        });
    }
    f_group.finish();

    let mut r_group = c.benchmark_group("r_symbol_cached_cgc");
    for (label, d) in [
        ("su3_octet_om2", &[1, 1][..]),
        ("su4_adjoint", &[1, 0, 1][..]),
    ] {
        let s = irr(d);
        let _ = r_symbol(&s, &s, &s).unwrap();
        r_group.bench_function(label, |b| {
            b.iter(|| black_box(r_symbol(black_box(&s), &s, &s).unwrap()))
        });
    }
    r_group.finish();
}

criterion_group!(benches, bench_cold, bench_hit, bench_cached_cgc);
criterion_main!(benches);
