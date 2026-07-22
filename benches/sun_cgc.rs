//! SU(N) CGC generation cost across representative `(N, dim)` pairs, plus the
//! warm cache-hit path. Not a CI gate; run with `cargo bench --features
//! cgc-gen`.

use criterion::{criterion_group, criterion_main, Criterion};
use racah::cache;
use racah::sun::{cgc, Irrep};
use std::hint::black_box;

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// (label, s1, s2, s3) representative channels spanning N and dim.
fn cases() -> Vec<(&'static str, Irrep, Irrep, Irrep)> {
    vec![
        ("su2_2x2->2_d3", irr(&[2]), irr(&[2]), irr(&[2])),
        ("su3_3x3b->8_d8", irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1])),
        (
            "su3_8x8->8_d8_om2",
            irr(&[1, 1]),
            irr(&[1, 1]),
            irr(&[1, 1]),
        ),
        ("su3_8x8->27_d27", irr(&[1, 1]), irr(&[1, 1]), irr(&[2, 2])),
        (
            "su4_4x4b->15_d15",
            irr(&[1, 0, 0]),
            irr(&[0, 0, 1]),
            irr(&[1, 0, 1]),
        ),
    ]
}

fn bench_generation(c: &mut Criterion) {
    let mut g = c.benchmark_group("cgc_generation_cold");
    for (label, s1, s2, s3) in cases() {
        g.bench_function(label, |b| {
            b.iter(|| {
                // Reset so every iteration pays the full SVD/QR/descent cost
                // (a warm hit would otherwise be measured after the first).
                cache::reset();
                black_box(cgc(black_box(&s1), black_box(&s2), black_box(&s3)).unwrap())
            })
        });
    }
    g.finish();
}

fn bench_cache_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("cgc_cache_hit");
    for (label, s1, s2, s3) in cases() {
        // Warm once, then every iteration is a hash-lookup + owned clone.
        let _ = cgc(&s1, &s2, &s3).unwrap();
        g.bench_function(label, |b| {
            b.iter(|| black_box(cgc(black_box(&s1), black_box(&s2), black_box(&s3)).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_generation, bench_cache_hit);
criterion_main!(benches);
