//! Exact SU(N) direct-product cold and warm paths for the #59 adjoint controls.

use criterion::{criterion_group, criterion_main, Criterion};
use racah::{
    cache,
    sun::{directproduct, Irrep},
};
use std::hint::black_box;

fn irrep(labels: &[i64]) -> Irrep {
    Irrep::from_dynkin(labels).unwrap()
}

fn bench_product(c: &mut Criterion) {
    for (name, a) in [
        ("su3_8x8", irrep(&[1, 1])),
        ("su4_15x15", irrep(&[1, 0, 1])),
    ] {
        c.bench_function(format!("sun_product_cold/{name}"), |b| {
            b.iter(|| {
                cache::reset();
                black_box(directproduct(black_box(&a), black_box(&a)).unwrap())
            })
        });
        let _ = directproduct(&a, &a).unwrap();
        c.bench_function(format!("sun_product_warm/{name}"), |b| {
            b.iter(|| black_box(directproduct(black_box(&a), black_box(&a)).unwrap()))
        });
    }
}

criterion_group!(benches, bench_product);
criterion_main!(benches);
