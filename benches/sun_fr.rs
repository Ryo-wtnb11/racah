//! SU(N) F-symbol generation cost across representative families, cold (full
//! four-CGC contraction, CGC caches cleared) and warm (derived-f64 F cache
//! hit). Not a CI gate; run with `cargo bench --features cgc-gen`.
//!
//! R is not benched separately: it is a single sparse join of two CGC, an order
//! of magnitude cheaper than a four-CGC F contraction, and needs no cache.

use criterion::{criterion_group, criterion_main, Criterion};
use racah::cache;
use racah::sun::{f_symbol, Irrep};
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

criterion_group!(benches, bench_cold, bench_hit);
criterion_main!(benches);
