//! B/C/D F-symbol generation cost, cold (fresh catalog + full CGC sweep +
//! four-CGC contraction) and warm (derived-f64 bcd F cache hit). Not a CI gate;
//! run with `cargo bench --features cgc-gen`.
//!
//! R is not benched separately: it is a single sparse join of two CGC, an order
//! of magnitude cheaper than a four-CGC F contraction, and needs no cache.

use criterion::{criterion_group, criterion_main, Criterion};
use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};
use racah::cache;
use std::hint::black_box;

fn irr(s: Series, d: &[i64]) -> Irrep {
    Irrep::from_dynkin(s, d).unwrap()
}

/// (label, series, rank, [a,b,c,d,e,f]) representative admissible sextets:
/// a multiplicity-free scalar on each of two series, and the OM = 2 D3 adjoint.
#[allow(clippy::type_complexity)]
fn cases() -> Vec<(&'static str, Series, usize, [Irrep; 6])> {
    vec![
        (
            // C2 = Sp(4): a = trivial forces the 1×1×1×1 identity block.
            "c2_vector_mfree_scalar",
            Series::C,
            2,
            [
                Irrep::trivial(Series::C, 2).unwrap(),
                irr(Series::C, &[0, 1]),
                irr(Series::C, &[0, 1]),
                irr(Series::C, &[2, 0]),
                irr(Series::C, &[0, 1]),
                irr(Series::C, &[2, 0]),
            ],
        ),
        (
            // B2 = SO(5) analogue.
            "b2_vector_mfree_scalar",
            Series::B,
            2,
            [
                Irrep::trivial(Series::B, 2).unwrap(),
                irr(Series::B, &[1, 0]),
                irr(Series::B, &[1, 0]),
                irr(Series::B, &[0, 2]),
                irr(Series::B, &[1, 0]),
                irr(Series::B, &[0, 2]),
            ],
        ),
        (
            // D3 = SO(6) adjoint (0,1,1): g⊗g→g has OM 2, so a 2-axis block.
            "d3_adjoint_om2",
            Series::D,
            3,
            [
                irr(Series::D, &[0, 1, 1]),
                irr(Series::D, &[0, 1, 1]),
                irr(Series::D, &[0, 1, 1]),
                irr(Series::D, &[0, 1, 1]),
                irr(Series::D, &[0, 1, 1]),
                irr(Series::D, &[0, 1, 1]),
            ],
        ),
    ]
}

fn bench_cold(c: &mut Criterion) {
    let mut g = c.benchmark_group("bcd_f_symbol_cold");
    // Cold F blocks (esp. the OM=2 D3 adjoint) can take seconds to materialize.
    g.sample_size(10);
    for (label, series, rank, s) in cases() {
        g.bench_function(label, |b| {
            b.iter(|| {
                // Fresh catalog + cleared caches: pay the full canonical-parent
                // CGC materialization (SVD sweeps) plus the four-CGC contraction.
                cache::reset();
                let mut cat = CanonicalCatalog::new(series, rank).unwrap();
                black_box(
                    f_symbol(
                        &mut cat,
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
    let mut g = c.benchmark_group("bcd_f_symbol_cache_hit");
    for (label, series, rank, s) in cases() {
        let mut cat = CanonicalCatalog::new(series, rank).unwrap();
        let _ = f_symbol(&mut cat, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5]).unwrap(); // warm
        g.bench_function(label, |b| {
            b.iter(|| {
                black_box(
                    f_symbol(
                        &mut cat,
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
