//! Timing for the prime-factorized 3j/6j engine (issue #3). Not a CI gate.
//!
//! Two comparisons against `wigner-symbols 0.5.1` on the doubled-spin <= 254
//! overlap domain (exact-value evaluation, no float rounding on either side),
//! plus a standalone timing of the thousands-tier 6j fixtures that the
//! reference crate cannot reach (its u8 label ceiling). Run with `cargo bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use racah::{wigner_3j, wigner_6j};
use wigner_symbols::{Wigner3jm, Wigner6j};

/// Admissible 6j label sets (doubled spins, all <= 254): equal-spin sextuples
/// at a spread of magnitudes so the timing reflects the k-sum length.
const SIX_J: &[[i32; 6]] = &[
    [2, 4, 4, 6, 4, 2],
    [20, 20, 20, 20, 20, 20],
    [60, 60, 60, 60, 60, 60],
    [120, 120, 120, 120, 120, 120],
    [200, 200, 200, 200, 200, 200],
    [254, 254, 254, 254, 254, 254],
];

/// Admissible 3j label sets `(dj,dj,dj; 0,0,0)` (doubled spins, even).
const THREE_J: &[[i32; 3]] = &[[20, 20, 20], [60, 60, 60], [120, 120, 120], [200, 200, 200]];

fn bench_6j(c: &mut Criterion) {
    let mut g = c.benchmark_group("6j_overlap");
    g.bench_function("racah", |b| {
        b.iter(|| {
            for &d in SIX_J {
                black_box(wigner_6j(
                    d[0] as u32,
                    d[1] as u32,
                    d[2] as u32,
                    d[3] as u32,
                    d[4] as u32,
                    d[5] as u32,
                ));
            }
        })
    });
    g.bench_function("wigner_symbols", |b| {
        b.iter(|| {
            for &d in SIX_J {
                let w = Wigner6j {
                    tj1: d[0],
                    tj2: d[1],
                    tj3: d[2],
                    tj4: d[3],
                    tj5: d[4],
                    tj6: d[5],
                };
                black_box(w.value());
            }
        })
    });
    g.finish();
}

fn bench_3j(c: &mut Criterion) {
    let mut g = c.benchmark_group("3j_overlap");
    g.bench_function("racah", |b| {
        b.iter(|| {
            for &d in THREE_J {
                black_box(wigner_3j(d[0] as u32, d[1] as u32, d[2] as u32, 0, 0, 0));
            }
        })
    });
    g.bench_function("wigner_symbols", |b| {
        b.iter(|| {
            for &d in THREE_J {
                let w = Wigner3jm {
                    tj1: d[0],
                    tm1: 0,
                    tj2: d[1],
                    tm2: 0,
                    tj3: d[2],
                    tm3: 0,
                };
                black_box(w.value());
            }
        })
    });
    g.finish();
}

/// The thousands-tier 6j fixtures: doubled spins > 600, beyond wigner-symbols'
/// reach, so racah is timed alone.
fn bench_6j_thousands(c: &mut Criterion) {
    let fixtures = include_str!("../tests/fixtures/su2_6j_large.txt");
    let labels: Vec<[u32; 6]> = fixtures
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<u32> = l
                .split_whitespace()
                .take(6)
                .filter_map(|s| s.parse().ok())
                .collect();
            <[u32; 6]>::try_from(f).ok()
        })
        .filter(|d| d.iter().any(|&x| x > 600))
        .collect();

    c.bench_function("6j_thousands_racah", |b| {
        b.iter(|| {
            for &d in &labels {
                black_box(wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]));
            }
        })
    });
}

/// The repeated-label regime (issue #5): the same small label sets recur many
/// thousands of times, as in tensor-network consumption. `racah_warm` primes
/// the cache once so every iteration is a hit (target: ~hash-lookup cost);
/// `racah_cold` clears the cache before each batch so every call recomputes the
/// big-rational sum. The gap between them is the cache's payoff — and it makes
/// the cold 3j-overlap deficit vs `wigner-symbols` irrelevant here, since a hit
/// touches no arithmetic. (`wigner-symbols` caches internally too, shown for
/// context: after its first batch it is likewise warm.)
fn bench_repeated_labels(c: &mut Criterion) {
    let mut g = c.benchmark_group("repeated_labels");

    let run_racah = || {
        for &d in SIX_J {
            black_box(wigner_6j(
                d[0] as u32,
                d[1] as u32,
                d[2] as u32,
                d[3] as u32,
                d[4] as u32,
                d[5] as u32,
            ));
        }
        for &d in THREE_J {
            black_box(wigner_3j(d[0] as u32, d[1] as u32, d[2] as u32, 0, 0, 0));
        }
    };

    // Warm: prime the cache once, then every measured iteration is a hit.
    racah::cache::reset();
    run_racah();
    g.bench_function("racah_warm", |b| b.iter(run_racah));

    // Cold: clear before each batch so every call misses and recomputes.
    g.bench_function("racah_cold", |b| {
        b.iter(|| {
            racah::cache::reset();
            run_racah();
        })
    });

    g.bench_function("wigner_symbols", |b| {
        b.iter(|| {
            for &d in SIX_J {
                let w = Wigner6j {
                    tj1: d[0],
                    tj2: d[1],
                    tj3: d[2],
                    tj4: d[3],
                    tj5: d[4],
                    tj6: d[5],
                };
                black_box(w.value());
            }
            for &d in THREE_J {
                let w = Wigner3jm {
                    tj1: d[0],
                    tm1: 0,
                    tj2: d[1],
                    tm2: 0,
                    tj3: d[2],
                    tm3: 0,
                };
                black_box(w.value());
            }
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_6j,
    bench_3j,
    bench_6j_thousands,
    bench_repeated_labels
);
criterion_main!(benches);
