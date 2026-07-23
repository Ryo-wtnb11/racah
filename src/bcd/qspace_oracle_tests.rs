//! QSpace CGC external anchor for the B/C/D intertwiner alignment (issue #29,
//! S3.5). Compares racah's **aligned** CGC against QSpace `getCG` fixtures
//! (`tests/fixtures/qspace_cgc.txt`, PR #31; unit-Frobenius per `(channel, OM
//! slice)`, product index `q = i1 + SIZE_a·i2` — the same "factor a fast"
//! Kronecker convention racah pins).
//!
//! # The robust form: the isotypic projector, behind the factor-basis dictionary
//!
//! The per-channel isotypic projector `P = Σ_μ Ĉ_μ·Ĉ_μᵀ` (columns rescaled to
//! QSpace's unit-Frobenius-per-slice normalization) is invariant under the whole
//! coupled-side gauge alignment fixes — the internal frame of `c` and the O(N)
//! outer-multiplicity mixing (§15.6 leaves the latter open, so a non-invariant
//! comparison could not test the OM≥2 channels at all). What `P` still depends on
//! is the per-**factor** basis: racah's S3.1 seed frame and QSpace's fixture frame
//! are two bases of the same factor irrep, related by an orthogonal
//! **factor-basis dictionary** `O_f`. For the defining rep the weights are
//! multiplicity-free, so `O_f` is a signed permutation: the permutation is fixed
//! by descending-weight order (QSpace's convention), and the `d` signs are the
//! dictionary this harness fits once per group and then holds fixed across every
//! vector² channel. With `O_v` applied, the projectors match to `1e-9` — the
//! external confirmation that alignment produces the QSpace values, not merely a
//! self-consistent gauge.
//!
//! Channels whose factors are themselves swept irreps in a degenerate frame (the
//! adjoint², incl. the SO(6) 84 and the OM=2 channel) need the adjoint
//! factor-basis dictionary, an O(k) block per degenerate weight space rather than
//! a sign — a bootstrap from QSpace's defining frame + the fixture's vector→adjoint
//! CGC. That is the S3.5 follow-up; here those channels are anchored structurally
//! (racah reproduces QSpace's coupled dimension and outer multiplicity, including
//! OM=2 for the D3 adjoint square), and the full projector match is left to the
//! dictionary bootstrap.

use super::{CanonicalCatalog, Irrep, Series};

/// One parsed fixture channel.
struct Channel {
    sym: String,
    series: Series,
    rank: usize,
    j1: Vec<i64>,
    j2: Vec<i64>,
    j3: Vec<i64>,
    d1: usize,
    d2: usize,
    d3: usize,
    om: usize,
    /// `(product row = i + d1·j, coupled col k, OM index m, value)`.
    entries: Vec<(usize, usize, usize, f64)>,
}

fn sym_to_series(sym: &str) -> Option<(Series, usize)> {
    match sym {
        "SO5" => Some((Series::B, 2)),
        "Sp4" => Some((Series::C, 2)),
        "SO6" => Some((Series::D, 3)),
        _ => None, // SU2 rows belong to the su2 suites
    }
}

fn ints(s: &str) -> Vec<i64> {
    s.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn parse_fixture() -> Vec<Channel> {
    let text = include_str!("../../tests/fixtures/qspace_cgc.txt");
    let mut out: Vec<Channel> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("---") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("CH ") {
            let seg: Vec<&str> = rest.split('|').collect();
            let sym = seg[0].trim().to_string();
            let Some((series, rank)) = sym_to_series(&sym) else {
                // Still record it so entry lines have a home, but mark rank 0.
                let dims = ints(seg[4]);
                out.push(Channel {
                    sym,
                    series: Series::B,
                    rank: 0,
                    j1: ints(seg[1]),
                    j2: ints(seg[2]),
                    j3: ints(seg[3]),
                    d1: dims[0] as usize,
                    d2: dims[1] as usize,
                    d3: dims[2] as usize,
                    om: dims.get(3).copied().unwrap_or(1) as usize,
                    entries: Vec::new(),
                });
                continue;
            };
            let dims = ints(seg[4]);
            out.push(Channel {
                sym,
                series,
                rank,
                j1: ints(seg[1]),
                j2: ints(seg[2]),
                j3: ints(seg[3]),
                d1: dims[0] as usize,
                d2: dims[1] as usize,
                d3: dims[2] as usize,
                om: dims.get(3).copied().unwrap_or(1) as usize,
                entries: Vec::new(),
            });
        } else {
            let t: Vec<&str> = line.split_whitespace().collect();
            let ch = out.last_mut().unwrap();
            let u = |x: &str| x.parse::<usize>().unwrap();
            let (i, j, k, m, v): (usize, usize, usize, usize, f64) = if t.len() == 4 {
                (u(t[0]), u(t[1]), u(t[2]), 0, t[3].parse().unwrap())
            } else {
                (u(t[0]), u(t[1]), u(t[2]), u(t[3]), t[4].parse().unwrap())
            };
            let row = i + ch.d1 * j;
            ch.entries.push((row, k, m, v));
        }
    }
    out
}

/// QSpace projector `P = Σ_m C_m·C_mᵀ` (row-major `d1d2 × d1d2`), unit-Frobenius
/// slices as stored.
fn qspace_projector(ch: &Channel) -> Vec<f64> {
    let rows = ch.d1 * ch.d2;
    let mut cm = vec![vec![0.0f64; rows * ch.d3]; ch.om];
    for &(row, col, m, v) in &ch.entries {
        cm[m][row * ch.d3 + col] = v;
    }
    let mut p = vec![0.0f64; rows * rows];
    for c in &cm {
        for a in 0..rows {
            for b in 0..rows {
                let mut acc = 0.0;
                for k in 0..ch.d3 {
                    acc += c[a * ch.d3 + k] * c[b * ch.d3 + k];
                }
                p[a * rows + b] += acc;
            }
        }
    }
    p
}

/// racah's aligned CGC copies for a channel, each column-major `rows × d3`, plus
/// the multiplicity. Rescaled to QSpace's unit-Frobenius-per-slice normalization
/// (racah copies are isometries, `‖·‖_F = √d3`).
fn racah_copies(ch: &Channel) -> (Vec<Vec<f64>>, usize) {
    let mut cat = CanonicalCatalog::new(ch.series, ch.rank).unwrap();
    let s1 = Irrep::from_dynkin(ch.series, &ch.j1).unwrap();
    let s2 = Irrep::from_dynkin(ch.series, &ch.j2).unwrap();
    let s3 = Irrep::from_dynkin(ch.series, &ch.j3).unwrap();
    let cgc = cat.cgc(&s1, &s2, &s3).unwrap();
    let (rows, d3) = cgc.copy_shape();
    assert_eq!(rows, ch.d1 * ch.d2);
    assert_eq!(d3, ch.d3);
    let scale = 1.0 / (d3 as f64).sqrt();
    let copies = (0..cgc.multiplicity())
        .map(|mu| cgc.copy(mu).iter().map(|x| x * scale).collect())
        .collect();
    (copies, cgc.multiplicity())
}

/// The descending-weight position of each state of irrep `f` in racah's basis —
/// QSpace's factor-basis order (`z2.FlipCols; sortRecs_float(-1)`, matching the
/// sweep's `descending_weight_perm`). `qpos[s]` is the QSpace index of racah
/// state `s`.
fn descending_positions(cat: &mut CanonicalCatalog, f: &Irrep) -> Vec<usize> {
    let g = cat.generators(f).unwrap();
    let d = g.dim();
    let nz = g.rank();
    let w: Vec<Vec<i64>> = (0..d)
        .map(|s| {
            (0..nz)
                .map(|j| g.cartan_diag(j)[s].round() as i64)
                .collect()
        })
        .collect();
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| {
        for c in (0..nz).rev() {
            match w[b][c].cmp(&w[a][c]) {
                std::cmp::Ordering::Equal => {}
                o => return o,
            }
        }
        a.cmp(&b)
    });
    let mut qpos = vec![0usize; d];
    for (pos, &s) in order.iter().enumerate() {
        qpos[s] = pos;
    }
    qpos
}

/// Transform racah's projector into QSpace's product basis under a signed
/// permutation `O_a = (qpos, signs)` on each factor (`P ↦ (O_a⊗O_a) P (O_a⊗O_a)ᵀ`),
/// applied by rebuilding the copies row-by-row, then compare to `pq`.
fn worst_projector_diff(
    copies: &[Vec<f64>],
    d1: usize,
    d2: usize,
    d3: usize,
    qpos: &[usize],
    signs: &[i8],
    pq: &[f64],
) -> f64 {
    let rows = d1 * d2;
    let mut p = vec![0.0f64; rows * rows];
    for col in copies {
        // Remap rows: racah a=(ma,mb) → QSpace A=(qpos[ma]+d1·qpos[mb]), sign flip.
        let mut c2 = vec![0.0f64; rows * d3];
        for a in 0..rows {
            let (ma, mb) = (a % d1, a / d1);
            let big = qpos[ma] + d1 * qpos[mb];
            let sgn = (signs[ma] * signs[mb]) as f64;
            for k in 0..d3 {
                c2[big * d3 + k] = sgn * col[a + k * rows];
            }
        }
        for a in 0..rows {
            for b in 0..rows {
                let mut acc = 0.0;
                for k in 0..d3 {
                    acc += c2[a * d3 + k] * c2[b * d3 + k];
                }
                p[a * rows + b] += acc;
            }
        }
    }
    p.iter()
        .zip(pq)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Precomputed per-channel data for the sign search: racah copies and the QSpace
/// projector, plus the product/coupled dimensions.
struct ChannelPrep {
    copies: Vec<Vec<f64>>,
    pq: Vec<f64>,
    d1: usize,
    d2: usize,
    d3: usize,
}

/// Fit the defining factor-basis dictionary (the `d` signs, global sign fixed)
/// jointly over a group's vector² channels, then assert every one matches QSpace.
fn check_defining_group(series: Series, rank: usize, sym: &str, channels: &[Channel]) {
    let vec_sq: Vec<&Channel> = channels
        .iter()
        .filter(|c| c.sym == sym && c.j1 == c.j2 && is_defining(&c.j1))
        .collect();
    assert!(!vec_sq.is_empty(), "{sym}: no vector² channels in fixture");

    let mut cat = CanonicalCatalog::new(series, rank).unwrap();
    let v = Irrep::from_dynkin(series, &defining_dynkin(rank)).unwrap();
    let qpos = descending_positions(&mut cat, &v);
    let d = v.dim().try_into().unwrap();

    // Precompute racah copies + QSpace projectors per channel.
    let data: Vec<ChannelPrep> = vec_sq
        .iter()
        .map(|ch| ChannelPrep {
            copies: racah_copies(ch).0,
            pq: qspace_projector(ch),
            d1: ch.d1,
            d2: ch.d2,
            d3: ch.d3,
        })
        .collect();

    // Search signs (global sign of the factor basis fixed at +1).
    let mut best: Option<(f64, Vec<i8>)> = None;
    for bits in 0u32..(1u32 << (d - 1)) {
        let mut signs = vec![1i8; d];
        for (s, sign) in signs.iter_mut().enumerate().skip(1) {
            if bits & (1 << (s - 1)) != 0 {
                *sign = -1;
            }
        }
        let worst = data
            .iter()
            .map(|p| worst_projector_diff(&p.copies, p.d1, p.d2, p.d3, &qpos, &signs, &p.pq))
            .fold(0.0f64, f64::max);
        if best.as_ref().is_none_or(|(bw, _)| worst < *bw) {
            best = Some((worst, signs));
        }
    }
    let (worst, signs) = best.unwrap();
    assert!(
        worst < 1e-9,
        "{sym}: no defining factor-basis dictionary reconciles the vector² \
         projectors with QSpace (best worst |ΔP| = {worst:e}, signs {signs:?})"
    );
    eprintln!(
        "QSpace anchor {sym}: {} vector² channels match under signed-perm dictionary, worst |ΔP| = {worst:e}",
        data.len()
    );
}

fn defining_dynkin(rank: usize) -> Vec<i64> {
    let mut d = vec![0i64; rank];
    d[0] = 1;
    d
}

fn is_defining(j: &[i64]) -> bool {
    j.first() == Some(&1) && j[1..].iter().all(|&x| x == 0)
}

#[test]
fn qspace_vector_products_match_under_factor_dictionary() {
    let channels = parse_fixture();
    check_defining_group(Series::B, 2, "SO5", &channels);
    check_defining_group(Series::C, 2, "Sp4", &channels);
    check_defining_group(Series::D, 3, "SO6", &channels);
}

/// Structural anchor for the non-defining-factor channels (adjoint², incl. the
/// SO(6) 84 = (0,2,2) and the OM=2 channel): racah reproduces QSpace's coupled
/// dimension and outer multiplicity for every fixture channel. The full projector
/// match for these needs the adjoint factor-basis dictionary (S3.5 follow-up).
#[test]
fn qspace_channel_structure_matches() {
    let mut checked = 0usize;
    for ch in parse_fixture() {
        if ch.rank == 0 {
            continue;
        }
        let mut cat = CanonicalCatalog::new(ch.series, ch.rank).unwrap();
        let s1 = Irrep::from_dynkin(ch.series, &ch.j1).unwrap();
        let s2 = Irrep::from_dynkin(ch.series, &ch.j2).unwrap();
        let s3 = Irrep::from_dynkin(ch.series, &ch.j3).unwrap();
        let cgc = cat.cgc(&s1, &s2, &s3).unwrap();
        let (rows, d3) = cgc.copy_shape();
        assert_eq!(
            rows,
            ch.d1 * ch.d2,
            "{:?} {:?}⊗{:?} rows",
            ch.series,
            ch.j1,
            ch.j2
        );
        assert_eq!(d3, ch.d3, "{:?} coupled dim for {:?}", ch.series, ch.j3);
        assert_eq!(
            cgc.multiplicity(),
            ch.om,
            "{:?} outer multiplicity for {:?}⊗{:?}→{:?}",
            ch.series,
            ch.j1,
            ch.j2,
            ch.j3
        );
        checked += 1;
    }
    assert!(
        checked >= 20,
        "expected the B/C/D channel set, got {checked}"
    );
}
