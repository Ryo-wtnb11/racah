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
//! **factor-basis dictionary** `O_f`.
//!
//! `O_f` is **derived and verified**, not fitted (the reviewer's P2 on the earlier
//! sign-fit). QSpace's own generator matrices (`Sp_k`, `Sz_k`) for the six needed
//! irreps are exported in `tests/fixtures/qspace_generators.txt`; the dictionary is
//! solved as the orthogonal intertwiner `Sp_q[i]·O = O·Sp_r[i]` by the same
//! weight-space Procrustes descent the production alignment uses
//! ([`solve_dictionary`]), then **verified element-wise** as an intertwiner
//! (residual `≤ 1e-10`) before it is used. A convention mismatch would blow that
//! residual, not silently pass. For the defining rep the weights are
//! multiplicity-free so `O_f` collapses to a signed permutation; for the adjoint
//! it carries genuine orthogonal blocks on the degenerate weight spaces.
//!
//! With the verified dictionaries applied, every fixture channel's full projector
//! matches QSpace to round-off (`< 1e-9`): the defining (vector²) products, the
//! SO(5)/SO(6) adjoint² products — **including the SO(6) 84 = (0,2,2) and the OM=2
//! 15-channel** — and the defining⊗adjoint cross products. Sp4's adjoint is [2 0]
//! (dim 10); the CGC fixture has no [2 0]⊗[2 0] channel, so Sp4 adjoint² is not
//! testable and stays covered by the structural anchor only. Every fixture channel
//! is additionally anchored structurally (coupled dimension + outer multiplicity)
//! by [`qspace_channel_structure_matches`].

use std::collections::HashMap;

use super::linalg::{matmul, svd, Dense};
use super::{CanonicalCatalog, Irrep, Series};

/// Post-verification headroom below the intertwiner tolerance the anchor demands.
/// A dictionary that verifies but sits *just* under `1e-10` is a watch item; the
/// tests log the actual residual so a shrinking margin is visible before it
/// bricks (reviewer's margin caveat, PR #34).
const DICT_RESIDUAL_TOL: f64 = 1e-10;

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

// ---- QSpace generator fixture (verified factor-basis dictionaries) ----------

/// One irrep's QSpace generator matrices: the `r` raising operators `Sp[i]`
/// (dense `d×d`) and the `r` Cartan diagonals `Sz[i]` (length `d`), in QSpace's
/// own RSet basis. Parsed from `tests/fixtures/qspace_generators.txt`.
struct QGen {
    sp: Vec<Dense>,
    sz: Vec<Vec<f64>>,
}

/// Parse the generator fixture into `(sym, dynkin) → QGen`.
fn parse_generators() -> HashMap<(String, Vec<i64>), QGen> {
    let text = include_str!("../../tests/fixtures/qspace_generators.txt");
    let mut out: HashMap<(String, Vec<i64>), QGen> = HashMap::new();
    let mut key: Option<(String, Vec<i64>)> = None;
    let mut dim = 0usize;
    let mut sp: Vec<Dense> = Vec::new();
    let mut sz: Vec<Vec<f64>> = Vec::new();
    // `Some(true)` inside an SP block, `Some(false)` inside an SZ block.
    let mut cur_sp: Option<bool> = None;
    let mut flush =
        |key: &mut Option<(String, Vec<i64>)>, sp: &mut Vec<Dense>, sz: &mut Vec<Vec<f64>>| {
            if let Some(k) = key.take() {
                out.insert(
                    k,
                    QGen {
                        sp: std::mem::take(sp),
                        sz: std::mem::take(sz),
                    },
                );
            }
        };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("---") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("IRREP ") {
            flush(&mut key, &mut sp, &mut sz);
            let seg: Vec<&str> = rest.split('|').collect();
            let sym = seg[0].trim().to_string();
            let dynkin = ints(seg[1]);
            dim = seg[2].trim().strip_prefix("dim=").unwrap().parse().unwrap();
            key = Some((sym, dynkin));
            cur_sp = None;
        } else if line.starts_with("Z ") {
            // Weights are recomputed from Sz below; the Z rows are documentation.
        } else if let Some(rest) = line.strip_prefix("OP ") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            match t[0] {
                "SP" => {
                    sp.push(Dense::zeros(dim, dim));
                    cur_sp = Some(true);
                }
                "SZ" => {
                    sz.push(vec![0.0; dim]);
                    cur_sp = Some(false);
                }
                _ => unreachable!("unknown OP {}", t[0]),
            }
        } else {
            let t: Vec<&str> = line.split_whitespace().collect();
            let (row, col, v): (usize, usize, f64) = (
                t[0].parse().unwrap(),
                t[1].parse().unwrap(),
                t[2].parse().unwrap(),
            );
            match cur_sp {
                Some(true) => sp.last_mut().unwrap().set(row, col, v),
                Some(false) => {
                    debug_assert_eq!(row, col, "Sz must be diagonal");
                    sz.last_mut().unwrap()[row] = v;
                }
                None => unreachable!("data line outside an OP block"),
            }
        }
    }
    flush(&mut key, &mut sp, &mut sz);
    out
}

/// racah's generator set for irrep `f` as plain `Dense`/`Vec<f64>` (a copy of the
/// catalog's generators, so it outlives the borrow).
fn racah_gen(cat: &mut CanonicalCatalog, f: &Irrep) -> QGen {
    let g = cat.generators(f).unwrap();
    let r = g.rank();
    QGen {
        sp: (0..r).map(|i| g.raising(i).clone()).collect(),
        sz: (0..r).map(|i| g.cartan_diag(i).to_vec()).collect(),
    }
}

/// The integer weight of state `s` under a generator set (`Sz[k]` diagonals).
fn weight_of(g: &QGen, s: usize) -> Vec<i64> {
    g.sz.iter().map(|zk| zk[s].round() as i64).collect()
}

/// The state-index lists of each weight space keyed by integer weight, plus the
/// weights in first-seen (descending) order.
type WeightSpaces = (Vec<Vec<i64>>, HashMap<Vec<i64>, Vec<usize>>);

/// Group states by integer weight, preserving first-seen (descending-weight) order.
fn weight_spaces(g: &QGen, dim: usize) -> WeightSpaces {
    let mut order: Vec<Vec<i64>> = Vec::new();
    let mut map: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for s in 0..dim {
        let w = weight_of(g, s);
        map.entry(w.clone()).or_insert_with(|| {
            order.push(w.clone());
            Vec::new()
        });
        map.get_mut(&w).unwrap().push(s);
    }
    (order, map)
}

/// Solve the orthogonal **factor-basis dictionary** `M` (`d×d`, rows = QSpace
/// state, cols = racah state) that intertwines racah's generators into QSpace's:
/// `Sp_q[i]·M = M·Sp_r[i]` for every simple root `i`, block-diagonal over weight
/// spaces (it commutes with the shared Cartans). Returns `(M, residual)` where
/// `residual` is the element-wise intertwiner check `max_i ‖Sp_q[i]·M − M·Sp_r[i]‖_∞`
/// — the fit-free verification the S3.5 anchor demands.
///
/// Same descent as the production `intertwiner` (§15.3 of `docs/gauge_soN.md`),
/// but across two *different* native state orderings (racah's vs QSpace's),
/// matched by weight value: from the 1-dim highest-weight space (block `= +1`)
/// down the ladder, each `Sp[i]` from an already-solved higher space gives the
/// exact relation `M_T·B = A·M_S = C`, solved by orthogonal Procrustes
/// `M_T = U·Vᵀ` from `SVD(C·Bᵀ)`. Degenerate weight spaces (the adjoint) yield a
/// genuine orthogonal block, not a sign.
fn solve_dictionary(gr: &QGen, gq: &QGen, dim: usize) -> (Dense, f64) {
    let r = gr.sp.len();
    let (order, r_spaces) = weight_spaces(gr, dim);
    let (_, q_spaces) = weight_spaces(gq, dim);

    // Root shift α_i of each raising op (from any nonzero racah entry).
    let alpha: Vec<Vec<i64>> = (0..r)
        .map(|i| {
            let sp = &gr.sp[i];
            for row in 0..dim {
                for col in 0..dim {
                    if sp.at(row, col).abs() > 1e-9 {
                        let (wr, wc) = (weight_of(gr, row), weight_of(gr, col));
                        return wr.iter().zip(&wc).map(|(a, b)| a - b).collect();
                    }
                }
            }
            vec![0; gr.sz.len()]
        })
        .collect();

    let mut m = Dense::zeros(dim, dim);
    let mut blocks: HashMap<Vec<i64>, Dense> = HashMap::new();
    for w in &order {
        let tr = &r_spaces[w];
        let tq = &q_spaces[w];
        let n_t = tr.len();
        assert_eq!(n_t, tq.len(), "weight {w:?}: multiplicity mismatch");
        let mut c_cols: Vec<f64> = Vec::new();
        let mut b_cols: Vec<f64> = Vec::new();
        let mut cols = 0usize;
        // `i` indexes three parallel structures (alpha, gq.sp, gr.sp).
        #[allow(clippy::needless_range_loop)]
        for i in 0..r {
            let ws: Vec<i64> = w.iter().zip(&alpha[i]).map(|(a, b)| a + b).collect();
            let (Some(sr), Some(sq), Some(ms)) =
                (r_spaces.get(&ws), q_spaces.get(&ws), blocks.get(&ws))
            else {
                continue;
            };
            let n_s = sr.len();
            // A = Sp_q[i]|_{Tq,Sq}, B = Sp_r[i]|_{Tr,Sr}; C = A·M_S.
            let mut a = Dense::zeros(n_t, n_s);
            let mut bmat = Dense::zeros(n_t, n_s);
            for (tl, &t) in tq.iter().enumerate() {
                for (sl, &s) in sq.iter().enumerate() {
                    a.set(tl, sl, gq.sp[i].at(t, s));
                }
            }
            for (tl, &t) in tr.iter().enumerate() {
                for (sl, &s) in sr.iter().enumerate() {
                    bmat.set(tl, sl, gr.sp[i].at(t, s));
                }
            }
            let c = matmul(&a, ms).unwrap();
            c_cols.extend_from_slice(&c.data);
            b_cols.extend_from_slice(&bmat.data);
            cols += n_s;
        }
        let block = if cols == 0 {
            identity(n_t) // highest-weight space: +1 fixes the (projector-irrelevant) global sign.
        } else {
            let cmat = Dense {
                rows: n_t,
                cols,
                data: c_cols,
            };
            let bmat = Dense {
                rows: n_t,
                cols,
                data: b_cols,
            };
            let prod = matmul(&cmat, &bmat.transpose()).unwrap();
            let (u, _s, vt) = svd(&prod).unwrap();
            matmul(&u, &vt).unwrap()
        };
        for (tl, &t) in tq.iter().enumerate() {
            for (ul, &u) in tr.iter().enumerate() {
                m.set(t, u, block.at(tl, ul));
            }
        }
        blocks.insert(w.clone(), block);
    }

    // Verify: element-wise intertwiner residual over every simple root.
    let mut residual = 0.0f64;
    for i in 0..r {
        let lhs = matmul(&gq.sp[i], &m).unwrap();
        let rhs = matmul(&m, &gr.sp[i]).unwrap();
        for (x, y) in lhs.data.iter().zip(&rhs.data) {
            residual = residual.max((x - y).abs());
        }
    }
    (m, residual)
}

fn identity(n: usize) -> Dense {
    let mut m = Dense::zeros(n, n);
    for i in 0..n {
        m.set(i, i, 1.0);
    }
    m
}

/// Solve + verify the dictionary for one factor irrep, asserting the residual is
/// below tolerance and logging the headroom (margin watch item).
fn dictionary_for(
    cat: &mut CanonicalCatalog,
    series: Series,
    sym: &str,
    dynkin: &[i64],
    gens: &HashMap<(String, Vec<i64>), QGen>,
) -> Dense {
    let f = Irrep::from_dynkin(series, dynkin).unwrap();
    let gr = racah_gen(cat, &f);
    let dim = gr.sp[0].rows;
    let gq = gens
        .get(&(sym.to_string(), dynkin.to_vec()))
        .unwrap_or_else(|| panic!("{sym} {dynkin:?}: no QSpace generators in fixture"));
    let (m, residual) = solve_dictionary(&gr, gq, dim);
    assert!(
        residual <= DICT_RESIDUAL_TOL,
        "{sym} {dynkin:?}: factor-basis dictionary is not a verified intertwiner \
         (residual {residual:e} > {DICT_RESIDUAL_TOL:e}) — a convention mismatch, \
         do not loosen the tolerance"
    );
    if residual == 0.0 {
        eprintln!("QSpace dictionary {sym} {dynkin:?}: verified intertwiner, residual 0 (exact)");
    } else {
        eprintln!(
            "QSpace dictionary {sym} {dynkin:?}: verified intertwiner, residual {residual:e} \
             (margin ×{:.0e} under {DICT_RESIDUAL_TOL:e})",
            DICT_RESIDUAL_TOL / residual
        );
    }
    m
}

/// racah's per-channel isotypic projector `P = Σ_μ C_μ·C_μᵀ`, with each factor
/// leg transformed into QSpace's basis by the verified dictionaries `m1` (for
/// factor `j1`), `m2` (for factor `j2`), emitted in QSpace's **fixture** product
/// order. Returned row-major `rows×rows`.
///
/// racah packs factor `j1` fast (`row = i_{j1} + dim(j1)·i_{j2}`, module docs);
/// the QSpace fixture packs *its* fast index over `ch.d1` states. For the
/// asymmetric (cross) channels `ch.d1 = dim(j2)`, i.e. the fixture packs the `j2`
/// leg fast — the reverse of racah — so the fixture linear index is detected from
/// the leg dims (`da`/`db`), not assumed. Symmetric channels have `da = db` and
/// the two orders coincide.
fn racah_projector_in_qspace_basis(ch: &Channel, m1: &Dense, m2: &Dense) -> Vec<f64> {
    let (da, db, d3) = (m1.rows, m2.rows, ch.d3); // da = dim(j1), db = dim(j2)
    let rows = da * db;
    assert_eq!(rows, ch.d1 * ch.d2, "{:?}⊗{:?} product size", ch.j1, ch.j2);
    // Which racah leg is the fixture's fast index? ch.d1 == da → j1 fast (same as
    // racah); ch.d1 == db → j2 fast (reversed). da==db makes the choice moot.
    let fixture_j1_fast = ch.d1 == da;
    let (copies, _) = racah_copies(ch);
    let mut p = vec![0.0f64; rows * rows];
    for col in &copies {
        // Transform each leg into QSpace's basis, place at the fixture index.
        let mut c2 = vec![0.0f64; rows * d3];
        for ia in 0..da {
            for ib in 0..db {
                let src = ia + da * ib; // racah's j1-fast packing
                for qa in 0..da {
                    let w1 = m1.at(qa, ia);
                    if w1 == 0.0 {
                        continue;
                    }
                    for qb in 0..db {
                        let w = w1 * m2.at(qb, ib);
                        if w == 0.0 {
                            continue;
                        }
                        let dst = if fixture_j1_fast {
                            qa + da * qb
                        } else {
                            qb + db * qa
                        };
                        for k in 0..d3 {
                            c2[dst * d3 + k] += w * col[src + k * rows];
                        }
                    }
                }
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
    p
}

/// Worst `|ΔP|` between racah's dictionary-transformed projector and QSpace's.
fn projector_diff(ch: &Channel, m1: &Dense, m2: &Dense) -> f64 {
    let pr = racah_projector_in_qspace_basis(ch, m1, m2);
    let pq = qspace_projector(ch);
    pr.iter()
        .zip(&pq)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Every fixture channel of `sym` whose two factors both appear in `factors`, with
/// the (dynkin of factor 1, dynkin of factor 2) so the caller picks the dictionaries.
fn channels_over<'a>(sym: &str, factors: &[Vec<i64>], channels: &'a [Channel]) -> Vec<&'a Channel> {
    channels
        .iter()
        .filter(|c| c.sym == sym && factors.contains(&c.j1) && factors.contains(&c.j2))
        .collect()
}

/// Assert the full projector `P = Σ_μ C_μ·C_μᵀ` matches QSpace for every fixture
/// channel of `sym` built from the given factor irreps, using the **verified**
/// dictionaries. `factors` are `(dynkin, dim-label)` pairs; the dictionary for
/// each is solved once and reused.
fn check_group(series: Series, rank: usize, sym: &str, factors: &[Vec<i64>], channels: &[Channel]) {
    let mut cat = CanonicalCatalog::new(series, rank).unwrap();
    let gens = parse_generators();
    let dicts: HashMap<Vec<i64>, Dense> = factors
        .iter()
        .map(|d| (d.clone(), dictionary_for(&mut cat, series, sym, d, &gens)))
        .collect();

    let chans = channels_over(sym, factors, channels);
    assert!(!chans.is_empty(), "{sym}: no channels over {factors:?}");
    for ch in chans {
        let m1 = &dicts[&ch.j1];
        let m2 = &dicts[&ch.j2];
        let worst = projector_diff(ch, m1, m2);
        assert!(
            worst < 1e-9,
            "{sym} {:?}⊗{:?}→{:?} (OM={}): projector mismatch under verified \
             dictionaries, worst |ΔP| = {worst:e}",
            ch.j1,
            ch.j2,
            ch.j3,
            ch.om
        );
        eprintln!(
            "QSpace anchor {sym} {:?}⊗{:?}→{:?} (OM={}): P matches, worst |ΔP| = {worst:e}",
            ch.j1, ch.j2, ch.j3, ch.om
        );
    }
}

/// Defining-factor anchor (item 1): the vector² channels, now against a
/// **verified** dictionary (fit-free, the reviewer's P2). One dictionary per group,
/// solved from QSpace's own generators and asserted as an element-wise intertwiner.
#[test]
fn qspace_vector_products_match_under_verified_dictionary() {
    let channels = parse_fixture();
    check_group(Series::B, 2, "SO5", &[vec![1, 0]], &channels);
    check_group(Series::C, 2, "Sp4", &[vec![1, 0]], &channels);
    check_group(Series::D, 3, "SO6", &[vec![1, 0, 0]], &channels);
}

/// Adjoint-factor anchor (item 2): the full `P = Σ C_μ C_μᵀ` projector match for
/// the SO(5) and SO(6) adjoint² channels — including the SO(6) 84 = (0,2,2) and
/// the OM=2 15-channel — plus the defining⊗adjoint cross products, all with
/// verified dictionaries (the adjoint dictionary carries genuine orthogonal blocks
/// on the degenerate weight spaces, not signs).
///
/// Sp4's adjoint is [2 0] (dim 10); the CGC fixture has no [2 0]⊗[2 0] channel, so
/// Sp4 adjoint² is not testable here and stays covered only by the structural
/// anchor. Sp4's [0 1] (dim 5) has no exported generators either.
#[test]
fn qspace_adjoint_products_match_under_verified_dictionary() {
    let channels = parse_fixture();
    check_group(Series::B, 2, "SO5", &[vec![1, 0], vec![0, 2]], &channels);
    check_group(
        Series::D,
        3,
        "SO6",
        &[vec![1, 0, 0], vec![0, 1, 1]],
        &channels,
    );
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
