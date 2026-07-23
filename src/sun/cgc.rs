//! SU(N) Clebsch-Gordan coefficient generation (Layer 2 of the `cgc-gen`
//! track): highest-weight nullspace, deterministic gauge canonicalization, and
//! reverse-lex weight-ladder descent.
//!
//! Ported from SUNRepresentations.jl v0.4.0 `src/clebschgordan.jl`. The
//! gauge produced here is a semver contract of this crate: any change that can
//! alter a returned coefficient value is a breaking release. See
//! `docs/gauge.md` for the full specification and per-choice citations.
//!
//! Pipeline (`clebschgordan.jl:_CGC`):
//! 1. `highest_weight_CGC` — build the sparse raising-operator system over the
//!    coupling subspace at `s3`'s highest weight, take its dense right
//!    nullspace (full SVD, cut at `TOL_NULLSPACE`), and gauge-fix the block.
//! 2. `lower_weight_CGC!` — descend to every lower weight in reverse
//!    lexicographic order, solving a per-weight least-squares system.
//! 3. `purge!` — drop coefficients below `TOL_PURGE`.
//!
//! Numerical seams (full SVD, positive-diagonal QR, least squares) route
//! through [`super::linalg`] (tenferro-linalg public APIs only).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::linalg::{self, Mat};
use super::{directproduct, GtPattern, Irrep, LadderEntry, SunError};

/// Absolute singular-value tolerance for the highest-weight nullspace rank cut.
/// Reference: `clebschgordan.jl:TOL_NULLSPACE = 1.0e-13`.
pub const TOL_NULLSPACE: f64 = 1.0e-13;

/// Pivot tolerance for the `cref!` column-echelon gauge step. Reference:
/// `clebschgordan.jl:TOL_GAUGE = 1.0e-11` (deliberately looser than
/// `TOL_NULLSPACE`, per the reference comment).
pub const TOL_GAUGE: f64 = 1.0e-11;

/// Magnitude below which an assembled coefficient is dropped as an approximate
/// zero. Reference: `clebschgordan.jl:TOL_PURGE = 1.0e-14`.
pub const TOL_PURGE: f64 = 1.0e-14;

/// Generation gate: worst tolerated CGC orthonormality residual. Not a
/// reference constant; sized well above the f64 SVD/QR round-off floor
/// (`~sqrt(dim) * eps`) so a true gauge/algebra defect trips it while faithful
/// round-off does not. See `docs/gauge.md`.
pub const TOL_ORTHO: f64 = 1.0e-9;

/// Generation gate: worst tolerated ladder-intertwiner residual (same sizing
/// rationale as `TOL_ORTHO`).
pub const TOL_LADDER: f64 = 1.0e-9;

/// One nonzero element of a sparse m-basis CGC tensor: value at
/// `(m1, m2, m3, mu)`, where `m1`/`m2`/`m3` are 0-based GT basis indices into
/// [`Irrep::patterns`] of `s1`/`s2`/`s3` and `mu` is the outer-multiplicity
/// (trailing) axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgcEntry {
    /// 0-based GT basis index in `s1`.
    pub m1: u32,
    /// 0-based GT basis index in `s2`.
    pub m2: u32,
    /// 0-based GT basis index in `s3`.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The (real, standard-gauge) coefficient.
    pub value: f64,
}

/// The Clebsch-Gordan coefficients coupling `s1 ⊗ s2 → s3`, as a sparse
/// m-basis tensor with the outer multiplicity carried on a trailing axis.
///
/// Shape is `[dim(s1), dim(s2), dim(s3), N]` with `N = N^{s3}_{s1 s2}` the
/// Layer 1 fusion multiplicity ([`directproduct`]). Only nonzero entries (after
/// the `TOL_PURGE` cut) are stored, sorted by `(m1, m2, m3, mu)`.
///
/// Coefficient values realize the SUNRepresentations.jl v0.4.0 gauge (see
/// `docs/gauge.md`); they are a versioned part of this crate's contract.
#[derive(Clone, Debug, PartialEq)]
pub struct Cgc {
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    dims: [usize; 4],
    entries: Vec<CgcEntry>,
}

impl Cgc {
    /// The left factor irrep `s1`.
    pub fn s1(&self) -> &Irrep {
        &self.s1
    }
    /// The right factor irrep `s2`.
    pub fn s2(&self) -> &Irrep {
        &self.s2
    }
    /// The coupled irrep `s3`.
    pub fn s3(&self) -> &Irrep {
        &self.s3
    }
    /// The outer multiplicity `N^{s3}_{s1 s2}` (length of the trailing axis).
    pub fn multiplicity(&self) -> usize {
        self.dims[3]
    }
    /// The tensor shape `[dim(s1), dim(s2), dim(s3), multiplicity]`.
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }
    /// The stored nonzero entries, sorted by `(m1, m2, m3, mu)`.
    pub fn entries(&self) -> &[CgcEntry] {
        &self.entries
    }
    /// The number of stored nonzero entries.
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }
    /// Retained-storage bytes (used by the byte-accounted cache): the entry
    /// vector plus the three irrep weight buffers. Over-counts rather than
    /// under-counts so a cache byte bound stays a true ceiling.
    pub(crate) fn storage_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<CgcEntry>()
            + std::mem::size_of::<Cgc>()
            + (self.s1.rank() + self.s2.rank() + self.s3.rank()) * std::mem::size_of::<i64>()
    }
}

// ---------------------------------------------------------------------------
// GT weight helpers (port of gtpatterns.jl:weight for a pattern).
// ---------------------------------------------------------------------------

/// The physical weight of a GT pattern, an `N`-tuple. Ported from
/// `gtpatterns.jl:weight`: component `l` (1-based) is `rowsum(l) - rowsum(l-1)`
/// with `rowsum(l) = Σ_{k=1..l} m[k, l]` and `rowsum(0) = 0`.
fn pattern_weight(p: &GtPattern, n: usize) -> Vec<i64> {
    let rowsum = |l: usize| -> i64 { (1..=l).map(|k| p.get(k, l)).sum() };
    let mut w = vec![0i64; n];
    for (l, wl) in w.iter_mut().enumerate() {
        *wl = rowsum(l + 1);
    }
    for l in (2..=n).rev() {
        w[l - 1] -= w[l - 2];
    }
    w
}

/// Map from weight tuple to the 0-based basis indices carrying it, over the GT
/// basis of `s`. Ports `clebschgordan.jl:weightmap`.
fn weight_map(patterns: &[GtPattern], n: usize) -> HashMap<Vec<i64>, Vec<usize>> {
    let mut m: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for (i, p) in patterns.iter().enumerate() {
        m.entry(pattern_weight(p, n)).or_default().push(i);
    }
    m
}

/// A GT ladder operator in sparse form: `by_col[c]` lists `(row, value)` of
/// column `c`; `elem[(row, col)]` reads a single matrix element.
struct LadderOp {
    by_col: HashMap<usize, Vec<(usize, f64)>>,
    elem: HashMap<(usize, usize), f64>,
}

fn build_ops(mats: Vec<Vec<LadderEntry>>) -> Vec<LadderOp> {
    mats.into_iter()
        .map(|mat| {
            let mut by_col: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
            let mut elem: HashMap<(usize, usize), f64> = HashMap::new();
            for e in mat {
                let v = e.value.to_f64();
                by_col.entry(e.col).or_default().push((e.row, v));
                elem.insert((e.row, e.col), v);
            }
            LadderOp { by_col, elem }
        })
        .collect()
}

fn is_trivial(s: &Irrep) -> bool {
    s.weight().iter().all(|&x| x == 0)
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Generate the Clebsch-Gordan coefficients for `s1 ⊗ s2 → s3`.
///
/// Returns the sparse m-basis [`Cgc`] tensor with the outer multiplicity on the
/// trailing axis. All multiplicity columns are generated together (they share
/// one nullspace) and gauge-fixed as a block, reproducing the
/// SUNRepresentations.jl v0.4.0 gauge (`docs/gauge.md`).
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if `s1`, `s2`, `s3` are not all SU(N) for one
///   `N`.
/// - [`SunError::NullspaceDimMismatch`] if the SVD rank cut disagrees with the
///   Layer 1 fusion multiplicity (`clebschgordan.jl` `@assert N123 == …`).
/// - [`SunError::NotOrthonormal`] / [`SunError::LadderInconsistent`] if a
///   generation gate fails.
/// - [`SunError::Linalg`] if a dense factorization backend call fails.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "cgc-gen")] {
/// use racah::sun::{cgc, Irrep};
/// // SU(2) 1/2 ⊗ 1/2 → 0 (the singlet), Dynkin (1) ⊗ (1) → (0).
/// let half = Irrep::from_dynkin(&[1]).unwrap();
/// let singlet = Irrep::from_dynkin(&[0]).unwrap();
/// let c = cgc(&half, &half, &singlet).unwrap();
/// assert_eq!(c.multiplicity(), 1);
/// # }
/// ```
pub fn cgc(s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Result<Cgc, SunError> {
    // Transparent, byte-accounted cache (WignerSymbols.jl / SUNRepresentations
    // `_get_CGC` model): a warm hit skips the whole SVD/QR/descent pipeline.
    // The cached value is an Arc<Cgc>; we hand back an owned clone to honor the
    // `-> Cgc` signature. A miss surfaces its error uncached (errors are not
    // stored).
    use std::sync::Arc;
    let key = (s1.clone(), s2.clone(), s3.clone());
    let cache = crate::cache::cache_cgc();
    if let Some(hit) = cache.get(&key) {
        return Ok((*hit).clone());
    }
    let value = Arc::new(generate(s1, s2, s3, None)?);
    let stored = cache.insert(key, value);
    Ok((*stored).clone())
}

/// The pure generation pipeline (no cache). `expected_override` forces the
/// multiplicity-gate target for the `#[cfg(test)]` forced-mismatch scenario;
/// `None` uses the Layer 1 fusion multiplicity.
fn generate(
    s1: &Irrep,
    s2: &Irrep,
    s3: &Irrep,
    expected_override: Option<usize>,
) -> Result<Cgc, SunError> {
    let n = s1.rank();
    if s2.rank() != n || s3.rank() != n {
        return Err(SunError::RankMismatch {
            a: n,
            b: if s2.rank() != n { s2.rank() } else { s3.rank() },
        });
    }
    let d1 = s1.patterns().len();
    let d2 = s2.patterns().len();
    let d3 = s3.patterns().len();
    let expected = expected_override.unwrap_or_else(|| {
        directproduct(s1, s2)
            .map(|p| p.get(s3).copied().unwrap_or(0) as usize)
            .unwrap_or(0)
    });

    // Trivial couplings (clebschgordan.jl:trivial_CGC): 1 ⊗ s → s and s ⊗ 1 → s
    // are identity embeddings, no linear algebra.
    if is_trivial(s1) {
        // Reference `_CGC`: isone(s1) => @assert s2 == s3. Without this the
        // trivial embedding would fabricate a mult-1 identity for a channel
        // whose real multiplicity is 0 (e.g. 1 ⊗ 3 → 3̄), silently skipping the
        // multiplicity gate. `expected` is the true fusion multiplicity (0 here).
        if s3 != s2 {
            return Err(SunError::NullspaceDimMismatch { expected, found: 1 });
        }
        return Ok(trivial_cgc(
            s1.clone(),
            s2.clone(),
            s3.clone(),
            [d1, d2, d3, 1],
            true,
        ));
    }
    if is_trivial(s2) {
        if s3 != s1 {
            return Err(SunError::NullspaceDimMismatch { expected, found: 1 });
        }
        return Ok(trivial_cgc(
            s1.clone(),
            s2.clone(),
            s3.clone(),
            [d1, d2, d3, 1],
            false,
        ));
    }

    let mut data: HashMap<(u32, u32, u32, u32), f64> = HashMap::new();
    let ctx = Ctx::new(s1, s2, s3);
    let n123 = highest_weight_cgc(&mut data, &ctx, expected)?;
    lower_weight_cgc(&mut data, &ctx, n123)?;
    purge(&mut data);

    let cgc = freeze(s1.clone(), s2.clone(), s3.clone(), [d1, d2, d3, n123], data);
    check_orthonormal(&cgc)?;
    check_ladder(&cgc, &ctx)?;
    Ok(cgc)
}

fn trivial_cgc(s1: Irrep, s2: Irrep, s3: Irrep, dims: [usize; 4], is_left: bool) -> Cgc {
    // isleft: 1 ⊗ s → s, CGC[0, m, m, 0] = 1. else s ⊗ 1 → s, CGC[m, 0, m, 0] = 1.
    let d = if is_left { dims[1] } else { dims[0] };
    let entries = (0..d)
        .map(|m| {
            let (m1, m2) = if is_left {
                (0, m as u32)
            } else {
                (m as u32, 0)
            };
            CgcEntry {
                m1,
                m2,
                m3: m as u32,
                mu: 0,
                value: 1.0,
            }
        })
        .collect();
    Cgc {
        s1,
        s2,
        s3,
        dims,
        entries,
    }
}

/// Precomputed per-pair data shared by the highest-weight and descent stages.
struct Ctx {
    n: usize,
    d3: usize,
    pats1: Vec<GtPattern>,
    creation1: Vec<LadderOp>,
    creation2: Vec<LadderOp>,
    annih1: Vec<LadderOp>,
    annih2: Vec<LadderOp>,
    annih3: Vec<LadderOp>,
    map1: HashMap<Vec<i64>, Vec<usize>>,
    map2: HashMap<Vec<i64>, Vec<usize>>,
    map3: HashMap<Vec<i64>, Vec<usize>>,
    w3_top: Vec<i64>,
    wshift: i64,
}

impl Ctx {
    fn new(s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Self {
        let n = s1.rank();
        let pats1 = s1.patterns();
        let pats2 = s2.patterns();
        let pats3 = s3.patterns();
        let sum: fn(&Irrep) -> i64 = |s| s.weight().iter().sum();
        let wshift = (sum(s1) + sum(s2) - sum(s3)) / n as i64;
        Ctx {
            n,
            d3: pats3.len(),
            creation1: build_ops(s1.creation()),
            creation2: build_ops(s2.creation()),
            annih1: build_ops(s1.annihilation()),
            annih2: build_ops(s2.annihilation()),
            annih3: build_ops(s3.annihilation()),
            map1: weight_map(&pats1, n),
            map2: weight_map(&pats2, n),
            map3: weight_map(&pats3, n),
            w3_top: s3.weight().to_vec(),
            pats1,
            wshift,
        }
    }
}

const EMPTY: &[usize] = &[];

fn get_indices<'a>(m: &'a HashMap<Vec<i64>, Vec<usize>>, w: &[i64]) -> &'a [usize] {
    m.get(w).map(Vec::as_slice).unwrap_or(EMPTY)
}

// ---------------------------------------------------------------------------
// Stage 1: highest-weight CGC (clebschgordan.jl:highest_weight_CGC).
// ---------------------------------------------------------------------------

fn highest_weight_cgc(
    data: &mut HashMap<(u32, u32, u32, u32), f64>,
    ctx: &Ctx,
    expected: usize,
) -> Result<usize, SunError> {
    let n = ctx.n;
    // Columns = coupling pairs (m1, m2) at the target (highest) weight of s3.
    let mut cols: Vec<(usize, usize)> = Vec::new();
    let mut col_index: HashMap<(usize, usize), usize> = HashMap::new();
    // Sparse raising-operator equations, keyed by row triple (l, m1', m2').
    let mut acc: Vec<((usize, usize, usize), usize, f64)> = Vec::new();
    let mut rowset: BTreeSet<(usize, usize, usize)> = BTreeSet::new();

    for (m1, pat1) in ctx.pats1.iter().enumerate() {
        let w1 = pattern_weight(pat1, n);
        // w2 = w3_top - w1 + wshift  (componentwise, wshift a scalar broadcast).
        let w2: Vec<i64> = (0..n).map(|c| ctx.w3_top[c] - w1[c] + ctx.wshift).collect();
        for &m2 in get_indices(&ctx.map2, &w2) {
            let col = cols.len();
            cols.push((m1, m2));
            col_index.insert((m1, m2), col);
            for l in 0..n - 1 {
                // Jp1 raises m1 (m2 fixed): column m1 of creation1[l].
                if let Some(entries) = ctx.creation1[l].by_col.get(&m1) {
                    for &(m1p, v) in entries {
                        let key = (l, m1p, m2);
                        rowset.insert(key);
                        acc.push((key, col, v));
                    }
                }
                // Jp2 raises m2 (m1 fixed): column m2 of creation2[l].
                if let Some(entries) = ctx.creation2[l].by_col.get(&m2) {
                    for &(m2p, v) in entries {
                        let key = (l, m1, m2p);
                        rowset.insert(key);
                        acc.push((key, col, v));
                    }
                }
            }
        }
    }

    let ncols = cols.len();
    let rows: Vec<(usize, usize, usize)> = rowset.into_iter().collect();
    let row_index: HashMap<(usize, usize, usize), usize> =
        rows.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut eqs = Mat::zeros(rows.len(), ncols);
    for (rk, col, v) in acc {
        eqs.add(row_index[&rk], col, v);
    }

    let solutions = linalg::nullspace(&eqs, TOL_NULLSPACE)?; // n_cols x N123
    let n123 = solutions.cols;
    if n123 != expected {
        return Err(SunError::NullspaceDimMismatch {
            expected,
            found: n123,
        });
    }

    // Gauge-fix the whole multiplicity block: first(qrpos!(cref!(solutions))).
    let fixed = gaugefix(solutions)?; // n_cols x N123

    // Scatter into CGC at m3 = highest-weight pattern of s3 = last basis index.
    let m3_top = (ctx.d3 - 1) as u32;
    for alpha in 0..n123 {
        for (i, &(m1, m2)) in cols.iter().enumerate() {
            let v = fixed.at(i, alpha);
            if v != 0.0 {
                *data
                    .entry((m1 as u32, m2 as u32, m3_top, alpha as u32))
                    .or_insert(0.0) += v;
            }
        }
    }
    Ok(n123)
}

/// `gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))`.
fn gaugefix(mut solutions: Mat) -> Result<Mat, SunError> {
    cref(&mut solutions, TOL_GAUGE);
    linalg::qr_positive_q(&solutions)
}

/// Column-pivoted reduced echelon form, ported pivot-for-pivot from
/// `clebschgordan.jl:cref!`. Operates in place on the `nr x nc` matrix.
///
/// Per pivot row `i`, the pivot column is the one with the largest
/// `|A[i, j]|` over the not-yet-pinned columns `j..nc`; ties resolve to the
/// leftmost such column (`findabsmax` uses a strict `>` update). A row whose
/// remaining max is `<= eps` is zeroed and skipped. This pivot selection *is*
/// the gauge; see `docs/gauge.md`.
fn cref(a: &mut Mat, eps: f64) {
    let (nr, nc) = (a.rows, a.cols);
    let (mut i, mut j) = (0usize, 0usize);
    while i < nr && j < nc {
        // Pivot column = largest |A[i, ·]| over the free columns j..nc, leftmost
        // on tie. Note (rows 0..nr vs the reference's i..nr in the swap/scale/
        // eliminate below): rows < i of a free column are already zero by the
        // time it is processed (each earlier pivot step zeroes its own row in
        // every other column), so operating on all rows is a no-op there and
        // matches the reference exactly.
        let (mval, off) = findabsmax((j..nc).map(|k| a.at(i, k)));
        let mj = j + off;
        if mval <= eps {
            if eps > 0.0 {
                for k in j..nc {
                    a.data[i + k * nr] = 0.0;
                }
            }
            i += 1;
        } else {
            for k in 0..nr {
                a.data.swap(k + j * nr, k + mj * nr);
            }
            let d = a.at(i, j);
            for k in 0..nr {
                a.data[k + j * nr] /= d;
            }
            for k in 0..nc {
                if k != j {
                    let dk = a.at(i, k);
                    if dk != 0.0 {
                        for l in 0..nr {
                            a.data[l + k * nr] -= dk * a.at(l, j);
                        }
                    }
                }
            }
            i += 1;
            j += 1;
        }
    }
}

/// `(max |v|, offset)` of the first entry of maximal absolute value in `vals`.
///
/// Ties resolve to the **leftmost** entry: the running maximum updates only on
/// a strict `>` (`clebschgordan.jl:findabsmax`). This leftmost rule is part of
/// the gauge specification (`docs/gauge.md §4a`). It is, however, **value-
/// neutral in the final `cref` output** — reduced column echelon form is unique,
/// so a different tie rule cannot change any returned coefficient; the rule is
/// pinned by a unit test at the selection site because no value fixture can
/// catch it.
fn findabsmax(vals: impl Iterator<Item = f64>) -> (f64, usize) {
    let mut m = f64::NEG_INFINITY;
    let mut mi = 0;
    for (k, v) in vals.enumerate() {
        if v.abs() > m {
            m = v.abs();
            mi = k;
        }
    }
    (m, mi)
}

// ---------------------------------------------------------------------------
// Stage 2: lower-weight descent (clebschgordan.jl:lower_weight_CGC!).
// ---------------------------------------------------------------------------

fn lower_weight_cgc(
    data: &mut HashMap<(u32, u32, u32, u32), f64>,
    ctx: &Ctx,
    n123: usize,
) -> Result<(), SunError> {
    let n = ctx.n;
    // Reverse-lexicographic weight order: parents (higher weights) come first
    // and are already solved. Skip the first (the highest weight, done above).
    let mut w3list: Vec<Vec<i64>> = ctx.map3.keys().cloned().collect();
    w3list.sort();
    w3list.reverse();

    for alpha in 0..n123 {
        for w3 in w3list.iter().skip(1) {
            let m3list = &ctx.map3[w3];
            let jmax = m3list.len();

            // Parent rows: one block per level l for the raised weight w3'.
            // imax = Σ_l |map3[w3' (l)]|.
            let mut parent_blocks: Vec<(usize, Vec<usize>)> = Vec::new();
            let mut imax = 0usize;
            for l in 0..n - 1 {
                let w3p = raise(w3, l);
                let list = get_indices(&ctx.map3, &w3p).to_vec();
                imax += list.len();
                parent_blocks.push((l, list));
            }

            let mut eqs = Mat::zeros(imax, jmax);
            // RHS over the unique (m1, m2) columns encountered.
            let mut mask_index: HashMap<(usize, usize), usize> = HashMap::new();
            let mut rhs_acc: Vec<(usize, (usize, usize), f64)> = Vec::new();

            let mut i = 0usize;
            for (l, m3plist) in &parent_blocks {
                let l = *l;
                let w3p = raise(w3, l);
                for &m3p in m3plist {
                    // eqs[i, j] = Jm3[m3, m3']  (annihilation3[l], element (m3, m3')).
                    for (j, &m3) in m3list.iter().enumerate() {
                        if let Some(&v) = ctx.annih3[l].elem.get(&(m3, m3p)) {
                            eqs.add(i, j, v);
                        }
                    }
                    // RHS: (Jm1 ⊗ I + I ⊗ Jm2) acting on parent CGC[m1',m2',m3',α].
                    for (w1p, m1plist) in &ctx.map1 {
                        let w2p: Vec<i64> = (0..n).map(|c| w3p[c] - w1p[c] + ctx.wshift).collect();
                        let m2plist = get_indices(&ctx.map2, &w2p);
                        if m2plist.is_empty() {
                            continue;
                        }
                        for &m2p in m2plist {
                            for &m1p in m1plist {
                                let cgc_coeff = *data
                                    .get(&(m1p as u32, m2p as u32, m3p as u32, alpha as u32))
                                    .unwrap_or(&0.0);
                                if cgc_coeff == 0.0 {
                                    continue;
                                }
                                // Apply Jm1: lower m1' at level l -> m1.
                                let w1 = lower(w1p, l);
                                for &m1 in get_indices(&ctx.map1, &w1) {
                                    if let Some(&jm1) = ctx.annih1[l].elem.get(&(m1, m1p)) {
                                        push_rhs(
                                            &mut mask_index,
                                            &mut rhs_acc,
                                            i,
                                            (m1, m2p),
                                            jm1 * cgc_coeff,
                                        );
                                    }
                                }
                                // Apply Jm2: lower m2' at level l -> m2.
                                let w2 = lower(&w2p, l);
                                for &m2 in get_indices(&ctx.map2, &w2) {
                                    if let Some(&jm2) = ctx.annih2[l].elem.get(&(m2, m2p)) {
                                        push_rhs(
                                            &mut mask_index,
                                            &mut rhs_acc,
                                            i,
                                            (m1p, m2),
                                            jm2 * cgc_coeff,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    i += 1;
                }
            }

            let nmask = mask_index.len();
            if nmask == 0 {
                continue;
            }
            // Ordered mask (column index -> (m1,m2)); mask_index gives the order.
            let mut mask: Vec<(usize, usize)> = vec![(0, 0); nmask];
            for (&pair, &c) in &mask_index {
                mask[c] = pair;
            }
            let mut rhs = Mat::zeros(imax, nmask);
            for (row, pair, val) in rhs_acc {
                rhs.add(row, mask_index[&pair], val);
            }

            // sols = eqs \ rhs  (least squares), shape jmax x nmask.
            let sols = linalg::lstsq(&eqs, &rhs)?;
            for (c, &(m1, m2)) in mask.iter().enumerate() {
                for (j, &m3) in m3list.iter().enumerate() {
                    let v = sols.at(j, c);
                    if v != 0.0 {
                        *data
                            .entry((m1 as u32, m2 as u32, m3 as u32, alpha as u32))
                            .or_insert(0.0) += v;
                    }
                }
            }
        }
    }
    Ok(())
}

fn push_rhs(
    mask_index: &mut HashMap<(usize, usize), usize>,
    rhs_acc: &mut Vec<(usize, (usize, usize), f64)>,
    row: usize,
    pair: (usize, usize),
    val: f64,
) {
    let next = mask_index.len();
    mask_index.entry(pair).or_insert(next);
    rhs_acc.push((row, pair, val));
}

/// Raise weight component `l` by 1 and `l+1` by -1 (0-based `l`). Ports the
/// `Base.setindex(w, w[l]+1, l); setindex(_, w[l+1]-1, l+1)` parent step.
fn raise(w: &[i64], l: usize) -> Vec<i64> {
    let mut v = w.to_vec();
    v[l] += 1;
    v[l + 1] -= 1;
    v
}

/// Lower weight component `l` by 1 and `l+1` by +1 (0-based `l`).
fn lower(w: &[i64], l: usize) -> Vec<i64> {
    let mut v = w.to_vec();
    v[l] -= 1;
    v[l + 1] += 1;
    v
}

// ---------------------------------------------------------------------------
// Stage 3: purge and freeze (clebschgordan.jl:purge!).
// ---------------------------------------------------------------------------

fn purge(data: &mut HashMap<(u32, u32, u32, u32), f64>) {
    data.retain(|_, v| v.abs() > TOL_PURGE);
}

fn freeze(
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    dims: [usize; 4],
    data: HashMap<(u32, u32, u32, u32), f64>,
) -> Cgc {
    let mut entries: Vec<CgcEntry> = data
        .into_iter()
        .map(|((m1, m2, m3, mu), value)| CgcEntry {
            m1,
            m2,
            m3,
            mu,
            value,
        })
        .collect();
    entries.sort_by_key(|e| (e.m1, e.m2, e.m3, e.mu));
    Cgc {
        s1,
        s2,
        s3,
        dims,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Generation gates.
// ---------------------------------------------------------------------------

/// Orthonormality gate: the CGC reshaped as `M[(m1,m2), (m3,α)]` is an
/// isometry, `Σ_{m1,m2} C[..,m3,α] C[..,m3',β] = δ_{m3 m3'} δ_{αβ}`. Overlap is
/// contracted over the coupling indices `(m1,m2)` only, per output column
/// `(m3,α)` -- not summed over `m3`.
fn check_orthonormal(cgc: &Cgc) -> Result<(), SunError> {
    // Columns keyed by (m3, mu); each is a map (m1,m2) -> value.
    let mut columns: BTreeMap<(u32, u32), HashMap<(u32, u32), f64>> = BTreeMap::new();
    for e in &cgc.entries {
        columns
            .entry((e.m3, e.mu))
            .or_default()
            .insert((e.m1, e.m2), e.value);
    }
    let keys: Vec<(u32, u32)> = columns.keys().copied().collect();
    let mut worst = 0.0f64;
    for (ia, ka) in keys.iter().enumerate() {
        let ca = &columns[ka];
        for kb in keys.iter().skip(ia) {
            let cb = &columns[kb];
            // Dot over shared (m1,m2); iterate the smaller column.
            let (small, big) = if ca.len() <= cb.len() {
                (ca, cb)
            } else {
                (cb, ca)
            };
            let mut dot = 0.0;
            for (idx, &v) in small {
                if let Some(&w) = big.get(idx) {
                    dot += v * w;
                }
            }
            let target = if ka == kb { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    if worst > TOL_ORTHO {
        return Err(SunError::NotOrthonormal { residual: worst });
    }
    Ok(())
}

/// Ladder-consistency spot check: the lowering intertwiner at level `l = 0`,
/// evaluated at the highest-weight parent `m3' = d3-1`, for every multiplicity
/// column. Verifies `Σ (Jm1⊗I + I⊗Jm2) C[·,·,m3',α] = Σ_m3 C[·,·,m3,α] Jm3[m3,m3']`
/// element-wise over the coupling states, coupling the highest-weight block to
/// its first descent.
fn check_ladder(cgc: &Cgc, ctx: &Ctx) -> Result<(), SunError> {
    if ctx.n < 2 || cgc.multiplicity() == 0 {
        return Ok(());
    }
    let l = 0usize;
    let m3p = (ctx.d3 - 1) as u32; // highest-weight pattern of s3
    let n123 = cgc.multiplicity();
    // Index CGC for random access.
    let mut idx: HashMap<(u32, u32, u32, u32), f64> = HashMap::new();
    for e in &cgc.entries {
        idx.insert((e.m1, e.m2, e.m3, e.mu), e.value);
    }
    let get = |m1: usize, m2: usize, m3: usize, a: usize| -> f64 {
        idx.get(&(m1 as u32, m2 as u32, m3 as u32, a as u32))
            .copied()
            .unwrap_or(0.0)
    };

    let mut worst = 0.0f64;
    for alpha in 0..n123 {
        // LHS[(m1,m2)] = Σ over lowering of the parent state m3'.
        let mut lhs: HashMap<(usize, usize), f64> = HashMap::new();
        // parent (m1',m2') states carrying nonzero CGC at m3'.
        for e in &cgc.entries {
            if e.m3 != m3p || e.mu as usize != alpha {
                continue;
            }
            let (m1p, m2p) = (e.m1 as usize, e.m2 as usize);
            let coeff = e.value;
            // Jm1 lowers m1'
            if let Some(col) = ctx.annih1[l].by_col.get(&m1p) {
                for &(m1, v) in col {
                    *lhs.entry((m1, m2p)).or_insert(0.0) += v * coeff;
                }
            }
            // Jm2 lowers m2'
            if let Some(col) = ctx.annih2[l].by_col.get(&m2p) {
                for &(m2, v) in col {
                    *lhs.entry((m1p, m2)).or_insert(0.0) += v * coeff;
                }
            }
        }
        // RHS[(m1,m2)] = Σ_m3 C[m1,m2,m3,α] Jm3[m3, m3'].
        let mut rhs: HashMap<(usize, usize), f64> = HashMap::new();
        if let Some(col) = ctx.annih3[l].by_col.get(&(m3p as usize)) {
            for &(m3, jm3) in col {
                // m3 is a child (lower) state; find CGC entries at that m3.
                for e in &cgc.entries {
                    if e.m3 as usize != m3 || e.mu as usize != alpha {
                        continue;
                    }
                    *rhs.entry((e.m1 as usize, e.m2 as usize)).or_insert(0.0) +=
                        get(e.m1 as usize, e.m2 as usize, m3, alpha) * jm3;
                }
            }
        }
        let mut keys: BTreeSet<(usize, usize)> = BTreeSet::new();
        keys.extend(lhs.keys().copied());
        keys.extend(rhs.keys().copied());
        for k in keys {
            let d = lhs.get(&k).copied().unwrap_or(0.0) - rhs.get(&k).copied().unwrap_or(0.0);
            worst = worst.max(d.abs());
        }
    }
    if worst > TOL_LADDER {
        return Err(SunError::LadderInconsistent { residual: worst });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(d: &[i64]) -> Irrep {
        Irrep::from_dynkin(d).unwrap()
    }

    #[test]
    fn forced_multiplicity_mismatch_is_typed_error() {
        // SU(3) 3 ⊗ 3̄ → 8 is multiplicity 1; forcing the gate target to 2 must
        // surface a typed NullspaceDimMismatch (the ported `@assert N123 == …`).
        let e = generate(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1]), Some(2)).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 2,
                found: 1
            }
        );
    }

    #[test]
    fn su3_adjoint_channel_generates_and_caches_equal() {
        let c = cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c.multiplicity(), 1);
        assert!(c.nnz() > 0);
        // Warm hit returns byte-identical values.
        let c2 = cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn su3_octet_squared_has_outer_multiplicity_two() {
        // 8 ⊗ 8 → 8 has N = 2: the multiplicity block is generated and
        // gauge-fixed together, and must come out orthonormal.
        let c = cgc(&irr(&[1, 1]), &irr(&[1, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c.multiplicity(), 2);
        check_orthonormal(&c).unwrap();
    }

    #[test]
    fn rank_mismatch_is_typed_error() {
        let e = generate(&irr(&[1, 0]), &irr(&[1, 0, 0]), &irr(&[1, 0]), None).unwrap_err();
        assert!(matches!(e, SunError::RankMismatch { .. }));
    }

    #[test]
    fn findabsmax_breaks_ties_leftmost() {
        // The cref pivot rule. RCEF is unique, so a wrong tie rule is
        // value-neutral in cref's output and no coefficient fixture can catch
        // it (docs/gauge.md §4a); pin it here at the selection site. Exact
        // tie -> leftmost. The `>`->`>=` mutant returns the rightmost tie and
        // fails all three.
        assert_eq!(findabsmax([2.0, 2.0, 1.0].into_iter()), (2.0, 0));
        assert_eq!(findabsmax([-3.0, 3.0, 3.0].into_iter()), (3.0, 0));
        assert_eq!(findabsmax([1.0, 4.0, 4.0, 2.0].into_iter()), (4.0, 1));
    }

    #[test]
    fn trivial_left_correct_target_is_identity() {
        // 1 ⊗ 3 → 3 : the sole admissible channel, an identity embedding.
        let triv = Irrep::trivial(3).unwrap();
        let c = cgc(&triv, &irr(&[1, 0]), &irr(&[1, 0])).unwrap();
        assert_eq!(c.multiplicity(), 1);
        assert_eq!(c.nnz(), 3); // identity on dim 3
    }

    #[test]
    fn trivial_left_wrong_target_is_typed_error() {
        // 1 ⊗ 3 → 3̄ : 3̄ ∉ 1 ⊗ 3 = {3} (mult 0). Must NOT fabricate an identity
        // (reference `@assert s2 == s3`).
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&triv, &irr(&[1, 0]), &irr(&[0, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }

    #[test]
    fn trivial_left_wrong_dim_target_is_typed_error() {
        // 1 ⊗ 3 → 8 : dims would be nonsense ([1,3,8,1]); reject.
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&triv, &irr(&[1, 0]), &irr(&[1, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }

    #[test]
    fn trivial_right_wrong_target_is_typed_error() {
        // 3 ⊗ 1 → 3̄ : symmetric guard (`@assert s1 == s3`).
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&irr(&[1, 0]), &triv, &irr(&[0, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }
}
