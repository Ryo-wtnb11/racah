//! SU(N) F- and R-symbols derived from Layer 2 Clebsch-Gordan coefficients
//! (Layer 3 of the `cgc-gen` track).
//!
//! Ported from SUNRepresentations.jl v0.4.0
//! (`~/.julia/packages/SUNRepresentations/BM32Z/src/sector.jl`):
//! - [`f_symbol`] ports `_Fsymbol` (`sector.jl:58-89`): the F-symbol as the
//!   contraction of four CGC over all magnetic indices, leaving the four outer
//!   multiplicity indices `[μ, ν, κ, λ]`.
//! - [`r_symbol`] ports `_Rsymbol` (`sector.jl:91-110`): the braiding matrix.
//!
//! The contraction wiring and the `[μ, ν, κ, λ]` axis order match
//! TensorKitSectors `sectors.jl:Fsymbol_from_fusiontensor` (`:406-418`), the
//! `GenericFusion` convention (see `sectors.jl:378-397` for the picture).
//!
//! The pentagon ([`check_pentagon`]) and hexagon ([`check_hexagon`]) gates port
//! `TensorKitSectors/sectors.jl:pentagon_equation` (`:786-819`) and
//! `hexagon_equation` (`:834-871`); [`check_f_unitarity`] is the F-move
//! unitarity gate. All three are shipped as public API: they double as
//! generation gates and as oracle harnesses (README "Self-check functions").
//!
//! # Contraction strategy: sparse key-matching accumulation
//!
//! CGC are sparse maps ([`Cgc::entries`]), and each F/R element is a sum over
//! matched magnetic-index keys. So the contraction is a sequence of sparse
//! joins over the shared magnetic indices — data-structure code, not a dense
//! numeric kernel — which is why it stays here rather than routing a densified
//! GEMM through tenferro (issue #16: "sparse-aware assembly is acceptable and
//! likely optimal"; a dense block would be mostly zeros and pay a
//! densify+GEMM+scatter tax the join avoids).
//!
//! # Conjugation
//!
//! `_Fsymbol`/`_Rsymbol` conjugate two of the CGC (`conj(D)`, `conj(C)` for F;
//! `conj(B)` for R). SUNRepresentations' SU(N) CGC are real `Float64` in the
//! standard gauge (`sectorscalartype = Float64`), so conjugation is the
//! identity and is elided; the port is value-identical.

use std::collections::HashMap;
use std::sync::Arc;

use super::{cgc, directproduct, Irrep, SunError};

/// CGC entries grouped by one magnetic index → `(idx_i, idx_j, mult, value)`.
type GroupBy4 = HashMap<u32, Vec<(u32, u32, u32, f64)>>;

/// A partial-contraction result keyed by the three shared magnetic indices
/// `(ma, mb, mc)` → list of `(mult_p, mult_q, value)`.
type PairGroup = HashMap<(u32, u32, u32), Vec<(u32, u32, f64)>>;

/// Verification-gate tolerances. Not reference constants: sized well above the
/// f64 round-off floor of the sparse contractions (a handful of products and
/// sums of `O(1)` coefficients, `~1e-13`) and far below any structural error
/// (a wrong sign/index is `O(1)`), so a genuine defect trips them while
/// faithful round-off does not. The pentagon/hexagon budgets are looser because
/// they compose several F/R blocks.
const TOL_F_UNITARY: f64 = 1.0e-9;
const TOL_PENTAGON: f64 = 1.0e-8;
const TOL_HEXAGON: f64 = 1.0e-8;

/// A dense F-symbol block `F^{abc}_d[e, f]`, a rank-4 array over the outer
/// multiplicity indices `[μ, ν, κ, λ]` in **row-major** order.
///
/// The axis lengths are `[N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}]`
/// (`μ, ν, κ, λ`), matching the TensorKitSectors `GenericFusion` convention
/// (`sectors.jl:Fsymbol_from_fusiontensor`). For a multiplicity-free family
/// (e.g. SU(2)) every axis is length 1, so the block holds the single scalar.
#[derive(Clone, Debug, PartialEq)]
pub struct FBlock {
    dims: [usize; 4],
    /// Row-major over `[μ, ν, κ, λ]`.
    data: Vec<f64>,
}

impl FBlock {
    fn zeros(dims: [usize; 4]) -> Self {
        FBlock {
            dims,
            data: vec![0.0; dims[0] * dims[1] * dims[2] * dims[3]],
        }
    }

    #[inline]
    fn flat(dims: [usize; 4], mu: usize, nu: usize, kappa: usize, lambda: usize) -> usize {
        ((mu * dims[1] + nu) * dims[2] + kappa) * dims[3] + lambda
    }

    /// The axis lengths `[N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}]` (`μ,ν,κ,λ`).
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    /// The row-major `[μ, ν, κ, λ]` coefficient data.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// The coefficient at multiplicity indices `(μ, ν, κ, λ)`.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of range for [`FBlock::dims`].
    pub fn at(&self, mu: usize, nu: usize, kappa: usize, lambda: usize) -> f64 {
        assert!(
            mu < self.dims[0] && nu < self.dims[1] && kappa < self.dims[2] && lambda < self.dims[3],
            "FBlock index out of range"
        );
        self.data[Self::flat(self.dims, mu, nu, kappa, lambda)]
    }
}

/// A dense R-symbol block `R^{ab}_c`, an `N^c_{ab} × N^c_{ba}` matrix in
/// **row-major** order (rows = `μ`, cols = `ν`).
///
/// `N^c_{ab} = N^c_{ba}` (fusion multiplicities are symmetric), so the matrix
/// is square. For a multiplicity-free family it is the single braiding phase.
#[derive(Clone, Debug, PartialEq)]
pub struct RBlock {
    n: usize,
    /// Row-major `n × n`.
    data: Vec<f64>,
}

impl RBlock {
    fn zeros(n: usize) -> Self {
        RBlock {
            n,
            data: vec![0.0; n * n],
        }
    }

    /// The multiplicity `N^c_{ab}` (both the row and column count).
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The row-major coefficient data (`N × N`, rows `μ`, cols `ν`).
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// The braiding coefficient at `(μ, ν)`.
    ///
    /// # Panics
    ///
    /// Panics if `mu` or `nu` is `>= dim()`.
    pub fn at(&self, mu: usize, nu: usize) -> f64 {
        assert!(mu < self.n && nu < self.n, "RBlock index out of range");
        self.data[mu * self.n + nu]
    }
}

// ---------------------------------------------------------------------------
// Multiplicity / rank helpers (the reference's compile-time `SUNIrrep{N}` and
// `Nsymbol` guards, re-erected as runtime typed errors -- issue #15).
// ---------------------------------------------------------------------------

/// `N^c_{ab}`, the fusion multiplicity of `a ⊗ b → c`. Errors
/// [`SunError::RankMismatch`] if `a`, `b`, `c` are not all SU(N) for one `N`
/// (the reference relies on the `SUNIrrep{N}` type parameter for this).
fn mult(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, SunError> {
    if c.rank() != a.rank() {
        return Err(SunError::RankMismatch {
            a: a.rank(),
            b: c.rank(),
        });
    }
    Ok(directproduct(a, b)?.get(c).copied().unwrap_or(0) as usize)
}

/// All six labels of an F request share one rank, or [`SunError::RankMismatch`].
fn require_same_rank(labels: &[&Irrep]) -> Result<(), SunError> {
    let n = labels[0].rank();
    for s in &labels[1..] {
        if s.rank() != n {
            return Err(SunError::RankMismatch { a: n, b: s.rank() });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// F-symbol (sector.jl:_Fsymbol, :58-89).
// ---------------------------------------------------------------------------

/// The F-symbol `F^{abc}_d[e, f]` as a dense `[μ, ν, κ, λ]` block.
///
/// Ports `sector.jl:_Fsymbol`. With `A = CGC(a,b,e)`, `B = CGC(e,c,d)`,
/// `C = CGC(b,c,f)`, `D = CGC(a,f,d)` (the last two conjugated; real, so
/// elided) the reference `@tensor` (line 85) is
///
/// ```text
/// F[μ,ν,κ,λ] = Σ  conj(D[ma,mf,λ]) conj(C[mb,mc,mf,κ]) A[ma,mb,me,μ] B[me,mc,ν]
///            ma,mb,me,mc,mf
/// ```
///
/// where `B`/`D` are the `_Fsymbol` slices at the first `d`-basis index
/// (`[:, :, 1, :]`, Julia 1-based → 0-based `m_d = 0`): the F-symbol is
/// independent of which `d`-state is fixed (the reference comment "computing
/// first diagonal element"), so the first is taken.
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if the six labels are not all SU(N) for one `N`.
/// - [`SunError::ZeroFusionChannel`] if any of the four vertices `a⊗b→e`,
///   `e⊗c→d`, `b⊗c→f`, `a⊗f→d` is empty. (The reference returns an all-zero
///   block here; this query API returns a typed error — issue #15.)
/// - [`SunError::NullspaceDimMismatch`] / [`SunError::NotOrthonormal`] /
///   [`SunError::LadderInconsistent`] / [`SunError::Linalg`] surfaced from an
///   underlying CGC generation.
pub fn f_symbol(
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
    e: &Irrep,
    f: &Irrep,
) -> Result<FBlock, SunError> {
    require_same_rank(&[a, b, c, d, e, f])?;
    // Guard: every vertex must be non-empty (the reference's
    // `Nsymbol(...) == 0 && return zeros` short-circuit becomes a typed error).
    let vertices = [
        (a, b, e), // N1 = μ
        (e, c, d), // N2 = ν
        (b, c, f), // N3 = κ
        (a, f, d), // N4 = λ
    ];
    for (x, y, z) in vertices {
        if mult(x, y, z)? == 0 {
            return Err(SunError::ZeroFusionChannel {
                a: x.dynkin(),
                b: y.dynkin(),
                c: z.dynkin(),
            });
        }
    }

    // Derived-f64 cache: key = the plain ordered six-label tuple.
    //
    // Why no Regge-style canonicalization: the 6j symmetry group that lets the
    // SU(2) F cache key on a canonical class (`su2.rs:FKey`) has no analogue for
    // GT-basis SU(N) F blocks — there is no implemented tetrahedral/Regge
    // canonical form for a rank-4 multiplicity block in this gauge, and the
    // block is not invariant under the naive label permutations anyway. So the
    // key is the six labels as given; distinct requests never collide, and no
    // canonicalization can silently merge two genuinely different blocks.
    let cache = crate::cache::cache_sun_f();
    let key = (
        a.clone(),
        b.clone(),
        c.clone(),
        d.clone(),
        e.clone(),
        f.clone(),
    );
    if let Some(hit) = cache.get(&key) {
        return Ok((*hit).clone());
    }
    let block = f_block_raw(a, b, c, d, e, f)?;
    let stored = cache.insert(key, Arc::new(block));
    Ok((*stored).clone())
}

/// The raw contraction with **reference** empty-vertex semantics (an all-zero
/// block, not an error). Used by the gates, which — like the reference
/// pentagon/hexagon loops — feed empty blocks through harmlessly. `f_symbol`
/// wraps this after converting empty vertices to a typed error.
fn f_block_raw(
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
    e: &Irrep,
    f: &Irrep,
) -> Result<FBlock, SunError> {
    let n1 = mult(a, b, e)?;
    let n2 = mult(e, c, d)?;
    let n3 = mult(b, c, f)?;
    let n4 = mult(a, f, d)?;
    let dims = [n1, n2, n3, n4];
    if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
        return Ok(FBlock::zeros(dims));
    }

    let cab = cgc(a, b, e)?; // A[ma,mb,me,μ]
    let cecd = cgc(e, c, d)?; // B slice at m_d=0: [me,mc,ν]
    let cbcf = cgc(b, c, f)?; // C[mb,mc,mf,κ]
    let cafd = cgc(a, f, d)?; // D slice at m_d=0: [ma,mf,λ]

    // Step 1: AB[(ma,mb,mc), (μ,ν)] = Σ_me A[ma,mb,me,μ] · B[me,mc,ν].
    // Group A and B by the shared magnetic index me, then cross each group.
    let mut a_by_me: GroupBy4 = HashMap::new();
    for x in cab.entries() {
        a_by_me
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (ma, mb, μ, vA)
    }
    // B is CGC(e,c,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut b_by_me: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in cecd.entries() {
        if x.m3 == 0 {
            b_by_me.entry(x.m1).or_default().push((x.m2, x.mu, x.value)); // (mc, ν, vB)
        }
    }
    // AB keyed by (ma,mb,mc) -> Vec<(μ, ν, value)>.
    let mut ab: PairGroup = HashMap::new();
    for (me, alist) in &a_by_me {
        let Some(blist) = b_by_me.get(me) else {
            continue;
        };
        for &(ma, mb, mu, va) in alist {
            for &(mc, nu, vb) in blist {
                ab.entry((ma, mb, mc)).or_default().push((mu, nu, va * vb));
            }
        }
    }

    // Step 2: CD[(ma,mb,mc), (κ,λ)] = Σ_mf C[mb,mc,mf,κ] · D[ma,mf,λ].
    let mut c_by_mf: GroupBy4 = HashMap::new();
    for x in cbcf.entries() {
        c_by_mf
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (mb, mc, κ, vC)
    }
    // D is CGC(a,f,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut d_by_mf: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in cafd.entries() {
        if x.m3 == 0 {
            d_by_mf.entry(x.m2).or_default().push((x.m1, x.mu, x.value)); // (ma, λ, vD)
        }
    }
    let mut cd: PairGroup = HashMap::new();
    for (mf, clist) in &c_by_mf {
        let Some(dlist) = d_by_mf.get(mf) else {
            continue;
        };
        for &(mb, mc, kappa, vc) in clist {
            for &(ma, lambda, vd) in dlist {
                cd.entry((ma, mb, mc))
                    .or_default()
                    .push((kappa, lambda, vc * vd));
            }
        }
    }

    // Step 3: F[μ,ν,κ,λ] = Σ_{ma,mb,mc} AB[(ma,mb,mc),(μ,ν)] · CD[(ma,mb,mc),(κ,λ)].
    let mut block = FBlock::zeros(dims);
    for (key, ablist) in &ab {
        let Some(cdlist) = cd.get(key) else {
            continue;
        };
        for &(mu, nu, vab) in ablist {
            for &(kappa, lambda, vcd) in cdlist {
                let idx = FBlock::flat(
                    dims,
                    mu as usize,
                    nu as usize,
                    kappa as usize,
                    lambda as usize,
                );
                block.data[idx] += vab * vcd;
            }
        }
    }
    Ok(block)
}

// ---------------------------------------------------------------------------
// R-symbol (sector.jl:_Rsymbol, :91-110).
// ---------------------------------------------------------------------------

/// The R-symbol `R^{ab}_c` as a dense `N^c_{ab} × N^c_{ba}` matrix.
///
/// Ports `sector.jl:_Rsymbol`. With `A = CGC(a,b,c)`, `B = CGC(b,a,c)` sliced at
/// `m_c = 0` (`[:, :, 1, :]`), the reference `@tensor R[μ; ν] :=
/// conj(B[mb,ma,ν]) A[ma,mb,μ]` (line 108) is
///
/// ```text
/// R[μ, ν] = Σ  A[ma, mb, μ] · B[mb, ma, ν]      (B conjugated; real, elided)
///          ma,mb
/// ```
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if `a`, `b`, `c` are not all SU(N) for one `N`.
/// - [`SunError::ZeroFusionChannel`] if `a ⊗ b → c` is empty.
/// - CGC generation errors are surfaced.
pub fn r_symbol(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<RBlock, SunError> {
    require_same_rank(&[a, b, c])?;
    if mult(a, b, c)? == 0 {
        return Err(SunError::ZeroFusionChannel {
            a: a.dynkin(),
            b: b.dynkin(),
            c: c.dynkin(),
        });
    }
    r_block_raw(a, b, c)
}

/// R contraction with reference empty-vertex semantics (zeros, not an error);
/// the gates feed empty blocks through harmlessly.
fn r_block_raw(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<RBlock, SunError> {
    let n1 = mult(a, b, c)?; // rows μ
    let n2 = mult(b, a, c)?; // cols ν  (== n1: fusion multiplicities are symmetric)
    if n1 == 0 || n2 == 0 {
        return Ok(RBlock::zeros(n1.max(n2)));
    }
    debug_assert_eq!(n1, n2, "N^c_ab == N^c_ba");

    let cab = cgc(a, b, c)?; // A[ma,mb,mc,μ]
    let cba = cgc(b, a, c)?; // B[mb,ma,mc,ν]

    // A slice at m_c = 0, keyed by (ma, mb) -> Vec<(μ, value)>.
    let mut a_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in cab.entries() {
        if x.m3 == 0 {
            a_map.entry((x.m1, x.m2)).or_default().push((x.mu, x.value));
        }
    }
    // B slice at m_c = 0, keyed by (ma, mb) = (B.m2, B.m1) -> Vec<(ν, value)>.
    let mut b_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in cba.entries() {
        if x.m3 == 0 {
            b_map.entry((x.m2, x.m1)).or_default().push((x.mu, x.value));
        }
    }

    let mut block = RBlock::zeros(n1);
    for (key, alist) in &a_map {
        let Some(blist) = b_map.get(key) else {
            continue;
        };
        for &(mu, va) in alist {
            for &(nu, vb) in blist {
                block.data[mu as usize * n1 + nu as usize] += va * vb;
            }
        }
    }
    Ok(block)
}

// ---------------------------------------------------------------------------
// Fusion-set helpers for the gates.
// ---------------------------------------------------------------------------

/// The irreps in `a ⊗ b` (fusion outputs), sorted deterministically.
fn products(a: &Irrep, b: &Irrep) -> Result<Vec<Irrep>, SunError> {
    Ok(directproduct(a, b)?.into_keys().collect())
}

/// `a ⊗ b ∩ c ⊗ d` (the pentagon/hexagon `intersect(⊗(...), ⊗(...))`).
fn intersect_products(a: &Irrep, b: &Irrep, c: &Irrep, d: &Irrep) -> Result<Vec<Irrep>, SunError> {
    let left = directproduct(a, b)?;
    let right = directproduct(c, d)?;
    Ok(left.into_keys().filter(|k| right.contains_key(k)).collect())
}

/// Per-gate-call memo for F/R blocks.
///
/// A pentagon/hexagon gate references the *same* F/R block from many index
/// combinations; without memoization each reference recomputes a four-CGC
/// contraction, which for OM≥2 SU(3) families (large intermediate irreps like
/// `27 ⊗ 8`) is the difference between seconds and many minutes. The blocks are
/// tiny (≤ `2⁴` f64), so the memo clones them out cheaply. It is *not* the
/// process-global `cache_sun_f`: gates use the raw zeros-for-`N=0` semantics,
/// which would pollute the public tier with blocks `f_symbol` never stores.
#[derive(Default)]
struct BlockMemo {
    f: HashMap<[Irrep; 6], FBlock>,
    r: HashMap<[Irrep; 3], RBlock>,
}

impl BlockMemo {
    fn f_block(
        &mut self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
        d: &Irrep,
        e: &Irrep,
        f: &Irrep,
    ) -> Result<FBlock, SunError> {
        let key = [
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            e.clone(),
            f.clone(),
        ];
        if let Some(bl) = self.f.get(&key) {
            return Ok(bl.clone());
        }
        let bl = f_block_raw(a, b, c, d, e, f)?;
        self.f.insert(key, bl.clone());
        Ok(bl)
    }

    fn r_block(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<RBlock, SunError> {
        let key = [a.clone(), b.clone(), c.clone()];
        if let Some(bl) = self.r.get(&key) {
            return Ok(bl.clone());
        }
        let bl = r_block_raw(a, b, c)?;
        self.r.insert(key, bl.clone());
        Ok(bl)
    }
}

// ---------------------------------------------------------------------------
// Gate 1: F-move unitarity.
// ---------------------------------------------------------------------------

/// Verify that the F-move for fixed outer labels `(a, b, c, d)` is unitary.
///
/// For fixed `a, b, c, d`, the F-symbols form a square matrix `M` with rows
/// indexed by `(e, μ, ν)` (`e ∈ a⊗b`, `μ ∈ [0,N^e_{ab})`, `ν ∈ [0,N^d_{ec})`)
/// and columns by `(f, κ, λ)` (`f ∈ b⊗c`, `κ ∈ [0,N^f_{bc})`,
/// `λ ∈ [0,N^d_{af})`), where `M[(e,μ,ν),(f,κ,λ)] = F^{abc}_d[e,f][μ,ν,κ,λ]`.
/// The two associativity bases of `a⊗b⊗c → d` are orthonormal, so `M` is
/// real-orthogonal: `M Mᵀ = I`. Both index sets have the same size
/// (`Σ_e N^e_{ab} N^d_{ec} = Σ_f N^f_{bc} N^d_{af}`), so `M` is square.
///
/// # Errors
///
/// [`SunError::FNotUnitary`] with the worst `|(M Mᵀ - I)_{ij}|` if the gate
/// fails; [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_f_unitarity(a: &Irrep, b: &Irrep, c: &Irrep, d: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c, d])?;

    // Rows: (e, μ, ν). Columns: (f, κ, λ).
    let mut rows: Vec<(Irrep, usize, usize)> = Vec::new();
    for e in products(a, b)? {
        let n_ab_e = mult(a, b, &e)?;
        let n_ec_d = mult(&e, c, d)?;
        for mu in 0..n_ab_e {
            for nu in 0..n_ec_d {
                rows.push((e.clone(), mu, nu));
            }
        }
    }
    let mut cols: Vec<(Irrep, usize, usize)> = Vec::new();
    for f in products(b, c)? {
        let n_bc_f = mult(b, c, &f)?;
        let n_af_d = mult(a, &f, d)?;
        for kappa in 0..n_bc_f {
            for lambda in 0..n_af_d {
                cols.push((f.clone(), kappa, lambda));
            }
        }
    }

    // M[row, col].
    let nr = rows.len();
    let nc = cols.len();
    let mut m = vec![0.0f64; nr * nc];
    let mut memo = BlockMemo::default();
    for (ri, (e, mu, nu)) in rows.iter().enumerate() {
        for (ci, (f, kappa, lambda)) in cols.iter().enumerate() {
            let block = memo.f_block(a, b, c, d, e, f)?;
            m[ri * nc + ci] = block.at(*mu, *nu, *kappa, *lambda);
        }
    }

    // worst |(M Mᵀ - I)_{ij}|.
    let mut worst = 0.0f64;
    for i in 0..nr {
        for j in 0..nr {
            let mut dot = 0.0;
            for k in 0..nc {
                dot += m[i * nc + k] * m[j * nc + k];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    if worst > TOL_F_UNITARY {
        return Err(SunError::FNotUnitary { residual: worst });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate 2: pentagon (TensorKitSectors sectors.jl:pentagon_equation, :786-819).
// ---------------------------------------------------------------------------

/// Verify the pentagon identity for the quadruple `(a, b, c, d)`.
///
/// Ports `TensorKitSectors/sectors.jl:pentagon_equation` (GenericFusion branch).
/// For every `f ∈ a⊗b`, `h ∈ c⊗d`, `g ∈ f⊗c`, `i ∈ b⊗h`,
/// `e ∈ (g⊗d) ∩ (a⊗i)`:
///
/// ```text
/// p1[λμν κρσ] = Σ_τ F(f,c,d,e,g,h)[λ,μ,ν,τ] · F(a,b,h,e,f,i)[κ,τ,ρ,σ]
/// p2[λμν κρσ] = Σ_{j∈b⊗c, α,β,τ}
///                 F(a,b,c,g,f,j)[κ,λ,α,β] · F(a,j,d,e,g,i)[β,μ,τ,σ]
///                 · F(b,c,d,i,j,h)[α,τ,ν,ρ]
/// ```
///
/// and requires `p1 ≈ p2`.
///
/// # Errors
///
/// [`SunError::PentagonViolation`] (worst residual) on failure;
/// [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_pentagon(a: &Irrep, b: &Irrep, c: &Irrep, d: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c, d])?;
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::default();

    for f in products(a, b)? {
        for h in products(c, d)? {
            for g in products(&f, c)? {
                for i in products(b, &h)? {
                    for e in intersect_products(&g, d, a, &i)? {
                        // Free-index dims: λ=N_fcg, μ=N_gde, ν=N_cdh, κ=N_abf,
                        // ρ=N_bhi, σ=N_aie.
                        let n_lambda = mult(&f, c, &g)?;
                        let n_mu = mult(&g, d, &e)?;
                        let n_nu = mult(c, d, &h)?;
                        let n_kappa = mult(a, b, &f)?;
                        let n_rho = mult(b, &h, &i)?;
                        let n_sigma = mult(a, &i, &e)?;
                        if [n_lambda, n_mu, n_nu, n_kappa, n_rho, n_sigma].contains(&0) {
                            continue; // empty output family -> vacuous
                        }

                        // p1: F1[λ,μ,ν,τ] · F2[κ,τ,ρ,σ], sum over τ (= N_fhe).
                        let f1 = memo.f_block(&f, c, d, &e, &g, &h)?; // [λ,μ,ν,τ]
                        let f2 = memo.f_block(a, b, &h, &e, &f, &i)?; // [κ,τ,ρ,σ]
                        let n_tau = f1.dims()[3];

                        // p2 factors, summed over j ∈ b⊗c and α,β,τ'.
                        let mut p2_terms: Vec<(FBlock, FBlock, FBlock)> = Vec::new();
                        for j in products(b, c)? {
                            let g1 = memo.f_block(a, b, c, &g, &f, &j)?; // [κ,λ,α,β]
                            let g2 = memo.f_block(a, &j, d, &e, &g, &i)?; // [β,μ,τ',σ]
                            let g3 = memo.f_block(b, c, d, &i, &j, &h)?; // [α,τ',ν,ρ]
                            p2_terms.push((g1, g2, g3));
                        }

                        for lambda in 0..n_lambda {
                            for mu in 0..n_mu {
                                for nu in 0..n_nu {
                                    for kappa in 0..n_kappa {
                                        for rho in 0..n_rho {
                                            for sigma in 0..n_sigma {
                                                let mut p1 = 0.0;
                                                for tau in 0..n_tau {
                                                    p1 += f1.at(lambda, mu, nu, tau)
                                                        * f2.at(kappa, tau, rho, sigma);
                                                }
                                                let mut p2 = 0.0;
                                                for (g1, g2, g3) in &p2_terms {
                                                    // dims: α=g1[2], β=g1[3], τ'=g2[2]
                                                    let n_alpha = g1.dims()[2];
                                                    let n_beta = g1.dims()[3];
                                                    let n_taup = g2.dims()[2];
                                                    for alpha in 0..n_alpha {
                                                        for beta in 0..n_beta {
                                                            for taup in 0..n_taup {
                                                                p2 += g1
                                                                    .at(kappa, lambda, alpha, beta)
                                                                    * g2.at(beta, mu, taup, sigma)
                                                                    * g3.at(alpha, taup, nu, rho);
                                                            }
                                                        }
                                                    }
                                                }
                                                worst = worst.max((p1 - p2).abs());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if worst > TOL_PENTAGON {
        return Err(SunError::PentagonViolation { residual: worst });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate 3: hexagon (TensorKitSectors sectors.jl:hexagon_equation, :834-871).
// ---------------------------------------------------------------------------

/// Verify both hexagon identities for the triple `(a, b, c)`.
///
/// Ports `TensorKitSectors/sectors.jl:hexagon_equation` (GenericFusion branch).
/// For every `e ∈ c⊗a`, `f ∈ c⊗b`, `d ∈ (e⊗b) ∩ (a⊗f)`, with
/// `F ≡ F(a,c,b,d,e,f)[λ,β,γ,ν]`:
///
/// ```text
/// RFR1[α,β,μ,ν] = Σ_{λ,γ} R(c,a,e)[α,λ] · F[λ,β,γ,ν] · R(c,b,f)[γ,μ]
/// RFR2[α,β,μ,ν] = Σ_{λ,γ} R(a,c,e)[α,λ] · F[λ,β,γ,ν] · R(b,c,f)[γ,μ]   (conj; real)
/// FRF1[α,β,μ,ν] = Σ_{g∈a⊗b, δ,σ,ψ}
///                   F(c,a,b,d,e,g)[α,β,δ,σ] · R(c,g,d)[σ,ψ] · F(a,b,c,d,g,f)[δ,ψ,μ,ν]
/// FRF2[α,β,μ,ν] = Σ ... R(g,c,d)[σ,ψ] ...   (conj; real)
/// ```
///
/// and requires `RFR1 ≈ FRF1` and `RFR2 ≈ FRF2`. R is real for SU(N), so the
/// two hexagons differ only in which R replaces which (`conj` is the identity).
///
/// # Errors
///
/// [`SunError::HexagonViolation`] (worst residual) on failure;
/// [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_hexagon(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c])?;
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::default();

    for e in products(c, a)? {
        let rcae = memo.r_block(c, a, &e)?; // [α,λ]
        let race = memo.r_block(a, c, &e)?; // [α,λ]
        for f in products(c, b)? {
            let rcbf = memo.r_block(c, b, &f)?; // [γ,μ]
            let rbcf = memo.r_block(b, c, &f)?; // [γ,μ]
            for d in intersect_products(&e, b, a, &f)? {
                // free dims: α=N_cae, β=N_ebd, μ=N_bcf, ν=N_afd.
                let n_alpha = mult(c, a, &e)?;
                let n_beta = mult(&e, b, &d)?;
                let n_mu = mult(b, c, &f)?;
                let n_nu = mult(a, &f, &d)?;
                if [n_alpha, n_beta, n_mu, n_nu].contains(&0) {
                    continue;
                }
                let facb = memo.f_block(a, c, b, &d, &e, &f)?; // [λ,β,γ,ν]
                let n_lam = facb.dims()[0]; // N_ace = N_cae
                let n_gam = facb.dims()[2]; // N_cbf

                // FRF factors over g ∈ a⊗b.
                let mut frf_terms: Vec<(FBlock, RBlock, RBlock, FBlock)> = Vec::new();
                for g in products(a, b)? {
                    let rcgd = memo.r_block(c, &g, &d)?;
                    let rgcd = memo.r_block(&g, c, &d)?;
                    let fcab = memo.f_block(c, a, b, &d, &e, &g)?; // [α,β,δ,σ]
                    let fabc = memo.f_block(a, b, c, &d, &g, &f)?; // [δ,ψ,μ,ν]
                    frf_terms.push((fcab, rcgd, rgcd, fabc));
                }

                for alpha in 0..n_alpha {
                    for beta in 0..n_beta {
                        for mu in 0..n_mu {
                            for nu in 0..n_nu {
                                // RFR1 / RFR2.
                                let mut rfr1 = 0.0;
                                let mut rfr2 = 0.0;
                                for lam in 0..n_lam {
                                    for gam in 0..n_gam {
                                        let fv = facb.at(lam, beta, gam, nu);
                                        rfr1 += rcae.at(alpha, lam) * fv * rcbf.at(gam, mu);
                                        rfr2 += race.at(alpha, lam) * fv * rbcf.at(gam, mu);
                                    }
                                }
                                // FRF1 / FRF2.
                                let mut frf1 = 0.0;
                                let mut frf2 = 0.0;
                                for (fcab, rcgd, rgcd, fabc) in &frf_terms {
                                    let n_delta = fcab.dims()[2]; // N_abg
                                    let n_sigma = fcab.dims()[3]; // N_cgd
                                    let n_psi = rcgd.dim(); // N_gcd (= N_cgd)
                                    for delta in 0..n_delta {
                                        for sigma in 0..n_sigma {
                                            let fc = fcab.at(alpha, beta, delta, sigma);
                                            for psi in 0..n_psi {
                                                let fa = fabc.at(delta, psi, mu, nu);
                                                frf1 += fc * rcgd.at(sigma, psi) * fa;
                                                frf2 += fc * rgcd.at(sigma, psi) * fa;
                                            }
                                        }
                                    }
                                }
                                worst = worst.max((rfr1 - frf1).abs());
                                worst = worst.max((rfr2 - frf2).abs());
                            }
                        }
                    }
                }
            }
        }
    }

    if worst > TOL_HEXAGON {
        return Err(SunError::HexagonViolation { residual: worst });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(d: &[i64]) -> Irrep {
        Irrep::from_dynkin(d).unwrap()
    }

    // ---- guard inventory: red-first ill-posed inputs ----

    #[test]
    fn f_symbol_zero_vertex_is_typed_error() {
        // SU(3): a⊗b→e with e ∉ a⊗b. 3⊗3 = 6 ⊕ 3̄, so e = 8 is empty.
        let three = irr(&[1, 0]);
        let eight = irr(&[1, 1]);
        let err = f_symbol(&three, &three, &three, &three, &eight, &three).unwrap_err();
        assert!(matches!(err, SunError::ZeroFusionChannel { .. }));
    }

    #[test]
    fn f_symbol_rank_mismatch_is_typed_error() {
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        let err = f_symbol(&su3, &su3, &su3, &su3, &su4, &su3).unwrap_err();
        assert!(matches!(err, SunError::RankMismatch { .. }));
    }

    #[test]
    fn r_symbol_zero_vertex_is_typed_error() {
        // 3 ⊗ 3 → 8 is empty (3⊗3 = 6 ⊕ 3̄).
        let three = irr(&[1, 0]);
        let eight = irr(&[1, 1]);
        let err = r_symbol(&three, &three, &eight).unwrap_err();
        assert!(matches!(err, SunError::ZeroFusionChannel { .. }));
    }

    #[test]
    fn r_symbol_rank_mismatch_is_typed_error() {
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        let err = r_symbol(&su3, &su4, &su3).unwrap_err();
        assert!(matches!(err, SunError::RankMismatch { .. }));
    }

    // ---- shapes ----

    #[test]
    fn su3_trivial_f_is_scalar_one() {
        // F with a = trivial: F^{1,b,c}_d[e=b, f=c] should be the identity
        // scalar (1×1×1×1 block, value 1) for admissible b,c.
        // a = 1: e = b = 3 forced, f = d ∈ 3⊗3 (take 6), so all vertices hold.
        let triv = Irrep::trivial(3).unwrap();
        let three = irr(&[1, 0]);
        let six = irr(&[2, 0]);
        let block = f_symbol(&triv, &three, &three, &six, &three, &six).unwrap();
        assert_eq!(block.dims(), [1, 1, 1, 1]);
        assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn su3_octet_cubed_f_block_is_2x2x2x2() {
        // 8⊗8 → 8 has multiplicity 2, so F(8,8,8,8,8,8) is a 2×2×2×2 block.
        let eight = irr(&[1, 1]);
        let block = f_symbol(&eight, &eight, &eight, &eight, &eight, &eight).unwrap();
        assert_eq!(block.dims(), [2, 2, 2, 2]);
    }

    #[test]
    fn su3_octet_r_block_is_2x2() {
        // 8⊗8 → 8 : R is a 2×2 braiding matrix.
        let eight = irr(&[1, 1]);
        let block = r_symbol(&eight, &eight, &eight).unwrap();
        assert_eq!(block.dim(), 2);
    }

    // ---- gates on small SU(3) families (self-consistency) ----

    #[test]
    fn su3_f_unitarity_multiplicity_free() {
        // 3⊗3⊗3 → various d: multiplicity-free F-move must be orthogonal.
        let three = irr(&[1, 0]);
        check_f_unitarity(&three, &three, &three, &irr(&[1, 0])).unwrap();
    }

    #[test]
    fn su3_f_unitarity_with_multiplicity() {
        // 8⊗8⊗8 → 8: the F-move mixes multiplicity indices; still orthogonal.
        let eight = irr(&[1, 1]);
        check_f_unitarity(&eight, &eight, &eight, &eight).unwrap();
    }

    #[test]
    fn su3_pentagon_multiplicity_free() {
        let three = irr(&[1, 0]);
        check_pentagon(&three, &three, &three, &three).unwrap();
    }

    #[test]
    fn su3_hexagon_multiplicity_free() {
        let three = irr(&[1, 0]);
        check_hexagon(&three, &three, &three).unwrap();
    }
}
