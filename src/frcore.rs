//! Family-generic F- and R-symbol core: the four-CGC contraction and the
//! pentagon/hexagon/F-unitarity gates, shared by the SU(N) (`crate::sun::fr`)
//! and B/C/D (`crate::bcd::fr`) surfaces.
//!
//! # Why one core, two callers
//!
//! The F-symbol is the contraction of four Clebsch–Gordan tensors over their
//! magnetic indices, leaving four outer multiplicity indices `[μ, ν, κ, λ]`
//! (SUNRepresentations.jl `sector.jl:_Fsymbol`, `:58-89`; the contraction wiring
//! and axis order match TensorKitSectors `sectors.jl:Fsymbol_from_fusiontensor`,
//! `:406-418`). The PR #17 review established that this logic depends on nothing
//! SU(N)-specific: only on an irrep label type, a fusion-multiplicity source
//! `N^c_{ab}`, a CGC provider returning **sparse real** `(m1, m2, m3, mu, value)`
//! entries, and `f64` scalars. Real-valuedness holds for B/C/D CGC too (the sweep
//! produces real orthogonal isometries), so conjugation is the identity and is
//! elided on both surfaces — value-identical to the reference. This module is
//! that logic, parameterized by the [`Family`] seam.
//!
//! # The `&mut self` seam (why it does not force interior mutability on SU(N))
//!
//! [`Family`] takes `&mut self`. That is the superset both callers satisfy: the
//! B/C/D provider is a `&mut CanonicalCatalog` (append-only generator
//! materialization is genuinely mutating), while the SU(N) provider is a
//! stateless zero-sized type whose methods delegate to the process-global CGC
//! cache via free functions and simply ignore the `&mut`. A `&mut` to a local ZST
//! is always available, so SU(N) pays no lock and grows no interior-mutability
//! cell — the mutability is real for one impl and vacuous for the other.
//!
//! # Contraction strategy: sparse key-matching accumulation
//!
//! CGC are sparse, and each F/R element is a sum over matched magnetic-index
//! keys, so the contraction is a sequence of sparse joins over shared magnetic
//! indices — data-structure code, not a dense numeric kernel (issue #16:
//! "sparse-aware assembly is acceptable and likely optimal"; a densified GEMM
//! would be mostly zeros and pay a densify+GEMM+scatter tax the join avoids).

use std::collections::HashMap;
use std::hash::Hash;

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
/// they compose several F/R blocks. Shared by both families: the B/C/D CGC come
/// from an SVD sweep and carry the same `f64` round-off character as the SU(N)
/// SVD/QR pipeline, so the same sizing rationale holds.
pub(crate) const TOL_F_UNITARY: f64 = 1.0e-9;
pub(crate) const TOL_PENTAGON: f64 = 1.0e-8;
pub(crate) const TOL_HEXAGON: f64 = 1.0e-8;

/// One nonzero entry of a sparse m-basis CGC tensor: `value` at magnetic indices
/// `(m1, m2, m3)` and outer-multiplicity `mu`. The family-neutral form the core
/// consumes; each [`Family`] adapts its native CGC representation to this.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MEntry {
    /// 0-based magnetic index in the left factor.
    pub m1: u32,
    /// 0-based magnetic index in the right factor.
    pub m2: u32,
    /// 0-based magnetic index in the coupled irrep.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The (real, standard-gauge) coefficient.
    pub value: f64,
}

/// The seam between the generic core and a representation family (SU(N) or
/// B/C/D). Exactly two implementors; not a public extension point.
pub(crate) trait Family {
    /// The irrep label type (used as a magnetic-block cache/memo key).
    type Irrep: Clone + Eq + Hash;
    /// The family's error type, surfaced unchanged by the core.
    type Error;

    /// `N^c_{ab}`, the fusion multiplicity of `a ⊗ b → c`. Errors on an
    /// ill-posed pairing (mixed rank/series/group) per the family's contract.
    fn mult(
        &mut self,
        a: &Self::Irrep,
        b: &Self::Irrep,
        c: &Self::Irrep,
    ) -> Result<usize, Self::Error>;

    /// The sparse real CGC entries coupling `a ⊗ b → c`. Called only for a
    /// non-empty channel (`mult > 0`), which the core checks first.
    fn cgc_entries(
        &mut self,
        a: &Self::Irrep,
        b: &Self::Irrep,
        c: &Self::Irrep,
    ) -> Result<Vec<MEntry>, Self::Error>;

    /// The fusion outputs of $a \otimes b$ (the irreps `c` with $N^c_{ab} > 0$), in a
    /// deterministic order.
    fn products(
        &mut self,
        a: &Self::Irrep,
        b: &Self::Irrep,
    ) -> Result<Vec<Self::Irrep>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Dense F/R blocks (family-neutral: just dims + row-major f64 data).
// ---------------------------------------------------------------------------

/// A dense F-symbol block $F^{abc}_d[e, f]$, a rank-4 array over the outer
/// multiplicity indices $[\mu, \nu, \kappa, \lambda]$ in **row-major** order.
///
/// The axis lengths are $[N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}]$
/// ($\mu, \nu, \kappa, \lambda$), matching the TensorKitSectors `GenericFusion` convention
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

    /// The axis lengths $[N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}]$ ($\mu,\nu,\kappa,\lambda$).
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

/// A dense R-symbol block $R^{ab}_c$, an $N^c_{ab} \times N^c_{ba}$ matrix in
/// **row-major** order (rows = $\mu$, cols = $\nu$).
///
/// $N^c_{ab} = N^c_{ba}$ (fusion multiplicities are symmetric), so the matrix
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
// F-symbol raw contraction (sector.jl:_Fsymbol, :58-89).
// ---------------------------------------------------------------------------

/// The raw F contraction with **reference** empty-vertex semantics (an all-zero
/// block, not an error): the reference pentagon/hexagon loops feed empty blocks
/// through harmlessly. The per-family `f_symbol` wrappers convert empty vertices
/// to a typed error before calling this.
///
/// With `A = CGC(a,b,e)`, `B = CGC(e,c,d)`, `C = CGC(b,c,f)`, `D = CGC(a,f,d)`
/// (the last two conjugated in the reference; real, so elided), and `B`/`D`
/// sliced at the first `d`-basis index (`m_d = 0`; the F-symbol is independent of
/// which `d`-state is fixed — the reference "first diagonal element"):
///
/// ```text
/// F[μ,ν,κ,λ] = Σ  D[ma,mf,λ] · C[mb,mc,mf,κ] · A[ma,mb,me,μ] · B[me,mc,ν]
///            ma,mb,me,mc,mf
/// ```
#[allow(clippy::too_many_arguments)]
pub(crate) fn f_block_raw<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
    d: &F::Irrep,
    e: &F::Irrep,
    f: &F::Irrep,
) -> Result<FBlock, F::Error> {
    let n1 = fam.mult(a, b, e)?;
    let n2 = fam.mult(e, c, d)?;
    let n3 = fam.mult(b, c, f)?;
    let n4 = fam.mult(a, f, d)?;
    let dims = [n1, n2, n3, n4];
    if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
        return Ok(FBlock::zeros(dims));
    }

    let cab = fam.cgc_entries(a, b, e)?; // A[ma,mb,me,μ]
    let cecd = fam.cgc_entries(e, c, d)?; // B slice at m_d=0: [me,mc,ν]
    let cbcf = fam.cgc_entries(b, c, f)?; // C[mb,mc,mf,κ]
    let cafd = fam.cgc_entries(a, f, d)?; // D slice at m_d=0: [ma,mf,λ]

    // Step 1: AB[(ma,mb,mc), (μ,ν)] = Σ_me A[ma,mb,me,μ] · B[me,mc,ν].
    // Group A and B by the shared magnetic index me, then cross each group.
    let mut a_by_me: GroupBy4 = HashMap::new();
    for x in &cab {
        a_by_me
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (ma, mb, μ, vA)
    }
    // B is CGC(e,c,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut b_by_me: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in &cecd {
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
    for x in &cbcf {
        c_by_mf
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (mb, mc, κ, vC)
    }
    // D is CGC(a,f,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut d_by_mf: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in &cafd {
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
// R-symbol raw contraction (sector.jl:_Rsymbol, :91-110).
// ---------------------------------------------------------------------------

/// The raw R contraction with reference empty-vertex semantics (zeros, not an
/// error); the gates feed empty blocks through harmlessly.
///
/// With `A = CGC(a,b,c)`, `B = CGC(b,a,c)` sliced at `m_c = 0`:
///
/// ```text
/// R[μ, ν] = Σ  A[ma, mb, μ] · B[mb, ma, ν]      (B conjugated; real, elided)
///          ma,mb
/// ```
pub(crate) fn r_block_raw<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
) -> Result<RBlock, F::Error> {
    let n1 = fam.mult(a, b, c)?; // rows μ
    let n2 = fam.mult(b, a, c)?; // cols ν  (== n1: fusion multiplicities are symmetric)
    if n1 == 0 || n2 == 0 {
        return Ok(RBlock::zeros(n1.max(n2)));
    }
    debug_assert_eq!(n1, n2, "N^c_ab == N^c_ba");

    let cab = fam.cgc_entries(a, b, c)?; // A[ma,mb,mc,μ]
    let cba = fam.cgc_entries(b, a, c)?; // B[mb,ma,mc,ν]

    // A slice at m_c = 0, keyed by (ma, mb) -> Vec<(μ, value)>.
    let mut a_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in &cab {
        if x.m3 == 0 {
            a_map.entry((x.m1, x.m2)).or_default().push((x.mu, x.value));
        }
    }
    // B slice at m_c = 0, keyed by (ma, mb) = (B.m2, B.m1) -> Vec<(ν, value)>.
    let mut b_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in &cba {
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
// Fusion-set helper for the gates.
// ---------------------------------------------------------------------------

/// `a ⊗ b ∩ c ⊗ d` (the pentagon/hexagon `intersect(⊗(...), ⊗(...))`).
fn intersect_products<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
    d: &F::Irrep,
) -> Result<Vec<F::Irrep>, F::Error> {
    let left = fam.products(a, b)?;
    let right = fam.products(c, d)?;
    Ok(left.into_iter().filter(|k| right.contains(k)).collect())
}

/// Per-gate-call memo for F/R blocks.
///
/// A pentagon/hexagon gate references the *same* F/R block from many index
/// combinations; without memoization each reference recomputes a four-CGC
/// contraction, which for OM≥2 families is the difference between seconds and
/// many minutes. The blocks are tiny, so the memo clones them out cheaply. It is
/// *not* a process-global cache: gates use the raw zeros-for-`N=0` semantics,
/// which would pollute a public value tier with blocks `f_symbol` never stores.
struct BlockMemo<I: Clone + Eq + Hash> {
    f: HashMap<[I; 6], FBlock>,
    r: HashMap<[I; 3], RBlock>,
}

impl<I: Clone + Eq + Hash> Default for BlockMemo<I> {
    fn default() -> Self {
        BlockMemo {
            f: HashMap::new(),
            r: HashMap::new(),
        }
    }
}

impl<I: Clone + Eq + Hash> BlockMemo<I> {
    #[allow(clippy::too_many_arguments)]
    fn f_block<F: Family<Irrep = I>>(
        &mut self,
        fam: &mut F,
        a: &I,
        b: &I,
        c: &I,
        d: &I,
        e: &I,
        f: &I,
    ) -> Result<FBlock, F::Error> {
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
        let bl = f_block_raw(fam, a, b, c, d, e, f)?;
        self.f.insert(key, bl.clone());
        Ok(bl)
    }

    fn r_block<F: Family<Irrep = I>>(
        &mut self,
        fam: &mut F,
        a: &I,
        b: &I,
        c: &I,
    ) -> Result<RBlock, F::Error> {
        let key = [a.clone(), b.clone(), c.clone()];
        if let Some(bl) = self.r.get(&key) {
            return Ok(bl.clone());
        }
        let bl = r_block_raw(fam, a, b, c)?;
        self.r.insert(key, bl.clone());
        Ok(bl)
    }
}

// ---------------------------------------------------------------------------
// Gate 1: F-move unitarity — worst |(M Mᵀ - I)_{ij}|.
// ---------------------------------------------------------------------------

/// The worst F-move unitarity residual for fixed outer labels `(a, b, c, d)`.
///
/// The F-symbols form a square matrix `M` with rows `(e, μ, ν)`
/// (`e ∈ a⊗b`, `μ ∈ [0,N^e_{ab})`, `ν ∈ [0,N^d_{ec})`) and columns `(f, κ, λ)`
/// (`f ∈ b⊗c`, `κ ∈ [0,N^f_{bc})`, `λ ∈ [0,N^d_{af})`). The two associativity
/// bases are orthonormal, so `M` is real-orthogonal (`M Mᵀ = I`); this returns
/// the worst `|(M Mᵀ - I)_{ij}|`. The per-family wrapper compares it to
/// [`TOL_F_UNITARY`] and raises the family's typed error.
pub(crate) fn f_unitarity_residual<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
    d: &F::Irrep,
) -> Result<f64, F::Error> {
    // Rows: (e, μ, ν). Columns: (f, κ, λ).
    let mut rows: Vec<(F::Irrep, usize, usize)> = Vec::new();
    for e in fam.products(a, b)? {
        let n_ab_e = fam.mult(a, b, &e)?;
        let n_ec_d = fam.mult(&e, c, d)?;
        for mu in 0..n_ab_e {
            for nu in 0..n_ec_d {
                rows.push((e.clone(), mu, nu));
            }
        }
    }
    let mut cols: Vec<(F::Irrep, usize, usize)> = Vec::new();
    for f in fam.products(b, c)? {
        let n_bc_f = fam.mult(b, c, &f)?;
        let n_af_d = fam.mult(a, &f, d)?;
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
            let block = memo.f_block(fam, a, b, c, d, e, f)?;
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
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Gate 2: pentagon (TensorKitSectors sectors.jl:pentagon_equation, :786-819).
// ---------------------------------------------------------------------------

/// The worst pentagon residual for the quadruple `(a, b, c, d)` (GenericFusion
/// branch of `TensorKitSectors/sectors.jl:pentagon_equation`). For every
/// `f ∈ a⊗b`, `h ∈ c⊗d`, `g ∈ f⊗c`, `i ∈ b⊗h`, `e ∈ (g⊗d) ∩ (a⊗i)`:
///
/// ```text
/// p1[λμν κρσ] = Σ_τ F(f,c,d,e,g,h)[λ,μ,ν,τ] · F(a,b,h,e,f,i)[κ,τ,ρ,σ]
/// p2[λμν κρσ] = Σ_{j∈b⊗c, α,β,τ}
///                 F(a,b,c,g,f,j)[κ,λ,α,β] · F(a,j,d,e,g,i)[β,μ,τ,σ]
///                 · F(b,c,d,i,j,h)[α,τ,ν,ρ]
/// ```
///
/// returns `max |p1 - p2|`.
pub(crate) fn pentagon_residual<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
    d: &F::Irrep,
) -> Result<f64, F::Error> {
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::default();

    for f in fam.products(a, b)? {
        for h in fam.products(c, d)? {
            for g in fam.products(&f, c)? {
                for i in fam.products(b, &h)? {
                    for e in intersect_products(fam, &g, d, a, &i)? {
                        // Free-index dims: λ=N_fcg, μ=N_gde, ν=N_cdh, κ=N_abf,
                        // ρ=N_bhi, σ=N_aie.
                        let n_lambda = fam.mult(&f, c, &g)?;
                        let n_mu = fam.mult(&g, d, &e)?;
                        let n_nu = fam.mult(c, d, &h)?;
                        let n_kappa = fam.mult(a, b, &f)?;
                        let n_rho = fam.mult(b, &h, &i)?;
                        let n_sigma = fam.mult(a, &i, &e)?;
                        if [n_lambda, n_mu, n_nu, n_kappa, n_rho, n_sigma].contains(&0) {
                            continue; // empty output family -> vacuous
                        }

                        // p1: F1[λ,μ,ν,τ] · F2[κ,τ,ρ,σ], sum over τ (= N_fhe).
                        let f1 = memo.f_block(fam, &f, c, d, &e, &g, &h)?; // [λ,μ,ν,τ]
                        let f2 = memo.f_block(fam, a, b, &h, &e, &f, &i)?; // [κ,τ,ρ,σ]
                        let n_tau = f1.dims()[3];

                        // p2 factors, summed over j ∈ b⊗c and α,β,τ'.
                        let mut p2_terms: Vec<(FBlock, FBlock, FBlock)> = Vec::new();
                        for j in fam.products(b, c)? {
                            let g1 = memo.f_block(fam, a, b, c, &g, &f, &j)?; // [κ,λ,α,β]
                            let g2 = memo.f_block(fam, a, &j, d, &e, &g, &i)?; // [β,μ,τ',σ]
                            let g3 = memo.f_block(fam, b, c, d, &i, &j, &h)?; // [α,τ',ν,ρ]
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
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Gate 3: hexagon (TensorKitSectors sectors.jl:hexagon_equation, :834-871).
// ---------------------------------------------------------------------------

/// The worst residual of both hexagon identities for the triple `(a, b, c)`
/// (GenericFusion branch of `TensorKitSectors/sectors.jl:hexagon_equation`).
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
/// returns `max(|RFR1 - FRF1|, |RFR2 - FRF2|)`. R is real, so the two hexagons
/// differ only in which R replaces which (`conj` is the identity).
pub(crate) fn hexagon_residual<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
) -> Result<f64, F::Error> {
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::default();

    for e in fam.products(c, a)? {
        let rcae = memo.r_block(fam, c, a, &e)?; // [α,λ]
        let race = memo.r_block(fam, a, c, &e)?; // [α,λ]
        for f in fam.products(c, b)? {
            let rcbf = memo.r_block(fam, c, b, &f)?; // [γ,μ]
            let rbcf = memo.r_block(fam, b, c, &f)?; // [γ,μ]
            for d in intersect_products(fam, &e, b, a, &f)? {
                // free dims: α=N_cae, β=N_ebd, μ=N_bcf, ν=N_afd.
                let n_alpha = fam.mult(c, a, &e)?;
                let n_beta = fam.mult(&e, b, &d)?;
                let n_mu = fam.mult(b, c, &f)?;
                let n_nu = fam.mult(a, &f, &d)?;
                if [n_alpha, n_beta, n_mu, n_nu].contains(&0) {
                    continue;
                }
                let facb = memo.f_block(fam, a, c, b, &d, &e, &f)?; // [λ,β,γ,ν]
                let n_lam = facb.dims()[0]; // N_ace = N_cae
                let n_gam = facb.dims()[2]; // N_cbf

                // FRF factors over g ∈ a⊗b.
                let mut frf_terms: Vec<(FBlock, RBlock, RBlock, FBlock)> = Vec::new();
                for g in fam.products(a, b)? {
                    let rcgd = memo.r_block(fam, c, &g, &d)?;
                    let rgcd = memo.r_block(fam, &g, c, &d)?;
                    let fcab = memo.f_block(fam, c, a, b, &d, &e, &g)?; // [α,β,δ,σ]
                    let fabc = memo.f_block(fam, a, b, c, &d, &g, &f)?; // [δ,ψ,μ,ν]
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
    Ok(worst)
}
