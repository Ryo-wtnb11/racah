//! The S3.2 decomposition sweep: a port of QSpace's `getSymmetryStates`
//! (`clebsch_aux.cc:53-348` @ `dd2cc7e`) plus `findMaxWeight`
//! (`clebsch_aux.cc:957-1045`), operating on given tensor-product generators
//! and producing, per coupled irrep, a CGC isometry, projected generators, a
//! Dynkin label and an outer-multiplicity index.
//!
//! Design authority: issue #18 (Rulings 1-4), spec: issue #23. The gauge — every
//! deterministic rule the sweep applies — is specified to re-derivation standard
//! in `docs/gauge_soN.md`; any value-affecting change here is a breaking release.
//!
//! # Scope
//!
//! S3.2 takes the **product generators** as input (the caller builds them from
//! two irrep generator sets via [`Generators::product`], the Kronecker
//! composition helper). It decomposes that product into irreducible multiplets.
//! The catalog/bootstrap deciding *which* products to decompose (S3.3), F/R
//! (S3.4), and QSpace `getCG` fixtures (S3.5) are separate, later layers.
//!
//! # Production gate (Ruling 1)
//!
//! For every discovered irrep `c` the sweep multiplicity `M^c_sweep` must equal
//! the exact fusion multiplicity `N^c_ab` from [`crate::bcd::directproduct`]
//! (S3.0), and the discovered support must equal the exact support — **both
//! directions** (a missing block is as fatal as a spurious one). This is not
//! optional or test-only; [`decompose`] takes the exact decomposition and gates
//! against it, returning [`SweepError::MultiplicityMismatch`] on any divergence.
//!
//! # Numerical seams
//!
//! `CoeffScalar = f64`. The QR column-orthonormalization (the factorization and
//! gauge-fixing step) and the block-level CGC contractions route through
//! `tenferro-linalg` (`super::linalg`); the Gram–Schmidt sweep arithmetic is
//! the gauge algorithm itself and, like `sun::cgc`'s `cref`, is carried in
//! plain `f64`. No hand-rolled factorization kernels.

use std::collections::BTreeMap;

use super::linalg::{matmul, qr_positive_q, tmatmul, Dense};
use super::seeds::Seed;
use super::{directproduct, Irrep, Series};

// ---- tolerance tier (QSpace CG_EPS ladder; provenance in gauge_soN.md) -----

/// QSpace `eps` in `getSymmetryStates` (`clebsch_aux.cc:76`): the working
/// threshold for "is this vector/overlap significant" in the sweep.
const EPS_SWEEP: f64 = 1e-8;
/// QSpace `eps2` in `getSymmetryStates` (`clebsch_aux.cc:76`): the tighter
/// warn/verify threshold (here used for the `U†U` identity and diagonality
/// checks — QSpace `isIdentityMatrix(eps2)` / `isDiagMatrix(eps2)`).
const EPS_VERIFY: f64 = 1e-10;
/// QSpace `CG_EPS1` (non-MPFR tier, `clebsch.hh:244`): the QR orthonormalization
/// tolerance (`OrthoNormalizeColsQR(FL, CG_EPS1)`).
const CG_EPS1: f64 = 1e-10;
/// FixRational integer-snap tolerance for the Cartan (Sz) eigenvalues and CGC
/// entries. QSpace snaps within `CG_SKIP_DEPS1 = 1e-12` at higher working
/// precision; in plain `f64` the sweep's round-off floor is `~1e-13`, so a
/// looser-but-still-safe `1e-6` snaps every genuine integer while a real
/// non-integer (a defect) stays far outside it. Integer-target only
/// (`clebsch_aux.cc:282` `FixRational(...,4)`), documented in gauge_soN.md.
const FIXRATIONAL_TOL: f64 = 1e-6;
/// findMaxWeight max-weight uniqueness threshold (`clebsch_aux.cc:1035`,
/// `recDiff2(0,i) > 1e-8`).
const EPS_MW_UNIQUE: f64 = 1e-8;

// ---- typed errors (guard inventory, issue #15) -----------------------------

/// Failure of the decomposition sweep. Every variant maps a QSpace `wblog(...,
/// "ERR ...")` abort in `getSymmetryStates`/`findMaxWeight` to a typed error
/// (guard inventory in the PR body), or is a racah production gate (Ruling 1).
///
/// Not `Eq`: several variants carry an `f64` residual for diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub enum SweepError {
    /// QSpace `if (!nz || np>nz) ERR` and `if (np!=r || nz!=r) ERR`
    /// (`clebsch_aux.cc:85,90`): the generator sets are empty or their counts
    /// disagree with each other or with the rank.
    InvalidGeneratorCounts {
        /// Number of raising operators `Sp`.
        np: usize,
        /// Number of Cartan operators `Sz`.
        nz: usize,
        /// The rank `r`.
        rank: usize,
    },
    /// [`Generators::product`] of two generator sets of different groups (series
    /// or rank) or inconsistent dimensions — an ill-posed input.
    GeneratorMismatch,
    /// QSpace `if (vi.sameUptoFac(v0)) ERR "failed to determine symmetry
    /// labels"` (`clebsch_aux.cc:128`): the seed vector is not a simultaneous
    /// Sz-eigenvector (not a weight vector), so its labels are ambiguous.
    SeedNotWeightVector {
        /// The multiplet index `it` (0-based) being started.
        multiplet: usize,
        /// The Cartan operator index that broke eigenvector-ness.
        cartan: usize,
    },
    /// QSpace `ERR "got overlap with V space"` (`clebsch_aux.cc:186`): a lowered
    /// vector had a residual overlap with the already-built multiplet space.
    OverlapWithVspace {
        /// Worst residual.
        residual: f64,
    },
    /// QSpace `ERR "got overlap with U"` (`clebsch_aux.cc:194`): a lowered
    /// vector had a residual overlap with the global accumulated space.
    OverlapWithUspace {
        /// Worst residual.
        residual: f64,
    },
    /// QSpace `ERR "V/Vi/U space out of bounds"` (`clebsch_aux.cc:214,222,246`):
    /// the accumulated space exceeded the total dimension `D` — an invariant
    /// break signalling a defective generator set.
    SpaceOutOfBounds,
    /// QSpace `ERR "failed to obtain symmetry multiplets"` (`clebsch_aux.cc:236`)
    /// and the `U.SIZE[1]!=D` dimension check: the sweep did not tile the whole
    /// product space (sum of block dims ≠ `d1·d2`).
    IncompleteDecomposition {
        /// Total dimension `D = d1·d2`.
        dim: usize,
        /// Dimension actually covered.
        covered: usize,
    },
    /// QSpace `ERR "new space not orthogonal"` (`clebsch_aux.cc:251`): `U†U ≠ I`
    /// beyond tolerance.
    NotOrthonormal {
        /// Worst `|(U†U − I)_{ij}|`.
        residual: f64,
    },
    /// QSpace `ERR "got non-diagonal z-operator"` (`clebsch_aux.cc:274`): a
    /// projected Cartan generator `V†(Sz V)` was not diagonal.
    NonDiagonalCartan {
        /// The block index.
        block: usize,
        /// Worst off-diagonal magnitude.
        residual: f64,
    },
    /// A Cartan eigenvalue (or a converted Dynkin label) was not integral within
    /// `FIXRATIONAL_TOL` — QSpace `num2int`/`FixRational` failure
    /// (`clebsch_aux.cc:282,983`).
    NonIntegerWeight {
        /// The block index.
        block: usize,
        /// The offending value.
        value: f64,
    },
    /// QSpace `ERR "maximum weight state not unique"` (`clebsch_aux.cc:1036`):
    /// two states in a block share the highest weight.
    MaxWeightNotUnique {
        /// The block index.
        block: usize,
    },
    /// A discovered irrep's Dynkin label was not a valid B/C/D tensor label
    /// (should be unreachable for a faithful sweep; surfaced not panicked).
    InvalidDiscoveredLabel {
        /// The offending Dynkin label.
        dynkin: Vec<i64>,
    },
    /// **Ruling 1 production gate.** The multiset of discovered irreps does not
    /// equal the exact tensor-product decomposition
    /// ([`crate::bcd::directproduct`]): for some `c`, `M^c_sweep ≠ N^c_ab`.
    /// Both directions are fatal (a missing block has `found = 0`, a spurious
    /// block has `expected = 0`).
    MultiplicityMismatch {
        /// The irrep `c` whose multiplicity disagreed, as a Dynkin label.
        dynkin: Vec<i64>,
        /// Exact `N^c_ab` from S3.0.
        expected: u32,
        /// `M^c_sweep` found by the sweep.
        found: u32,
    },
    /// A projected generator set failed the commutator relations (S3.1's check,
    /// in `f64` — the projected `Sp` are generally non-integer). The worst
    /// residual is carried.
    CommutatorResidual {
        /// The block index.
        block: usize,
        /// Worst commutator residual.
        residual: f64,
    },
    /// A dense factorization/contraction routed through `tenferro-linalg`
    /// failed. Surfaced (not panicked) because the floating-point stages are
    /// verification-gated.
    Linalg(String),
}

impl std::fmt::Display for SweepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SweepError::InvalidGeneratorCounts { np, nz, rank } => {
                write!(f, "invalid generator counts: np={np}, nz={nz}, rank={rank}")
            }
            SweepError::GeneratorMismatch => {
                write!(f, "product of generator sets of mismatched groups/dims")
            }
            SweepError::SeedNotWeightVector { multiplet, cartan } => write!(
                f,
                "seed for multiplet {multiplet} is not an Sz[{cartan}] eigenvector \
                 (cannot determine symmetry labels)"
            ),
            SweepError::OverlapWithVspace { residual } => {
                write!(
                    f,
                    "lowered vector overlaps the multiplet space (residual {residual:e})"
                )
            }
            SweepError::OverlapWithUspace { residual } => {
                write!(
                    f,
                    "lowered vector overlaps the accumulated space (residual {residual:e})"
                )
            }
            SweepError::SpaceOutOfBounds => write!(f, "accumulated space exceeded D"),
            SweepError::IncompleteDecomposition { dim, covered } => write!(
                f,
                "sweep covered {covered} of {dim} dimensions (incomplete decomposition)"
            ),
            SweepError::NotOrthonormal { residual } => {
                write!(
                    f,
                    "accumulated space not orthonormal: U†U−I residual {residual:e}"
                )
            }
            SweepError::NonDiagonalCartan { block, residual } => write!(
                f,
                "block {block}: projected Cartan not diagonal (residual {residual:e})"
            ),
            SweepError::NonIntegerWeight { block, value } => {
                write!(f, "block {block}: non-integer weight/label {value}")
            }
            SweepError::MaxWeightNotUnique { block } => {
                write!(f, "block {block}: maximum-weight state not unique")
            }
            SweepError::InvalidDiscoveredLabel { dynkin } => {
                write!(f, "discovered irrep has an invalid Dynkin label {dynkin:?}")
            }
            SweepError::MultiplicityMismatch {
                dynkin,
                expected,
                found,
            } => write!(
                f,
                "multiplicity gate: irrep {dynkin:?} exact N={expected} but sweep M={found}"
            ),
            SweepError::CommutatorResidual { block, residual } => write!(
                f,
                "block {block}: projected generators fail commutators (residual {residual:e})"
            ),
            SweepError::Linalg(msg) => write!(f, "dense factorization failed: {msg}"),
        }
    }
}

impl std::error::Error for SweepError {}

// ---- generator sets & Kronecker product composition ------------------------

/// A generator set of one B/C/D irrep on its carrier space: the `r` raising
/// operators `Sp[i]` (dense `D×D`) and the `r` Cartan diagonals `Sz[i]` (length
/// `D`), all `f64`. The sweep's input.
///
/// The defining-rep set comes from a [`Seed`] via [`Generators::from_seed`];
/// products of two sets via [`Generators::product`]. Higher-irrep generator sets
/// are produced by the sweep itself ([`Block::generators`](Block::generators)) and, in S3.3, stored
/// in the catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct Generators {
    series: Series,
    rank: usize,
    dim: usize,
    /// `sp[i]` = the `i`-th raising operator, dense column-major `D×D`.
    sp: Vec<Dense>,
    /// `sz[i]` = the `i`-th Cartan diagonal, length `D`.
    sz: Vec<Vec<f64>>,
}

impl Generators {
    /// The series (`B`, `C` or `D`).
    pub fn series(&self) -> Series {
        self.series
    }
    /// The rank `r`.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// The `i`-th Cartan diagonal. Pins the Kronecker composite-index
    /// convention at its site (§1 of `docs/gauge_soN.md`) and lets the S3.3
    /// catalog compare a rediscovered block's Cartan spectrum against a stored
    /// generator set (the debug-assert that replaces QSpace's `normDiff`
    /// cross-copy check, `clebsch.cc:6710-6718 @ dd2cc7e`).
    pub(crate) fn cartan_diag(&self, i: usize) -> &[f64] {
        &self.sz[i]
    }

    /// Worst residual of the S3.1 commutator relations satisfied by this
    /// generator set (`f64` analogue — the projected `Sp` are generally
    /// irrational). Exposed for the S3.3 chain-depth error bench (issue #18
    /// watch item): a catalog entry materialized through a deep canonical-parent
    /// chain accumulates round-off, and this reports it. The sweep already gates
    /// every stored set at `≤ EPS_SWEEP`, so a stored set's residual is bounded.
    #[cfg(test)]
    pub(crate) fn max_commutator_residual(&self) -> f64 {
        commutator_residual(&self.sp, &self.sz, self.dim)
    }

    /// The carrier dimension `D`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The defining-representation generator set for `seed` (dense `f64` from the
    /// exact integer [`Seed`]).
    pub fn from_seed(seed: &Seed) -> Generators {
        let d = seed.dim();
        let sp: Vec<Dense> = seed
            .raising()
            .iter()
            .map(|recs| {
                let mut m = Dense::zeros(d, d);
                for &(row, col, v) in recs {
                    m.set(row, col, v as f64);
                }
                m
            })
            .collect();
        let sz: Vec<Vec<f64>> = seed
            .cartan()
            .iter()
            .map(|diag| diag.iter().map(|&x| x as f64).collect())
            .collect();
        Generators {
            series: seed.series(),
            rank: seed.rank(),
            dim: d,
            sp,
            sz,
        }
    }

    /// The trivial (vacuum) generator set of `series` at rank `r`: a
    /// 1-dimensional carrier on which every raising and Cartan operator is zero.
    /// The `≺`-minimal base case for the S3.3 catalog (§14 of `docs/gauge_soN.md`).
    pub fn trivial(series: Series, r: usize) -> Generators {
        Generators {
            series,
            rank: r,
            dim: 1,
            sp: (0..r).map(|_| Dense::zeros(1, 1)).collect(),
            sz: (0..r).map(|_| vec![0.0]).collect(),
        }
    }

    /// The Kronecker product-generator set of `a ⊗ b`:
    /// `Sp[i] = Sp_a[i] ⊗ 1_b + 1_a ⊗ Sp_b[i]`, likewise for `Sz`
    /// (QSpace `clebsch.cc:6649-6656` @ `dd2cc7e`).
    ///
    /// **Kronecker convention (gauge — pinned here and in gauge_soN.md).** The
    /// product basis index of `|m_a, m_b⟩` is `m_a + d_a · m_b`: the **first**
    /// factor (`a`) is the *fast* index, the second (`b`) the slow index. This
    /// matches QSpace's `wbsparray::setRec_kron` (`q[i] = i1[i] + SIZE_a·i2[i]`,
    /// `wbsparray.cc:3210`), which is the reverse of the textbook `kron(A,B)`
    /// (first factor slow). A different convention is a different gauge.
    ///
    /// Errors ([`SweepError::GeneratorMismatch`]) if the two sets are of
    /// different series or rank.
    pub fn product(a: &Generators, b: &Generators) -> Result<Generators, SweepError> {
        if a.series != b.series || a.rank != b.rank {
            return Err(SweepError::GeneratorMismatch);
        }
        let r = a.rank;
        let (da, db) = (a.dim, b.dim);
        let d = da * db;
        // combined(m_a, m_b) = m_a + d_a * m_b   (factor a fast).
        let comb = |ma: usize, mb: usize| ma + da * mb;

        let mut sp = Vec::with_capacity(r);
        for i in 0..r {
            let mut m = Dense::zeros(d, d);
            // Sp_a[i] ⊗ 1_b : acts on the fast index, spectator mb.
            for c in 0..da {
                for rr in 0..da {
                    let v = a.sp[i].at(rr, c);
                    if v != 0.0 {
                        for mb in 0..db {
                            let cur = m.at(comb(rr, mb), comb(c, mb));
                            m.set(comb(rr, mb), comb(c, mb), cur + v);
                        }
                    }
                }
            }
            // 1_a ⊗ Sp_b[i] : acts on the slow index, spectator ma.
            for c in 0..db {
                for rr in 0..db {
                    let v = b.sp[i].at(rr, c);
                    if v != 0.0 {
                        for ma in 0..da {
                            let cur = m.at(comb(ma, rr), comb(ma, c));
                            m.set(comb(ma, rr), comb(ma, c), cur + v);
                        }
                    }
                }
            }
            sp.push(m);
        }

        let mut sz = Vec::with_capacity(r);
        for i in 0..r {
            let mut diag = vec![0.0; d];
            for mb in 0..db {
                for ma in 0..da {
                    diag[comb(ma, mb)] = a.sz[i][ma] + b.sz[i][mb];
                }
            }
            sz.push(diag);
        }

        Ok(Generators {
            series: a.series,
            rank: r,
            dim: d,
            sp,
            sz,
        })
    }

    /// Validate the generator counts against the rank (QSpace guards at
    /// `clebsch_aux.cc:85,90`).
    fn check_counts(&self) -> Result<(), SweepError> {
        let (np, nz) = (self.sp.len(), self.sz.len());
        if nz == 0 || np > nz || np != self.rank || nz != self.rank {
            return Err(SweepError::InvalidGeneratorCounts {
                np,
                nz,
                rank: self.rank,
            });
        }
        Ok(())
    }
}

// ---- results ---------------------------------------------------------------

/// One discovered irreducible multiplet in the product decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    irrep: Irrep,
    /// CGC isometry `V`, column-major `d1·d2 × d3` (each column a coupled state,
    /// in descending-weight order, max-weight first).
    cgc: Dense,
    /// Projected generators of the multiplet (`Sp[j] = V†(Sp_prod[j] V)`, etc.).
    gens: Generators,
    /// Cartan eigenvalues per state, `d3 × nz` (row = state, integer-snapped).
    z: Dense,
    /// Outer-multiplicity index `(index, size)`: `index ∈ [0, size)`, `size` =
    /// number of blocks sharing this highest weight.
    om: (usize, usize),
}

impl Block {
    /// The coupled irrep `c`.
    pub fn irrep(&self) -> &Irrep {
        &self.irrep
    }
    /// The block dimension `d3 = dim(c)`.
    pub fn dim(&self) -> usize {
        self.irrep.dim().try_into().unwrap_or(usize::MAX)
    }
    /// The CGC isometry as a flat column-major `d1·d2 × d3` buffer.
    pub fn cgc(&self) -> &[f64] {
        &self.cgc.data
    }
    /// `(rows, cols)` of the CGC isometry (`d1·d2`, `d3`).
    pub fn cgc_shape(&self) -> (usize, usize) {
        (self.cgc.rows, self.cgc.cols)
    }
    /// The projected generator set of this multiplet.
    pub fn generators(&self) -> &Generators {
        &self.gens
    }
    /// The Cartan eigenvalue of state `s` under Cartan operator `j`.
    pub fn weight(&self, s: usize, j: usize) -> f64 {
        self.z.at(s, j)
    }
    /// The outer-multiplicity index `(index, size)`.
    pub fn outer_multiplicity(&self) -> (usize, usize) {
        self.om
    }
}

/// The full decomposition of a product: the discovered blocks in sweep
/// (discovery) order, having passed every production gate.
#[derive(Clone, Debug, PartialEq)]
pub struct Decomposition {
    blocks: Vec<Block>,
}

impl Decomposition {
    /// The discovered blocks, in sweep order.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
    /// The multiset of discovered irreps with their sweep multiplicities.
    pub fn multiplicities(&self) -> BTreeMap<Irrep, u32> {
        let mut m = BTreeMap::new();
        for b in &self.blocks {
            *m.entry(b.irrep.clone()).or_insert(0) += 1;
        }
        m
    }
}

// ---- public entry points ---------------------------------------------------

/// Decompose the product of the two defining-rep generator sets `a` and `b`
/// (built from [`Seed`]s), gated against the exact S3.0 decomposition.
///
/// Convenience wrapper over [`decompose`]: composes the product generators,
/// derives the two irrep labels from the seeds' defining label, and computes the
/// exact `N^c_ab` gate via [`crate::bcd::directproduct`].
pub fn decompose_defining_product(a: &Seed, b: &Seed) -> Result<Decomposition, SweepError> {
    let ga = Generators::from_seed(a);
    let gb = Generators::from_seed(b);
    let prod = Generators::product(&ga, &gb)?;
    // The defining label is the vector/fundamental (1,0,…,0).
    let mut dynkin = vec![0i64; a.rank()];
    dynkin[0] = 1;
    let ia = Irrep::from_dynkin(a.series(), &dynkin).map_err(|_| {
        SweepError::InvalidDiscoveredLabel {
            dynkin: dynkin.clone(),
        }
    })?;
    let expected = directproduct(&ia, &ia).map_err(|_| SweepError::GeneratorMismatch)?;
    decompose(&prod, &expected)
}

/// Decompose `product` into irreducible multiplets, gated against the exact
/// decomposition `expected` (`N^c_ab` from [`crate::bcd::directproduct`],
/// Ruling 1).
///
/// Runs the sweep (`get_symmetry_states`), assigns Dynkin labels and
/// outer-multiplicity indices, and enforces every production gate:
/// dimension bookkeeping, `U†U` orthonormality, Cartan diagonality, max-weight
/// uniqueness, `M^c_sweep == N^c_ab` (both directions), and the projected
/// generators' commutator relations.
pub fn decompose(
    product: &Generators,
    expected: &BTreeMap<Irrep, u32>,
) -> Result<Decomposition, SweepError> {
    let blocks = get_symmetry_states(product)?;

    // Ruling 1: both-direction multiplicity gate.
    let found: BTreeMap<Irrep, u32> = {
        let mut m = BTreeMap::new();
        for b in &blocks {
            *m.entry(b.irrep.clone()).or_insert(0) += 1;
        }
        m
    };
    for (c, &n) in expected {
        let got = found.get(c).copied().unwrap_or(0);
        if got != n {
            return Err(SweepError::MultiplicityMismatch {
                dynkin: c.dynkin(),
                expected: n,
                found: got,
            });
        }
    }
    for (c, &got) in &found {
        if expected.get(c).copied().unwrap_or(0) != got {
            return Err(SweepError::MultiplicityMismatch {
                dynkin: c.dynkin(),
                expected: expected.get(c).copied().unwrap_or(0),
                found: got,
            });
        }
    }

    Ok(Decomposition { blocks })
}

// ---- the sweep (getSymmetryStates port) ------------------------------------

/// Port of `getSymmetryStates` + the per-block `findMaxWeight`/label/sign/OM
/// steps. Returns the discovered blocks in sweep order, having passed the
/// per-block gates (orthonormality, diagonality, uniqueness) but before the
/// exact-multiplicity gate.
fn get_symmetry_states(g: &Generators) -> Result<Vec<Block>, SweepError> {
    g.check_counts()?;
    let d = g.dim;
    let r = g.rank;
    let nz = g.sz.len();

    // U = accumulated orthonormal basis of the whole space, D × (covered).
    let mut u = Dense::zeros(d, 0);
    // Per-multiplet CGC isometries, in discovery order.
    let mut multiplets: Vec<Dense> = Vec::new();
    // Seed index, PERSISTENT across multiplets (lowest uncovered basis index).
    let mut i0: usize = 0;
    let mut it: usize = 0;

    while it < d {
        // ---- rung 1: seed = lowest basis index not in span(U) --------------
        let mut v0 = Dense::zeros(d, 1);
        let mut have_seed = false;
        while i0 < d {
            let seed = Dense::unit(d, i0);
            if i0 == 0 && it == 0 {
                v0 = seed;
                have_seed = true;
                break;
            }
            // x = |U† e_{i0}|; == 1 iff e_{i0} ∈ span(U) (U orthonormal).
            let proj = tmatmul(&u, &seed)?;
            if (proj.norm() - 1.0).abs() < EPS_SWEEP {
                i0 += 1;
                continue;
            }
            v0 = seed;
            have_seed = true;
            break;
        }
        if !have_seed {
            break;
        }

        // ---- rung 1b: orthogonalize seed against U (GS2) -------------------
        if it > 0 {
            project_out(&u, &mut v0)?;
            normalize(&mut v0);
            project_out(&u, &mut v0)?;
            normalize(&mut v0);
        }

        // Seed must be a simultaneous Sz-eigenvector (a weight vector).
        for (j, szj) in g.sz.iter().enumerate() {
            let mut vi = v0.clone();
            apply_diag(szj, &mut vi);
            if vi.norm() < EPS_SWEEP {
                continue;
            }
            if !parallel(&vi, &v0) {
                return Err(SweepError::SeedNotWeightVector {
                    multiplet: it,
                    cartan: j,
                });
            }
        }

        // ---- rung 2: raise to max weight (ascending Sp order) --------------
        let mut found = true;
        while found {
            found = false;
            for spi in &g.sp {
                let mut vi = apply(spi, &v0)?;
                let x = vi.norm();
                if x > EPS_SWEEP {
                    scale(&mut vi, 1.0 / x);
                    v0 = vi;
                    found = true;
                }
            }
        }

        // ---- rung 3: sweep down (lowering + GS2 + QR twice) ----------------
        let mut vblock = v0.clone(); // V: the whole multiplet (this pass)
        let mut frontier = v0; // current lowering frontier
        loop {
            let mut level = Dense::zeros(d, 0); // Vi: new vectors this level
            let mut any = false;
            for spi in &g.sp {
                // lowering = Sp† applied to the frontier block
                let mut vi = apply_dagger(spi, &frontier)?;
                // skip if the whole block is tiny
                if col_rms(&vi) < EPS_SWEEP {
                    continue;
                }
                // GS pass 1: self (Vi) → pass (V) → global (U)
                project_out(&level, &mut vi)?;
                let ov = overlap(&vblock, &vi)?;
                if ov > EPS_SWEEP {
                    return Err(SweepError::OverlapWithVspace { residual: ov });
                }
                project_out(&vblock, &mut vi)?;
                let ou = overlap(&u, &vi)?;
                if ou > EPS_SWEEP {
                    return Err(SweepError::OverlapWithUspace { residual: ou });
                }
                project_out(&u, &mut vi)?;
                // drop tiny columns, then QR (positive-diagonal)
                vi = skip_tiny_cols(&vi, EPS_SWEEP);
                if vi.cols == 0 {
                    continue;
                }
                vi = qr_positive_q(&vi, CG_EPS1)?;
                // GS pass 2 (reverse order: global → pass → self), then QR
                project_out(&u, &mut vi)?;
                project_out(&vblock, &mut vi)?;
                project_out(&level, &mut vi)?;
                vi = qr_positive_q(&vi, CG_EPS1)?;
                level.cat_cols(&vi);
                if level.cols > d {
                    return Err(SweepError::SpaceOutOfBounds);
                }
                any = true;
            }
            if any {
                vblock.cat_cols(&level);
                if vblock.cols > d {
                    return Err(SweepError::SpaceOutOfBounds);
                }
                frontier = level;
            } else {
                break;
            }
        }

        // ---- accumulate: U ← [U, V] ---------------------------------------
        u.cat_cols(&vblock);
        if u.cols > d {
            return Err(SweepError::SpaceOutOfBounds);
        }
        multiplets.push(vblock);
        it += 1;
        if u.cols == d {
            break;
        }
    }

    let nt = multiplets.len();
    if u.cols != d || nt == 0 {
        return Err(SweepError::IncompleteDecomposition {
            dim: d,
            covered: u.cols,
        });
    }

    // ---- rung 4: global orthogonality U†U == I ----------------------------
    {
        let utu = tmatmul(&u, &u)?;
        let mut worst = 0.0f64;
        for i in 0..utu.rows {
            for j in 0..utu.cols {
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((utu.at(i, j) - target).abs());
            }
        }
        if worst > EPS_VERIFY {
            return Err(SweepError::NotOrthonormal { residual: worst });
        }
    }

    // ---- rungs 5-8: project generators, labels, sort, sign, OM ------------
    let mut blocks: Vec<Block> = Vec::with_capacity(nt);
    for (bi, v) in multiplets.into_iter().enumerate() {
        let d0 = v.cols;

        // rung 5: project generators. R.Sp[j] = V†(Sp V); R.Sz[j] = V†(Sz V).
        let mut rsp: Vec<Dense> = Vec::with_capacity(r);
        for spi in &g.sp {
            let spv = apply(spi, &v)?;
            rsp.push(tmatmul(&v, &spv)?);
        }
        let mut rsz_diag: Vec<Vec<f64>> = Vec::with_capacity(nz);
        let mut zmat = Dense::zeros(d0, nz);
        for (j, szj) in g.sz.iter().enumerate() {
            let mut szv = v.clone();
            apply_diag(szj, &mut szv);
            let rszj = tmatmul(&v, &szv)?; // d0 × d0
                                           // diagonality gate
            let mut worst = 0.0f64;
            for a in 0..d0 {
                for b in 0..d0 {
                    if a != b {
                        worst = worst.max(rszj.at(a, b).abs());
                    }
                }
            }
            if worst > EPS_VERIFY {
                return Err(SweepError::NonDiagonalCartan {
                    block: bi,
                    residual: worst,
                });
            }
            let diag: Vec<f64> = (0..d0).map(|a| snap_int(rszj.at(a, a))).collect();
            for (a, &val) in diag.iter().enumerate() {
                if (val - rszj.at(a, a)).abs() > FIXRATIONAL_TOL {
                    return Err(SweepError::NonIntegerWeight {
                        block: bi,
                        value: rszj.at(a, a),
                    });
                }
                zmat.set(a, j, val);
            }
            rsz_diag.push(diag);
        }

        // rung 6: findMaxWeight → Dynkin label + descending-weight sort perm.
        let (irrep, perm) = find_max_weight(g.series, r, &zmat, bi)?;

        // apply the sort permutation to V columns, Z rows, R.Sp, R.Sz.
        let v_sorted = permute_cols(&v, &perm);
        let z_sorted = permute_rows(&zmat, &perm);
        let rsp_sorted: Vec<Dense> = rsp.iter().map(|m| permute_both(m, &perm)).collect();
        let rsz_sorted: Vec<Vec<f64>> = rsz_diag
            .iter()
            .map(|diag| perm.iter().map(|&p| diag[p]).collect())
            .collect();

        // rung 7: sign convention — first significant CGC entry positive.
        let mut v_signed = v_sorted;
        range_sign_convention(&mut v_signed.data);
        // integer-snap CGC entries (FixRational, integer-target only where they
        // land on integers; non-integers left as-is — CGCs are generally
        // irrational, so only exact 0/±1 style entries snap).
        for x in v_signed.data.iter_mut() {
            *x = snap_int(*x);
        }

        // rung 5 (gate): projected generators satisfy the commutator relations.
        let residual = commutator_residual(&rsp_sorted, &rsz_sorted, d0);
        if residual > EPS_SWEEP {
            return Err(SweepError::CommutatorResidual {
                block: bi,
                residual,
            });
        }

        let gens = Generators {
            series: g.series,
            rank: r,
            dim: d0,
            sp: rsp_sorted,
            sz: rsz_sorted,
        };

        blocks.push(Block {
            irrep,
            cgc: v_signed,
            gens,
            z: z_sorted,
            om: (0, 1), // filled in below
        });
    }

    // ---- rung 8: outer-multiplicity assignment ----------------------------
    assign_outer_multiplicity(&mut blocks);

    Ok(blocks)
}

// ---- findMaxWeight port ----------------------------------------------------

/// Port of `findMaxWeight` (`clebsch_aux.cc:957-1045` @ `dd2cc7e`).
///
/// `z` is `d0 × nz` (row = state, col = Cartan operator; integer-snapped). The
/// max-weight state is the row that is lexicographically largest reading the
/// columns in **reversed** order (QSpace `z2.FlipCols()` then
/// `sortRecs_float(P,-1)`); the returned permutation sorts all states in that
/// descending order (max weight first). The row of `z` at the max-weight state
/// is converted to a Dynkin label by the per-series formula.
fn find_max_weight(
    series: Series,
    r: usize,
    z: &Dense,
    block: usize,
) -> Result<(Irrep, Vec<usize>), SweepError> {
    let d0 = z.rows;
    let nz = z.cols;
    let perm = descending_weight_perm(z);
    let k = perm[0];

    // max-weight uniqueness: row k must differ from the 2nd-sorted row.
    if d0 > 1 {
        let k2 = perm[1];
        let diff2: f64 = (0..nz).map(|c| (z.at(k, c) - z.at(k2, c)).powi(2)).sum();
        if diff2 <= EPS_MW_UNIQUE {
            return Err(SweepError::MaxWeightNotUnique { block });
        }
    }

    // qm = z.row(k) in ORIGINAL column order; convert to Dynkin per series.
    let qm: Vec<f64> = (0..nz).map(|c| z.at(k, c)).collect();
    let dynkin = to_dynkin(series, r, &qm, block)?;
    let irrep = Irrep::from_dynkin(series, &dynkin)
        .map_err(|_| SweepError::InvalidDiscoveredLabel { dynkin })?;
    Ok((irrep, perm))
}

/// The permutation sorting states into descending-weight order (QSpace
/// `z2.FlipCols(); z2.sortRecs_float(P,-1)`, `clebsch_aux.cc:969-970 @ dd2cc7e`):
/// lexicographic comparison on the Cartan columns read in **reversed** order
/// (column `nz-1` first, …, column `0` last), descending. **Tie-break (gauge):**
/// equal weight rows keep ascending original basis index (a stable, deterministic
/// total order). Extracted for a site test because the tie-break is value-neutral
/// to the *decomposition* (it only reorders states of an identical weight), so no
/// label/multiplicity oracle can catch a change to it.
fn descending_weight_perm(z: &Dense) -> Vec<usize> {
    let nz = z.cols;
    let mut perm: Vec<usize> = (0..z.rows).collect();
    perm.sort_by(|&a, &b| {
        for c in (0..nz).rev() {
            match z
                .at(b, c)
                .partial_cmp(&z.at(a, c))
                .unwrap_or(std::cmp::Ordering::Equal)
            {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        a.cmp(&b) // tie-break: ascending original index
    });
    perm
}

/// Convert the max-weight Cartan eigenvalues `qm` (QSpace `Sz` basis) to Dynkin
/// labels, per series (`findMaxWeight` `q.type` branches, `clebsch_aux.cc:977-
/// 1031`). Integer-target: each result must be integral within `FIXRATIONAL_TOL`.
fn to_dynkin(series: Series, r: usize, qm: &[f64], block: usize) -> Result<Vec<i64>, SweepError> {
    let int = |x: f64| -> Result<i64, SweepError> {
        let q = x.round();
        if (x - q).abs() > FIXRATIONAL_TOL {
            return Err(SweepError::NonIntegerWeight { block, value: x });
        }
        Ok(q as i64)
    };
    let mut q = qm.to_vec();
    match series {
        Series::C => {
            // SpN branch: q[i] = (q[i]-q[i-1])/(i+1) for i=r-1..1; q[0]=q[0].
            let mut out = vec![0i64; r];
            for i in (1..r).rev() {
                out[i] = int((q[i] - q[i - 1]) / ((i + 1) as f64))?;
            }
            out[0] = int(q[0])?;
            Ok(out)
        }
        Series::B => {
            // SON branch.
            let l = (r - 1) / 2;
            let x = int(2.0 * q[0])?;
            let mut out = vec![0i64; r];
            for i in 1..r {
                out[i - 1] = int(q[i] - q[i - 1])?;
            }
            out[r - 1] = x;
            for i in 0..l {
                out.swap(i, r - 2 - i);
            }
            Ok(out)
        }
        Series::D => {
            // SEN branch.
            let l = (r - 1) / 2;
            let x = int(q[0] + q[1])?;
            let mut out = vec![0i64; r];
            for i in 1..r {
                out[i - 1] = int(q[i] - q[i - 1])?;
            }
            out[r - 1] = x;
            for i in 0..l {
                out.swap(i, r - 2 - i);
            }
            // q is unused after this; silence the "never read" on the moved vec.
            let _ = &mut q;
            Ok(out)
        }
    }
}

// ---- sign convention (rangeSignConvention / signFirstVal) ------------------

/// `signFirstVal` (`clebsch_aux.cc:29-40`): +1 unless the first entry with
/// `|x| > eps1` is negative. `eps1 = CG_EPS1`, `eps2 = CG_EPS2` (warn tier;
/// the warn is elided here — a silently-degraded value is a gate elsewhere).
fn sign_first_val(d: &[f64]) -> i32 {
    for &x in d {
        if x.abs() > CG_EPS1 {
            return if x < 0.0 { -1 } else { 1 };
        }
    }
    1
}

/// `rangeSignConvention` (`clebsch_aux.cc:43-51`): flip the whole vector's sign
/// if its first significant entry is negative, so the block's first significant
/// CGC entry is positive.
fn range_sign_convention(d: &mut [f64]) {
    if sign_first_val(d) < 0 {
        for x in d.iter_mut() {
            *x = -*x;
        }
    }
}

// ---- outer-multiplicity harvest (clebsch_aux.cc:331-345) -------------------

/// Assign `(index, size)` outer-multiplicity labels: blocks sharing the same
/// irrep (highest weight) get `index = 0,1,…` in discovery order and a common
/// `size`.
fn assign_outer_multiplicity(blocks: &mut [Block]) {
    let mut counts: BTreeMap<Irrep, usize> = BTreeMap::new();
    for b in blocks.iter() {
        *counts.entry(b.irrep.clone()).or_insert(0) += 1;
    }
    let mut seen: BTreeMap<Irrep, usize> = BTreeMap::new();
    for b in blocks.iter_mut() {
        let size = counts[&b.irrep];
        let idx = seen.entry(b.irrep.clone()).or_insert(0);
        b.om = (*idx, size);
        *idx += 1;
    }
}

// ---- projected-generator commutator gate (f64 analogue of S3.1) ------------

/// Worst residual of the commutator relations the projected generators must
/// satisfy: `[Sz_j, Sp_i] = d_{i,j} Sp_i` (root eigenvector) and
/// `[Sp_i, Sp_i†] = Σ_k f_{i,k} Sz_k` (Cartan span). `f64` because the projected
/// `Sp` are generally irrational; S3.1's exact `check_commutators` is the
/// integer analogue for seeds.
fn commutator_residual(sp: &[Dense], sz_diag: &[Vec<f64>], d0: usize) -> f64 {
    let r = sp.len();
    let mut worst = 0.0f64;
    // build dense Sz.
    let szd: Vec<Dense> = sz_diag
        .iter()
        .map(|diag| {
            let mut m = Dense::zeros(d0, d0);
            for (a, &v) in diag.iter().enumerate() {
                m.set(a, a, v);
            }
            m
        })
        .collect();
    // [Sz_j, Sp_i] proportional to Sp_i.
    for spi in sp.iter() {
        for szj in szd.iter() {
            let c = dense_commutator(szj, spi);
            // eigenvalue d from the largest |Sp_i| entry.
            let (mut br, mut bc, mut bv) = (0usize, 0usize, 0.0f64);
            for a in 0..d0 {
                for b in 0..d0 {
                    if spi.at(a, b).abs() > bv.abs() {
                        bv = spi.at(a, b);
                        br = a;
                        bc = b;
                    }
                }
            }
            if bv.abs() < EPS_SWEEP {
                continue;
            }
            let dz = c.at(br, bc) / bv;
            for a in 0..d0 {
                for b in 0..d0 {
                    worst = worst.max((c.at(a, b) - dz * spi.at(a, b)).abs());
                }
            }
        }
    }
    // [Sp_i, Sp_i†] ∈ span(Sz): project onto Sz basis, check residual.
    for spi in sp.iter() {
        let spt = spi.transpose();
        let comm = dense_commutator(spi, &spt);
        // Frobenius-project the diagonal onto each Sz_j, then residual.
        let mut recon = Dense::zeros(d0, d0);
        for szj in szd.iter() {
            let mut num = 0.0;
            let mut den = 0.0;
            for a in 0..d0 {
                num += comm.at(a, a) * szj.at(a, a);
                den += szj.at(a, a) * szj.at(a, a);
            }
            let f = if den > EPS_SWEEP { num / den } else { 0.0 };
            for a in 0..d0 {
                let cur = recon.at(a, a);
                recon.set(a, a, cur + f * szj.at(a, a));
            }
        }
        for a in 0..d0 {
            for b in 0..d0 {
                worst = worst.max((comm.at(a, b) - recon.at(a, b)).abs());
            }
        }
    }
    let _ = r;
    worst
}

fn dense_commutator(a: &Dense, b: &Dense) -> Dense {
    let d = a.rows;
    let mut c = Dense::zeros(d, d);
    for i in 0..d {
        for k in 0..d {
            let aik = a.at(i, k);
            let bik = b.at(i, k);
            if aik == 0.0 && bik == 0.0 {
                continue;
            }
            for j in 0..d {
                // AB - BA
                let cur = c.at(i, j);
                c.set(i, j, cur + aik * b.at(k, j) - bik * a.at(k, j));
            }
        }
    }
    c
}

// ---- small dense/vector helpers (plain f64: the gauge algorithm itself) ----

/// Apply the raising operator `sp` (dense) to the block `x`: `sp · x`.
fn apply(sp: &Dense, x: &Dense) -> Result<Dense, SweepError> {
    // plain sparse-ish dense matmul; sp is small (D×D), x is D×k.
    let d = sp.rows;
    let mut out = Dense::zeros(d, x.cols);
    for i in 0..d {
        for k in 0..d {
            let v = sp.at(i, k);
            if v == 0.0 {
                continue;
            }
            for c in 0..x.cols {
                let cur = out.at(i, c);
                out.set(i, c, cur + v * x.at(k, c));
            }
        }
    }
    Ok(out)
}

/// Apply the lowering operator `sp†` (transpose of a real raising op) to `x`.
fn apply_dagger(sp: &Dense, x: &Dense) -> Result<Dense, SweepError> {
    let d = sp.rows;
    let mut out = Dense::zeros(d, x.cols);
    for k in 0..d {
        for i in 0..d {
            let v = sp.at(k, i); // (sp†)[i,k] = sp[k,i]
            if v == 0.0 {
                continue;
            }
            for c in 0..x.cols {
                let cur = out.at(i, c);
                out.set(i, c, cur + v * x.at(k, c));
            }
        }
    }
    Ok(out)
}

/// Apply a diagonal Cartan operator (elementwise scaling of the rows).
fn apply_diag(diag: &[f64], x: &mut Dense) {
    for c in 0..x.cols {
        for (i, &d) in diag.iter().enumerate() {
            let cur = x.at(i, c);
            x.set(i, c, cur * d);
        }
    }
}

/// `x -= Q (Qᵀ x)` for orthonormal `Q` (project columns of `x` out of span(Q)).
fn project_out(q: &Dense, x: &mut Dense) -> Result<(), SweepError> {
    if q.cols == 0 {
        return Ok(());
    }
    let qtx = tmatmul(q, x)?; // n × k
    let qqtx = matmul(q, &qtx)?; // D × k
    for i in 0..x.data.len() {
        x.data[i] -= qqtx.data[i];
    }
    Ok(())
}

/// Worst-column overlap magnitude `|Qᵀ x|_∞` (max abs entry), for the guard
/// checks (QSpace `x1.aMax()` / `sqrt(norm2/vi2)`).
fn overlap(q: &Dense, x: &Dense) -> Result<f64, SweepError> {
    if q.cols == 0 {
        return Ok(0.0);
    }
    let qtx = tmatmul(q, x)?;
    Ok(qtx.data.iter().fold(0.0f64, |m, &v| m.max(v.abs())))
}

/// Normalize a single-column vector in place.
fn normalize(v: &mut Dense) {
    let n = v.norm();
    if n > 0.0 {
        scale(v, 1.0 / n);
    }
}

fn scale(v: &mut Dense, s: f64) {
    for x in v.data.iter_mut() {
        *x *= s;
    }
}

/// Root-mean-square per column: `sqrt(‖x‖² / cols)` (QSpace `sqrt(vi2/SIZE[1])`).
fn col_rms(x: &Dense) -> f64 {
    if x.cols == 0 {
        return 0.0;
    }
    (x.data.iter().map(|v| v * v).sum::<f64>() / x.cols as f64).sqrt()
}

/// Whether `a` is parallel to `b` up to a scalar (both single columns).
/// The negation of QSpace's `sameUptoFac` == 0 test.
fn parallel(a: &Dense, b: &Dense) -> bool {
    // a ∥ b iff ‖a‖‖b‖ - |⟨a,b⟩| ≈ 0.
    let na = a.norm();
    let nb = b.norm();
    if na < EPS_SWEEP || nb < EPS_SWEEP {
        return true; // a tiny vector carries no label constraint
    }
    let dot: f64 = a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum();
    (na * nb - dot.abs()).abs() < EPS_SWEEP * na * nb
}

/// Drop columns whose norm is `< eps` (QSpace `SkipTinyCols`).
fn skip_tiny_cols(x: &Dense, eps: f64) -> Dense {
    let keep: Vec<usize> = (0..x.cols)
        .filter(|&j| x.col(j).iter().map(|v| v * v).sum::<f64>().sqrt() >= eps)
        .collect();
    x.select_cols(&keep)
}

fn permute_cols(m: &Dense, perm: &[usize]) -> Dense {
    m.select_cols(perm)
}

fn permute_rows(m: &Dense, perm: &[usize]) -> Dense {
    let mut out = Dense::zeros(m.rows, m.cols);
    for (ro, &r) in perm.iter().enumerate() {
        for c in 0..m.cols {
            out.set(ro, c, m.at(r, c));
        }
    }
    out
}

fn permute_both(m: &Dense, perm: &[usize]) -> Dense {
    let mut out = Dense::zeros(m.rows, m.cols);
    for (ro, &r) in perm.iter().enumerate() {
        for (co, &c) in perm.iter().enumerate() {
            out.set(ro, co, m.at(r, c));
        }
    }
    out
}

/// Snap `x` to the nearest integer if within `FIXRATIONAL_TOL`, else leave it.
fn snap_int(x: f64) -> f64 {
    let q = x.round();
    if (x - q).abs() <= FIXRATIONAL_TOL {
        q
    } else {
        x
    }
}

#[cfg(test)]
mod tests;
