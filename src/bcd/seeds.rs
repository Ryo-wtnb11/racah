//! Defining-representation generator seeds for the B/C/D series (Layer S3.1 of
//! the `cgc-gen` track; design authority: issue #18 rulings, spec: issue #21).
//!
//! This module ports, entry-for-entry, the exact sparse raising operators
//! `Sp[i]` and Cartan generators `Sz[i]` of the *defining* representation for
//!
//! - `C_r = Sp(2r)`   — QSpace `clebsch.cc:Setup_SpN` (`:7145-7244` @ `dd2cc7e`),
//! - `B_r = SO(2r+1)` — QSpace `clebsch.cc:Setup_SON` (`:7246-7348` @ `dd2cc7e`),
//! - `D_r = SO(2r)`   — QSpace `clebsch.cc:Setup_SEN` (`:7350-7457` @ `dd2cc7e`),
//!
//! together with [`check_commutators`], the exact self-check that gates them —
//! the Rust analogue of QSpace's `initCommRel` / `checkCommRel`
//! (`clebsch.cc:5949-6120` @ `dd2cc7e`). This self-check is the load-bearing
//! gate the numeric sweep (S3.2) will reuse against generated matrices.
//!
//! # Exactness: every entry is an integer at `dd2cc7e`
//!
//! In QSpace all ladder entries (`P.setRec(..,1.)`) and all Cartan diagonals
//! (`Z[..]`) of these three Setups are **integers** — value `±1`, `±2` on the
//! Cartan diagonals, `+1` on every ladder entry. There is *no* `sqrt(2)` short
//! root, no fractional ladder normalization: QSpace's convention places the
//! whole scale into the (integer, mutually orthogonal) Cartan generators and
//! keeps unit ladder entries. Consequently every quantity in this layer —
//! seed entries, commutators, and the derived Cartan/root structure constants —
//! is exact integer or rational (`Ratio<i64>`); [`crate::exact::SignedSqrtRational`]
//! is *not* needed for the entries, and no float appears anywhere. Should a
//! future QSpace revision introduce an irrational ladder normalization, the
//! entry type here must be widened and this note revisited.
//!
//! # Basis convention: QSpace, not Chevalley
//!
//! QSpace's `Sz[i]` are **not** the Chevalley coroots `H_i`. They are integer,
//! traceless, mutually Frobenius-orthogonal diagonal generators; the ladder
//! operators `Sp[i]` carry unit entries. Consequently the structure constants
//! this layer checks are *not* the textbook Cartan matrix:
//!
//! - `[Sp_i, Sp_i^†] = Σ_k f_{i,k} Sz_k` — a linear combination of the Cartan
//!   generators with rational coefficients `f_{i,k}` obtained by Frobenius
//!   projection (QSpace `initCommRel` `CR[i]`, `clebsch.cc:5971-5987`).
//! - `[Sz_j, Sp_i] = d_{i,j} Sp_i` — each `Sp_i` is a common `ad(Sz_j)`
//!   eigenvector with rational "root component" `d_{i,j}` (QSpace `initCommRel`
//!   `DZ`, `clebsch.cc:5989-6003`).
//! - `⟨Sz_i, Sz_j⟩_F = 0` for `i ≠ j`, and `[Sz_i, Sz_j] = 0` (QSpace
//!   `checkCommRel`, `clebsch.cc:6032-6050`; the latter is vacuous here since
//!   the `Sz` are diagonal).
//!
//! [`check_commutators`] verifies *exactly these three QSpace relations*, with
//! the coefficients derived by the same Frobenius-projection recipe QSpace uses
//! — deliberately **not** substituting a textbook Chevalley normalization,
//! because S3.2 ports QSpace's sweep against precisely these matrices. The
//! derived `f` and `d` are returned in [`CommReport`] so downstream layers and
//! tests can inspect the actual (QSpace-basis) root system.
//!
//! # References
//!
//! - QSpace v4 (Weichselbaum), `Source/clebsch.cc` @ `dd2cc7e`:
//!   `Setup_SpN`/`Setup_SON`/`Setup_SEN` (the seeds) and
//!   `initCommRel`/`checkCommRel` (the self-check).
//! - `Source/wbsparray.hh:633` — `froNorm2(B) = Σ_ij this_ij·B_ij` (the real
//!   Frobenius inner product used for the Cartan projection).

use num_rational::Ratio;
use num_traits::Zero;

use super::{BcdError, Series};

/// The exact defining-representation generator seed for one B/C/D group.
///
/// Holds the `r` simple-root raising operators `Sp[i]` (as sparse
/// `(row, col, value)` records — the natural exact form of QSpace's `setRec`)
/// and the `r` Cartan generators `Sz[i]` (as their integer diagonals), all over
/// the `dim`-dimensional defining representation. Every entry is an exact
/// integer (see module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seed {
    series: Series,
    rank: usize,
    dim: usize,
    /// `sp[i]` = nonzero `(row, col, value)` records of the `i`-th raising op.
    sp: Vec<Vec<(usize, usize, i64)>>,
    /// `sz[i]` = the `dim`-length integer diagonal of the `i`-th Cartan op.
    sz: Vec<Vec<i64>>,
}

impl Seed {
    /// The series (`B`, `C` or `D`).
    pub fn series(&self) -> Series {
        self.series
    }
    /// The rank `r` (number of `Sp`/`Sz` generators).
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// The defining-representation dimension `D` (matrix size): `2r` for `C`/`D`,
    /// `2r+1` for `B`. Equal to the Weyl dimension of the defining label.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// The raising operators `Sp[0..r]`, each a list of nonzero
    /// `(row, col, value)` records.
    pub fn raising(&self) -> &[Vec<(usize, usize, i64)>] {
        &self.sp
    }
    /// The Cartan generators `Sz[0..r]`, each as its `D`-length integer diagonal.
    pub fn cartan(&self) -> &[Vec<i64>] {
        &self.sz
    }

    /// Test-only mutable access, for the mutation-sanity tests that corrupt one
    /// entry and assert [`check_commutators`] then rejects the seed.
    #[cfg(test)]
    fn raising_mut(&mut self) -> &mut [Vec<(usize, usize, i64)>] {
        &mut self.sp
    }
    #[cfg(test)]
    fn cartan_mut(&mut self) -> &mut [Vec<i64>] {
        &mut self.sz
    }
}

/// The exact defining-rep generator seed for `series` at rank `r`.
///
/// Ports the sparse `Sp`/`Sz` seed matrices entry-for-entry from QSpace's
/// `Setup_SpN`/`Setup_SON`/`Setup_SEN` (@ `dd2cc7e`). The defining irrep is the
/// vector (`B`, `D`) / fundamental (`C`) label `(1,0,…,0)`; its dimension is
/// `2r` (`C`, `D`) or `2r+1` (`B`).
///
/// Returns [`BcdError::ExcludedRank`] (with SU(2)/SU(2)×SU(2) redirection) for
/// the low-rank isomorphisms `B_1 = SO(3)`, `C_1 = Sp(2)`, `D_2 = SO(4)`,
/// inheriting the S3.0 rank guard.
///
/// # Guard inventory (issue #15; QSpace asserts around the Setups)
///
/// - QSpace `if (D<2||D>10)` (`Sp`) / `if (D<3||D>12)` (`SO`/`SEN`): the *lower*
///   bound is the low-rank-isomorphism guard, mapped to
///   [`BcdError::ExcludedRank`]. The *upper* bound is a QSpace fixed-buffer /
///   `dmax` build artifact, **not** a mathematical constraint — N/A here; seeds
///   are generated for any admissible rank.
/// - QSpace `initCommRel` `if (R.Sp[i].isEmpty()) ERR` and
///   `if (C.norm2()<1e-10) ERR "[Sp,Sp'] has norm 0"`: mapped into
///   [`check_commutators`] as [`BcdError::CommutatorViolation`].
/// - QSpace `checkCommRel` z-orthogonality / `[Z,Z]` / `CR` / `[Z,Sp]`
///   consistency `ERR`s: each mapped to a [`BcdError::CommutatorViolation`]
///   relation in [`check_commutators`].
pub fn defining_seed(series: Series, r: usize) -> Result<Seed, BcdError> {
    if r < series.min_rank() {
        return Err(BcdError::ExcludedRank {
            series,
            rank: r,
            redirect: series.low_rank_redirect(),
        });
    }
    let seed = match series {
        Series::C => setup_spn(r),
        Series::B => setup_son(r),
        Series::D => setup_sen(r),
    };
    Ok(seed)
}

/// `Setup_SpN` — `Sp(2r)`, `D = 2r` (clebsch.cc:7145-7244 @ dd2cc7e).
///
/// Paired `±` short-root blocks (two ladder entries) for `i<r`; a single
/// long-root ladder entry for `i=r`. The Cartan diagonals are the QSpace
/// integer construction: upper half `Z[i]=-i, Z[0..i-1]=1` (`i<r`) or all `1`
/// (`i=r`), lower half the negated mirror `Z[r+j] = -Z[r-1-j]`.
fn setup_spn(r: usize) -> Seed {
    let d = 2 * r;
    let mut sp = Vec::with_capacity(r);
    let mut sz = Vec::with_capacity(r);
    // C++ loop `for (i=1; i<=r; i++)`; index into Sp/Sz is `i-1`.
    for i in 1..=r {
        let mut z = vec![0i64; d];
        let p: Vec<(usize, usize, i64)> = if i < r {
            z[i] = -(i as i64); // C++: Z[i]=-int(i) (i is 1-based, used as 0-based index)
            z[..i].fill(1); // C++: for (j=0;j<i;++j) Z[j]=1
            vec![(i - 1, i, 1), (2 * r - i - 1, 2 * r - i, 1)]
        } else {
            for z_j in z.iter_mut().take(r) {
                *z_j = 1;
            }
            vec![(r - 1, r, 1)]
        };
        for j in 0..r {
            z[r + j] = -z[r - 1 - j];
        }
        sp.push(p);
        sz.push(z);
    }
    Seed {
        series: Series::C,
        rank: r,
        dim: d,
        sp,
        sz,
    }
}

/// `Setup_SON` — `SO(2r+1)`, `D = 2r+1` (clebsch.cc:7246-7348 @ dd2cc7e).
///
/// `2×2` weight pairs `Z[i2]=1, Z[i2+1]=-1` (`i2 = 2(i-1)`); paired ladder
/// entries for `i<r`; the short-root ladder touching the zero-weight state
/// (index `D-1`) for `i=r`.
fn setup_son(r: usize) -> Seed {
    let d = 2 * r + 1;
    let mut sp = Vec::with_capacity(r);
    let mut sz = Vec::with_capacity(r);
    for i in 1..=r {
        let i2 = 2 * (i - 1);
        let mut z = vec![0i64; d];
        z[i2] = 1;
        z[i2 + 1] = -1;
        let p = if i < r {
            vec![(i2 + 2, i2, 1), (i2 + 1, i2 + 3, 1)]
        } else {
            vec![(i2 + 2, 1, 1), (0, i2 + 2, 1)]
        };
        sp.push(p);
        sz.push(z);
    }
    Seed {
        series: Series::B,
        rank: r,
        dim: d,
        sp,
        sz,
    }
}

/// `Setup_SEN` — `SO(2r)`, `D = 2r` (clebsch.cc:7350-7457 @ dd2cc7e).
///
/// Same `2×2` weight pairs and interior ladder as `SON`, but the `i=r` node is
/// the D-series **fork**: fixed entries `(2,1)` and `(0,3)` (independent of `r`),
/// attaching the last simple root away from the tail of the chain.
fn setup_sen(r: usize) -> Seed {
    let d = 2 * r;
    let mut sp = Vec::with_capacity(r);
    let mut sz = Vec::with_capacity(r);
    for i in 1..=r {
        let i2 = 2 * (i - 1);
        let mut z = vec![0i64; d];
        z[i2] = 1;
        z[i2 + 1] = -1;
        let p = if i < r {
            vec![(i2 + 2, i2, 1), (i2 + 1, i2 + 3, 1)]
        } else {
            vec![(2, 1, 1), (0, 3, 1)]
        };
        sp.push(p);
        sz.push(z);
    }
    Seed {
        series: Series::D,
        rank: r,
        dim: d,
        sp,
        sz,
    }
}

// ---- commutator self-check (QSpace initCommRel / checkCommRel) ------------

/// The exact Cartan/root structure constants derived while checking a [`Seed`],
/// in QSpace's (non-Chevalley) basis — see module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommReport {
    /// `cartan_coeffs[i][k] = f_{i,k}` with `[Sp_i, Sp_i^†] = Σ_k f_{i,k} Sz_k`
    /// (Frobenius projection; QSpace `CR`). `r × r`, exact rationals.
    pub cartan_coeffs: Vec<Vec<Ratio<i64>>>,
    /// `root_weights[i][j] = d_{i,j}` with `[Sz_j, Sp_i] = d_{i,j} Sp_i`
    /// (QSpace `DZ`). `r × r`, exact rationals — the `i`-th root in the `Sz`
    /// basis.
    pub root_weights: Vec<Vec<Ratio<i64>>>,
}

/// Verify — exactly — the QSpace commutator relations a [`Seed`] must satisfy,
/// returning the derived Cartan/root structure constants ([`CommReport`]) or the
/// first [`BcdError::CommutatorViolation`].
///
/// This is the Rust analogue of QSpace's `initCommRel` + `checkCommRel`
/// (`clebsch.cc:5949-6120` @ `dd2cc7e`) and the foundation of the S3.2 sweep's
/// gate. Because every seed entry is an integer, the whole check is exact
/// (`i64` matrix arithmetic; `Ratio<i64>` structure constants); there are no
/// float tolerances. Relations checked (see module docs for the basis
/// convention):
///
/// 1. **Cartan orthogonality** `⟨Sz_i, Sz_j⟩_F = 0` (`i<j`), and each `Sz_i`
///    nonzero. (`[Sz_i,Sz_j]=0` is vacuous for diagonal `Sz` — N/A.)
/// 2. **Ladder–Cartan** `[Sp_i, Sp_i^†] = Σ_k f_{i,k} Sz_k`, `f` by Frobenius
///    projection, verified by exact residual `= 0`; and `[Sp_i,Sp_i^†] ≠ 0`.
/// 3. **Root** `[Sz_j, Sp_i] = d_{i,j} Sp_i`, `d` from the ratio of a matching
///    entry, verified by exact residual `= 0`.
pub fn check_commutators(seed: &Seed) -> Result<CommReport, BcdError> {
    let d = seed.dim;
    let r = seed.rank;
    let series = seed.series;
    let viol = |relation: &'static str, i: usize, j: usize| BcdError::CommutatorViolation {
        series,
        relation,
        i,
        j,
    };

    // Dense forms: Sp_i and its transpose (= conjugate, entries are real).
    let sp_dense: Vec<Vec<i64>> = seed.sp.iter().map(|p| dense(p, d)).collect();

    // (1) Cartan orthogonality + nonzero.
    for i in 0..r {
        if seed.sz[i].iter().all(|&x| x == 0) {
            return Err(viol("cartan is zero", i, i));
        }
        for j in i + 1..r {
            if fro_diag(&seed.sz[i], &seed.sz[j]) != 0 {
                return Err(viol("cartan not mutually orthogonal", i, j));
            }
        }
    }

    // (2) [Sp_i, Sp_i^†] = Σ_k f_{i,k} Sz_k.
    let mut cartan_coeffs = vec![vec![Ratio::<i64>::zero(); r]; r];
    for i in 0..r {
        let spt = transpose(&sp_dense[i], d);
        let c = commutator(&sp_dense[i], &spt, d);
        if c.iter().all(|&x| x == 0) {
            return Err(viol("[Sp,Sp^dagger] has norm 0", i, i));
        }
        // f_{i,k} = ⟨C, Sz_k⟩_F / ⟨Sz_k, Sz_k⟩_F  (Frobenius projection).
        let c_diag: Vec<i64> = (0..d).map(|p| c[p * d + p]).collect();
        cartan_coeffs[i] = seed
            .sz
            .iter()
            .map(|szk| Ratio::new(fro_diag(&c_diag, szk), fro_diag(szk, szk)))
            .collect();
        // Residual C - Σ f_k Sz_k must vanish exactly (also proves C diagonal).
        for row in 0..d {
            for col in 0..d {
                let mut res = Ratio::from_integer(c[row * d + col]);
                if row == col {
                    for (fk, szk) in cartan_coeffs[i].iter().zip(&seed.sz) {
                        res -= *fk * szk[row];
                    }
                }
                if !res.is_zero() {
                    return Err(viol("[Sp,Sp^dagger] not in span(Sz)", i, i));
                }
            }
        }
    }

    // (3) [Sz_j, Sp_i] = d_{i,j} Sp_i.
    let mut root_weights = vec![vec![Ratio::<i64>::zero(); r]; r];
    for i in 0..r {
        // `j` indexes both `seed.sz[j]` and `root_weights[i][j]` and drives the
        // per-`j` commutator build; an index loop is the clear form here.
        #[allow(clippy::needless_range_loop)]
        for j in 0..r {
            let szj = diag(&seed.sz[j], d);
            let bc = commutator(&szj, &sp_dense[i], d);
            // d from the first nonzero Sp_i entry (guaranteed to exist).
            let (r0, c0, v0) = seed.sp[i][0];
            let dz = Ratio::new(bc[r0 * d + c0], v0);
            root_weights[i][j] = dz;
            for row in 0..d {
                for col in 0..d {
                    let res =
                        Ratio::from_integer(bc[row * d + col]) - dz * sp_dense[i][row * d + col];
                    if !res.is_zero() {
                        return Err(viol("[Sz,Sp] not proportional to Sp", i, j));
                    }
                }
            }
        }
    }

    Ok(CommReport {
        cartan_coeffs,
        root_weights,
    })
}

// ---- tiny dense i64 matrix helpers (D <= 12; correctness over speed) ------

/// Row-major `D×D` dense matrix from sparse `(row, col, value)` records.
fn dense(recs: &[(usize, usize, i64)], d: usize) -> Vec<i64> {
    let mut m = vec![0i64; d * d];
    for &(row, col, v) in recs {
        m[row * d + col] = v;
    }
    m
}

/// Row-major `D×D` dense matrix with the given diagonal.
fn diag(diagonal: &[i64], d: usize) -> Vec<i64> {
    let mut m = vec![0i64; d * d];
    for (p, &v) in diagonal.iter().enumerate() {
        m[p * d + p] = v;
    }
    m
}

/// Transpose of a row-major `D×D` matrix (real conjugate-transpose).
fn transpose(a: &[i64], d: usize) -> Vec<i64> {
    let mut t = vec![0i64; d * d];
    for row in 0..d {
        for col in 0..d {
            t[col * d + row] = a[row * d + col];
        }
    }
    t
}

/// `A·B` for row-major `D×D` matrices.
fn matmul(a: &[i64], b: &[i64], d: usize) -> Vec<i64> {
    let mut m = vec![0i64; d * d];
    for row in 0..d {
        for k in 0..d {
            let aik = a[row * d + k];
            if aik == 0 {
                continue;
            }
            for col in 0..d {
                m[row * d + col] += aik * b[k * d + col];
            }
        }
    }
    m
}

/// `[A, B] = A·B − B·A`.
fn commutator(a: &[i64], b: &[i64], d: usize) -> Vec<i64> {
    let ab = matmul(a, b, d);
    let ba = matmul(b, a, d);
    (0..d * d).map(|p| ab[p] - ba[p]).collect()
}

/// Frobenius inner product of two diagonals `Σ_k u_k v_k` (both `Sz` are
/// diagonal, so only diagonals contribute).
fn fro_diag(u: &[i64], v: &[i64]) -> i64 {
    u.iter().zip(v).map(|(&a, &b)| a * b).sum()
}

#[cfg(test)]
mod tests;
