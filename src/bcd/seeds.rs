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
//! (`Z[..]`) of these three Setups are **integers** — small integers of
//! magnitude `< r` on the Cartan diagonals (`SON`/`SEN` use only `±1`; `SpN`'s
//! `Z[i] = -i` reaches `-(r-1)`, e.g. `-3` at `C_4`), `+1` on every ladder
//! entry. There is *no* `sqrt(2)` short
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
use num_traits::{One, Zero};

use crate::group::GlobalForm;

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
    /// `Sz[i] = sz_scale · sz[i]`. `1` for every defining seed; `1/2` for the
    /// spinor seeds, whose Cartan eigenvalues are half-integers.
    sz_scale: Ratio<i64>,
    /// `Sp[i] = √(sp_scale2[i]) · sp[i]`. `1` everywhere except the `B_r`
    /// spinor seed's short-root generator, where it is `1/2` — see
    /// [`spinor_seeds`] for why the scale is a square root and why that is the
    /// only irrational entry in the layer.
    sp_scale2: Vec<Ratio<i64>>,
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
    /// The Cartan generators `Sz[0..r]`, each as its `D`-length integer
    /// diagonal of **records** — the operator is
    /// [`cartan_scale`](Self::cartan_scale) times this.
    pub fn cartan(&self) -> &[Vec<i64>] {
        &self.sz
    }

    /// The common rational scale of the [`cartan`](Self::cartan) records:
    /// `Sz[i] = cartan_scale · cartan()[i]`. `1` for every defining seed,
    /// `1/2` for a spinor seed.
    pub fn cartan_scale(&self) -> Ratio<i64> {
        self.sz_scale
    }

    /// The **squared** rational scale of raising operator `i`:
    /// `Sp[i] = √(raising_scale2(i)) · raising()[i]`. `1` for every defining
    /// seed and for every long-root generator; `1/2` for the `B_r` spinor
    /// seed's short-root generator.
    pub fn raising_scale2(&self, i: usize) -> Ratio<i64> {
        self.sp_scale2[i]
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
/// - QSpace `checkCommRel` `if (np!=nz || nz!=qt.sub) ERR` (the
///   `Sp`/`Sz`/`sub` arity check): **N/A** — [`Seed`] always builds exactly `r`
///   `Sp` and `r` `Sz` generators, so `#Sp = #Sz = rank` holds by construction.
pub fn defining_seed(series: Series, r: usize) -> Result<Seed, BcdError> {
    if r < series.min_rank() {
        return Err(BcdError::ExcludedRank {
            series,
            rank: r,
            // The seed layer works on the **cover** (it is where the spinor
            // base cases live), so the simply-connected redirect is the right
            // one: `B_1 = Spin(3) ≅ SU(2)`, `D_2 = Spin(4) ≅ SU(2)×SU(2)`.
            redirect: series.low_rank_redirect(GlobalForm::SimplyConnected),
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
        sz_scale: Ratio::one(),
        sp_scale2: vec![Ratio::one(); r],
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
        sz_scale: Ratio::one(),
        sp_scale2: vec![Ratio::one(); r],
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
        sz_scale: Ratio::one(),
        sp_scale2: vec![Ratio::one(); r],
    }
}

// ---- spinor base cases (issue #54; docs/gauge_soN.md §16) ------------------

/// The exact **spinor** base-case generator seeds of `series` at rank `r`, each
/// paired with the Dynkin label of the irrep it carries — the second base case
/// of the bootstrap (`docs/gauge_soN.md` §14.2, §16).
///
/// - `B_r`: one seed, the spinor `ω_r = (0,…,0,1)`, dimension `2^r`.
/// - `D_r`: two seeds, the half-spinors `ω_{r-1}` and `ω_r`, dimension
///   `2^{r-1}` each, returned in ascending Dynkin-label order.
/// - `C_r`: none — `Sp(2r)` is simply connected, it has no spinor sector.
///
/// Returns [`BcdError::ExcludedRank`] for the low-rank isomorphisms, exactly as
/// [`defining_seed`] does.
///
/// # The construction (normative: `docs/gauge_soN.md` §16)
///
/// The carrier is the fermionic Fock space of `r` modes — the Clifford module
/// of `so(N)`. A basis state is an occupation string `n ∈ {0,1}^r` stored at
/// index `m = Σ_k n_k 2^{k-1}` (mode `k` in bit `k-1`), and its ε-basis weight
/// is `λ_k = n_k − ½`. The generators are the standard Jordan–Wigner fermion
/// operators, placed on QSpace's node order (`Sp[i]` carries the simple root
/// `α_{r-i}` for `i < r`, `α_r` for `i = r`; `Sz[j]` measures `λ_{r+1-j}`; see
/// §7's `findMaxWeight` conversion):
///
/// ```text
/// Sz[j] = n_{r+1-j} − ½
/// Sp[i] = c†_{r-i} c_{r-i+1}        (i < r,  root ε_{r-i} − ε_{r-i+1})
/// Sp[r] = c†_r / √2                 (B_r,    root ε_r)
/// Sp[r] = c†_{r-1} c†_r             (D_r,    root ε_{r-1} + ε_r)
/// ```
///
/// For `D_r` the generators are all fermion-number-even, so the Fock space
/// splits into the two half-spin sectors by the parity of `Σ_k n_k`; each is
/// listed in ascending `m`.
///
/// **The one irrational scale.** The module docs above record that no defining
/// seed needs a `√2`. The `B_r` spinor seed does, and only there: QSpace's
/// `Sp[r]` is normalized so that `[Sp_r, Sp_r†] = Sz_1 = ½ α_r^∨` (read off the
/// defining seed, where the short-root string is a spin-1 triplet with matrix
/// elements `√2`). In the spinor the same string is a doublet with matrix
/// elements `1`, so the seed entry is `1/√2`. It is carried exactly as the
/// squared rational scale [`Seed::raising_scale2`], which is all the
/// self-check ever needs: the scale cancels in `[Sz_j, Sp_i] = d_{i,j} Sp_i`
/// and enters `[Sp_i, Sp_i†] = Σ_k f_{i,k} Sz_k` only quadratically.
pub fn spinor_seeds(series: Series, r: usize) -> Result<Vec<(Vec<i64>, Seed)>, BcdError> {
    if r < series.min_rank() {
        return Err(BcdError::ExcludedRank {
            series,
            rank: r,
            redirect: series.low_rank_redirect(GlobalForm::SimplyConnected),
        });
    }
    let mut out = match series {
        Series::C => Vec::new(),
        Series::B => vec![fock_seed(Series::B, r, None)],
        Series::D => vec![
            fock_seed(Series::D, r, Some(0)),
            fock_seed(Series::D, r, Some(1)),
        ],
    };
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Annihilate mode `k` (1-based) in occupation string `m`, with the
/// Jordan–Wigner sign `(-1)^{Σ_{l<k} n_l}`; `None` if the mode is empty.
fn ann(m: usize, k: usize) -> Option<(usize, i64)> {
    let bit = 1usize << (k - 1);
    if m & bit == 0 {
        return None;
    }
    let below = (m & (bit - 1)).count_ones();
    Some((m ^ bit, if below.is_multiple_of(2) { 1 } else { -1 }))
}

/// Create mode `k` (1-based) in `m`, same sign rule; `None` if already filled.
fn cre(m: usize, k: usize) -> Option<(usize, i64)> {
    let bit = 1usize << (k - 1);
    if m & bit != 0 {
        return None;
    }
    let below = (m & (bit - 1)).count_ones();
    Some((m | bit, if below.is_multiple_of(2) { 1 } else { -1 }))
}

/// Build the spinor seed on the Fock space of `r` modes, restricted to the
/// fermion-parity `sector` when given (the two `D_r` half-spinors), and pair it
/// with the Dynkin label of its highest weight.
fn fock_seed(series: Series, r: usize, sector: Option<u32>) -> (Vec<i64>, Seed) {
    // Carrier basis: the occupation strings of the sector, in the sweep's
    // **descending-weight** order (§7) — the order every non-base generator set
    // is produced in, so a rediscovered spinor block is coherent with this seed
    // and the §15 coherence guard applies to it unchanged. That order is
    // lexicographic on the Cartan columns read in reverse, i.e. descending in
    // (λ_1, …, λ_r) = (n_1, …, n_r), which is descending in the bit-reversed
    // occupation string. A spinor's weights are non-degenerate, so there are no
    // ties and the tie-break rule never fires.
    let mut states: Vec<usize> = (0..1usize << r)
        .filter(|m| sector.is_none_or(|p| m.count_ones() % 2 == p))
        .collect();
    let rev_key = |m: &usize| -> std::cmp::Reverse<Vec<u8>> {
        std::cmp::Reverse((1..=r).map(|k| ((m >> (k - 1)) & 1) as u8).rev().collect())
    };
    states.sort_by_key(rev_key);
    let d = states.len();
    let mut pos = vec![usize::MAX; 1usize << r];
    for (p, &m) in states.iter().enumerate() {
        pos[m] = p;
    }

    // Sz[j] = n_{r+1-j} − ½, stored as the record 2λ = 2n−1 with scale ½.
    let sz: Vec<Vec<i64>> = (0..r)
        .map(|j0| {
            let bit = 1usize << (r - 1 - j0); // mode k = r-j0, i.e. bit k-1
            states
                .iter()
                .map(|&m| if m & bit != 0 { 1 } else { -1 })
                .collect()
        })
        .collect();

    // Sp[i] on QSpace's node order.
    let mut sp: Vec<Vec<(usize, usize, i64)>> = Vec::with_capacity(r);
    let mut sp_scale2 = vec![Ratio::<i64>::one(); r];
    for i in 1..=r {
        let mut recs = Vec::new();
        for (col, &m) in states.iter().enumerate() {
            let acted = if i < r {
                // c†_{r-i} c_{r-i+1}
                ann(m, r - i + 1).and_then(|(m1, s1)| cre(m1, r - i).map(|(m2, s2)| (m2, s1 * s2)))
            } else if series == Series::B {
                // c†_r (times 1/√2, carried in the scale)
                cre(m, r)
            } else {
                // c†_{r-1} c†_r
                cre(m, r).and_then(|(m1, s1)| cre(m1, r - 1).map(|(m2, s2)| (m2, s1 * s2)))
            };
            if let Some((m2, sign)) = acted {
                recs.push((pos[m2], col, sign));
            }
        }
        recs.sort_unstable();
        sp.push(recs);
    }
    if series == Series::B {
        sp_scale2[r - 1] = Ratio::new(1, 2);
    }

    let seed = Seed {
        series,
        rank: r,
        dim: d,
        sp,
        sz,
        sz_scale: Ratio::new(1, 2),
        sp_scale2,
    };
    (highest_weight_dynkin(&seed, &states, r), seed)
}

/// The Dynkin label of the seed's highest weight: the unique carrier state
/// annihilated by every raising operator. Derived from the built matrices
/// rather than asserted, so a mis-built seed cannot silently claim a label.
///
/// Panics only on an internal inconsistency (no unique highest-weight state),
/// which is unreachable for an irreducible carrier — a loud invariant, in the
/// sense of the guard inventory.
fn highest_weight_dynkin(seed: &Seed, states: &[usize], r: usize) -> Vec<i64> {
    let mut hw: Option<usize> = None;
    for col in 0..states.len() {
        if seed
            .sp
            .iter()
            .all(|recs| !recs.iter().any(|&(_, c, _)| c == col))
        {
            assert!(hw.is_none(), "spinor seed carrier is not irreducible");
            hw = Some(col);
        }
    }
    let hw = hw.expect("an irreducible carrier has a highest-weight state");
    let m = states[hw];
    // 2λ_k = 2n_k − 1.
    let two_lambda: Vec<i64> = (1..=r)
        .map(|k| if m & (1usize << (k - 1)) != 0 { 1 } else { -1 })
        .collect();
    super::two_partition_to_dynkin(seed.series, &two_lambda)
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
        // f_{i,k} = ⟨C, Sz_k⟩_F / ⟨Sz_k, Sz_k⟩_F  (Frobenius projection). The
        // projection is done on the integer **records**, so the residual below
        // is exact and scale-free; the reported coefficient then carries the
        // scales, `f = (g_i / s) · f_records` with `Sp_i = √g_i · sp_i` and
        // `Sz_k = s · sz_k`.
        let c_diag: Vec<i64> = (0..d).map(|p| c[p * d + p]).collect();
        let scale = seed.sp_scale2[i] / seed.sz_scale;
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
        for f in cartan_coeffs[i].iter_mut() {
            *f *= scale;
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
            // On the integer records; the reported root component carries the
            // Cartan scale (the `Sp` scale cancels between the two sides).
            let dz = Ratio::new(bc[r0 * d + c0], v0);
            root_weights[i][j] = dz * seed.sz_scale;
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
