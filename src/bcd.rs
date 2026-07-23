//! Exact SO(N)/Sp(2N) representation combinatorics for the B, C, D Cartan
//! series (Layer S3.0 of the `cgc-gen` track; design authority: issue #18
//! rulings, spec: issue #19).
//!
//! Pure integer/rational arithmetic — no floats anywhere in this module. It
//! provides irrep label types, exact Weyl dimensions, duals, Frobenius–Schur
//! indicators, exact weight multiplicities (Freudenthal recursion) and the
//! exact tensor-product decomposition `N^c_ab` (Brauer–Klimyk / Racah–Speiser
//! over Weyl characters). This is the production `N`-symbol that the numeric
//! sweep (S3.2) is checked against (`M^sweep == N^exact`, Ruling 1).
//!
//! # Published object (issue #18, Ruling 3)
//!
//! The object is the set of finite-dimensional **linear** representations of
//! the compact groups SO(2r+1) (series `B_r`), Sp(2r) (series `C_r`) and
//! SO(2r) (series `D_r`). Their irreps are exactly the **tensor** irreps —
//! integer highest weights in the orthonormal (ε) basis. Spinor
//! representations are representations of the covering group Spin(N), not of
//! SO(N); they are out of scope *by definition of the object*, not as a first
//! cut. A spinor Dynkin label is rejected with
//! [`BcdError::SpinorLabel`](crate::bcd::BcdError::SpinorLabel).
//!
//! # Conventions and normalization
//!
//! An [`Irrep`](crate::bcd::Irrep) stores the highest weight as an integer **partition** `λ` in
//! the orthonormal ε-basis (Bourbaki/Fulton–Harris convention), length `r`:
//!
//! - `B_r`, `C_r`: `λ₁ ≥ λ₂ ≥ … ≥ λ_r ≥ 0`.
//! - `D_r`: `λ₁ ≥ … ≥ λ_{r-1} ≥ |λ_r|`, and `λ_r` may be negative — the sign
//!   of `λ_r` is the D-series chirality (the two `±λ_r` labels are the
//!   analog of the two spinor chiralities, but here for tensor irreps).
//!
//! Integer **Dynkin** labels `a = (a₁,…,a_r)`, `aᵢ = ⟨λ, αᵢ^∨⟩`, relate to the
//! partition by (Fulton–Harris §18.1, roots/coroots; cross-check against the
//! QSpace `findMaxWeight` z→Dynkin maps, `clebsch_aux.cc:977–1031`):
//!
//! - `B_r`: `aᵢ = λᵢ − λ_{i+1}` (`i<r`), `a_r = 2λ_r`. Tensor ⇔ `a_r` even
//!   (`a_r` odd is the spinor `ω_r`).
//! - `C_r`: `aᵢ = λᵢ − λ_{i+1}` (`i<r`), `a_r = λ_r`. Every non-negative
//!   integer Dynkin label is a tensor irrep (Sp(2r) is simply connected).
//! - `D_r`: `aᵢ = λᵢ − λ_{i+1}` (`i≤r-2`), `a_{r-1} = λ_{r-1} − λ_r`,
//!   `a_r = λ_{r-1} + λ_r`. Tensor ⇔ `a_{r-1} ≡ a_r (mod 2)` (odd sum is the
//!   spinor lattice).
//!
//! # Excluded low ranks (guard inventory, issue #15; QSpace
//! `clebsch_aux.cc:990/1001/1018`)
//!
//! - `B_1 = SO(3) ≅ SU(2)` — rejected, redirect to SU(2).
//! - `C_1 = Sp(2) ≅ SU(2)` — rejected, redirect to SU(2).
//! - `D_2 = SO(4) ≅ SU(2)×SU(2)` — rejected, redirect to SU(2)×SU(2).
//!
//! # References
//!
//! - W. Fulton, J. Harris, *Representation Theory* (GTM 129), §§18, 24
//!   (root systems B/C/D, the Weyl dimension formula, weight multiplicities).
//! - J. Humphreys, *Introduction to Lie Algebras and Representation Theory*,
//!   §13.4 (Freudenthal's recursion), §22.3 / §24 (character arithmetic,
//!   the Racah–Speiser / Brauer–Klimyk sign rule via the dot action of the
//!   Weyl group).
//! - QSpace v4 (Weichselbaum), `Source/clebsch_aux.cc` at revision `dd2cc7e`
//!   (the revision all `clebsch_aux.cc:LINE` citations in this module refer
//!   to): `wdim_C/B/D` (`:458/486/524`) and `findMaxWeight` label maps and
//!   low-rank redirects (`:957–1045`, guards at `:990/1001/1018`) — the
//!   numerical oracle whose dimension values this module reproduces.
//!
//! # Example
//!
//! F/R generation for B/C/D runs through a per-(series, rank)
//! [`CanonicalCatalog`](crate::bcd::CanonicalCatalog)
//! that caches the aligned CGC. This computes an Sp(4) (`C_2`) F-symbol block;
//! with `a` trivial it is the `1×1×1×1` identity (value 1):
//!
//! ```
//! use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};
//!
//! let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4)
//! let triv = Irrep::trivial(Series::C, 2).unwrap();
//! let v = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // vector
//! let adj = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap(); // in v ⊗ v
//!
//! let block = f_symbol(&mut cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
//! assert_eq!(block.dims(), [1, 1, 1, 1]);
//! assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::One;

/// The three orthogonal/symplectic Cartan series covered here.
///
/// `B_r = SO(2r+1)`, `C_r = Sp(2r)`, `D_r = SO(2r)`. The name `bcd` is used
/// for the module (rather than `son`) because `C = Sp` is *not* an SO series;
/// only the Cartan-letter naming covers all three families honestly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Series {
    /// `B_r = SO(2r+1)` (odd orthogonal).
    B,
    /// `C_r = Sp(2r)` (symplectic).
    C,
    /// `D_r = SO(2r)` (even orthogonal).
    D,
}

impl Series {
    fn name(self) -> &'static str {
        match self {
            Series::B => "B (SO(2r+1))",
            Series::C => "C (Sp(2r))",
            Series::D => "D (SO(2r))",
        }
    }

    /// The minimum rank at which the series is not an excluded isomorphism.
    fn min_rank(self) -> usize {
        match self {
            Series::B | Series::C => 2,
            Series::D => 3,
        }
    }

    /// Redirection guidance for the excluded low rank.
    fn low_rank_redirect(self) -> &'static str {
        match self {
            // B_1 = SO(3) ≅ SU(2); C_1 = Sp(2) ≅ SU(2).
            Series::B | Series::C => "use SU(2) instead",
            // D_2 = SO(4) ≅ SU(2)×SU(2).
            Series::D => "use SU(2)×SU(2) instead",
        }
    }
}

/// Error for a malformed or out-of-scope B/C/D irrep label, or an ill-posed
/// product. Public constructors never panic; they return this instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BcdError {
    /// A rank-0 (empty) Dynkin label.
    EmptyLabel,
    /// A Dynkin label with a negative component (dominant integral weights
    /// have non-negative Dynkin labels).
    NegativeDynkin {
        /// The offending Dynkin label.
        dynkin: Vec<i64>,
    },
    /// A rank that is one of the excluded low-rank isomorphisms
    /// (`SO(3)`, `Sp(2)`, `SO(4)`), which are handled by the SU(2) machinery.
    /// Carries redirection guidance (guard inventory, issue #15).
    ExcludedRank {
        /// The series.
        series: Series,
        /// The offending rank `r`.
        rank: usize,
        /// Where to go instead (e.g. `"use SU(2) instead"`).
        redirect: &'static str,
    },
    /// A Dynkin label that denotes a spinor representation of `Spin(N)`, not a
    /// tensor (linear) representation of `SO(N)`. Out of scope by Ruling 3.
    /// (`B_r`: `a_r` odd; `D_r`: `a_{r-1} + a_r` odd.)
    SpinorLabel {
        /// The series.
        series: Series,
        /// The offending Dynkin label.
        dynkin: Vec<i64>,
    },
    /// A [`directproduct`] of irreps from different series or ranks (distinct
    /// groups share no product): an ill-posed input, not a zero fusion.
    GroupMismatch {
        /// The first operand as `(series, rank)`.
        a: (Series, usize),
        /// The second operand as `(series, rank)`.
        b: (Series, usize),
    },
    /// A defining-rep generator seed failed the exact commutator self-check
    /// ([`check_commutators`]) — the Rust analogue of QSpace's `checkCommRel`
    /// error. Names the violated relation and the generator indices involved.
    CommutatorViolation {
        /// The series.
        series: Series,
        /// The violated relation (e.g. `"cartan not mutually orthogonal"`).
        relation: &'static str,
        /// First generator index.
        i: usize,
        /// Second generator index.
        j: usize,
    },
}

impl fmt::Display for BcdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BcdError::EmptyLabel => write!(f, "B/C/D irrep label must be non-empty"),
            BcdError::NegativeDynkin { dynkin } => {
                write!(f, "Dynkin label has a negative component: {dynkin:?}")
            }
            BcdError::ExcludedRank {
                series,
                rank,
                redirect,
            } => write!(
                f,
                "series {} rank {rank} is an excluded low-rank isomorphism: {redirect}",
                series.name()
            ),
            BcdError::SpinorLabel { series, dynkin } => write!(
                f,
                "Dynkin label {dynkin:?} is a spinor of Spin(N), not a tensor irrep of \
                 series {} — spinors belong to the covering group and are out of scope",
                series.name()
            ),
            BcdError::GroupMismatch { a, b } => write!(
                f,
                "directproduct across distinct groups {:?} and {:?}",
                a, b
            ),
            BcdError::CommutatorViolation {
                series,
                relation,
                i,
                j,
            } => write!(
                f,
                "series {} defining-seed commutator self-check failed: {relation} \
                 (generators i={i}, j={j})",
                series.name()
            ),
        }
    }
}

impl std::error::Error for BcdError {}

/// An irreducible tensor representation of `SO(2r+1)`, `Sp(2r)` or `SO(2r)`,
/// labelled by its highest weight (an integer partition in the ε-basis; see
/// module docs for the normalization and chirality convention).
///
/// `Ord`/`Hash` are on `(series, weight)`, so two `Irrep`s are equal iff they
/// denote the same irrep; the order is deterministic (used as a map key).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Irrep {
    series: Series,
    /// Highest weight `λ` in the ε-basis, length `r`.
    weight: Box<[i64]>,
}

impl Irrep {
    /// Construct from the `r` integer Dynkin labels of `series` (`r =
    /// dynkin.len()`).
    ///
    /// Rejects: an empty label ([`BcdError::EmptyLabel`]); a negative component
    /// ([`BcdError::NegativeDynkin`]); an excluded low rank
    /// ([`BcdError::ExcludedRank`], with redirection); a spinor label
    /// ([`BcdError::SpinorLabel`]).
    pub fn from_dynkin(series: Series, dynkin: &[i64]) -> Result<Self, BcdError> {
        if dynkin.is_empty() {
            return Err(BcdError::EmptyLabel);
        }
        let r = dynkin.len();
        if r < series.min_rank() {
            return Err(BcdError::ExcludedRank {
                series,
                rank: r,
                redirect: series.low_rank_redirect(),
            });
        }
        if dynkin.iter().any(|&a| a < 0) {
            return Err(BcdError::NegativeDynkin {
                dynkin: dynkin.to_vec(),
            });
        }
        // Tensor-irrep (integer-partition) constraint; violation is a spinor.
        let spinor = match series {
            Series::B => dynkin[r - 1] % 2 != 0,
            Series::C => false,
            Series::D => (dynkin[r - 2] + dynkin[r - 1]) % 2 != 0,
        };
        if spinor {
            return Err(BcdError::SpinorLabel {
                series,
                dynkin: dynkin.to_vec(),
            });
        }
        let weight = dynkin_to_partition(series, dynkin);
        Ok(Irrep {
            series,
            weight: weight.into_boxed_slice(),
        })
    }

    /// The trivial (vacuum) irrep of `series` at rank `r` — the zero weight.
    pub fn trivial(series: Series, r: usize) -> Result<Self, BcdError> {
        Self::from_dynkin(series, &vec![0i64; r])
    }

    /// Construct directly from an ε-basis integer partition `weight`, bypassing
    /// the Dynkin validation. `pub(crate)` for the S3.3 catalog's bounded-dim
    /// irrep enumeration (`bcd::catalog`), which only ever produces valid
    /// integer dominant weights of this family; not a public constructor
    /// (the public path is [`Irrep::from_dynkin`], which validates).
    pub(crate) fn from_weight(series: Series, weight: Vec<i64>) -> Irrep {
        Irrep {
            series,
            weight: weight.into_boxed_slice(),
        }
    }

    /// The series (`B`, `C` or `D`).
    pub fn series(&self) -> Series {
        self.series
    }

    /// The rank `r`.
    pub fn rank(&self) -> usize {
        self.weight.len()
    }

    /// The highest weight `λ` as an integer partition in the ε-basis.
    pub fn partition(&self) -> &[i64] {
        &self.weight
    }

    /// The `r` integer Dynkin labels.
    pub fn dynkin(&self) -> Vec<i64> {
        partition_to_dynkin(self.series, &self.weight)
    }

    /// The exact Weyl dimension.
    ///
    /// Computed from the Weyl dimension formula (Fulton–Harris §24.3,
    /// eq. 24.30) `dim = ∏_{α>0} ⟨λ+ρ,α⟩ / ⟨ρ,α⟩`, evaluated exactly over the
    /// positive roots as a `Ratio<BigInt>` (the product is integral). This
    /// reproduces the QSpace values `wdim_B/C/D` (`clebsch_aux.cc:458–559`).
    pub fn dim(&self) -> BigInt {
        let r = self.rank();
        let two_rho = two_rho(self.series, r);
        let mut acc = Ratio::<BigInt>::one();
        for alpha in positive_roots(self.series, r) {
            // ⟨λ+ρ,α⟩ / ⟨ρ,α⟩ = (2⟨λ,α⟩ + ⟨2ρ,α⟩) / ⟨2ρ,α⟩ — all integers.
            let two_lam = 2 * dot(&self.weight, &alpha);
            let two_rho_a = dot(&two_rho, &alpha);
            acc *= Ratio::new(BigInt::from(two_lam + two_rho_a), BigInt::from(two_rho_a));
        }
        acc.to_integer()
    }

    /// The dual (complex-conjugate) irrep.
    ///
    /// Derivation (Fulton–Harris §26; Bourbaki, `-w₀` = the diagram
    /// automorphism): the dual highest weight is `-w₀(λ)`, where `w₀` is the
    /// longest Weyl element.
    ///
    /// - `B_r`, `C_r`, and `D_r` with `r` **even**: `-w₀ = 1`, so every tensor
    ///   irrep is **self-dual**.
    /// - `D_r` with `r` **odd**: `-w₀` is the order-2 diagram automorphism that
    ///   swaps the last two nodes, i.e. `λ_r ↦ -λ_r` in the ε-basis
    ///   (equivalently, swap the last two Dynkin labels). This is the chirality
    ///   flip: `so(6) = D_3` has `dual((0,2,0)) = (0,0,2)` (a tensor chiral
    ///   pair) and the vector `(1,0,0)` self-dual.
    pub fn dual(&self) -> Irrep {
        let mut w = self.weight.to_vec();
        if self.series == Series::D && self.rank() % 2 == 1 {
            let last = self.rank() - 1;
            w[last] = -w[last];
        }
        Irrep {
            series: self.series,
            weight: w.into_boxed_slice(),
        }
    }

    /// The Frobenius–Schur indicator: `+1` (real/orthogonal), `-1`
    /// (quaternionic/symplectic), or `0` (complex, i.e. not self-dual).
    ///
    /// Derivation:
    /// - **`B_r`, `D_r` (`SO(N)`)**: every tensor irrep is realized inside a
    ///   tensor power of the *real* defining (vector) representation, hence is
    ///   real. So the indicator is `+1` for self-dual labels and `0` for the
    ///   non-self-dual `D_r` (`r` odd, `λ_r ≠ 0`) chiral pair. No tensor irrep
    ///   of `SO(N)` is quaternionic.
    /// - **`C_r` (`Sp(2r)`)**: self-dual, so `±1`. The value is the sign by
    ///   which the central element `-I ∈ Sp(2r)` acts, which is
    ///   `(-1)^{Σ_i λ_i}`. In Dynkin labels `Σ_i λ_i = Σ_{j=1}^r j·a_j`, whose
    ///   parity equals that of `Σ_{j odd} a_j`; hence the irrep is
    ///   **quaternionic iff the sum of the odd-position Dynkin labels is odd**
    ///   (matching the standard `Sp(2r)` reality rule; the vector `(1,0,…)` is
    ///   quaternionic and the adjoint `(2,0,…)` is real).
    pub fn frobenius_schur(&self) -> i32 {
        match self.series {
            Series::B => 1,
            Series::D => {
                if *self == self.dual() {
                    1
                } else {
                    0
                }
            }
            Series::C => {
                let sum_lambda: i64 = self.weight.iter().sum();
                if sum_lambda % 2 == 0 {
                    1
                } else {
                    -1
                }
            }
        }
    }

    /// Exact dominant-weight multiplicities of this irrep, computed by
    /// Freudenthal's recursion (Humphreys §13.4) in integer arithmetic.
    ///
    /// Keys are dominant weights `μ` (ε-basis) with `μ ≤ λ`; values are their
    /// multiplicities `m_λ(μ) ≥ 1`. Every weight of the irrep is a Weyl-image
    /// of exactly one key, with the same multiplicity.
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        freudenthal(self.series, &self.weight)
    }
}

// ---- label ↔ partition maps ----------------------------------------------

/// Partition (ε-basis) `λ` from Dynkin labels `a`. Assumes the tensor
/// constraint already validated (so all `λ` are integers).
fn dynkin_to_partition(series: Series, a: &[i64]) -> Vec<i64> {
    let r = a.len();
    let mut lam = vec![0i64; r];
    match series {
        Series::B => {
            // λ_r = a_r/2, λ_i = λ_{i+1} + a_i.
            lam[r - 1] = a[r - 1] / 2;
            for i in (0..r - 1).rev() {
                lam[i] = lam[i + 1] + a[i];
            }
        }
        Series::C => {
            // λ_r = a_r, λ_i = λ_{i+1} + a_i.
            lam[r - 1] = a[r - 1];
            for i in (0..r - 1).rev() {
                lam[i] = lam[i + 1] + a[i];
            }
        }
        Series::D => {
            // λ_{r-1} = (a_{r-1}+a_r)/2, λ_r = (a_r-a_{r-1})/2, λ_i = λ_{i+1}+a_i.
            lam[r - 1] = (a[r - 1] - a[r - 2]) / 2;
            lam[r - 2] = (a[r - 1] + a[r - 2]) / 2;
            for i in (0..r - 2).rev() {
                lam[i] = lam[i + 1] + a[i];
            }
        }
    }
    lam
}

/// Dynkin labels `a` from a partition `λ` (inverse of [`dynkin_to_partition`]).
fn partition_to_dynkin(series: Series, lam: &[i64]) -> Vec<i64> {
    let r = lam.len();
    let mut a = vec![0i64; r];
    for i in 0..r - 1 {
        a[i] = lam[i] - lam[i + 1];
    }
    match series {
        Series::B => a[r - 1] = 2 * lam[r - 1],
        Series::C => a[r - 1] = lam[r - 1],
        Series::D => {
            a[r - 2] = lam[r - 2] - lam[r - 1];
            a[r - 1] = lam[r - 2] + lam[r - 1];
        }
    }
    a
}

// ---- root system in the ε-basis ------------------------------------------

/// Euclidean inner product of two ε-basis integer vectors (roots/weights are
/// carried as their ε-coefficient vectors, e.g. `2e_i` as a `2` in slot `i`).
fn dot(u: &[i64], v: &[i64]) -> i64 {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

/// `2ρ` (twice the Weyl vector, integer-valued) in the ε-basis:
/// `B: 2r-2i+1`, `C: 2r-2i+2`, `D: 2r-2i` (`i = 1..r`).
fn two_rho(series: Series, r: usize) -> Vec<i64> {
    (0..r)
        .map(|i0| {
            let i = i0 as i64 + 1;
            let rr = r as i64;
            match series {
                Series::B => 2 * rr - 2 * i + 1,
                Series::C => 2 * rr - 2 * i + 2,
                Series::D => 2 * rr - 2 * i,
            }
        })
        .collect()
}

/// The positive roots of the series in the ε-basis, as integer coefficient
/// vectors (Fulton–Harris §18). Common to all: `e_i − e_j`, `e_i + e_j`
/// (`i<j`); plus `e_i` for `B`, `2e_i` for `C`, none for `D`.
fn positive_roots(series: Series, r: usize) -> Vec<Vec<i64>> {
    let mut roots = Vec::new();
    for i in 0..r {
        for j in i + 1..r {
            let mut minus = vec![0i64; r];
            minus[i] = 1;
            minus[j] = -1;
            roots.push(minus);
            let mut plus = vec![0i64; r];
            plus[i] = 1;
            plus[j] = 1;
            roots.push(plus);
        }
    }
    match series {
        Series::B => {
            for i in 0..r {
                let mut e = vec![0i64; r];
                e[i] = 1;
                roots.push(e);
            }
        }
        Series::C => {
            for i in 0..r {
                let mut e = vec![0i64; r];
                e[i] = 2;
                roots.push(e);
            }
        }
        Series::D => {}
    }
    roots
}

// ---- Weyl-group dominant conjugation -------------------------------------

/// Sort a vector descending, returning the sorted vector and the sign of the
/// sorting permutation (`+1`/`-1`). For distinct entries this is `sgn(perm)`;
/// callers that need the sign guarantee distinctness first.
fn sort_desc_with_parity(v: &[i64]) -> (Vec<i64>, i32) {
    let n = v.len();
    let mut inv = 0usize;
    for i in 0..n {
        for j in i + 1..n {
            if v[i] < v[j] {
                inv += 1;
            }
        }
    }
    let mut out = v.to_vec();
    out.sort_unstable_by(|a, b| b.cmp(a));
    (out, if inv.is_multiple_of(2) { 1 } else { -1 })
}

/// The dominant Weyl-orbit representative of a weight `v` (ε-basis), ignoring
/// singularity (weights may lie on walls; the dominant representative is still
/// well defined). Used for Freudenthal multiplicity lookups.
///
/// - `B`/`C`: `|v|` sorted descending (Weyl group = all signed permutations).
/// - `D`: `|v|` sorted descending, last entry negated iff an odd number of
///   components were negative **and** the smallest `|v|` is non-zero (Weyl
///   group = *even* sign changes; a zero component frees the parity).
fn weyl_dominant(series: Series, v: &[i64]) -> Vec<i64> {
    let negcount = v.iter().filter(|&&x| x < 0).count();
    let absv: Vec<i64> = v.iter().map(|x| x.abs()).collect();
    let (mut sorted, _) = sort_desc_with_parity(&absv);
    if series == Series::D && !negcount.is_multiple_of(2) {
        let last = sorted.len() - 1;
        if sorted[last] != 0 {
            sorted[last] = -sorted[last];
        }
    }
    sorted
}

/// The dominant conjugate of a **ρ-shifted** vector `two_v = 2(a+ρ+μ)`
/// (carried at twice scale so `B`'s half-integer `ρ` stays integral),
/// together with `det(w) = ±1`, or `None` if `two_v` is Weyl-singular (lies on
/// a reflection wall — contributes `0` to the Racah–Speiser sum).
///
/// Singular ⇔ two components equal in absolute value (wall `e_i ± e_j`), or,
/// for `B`/`C`, a zero component (wall `e_i` resp. `2e_i`).
fn dominant_conjugate_signed(series: Series, two_v: &[i64]) -> Option<(Vec<i64>, i32)> {
    let negcount = two_v.iter().filter(|&&x| x < 0).count();
    let absv: Vec<i64> = two_v.iter().map(|x| x.abs()).collect();
    // Wall e_i±e_j: two equal absolute values.
    for i in 0..absv.len() {
        for j in i + 1..absv.len() {
            if absv[i] == absv[j] {
                return None;
            }
        }
    }
    let (mut sorted, perm_sign) = sort_desc_with_parity(&absv);
    match series {
        Series::B | Series::C => {
            // Wall e_i (B) / 2e_i (C): a zero component.
            if absv.contains(&0) {
                return None;
            }
            // det(w) = sgn(perm) · (-1)^{#flips}, #flips = #negatives.
            let sign = perm_sign * if negcount.is_multiple_of(2) { 1 } else { -1 };
            Some((sorted, sign))
        }
        Series::D => {
            // det(w) = sgn(perm) (even sign changes have det +1). Choose the
            // last sign to match the even-flip parity of the orbit: negative
            // iff #negatives is odd and the smallest |·| is non-zero.
            let last = sorted.len() - 1;
            if !negcount.is_multiple_of(2) && sorted[last] != 0 {
                sorted[last] = -sorted[last];
            }
            Some((sorted, perm_sign))
        }
    }
}

// ---- Freudenthal weight multiplicities -----------------------------------

/// Exact dominant-weight multiplicities of the irrep with highest weight `λ`,
/// by Freudenthal's recursion (Humphreys §13.4), in integer arithmetic.
///
/// `ponytail:` weight coordinates and inner products are tiny for the ranks in
/// scope; multiplicities are accumulated in `i128`. Upgrade to `BigInt` here
/// only if an application drives rank/label high enough to overflow (weights
/// would have to reach thousands).
fn freudenthal(series: Series, lambda: &[i64]) -> BTreeMap<Vec<i64>, u64> {
    let r = lambda.len();
    let two_rho = two_rho(series, r);
    let roots = positive_roots(series, r);

    // Dominant weights μ ≤ λ (same root lattice), with their depth = height of
    // λ-μ in simple roots. Enumerate a box of dominant weights and keep those
    // with λ-μ a non-negative *integer* combination of simple roots.
    let mut doms: Vec<(i64, Vec<i64>)> = enumerate_dominant_below(series, lambda)
        .into_iter()
        .map(|mu| (depth(series, lambda, &mu), mu))
        .collect();
    doms.sort();

    // ⟨λ+ρ,λ+ρ⟩ contribution that survives the difference: ⟨λ,λ⟩ + ⟨λ,2ρ⟩.
    let casimir = |w: &[i64]| -> i128 { (dot(w, w) + dot(w, &two_rho)) as i128 };
    let cas_lambda = casimir(lambda);

    let mut mult: BTreeMap<Vec<i64>, u64> = BTreeMap::new();
    for (_, mu) in &doms {
        if mu == lambda {
            mult.insert(mu.clone(), 1);
            continue;
        }
        let denom = cas_lambda - casimir(mu);
        debug_assert!(denom > 0, "Freudenthal denominator must be positive");
        let mut num: i128 = 0;
        for alpha in &roots {
            let aa = dot(alpha, alpha) as i128;
            let mu_a = dot(mu, alpha) as i128;
            let mut k: i128 = 1;
            loop {
                // μ + kα
                let shifted: Vec<i64> = mu
                    .iter()
                    .zip(alpha)
                    .map(|(&m, &al)| m + (k as i64) * al)
                    .collect();
                let dom = weyl_dominant(series, &shifted);
                match mult.get(&dom) {
                    Some(&m) if m > 0 => {
                        num += 2 * (mu_a + k * aa) * (m as i128);
                        k += 1;
                    }
                    _ => break,
                }
            }
        }
        debug_assert_eq!(num % denom, 0, "Freudenthal must divide exactly");
        let m = num / denom;
        if m > 0 {
            mult.insert(mu.clone(), m as u64);
        }
    }
    mult
}

/// Height of `λ − μ` in the simple-root basis (its coefficient sum), assuming
/// `μ ≤ λ` so all coefficients are non-negative integers.
fn depth(series: Series, lambda: &[i64], mu: &[i64]) -> i64 {
    let d: Vec<i64> = lambda.iter().zip(mu).map(|(&l, &m)| l - m).collect();
    simple_root_coeffs(series, &d)
        .map(|c| c.iter().sum())
        .unwrap_or(-1)
}

/// Coefficients `c` with `d = Σ cᵢ αᵢ` (simple roots, ε-basis), or `None` if
/// `d` is not a non-negative integer combination. Closed forms from the
/// simple-root structure (Fulton–Harris §18):
/// - `B`/`C`/`D` share `cⱼ = Σ_{i≤j} dᵢ` for the `e_i − e_{i+1}` part;
///   the short/spin root closes the last one(s).
fn simple_root_coeffs(series: Series, d: &[i64]) -> Option<Vec<i64>> {
    let r = d.len();
    // Prefix sums = coefficients of e_i - e_{i+1} chain.
    let mut c = vec![0i64; r];
    let mut acc = 0i64;
    for i in 0..r {
        acc += d[i];
        c[i] = acc; // provisional; last one(s) fixed per series below
    }
    let total: i64 = d.iter().sum();
    match series {
        Series::B => {
            // α_r = e_r, c_r = Σ d_i = total (already c[r-1]).
        }
        Series::C => {
            // α_r = 2e_r, c_r = total/2.
            if total % 2 != 0 {
                return None;
            }
            c[r - 1] = total / 2;
        }
        Series::D => {
            // α_r = e_{r-1}+e_r: c_r = total/2, c_{r-1} = total/2 - d_r.
            if total % 2 != 0 {
                return None;
            }
            c[r - 1] = total / 2;
            c[r - 2] = total / 2 - d[r - 1];
        }
    }
    if c.iter().all(|&x| x >= 0) {
        Some(c)
    } else {
        None
    }
}

/// All dominant weights `μ` with `μ ≤ λ` (dominance) and `λ − μ` in the root
/// lattice, by enumerating a bounded box of dominant partitions and filtering
/// with [`simple_root_coeffs`].
fn enumerate_dominant_below(series: Series, lambda: &[i64]) -> Vec<Vec<i64>> {
    let r = lambda.len();
    let hi = lambda[0]; // μ ≤ λ ⇒ μ₁ ≤ λ₁; all |μ_i| ≤ λ₁.
    let mut out = Vec::new();
    let mut cur = vec![0i64; r];
    enum_dom_rec(series, lambda, hi, 0, &mut cur, &mut out);
    out
}

fn enum_dom_rec(
    series: Series,
    lambda: &[i64],
    hi: i64,
    pos: usize,
    cur: &mut Vec<i64>,
    out: &mut Vec<Vec<i64>>,
) {
    let r = cur.len();
    if pos == r {
        if simple_root_coeffs(series, &sub(lambda, cur)).is_some() {
            out.push(cur.clone());
        }
        return;
    }
    // Dominant: μ_pos ≤ μ_{pos-1} (and ≤ hi). For D the last slot also allows
    // negatives down to -μ_{r-2} (chirality); for B/C the floor is 0.
    let upper = if pos == 0 { hi } else { cur[pos - 1] };
    let lower = if series == Series::D && pos == r - 1 {
        -cur[pos - 1]
    } else {
        0
    };
    for v in (lower..=upper).rev() {
        cur[pos] = v;
        enum_dom_rec(series, lambda, hi, pos + 1, cur, out);
    }
    cur[pos] = 0;
}

fn sub(a: &[i64], b: &[i64]) -> Vec<i64> {
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

// ---- Weyl orbit ----------------------------------------------------------

/// All distinct Weyl-group images of a dominant weight `mu` (ε-basis):
/// signed permutations for `B`/`C`, even-signed permutations for `D`.
fn weyl_orbit(series: Series, mu: &[i64]) -> Vec<Vec<i64>> {
    let r = mu.len();
    let mut set: HashSet<Vec<i64>> = HashSet::new();
    for signs in 0u32..(1u32 << r) {
        let flips = signs.count_ones() as usize;
        if series == Series::D && !flips.is_multiple_of(2) {
            continue;
        }
        let signed: Vec<i64> = (0..r)
            .map(|i| if signs & (1 << i) != 0 { -mu[i] } else { mu[i] })
            .collect();
        permute_into(&signed, &mut set);
    }
    set.into_iter().collect()
}

/// Insert every permutation of `v` into `set`.
fn permute_into(v: &[i64], set: &mut HashSet<Vec<i64>>) {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    permute_rec(v, &mut idx, 0, set);
}

fn permute_rec(v: &[i64], idx: &mut Vec<usize>, k: usize, set: &mut HashSet<Vec<i64>>) {
    let n = idx.len();
    if k == n {
        set.insert(idx.iter().map(|&i| v[i]).collect());
        return;
    }
    for i in k..n {
        idx.swap(k, i);
        permute_rec(v, idx, k + 1, set);
        idx.swap(k, i);
    }
}

// ---- Brauer–Klimyk / Racah–Speiser product decomposition -----------------

/// Exact tensor-product decomposition: the fusion multiplicities `N^c_ab` of
/// `a ⊗ b`, keyed by the resulting irrep `c`.
///
/// Requires `a` and `b` to label the same group (same series and rank); a
/// mismatch is an ill-posed input across distinct groups and returns
/// [`BcdError::GroupMismatch`].
///
/// Algorithm (Racah–Speiser / Brauer–Klimyk, Humphreys §24): for every weight
/// `μ` of `b` (multiplicity `m_b(μ)` from Freudenthal, expanded over its Weyl
/// orbit), form `ξ = a + μ + ρ`. If `ξ` is Weyl-singular it contributes `0`;
/// otherwise let `w` be the Weyl element making `ξ` dominant, and add
/// `det(w)·m_b(μ)` to the coefficient of the irrep with highest weight
/// `w(ξ) − ρ`. All arithmetic is exact integer; `ρ`-shifts are carried at
/// twice scale to keep `B`'s half-integer `ρ` integral.
pub fn directproduct(a: &Irrep, b: &Irrep) -> Result<BTreeMap<Irrep, u32>, BcdError> {
    if a.series != b.series || a.rank() != b.rank() {
        return Err(BcdError::GroupMismatch {
            a: (a.series, a.rank()),
            b: (b.series, b.rank()),
        });
    }
    let series = a.series;
    let r = a.rank();
    let two_rho = two_rho(series, r);
    // 2(a + ρ), constant across the μ loop.
    let two_a_rho: Vec<i64> = a
        .weight
        .iter()
        .zip(&two_rho)
        .map(|(&av, &tr)| 2 * av + tr)
        .collect();

    let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
    for (mu, &m) in &b.weight_multiplicities() {
        let m = m as i64;
        for omega in weyl_orbit(series, mu) {
            // 2ξ = 2(a+ρ) + 2μ.
            let two_xi: Vec<i64> = two_a_rho
                .iter()
                .zip(&omega)
                .map(|(&ar, &w)| ar + 2 * w)
                .collect();
            if let Some((dom, sign)) = dominant_conjugate_signed(series, &two_xi) {
                // c = (dom - 2ρ)/2.
                let c: Vec<i64> = dom
                    .iter()
                    .zip(&two_rho)
                    .map(|(&d, &tr)| (d - tr) / 2)
                    .collect();
                *acc.entry(c).or_insert(0) += sign as i64 * m;
            }
        }
    }

    let mut result: BTreeMap<Irrep, u32> = BTreeMap::new();
    for (c, n) in acc {
        debug_assert!(n >= 0, "Racah–Speiser multiplicity must be non-negative");
        if n > 0 {
            result.insert(
                Irrep {
                    series,
                    weight: c.into_boxed_slice(),
                },
                n as u32,
            );
        }
    }
    Ok(result)
}

mod seeds;
pub use seeds::{check_commutators, defining_seed, CommReport, Seed};

// The decomposition sweep (S3.2) and its dense linalg seam depend on
// `tenferro-linalg`, so they live behind the `cgc-gen` feature like the SU(N)
// CGC pipeline; the base crate stays dependency-light.
#[cfg(feature = "cgc-gen")]
mod linalg;
#[cfg(feature = "cgc-gen")]
mod sweep;
#[cfg(feature = "cgc-gen")]
pub use sweep::{
    decompose, decompose_defining_product, Block, Decomposition, Generators, SweepError,
};

// The S3.3 canonical catalog (append-only generator ownership) sits on top of
// the sweep, so it shares the `cgc-gen` feature gate.
#[cfg(feature = "cgc-gen")]
mod catalog;
#[cfg(feature = "cgc-gen")]
pub use catalog::{CanonicalCatalog, CatalogCgc, CatalogError};

// S3.4 (#27): the B/C/D binding of the family-generic F/R core, over the S3.3
// catalog's canonical CGC.
#[cfg(feature = "cgc-gen")]
mod fr;
#[cfg(feature = "cgc-gen")]
pub use fr::{
    cgc_sweeps, check_f_unitarity, check_hexagon, check_pentagon, f_symbol, r_symbol, FBlock,
    FrError, RBlock,
};

#[cfg(test)]
mod tests;

// S3.5 external anchor: QSpace CGC oracle, behind the factor-basis dictionary.
#[cfg(all(test, feature = "cgc-gen"))]
mod qspace_oracle_tests;
