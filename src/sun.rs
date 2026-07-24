//! SU(N) irreps and their Clebsch–Gordan / recoupling coefficients, built by
//! the Gelfand–Tsetlin (GT) construction.
//!
//! An [`Irrep`](crate::sun::Irrep) is an SU(N) highest weight; from it this
//! module gives the Weyl [`dimension`](crate::sun::Irrep::dim), the
//! [`dual`](crate::sun::Irrep::dual), the GT basis
//! ([`patterns`](crate::sun::Irrep::patterns)), Littlewood–Richardson products (fusion
//! multiplicities), and — the point of the module — the Clebsch–Gordan
//! coefficients [`cgc`](crate::sun::cgc) and the recoupling
//! [`f_symbol`](crate::sun::f_symbol) / [`r_symbol`](crate::sun::r_symbol),
//! with outer-multiplicity indices. Values are exact rationals (labels, dimensions,
//! GT ladder matrices) up to the CGC nullspace solve, which is
//! verification-gated floating point.
//!
//! The GT construction applies to SU(N) because the unitary chain
//! `U(N) ⊃ U(N-1) ⊃ … ⊃ U(1)` is multiplicity-free (the intermediate U(1) charge
//! separates copies the SU chain alone would repeat): GT patterns label basis
//! states of an SU(N) irrep uniquely, so the ladder operators have exact
//! closed-form matrix elements. See [`docs/theory.md`] §5 for the rationale and
//! [`docs/references.md`] for the port provenance.
//!
//! [`docs/theory.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/theory.md
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/racah/blob/main/docs/references.md
//!
//! # Conventions
//!
//! ## Label normalization invariant
//!
//! An [`Irrep`](crate::sun::Irrep) stores the SU(N) highest weight as a *normalized* weight
//! `λ = (λ₁ ≥ λ₂ ≥ … ≥ λ_N)` with `λ_N = 0` and all `λ_i ≥ 0`, of length `N`
//! (`= rank`). This matches `sunirrep.jl`'s `weight(s)` (`_dynkin_to_weight`
//! produces a nonincreasing tuple with last component 0). Weight input is
//! shift-invariant: any representative is accepted and normalized by
//! subtracting `λ_N`. The Dynkin labels `aᵢ = λᵢ − λᵢ₊₁` (all `≥ 0`) are
//! derivable via [`Irrep::dynkin`](crate::sun::Irrep::dynkin).
//!
//! # Example
//!
//! Irreps are built from Dynkin labels (length `N-1`). This computes an
//! SU(3) F-symbol block for the sextet `1 ⊗ 3 ⊗ 3 → 6`; with `a` trivial the
//! move is the `1×1×1×1` identity (value 1):
//!
//! ```
//! use racah::sun::{f_symbol, Irrep};
//!
//! let triv = Irrep::trivial(3).unwrap(); // SU(3) singlet
//! let three = Irrep::from_dynkin(&[1, 0]).unwrap(); // fundamental
//! let six = Irrep::from_dynkin(&[2, 0]).unwrap();
//!
//! let block = f_symbol(&triv, &three, &three, &six, &three, &six).unwrap();
//! assert_eq!(block.dims(), [1, 1, 1, 1]);
//! assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::SignedSqrtRational;

mod cgc;
mod fr;
mod linalg;

pub use cgc::{cgc, Cgc, CgcEntry};
pub use fr::{
    check_f_unitarity, check_hexagon, check_pentagon, f_symbol, r_symbol, FBlock, RBlock,
};

/// Error for a malformed SU(N) irrep label. The public constructors never
/// panic; they return this instead.
///
/// Not `Eq`: the generation gates ([`SunError::NotOrthonormal`] and friends)
/// carry an `f64` residual for diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub enum SunError {
    /// A rank-0 label (empty weight / empty Dynkin for `N = 1` is a single
    /// zero, so a genuinely empty slice is rejected here).
    EmptyLabel,
    /// A weight that is not nonincreasing (`λᵢ < λᵢ₊₁` for some `i`), i.e. its
    /// implied Dynkin label would be negative.
    NotNonincreasing {
        /// The offending (unnormalized) weight.
        weight: Vec<i64>,
    },
    /// A Dynkin label with a negative component.
    NegativeDynkin {
        /// The offending Dynkin label.
        dynkin: Vec<i64>,
    },
    /// A [`directproduct`] of two irreps of different rank (distinct SU(N)
    /// groups have no common product; this is an ill-posed input, not a
    /// zero-channel fusion).
    RankMismatch {
        /// Rank `N` of the first irrep.
        a: usize,
        /// Rank `N` of the second irrep.
        b: usize,
    },
    /// The highest-weight nullspace dimension did not equal the Layer 1 fusion
    /// multiplicity `N^{s3}_{s1 s2}`. Ported gate from `clebschgordan.jl`'s
    /// `@assert N123 == directproduct(s1, s2)[s3]`; a mismatch means the
    /// numerical rank cut disagrees with the exact combinatorics.
    NullspaceDimMismatch {
        /// Expected multiplicity from [`directproduct`].
        expected: usize,
        /// Nullspace dimension found by the SVD rank cut.
        found: usize,
    },
    /// Assembled CGC columns are not orthonormal to tolerance (generation
    /// gate). The value is the worst `|<α|β> - δ_{αβ}|` over multiplicity
    /// columns.
    NotOrthonormal {
        /// Worst orthonormality residual.
        residual: f64,
    },
    /// The ladder-descent consistency spot check exceeded tolerance
    /// (generation gate): a lowered highest-weight relation was not reproduced
    /// by the descended coefficients.
    LadderInconsistent {
        /// Worst ladder residual.
        residual: f64,
    },
    /// A dense factorization routed through `tenferro-linalg` failed. Carries
    /// the backend's message; surfaced rather than panicked because the
    /// floating-point stages are verification-gated, not proven-unreachable.
    Linalg(String),
    /// An F- or R-symbol was requested for labels where one of the required
    /// fusion vertices is empty (`N^c_{ab} = 0`). The reference `_Fsymbol` /
    /// `_Rsymbol` return an all-zero block *by construction* when a channel is
    /// empty; this crate exposes a query API, so an empty vertex is an
    /// ill-posed question and becomes a typed error (issue #15 guard class —
    /// the reference's `Nsymbol(...) == 0 && return zeros` short-circuit).
    /// Carries the offending vertex `a ⊗ b → c` as Dynkin labels.
    ZeroFusionChannel {
        /// Dynkin label of the vertex's left factor.
        a: Vec<i64>,
        /// Dynkin label of the vertex's right factor.
        b: Vec<i64>,
        /// Dynkin label of the vertex's coupled irrep.
        c: Vec<i64>,
    },
    /// The F-move matrix (rows `(e, μ, ν)`, cols `(f, κ, λ)` for fixed outer
    /// labels `a, b, c, d`) failed the unitarity gate. The value is the worst
    /// `|(F Fᵀ - I)_{ij}|`.
    FNotUnitary {
        /// Worst unitarity residual.
        residual: f64,
    },
    /// The pentagon identity spot check exceeded tolerance for the sampled
    /// `(a, b, c, d)` family. The value is the worst two-sided residual.
    PentagonViolation {
        /// Worst pentagon residual.
        residual: f64,
    },
    /// A hexagon identity spot check exceeded tolerance for the sampled
    /// `(a, b, c)` family. The value is the worst two-sided residual.
    HexagonViolation {
        /// Worst hexagon residual.
        residual: f64,
    },
}

impl fmt::Display for SunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SunError::EmptyLabel => write!(f, "SU(N) irrep label must be non-empty"),
            SunError::NotNonincreasing { weight } => {
                write!(f, "SU(N) weight is not nonincreasing: {weight:?}")
            }
            SunError::NegativeDynkin { dynkin } => {
                write!(f, "SU(N) Dynkin label has a negative component: {dynkin:?}")
            }
            SunError::RankMismatch { a, b } => {
                write!(
                    f,
                    "directproduct of SU({a}) and SU({b}) irreps (rank mismatch)"
                )
            }
            SunError::NullspaceDimMismatch { expected, found } => write!(
                f,
                "CGC nullspace dimension {found} != fusion multiplicity {expected}"
            ),
            SunError::NotOrthonormal { residual } => {
                write!(f, "CGC columns not orthonormal (residual {residual:e})")
            }
            SunError::LadderInconsistent { residual } => {
                write!(f, "CGC ladder-descent inconsistent (residual {residual:e})")
            }
            SunError::Linalg(msg) => write!(f, "dense factorization failed: {msg}"),
            SunError::ZeroFusionChannel { a, b, c } => write!(
                f,
                "empty fusion vertex {a:?} ⊗ {b:?} → {c:?} (N = 0) in an F/R request"
            ),
            SunError::FNotUnitary { residual } => {
                write!(f, "F-move matrix not unitary (residual {residual:e})")
            }
            SunError::PentagonViolation { residual } => {
                write!(f, "pentagon identity violated (residual {residual:e})")
            }
            SunError::HexagonViolation { residual } => {
                write!(f, "hexagon identity violated (residual {residual:e})")
            }
        }
    }
}

impl std::error::Error for SunError {}

/// An irreducible representation of SU(N), labelled by its normalized highest
/// weight (see module docs for the invariant).
///
/// `Ord`/`Hash` are on the normalized weight, so two `Irrep`s are equal iff
/// they denote the same irrep; the order is deterministic (used as a map key).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Irrep {
    /// Normalized highest weight, length `N`, nonincreasing, last entry `0`.
    weight: Box<[i64]>,
}

impl Irrep {
    /// Construct from an `N`-component highest weight (any shift representative).
    ///
    /// Normalizes by subtracting the last component. Rejects an empty slice
    /// ([`SunError::EmptyLabel`]) or a non-nonincreasing weight
    /// ([`SunError::NotNonincreasing`]).
    pub fn from_weight(weight: &[i64]) -> Result<Self, SunError> {
        if weight.is_empty() {
            return Err(SunError::EmptyLabel);
        }
        for w in weight.windows(2) {
            if w[0] < w[1] {
                return Err(SunError::NotNonincreasing {
                    weight: weight.to_vec(),
                });
            }
        }
        let last = weight[weight.len() - 1];
        let norm: Box<[i64]> = weight.iter().map(|x| x - last).collect();
        Ok(Irrep { weight: norm })
    }

    /// Construct from the `N-1` Dynkin labels `aᵢ = λᵢ − λᵢ₊₁` (all `≥ 0`).
    ///
    /// `N = dynkin.len() + 1`. Mirrors `_dynkin_to_weight`: the weight is the
    /// suffix sums with a trailing `0`. An empty slice yields the SU(1) trivial
    /// irrep. Rejects a negative component ([`SunError::NegativeDynkin`]).
    pub fn from_dynkin(dynkin: &[i64]) -> Result<Self, SunError> {
        if dynkin.iter().any(|&a| a < 0) {
            return Err(SunError::NegativeDynkin {
                dynkin: dynkin.to_vec(),
            });
        }
        let n = dynkin.len() + 1;
        let mut w = vec![0i64; n];
        for i in (0..n - 1).rev() {
            w[i] = w[i + 1] + dynkin[i];
        }
        Ok(Irrep {
            weight: w.into_boxed_slice(),
        })
    }

    /// The trivial (vacuum) SU(N) irrep — the all-zero weight of length `N`.
    /// `N` must be `≥ 1` ([`SunError::EmptyLabel`] otherwise).
    pub fn trivial(n: usize) -> Result<Self, SunError> {
        if n == 0 {
            return Err(SunError::EmptyLabel);
        }
        Ok(Irrep {
            weight: vec![0i64; n].into_boxed_slice(),
        })
    }

    /// The rank `N` of the SU(N) group.
    pub fn rank(&self) -> usize {
        self.weight.len()
    }

    /// The normalized highest weight (length `N`, nonincreasing, last `0`).
    pub fn weight(&self) -> &[i64] {
        &self.weight
    }

    /// The `N-1` Dynkin labels `aᵢ = λᵢ − λᵢ₊₁`.
    pub fn dynkin(&self) -> Vec<i64> {
        self.weight.windows(2).map(|w| w[0] - w[1]).collect()
    }

    /// The Weyl dimension, exact.
    ///
    /// Ported from `sector.jl:dim`:
    /// `∏_{k₂=2..N, k₁=1..k₂-1} (k₂-k₁ + λ_{k₁} - λ_{k₂}) / (k₂-k₁)`.
    /// The product is an integer; we accumulate as a `Ratio<BigInt>` and return
    /// the (guaranteed-integral) value as a `BigInt` (no `u*` cap on `N`).
    pub fn dim(&self) -> BigInt {
        let w = &self.weight;
        let n = w.len();
        let mut acc = Ratio::<BigInt>::one();
        for k2 in 2..=n {
            for k1 in 1..k2 {
                let d = (k2 - k1) as i64;
                let numer = d + w[k1 - 1] - w[k2 - 1];
                acc *= Ratio::new(BigInt::from(numer), BigInt::from(d));
            }
        }
        acc.to_integer()
    }

    /// The dual (conjugate) irrep. Ported from `sector.jl:dual`: reverse the
    /// Dynkin labels. Reversing a nonnegative Dynkin label stays valid, so the
    /// reconstruction cannot fail.
    pub fn dual(&self) -> Irrep {
        let mut d = self.dynkin();
        d.reverse();
        Irrep::from_dynkin(&d).expect("reversed nonnegative Dynkin label is valid")
    }

    /// All GT patterns of this irrep, in the reference basis order.
    ///
    /// Ported from `gtpatterns.jl:GTPatternIterator` /
    /// `basis(s) = GTPatternIterator{N}(weight(s))`, reproduced index-for-index.
    /// The order is load-bearing — the CGC gauge is a deterministic function of
    /// it — so it is pinned by checked-in fixtures. See the private
    /// `gt_enumerate` for the recursion.
    pub fn patterns(&self) -> Vec<GtPattern> {
        let n = self.rank();
        gt_enumerate(&self.weight)
            .into_iter()
            .map(|data| GtPattern {
                n,
                data: data.into_boxed_slice(),
            })
            .collect()
    }

    /// The GT creation (raising) matrices, one per level `l = 1..N-1`.
    ///
    /// Result index `l-1` holds the sparse entries of the raising operator that
    /// increments `m[k, l]`. Entry rows/cols are 0-based indices into
    /// [`Irrep::patterns`]. Ported from `gtpatterns.jl:creation`; the entry is
    /// `signedroot(coef)` of the exact GT rational `coef`, carried as a
    /// [`SignedSqrtRational`] (its `signed_square()` equals `coef`).
    pub fn creation(&self) -> Vec<Vec<LadderEntry>> {
        let n = self.rank();
        let pats = self.patterns();
        // Basis index of each pattern (the reference `table`).
        let table: HashMap<&GtPattern, usize> =
            pats.iter().enumerate().map(|(i, m)| (m, i)).collect();
        let mut result: Vec<Vec<LadderEntry>> = vec![Vec::new(); n.saturating_sub(1)];

        for (i, m) in pats.iter().enumerate() {
            if n < 2 {
                break;
            }
            for l in 1..=(n - 1) {
                for k in 1..=l {
                    // coef = -1 * ∏_{k'} [ (m[k',l+1] - m[k,l] + k - k')
                    //                     · (m[k',l-1] - m[k,l] + k - k' - 1) if k'≤l-1 ]
                    //             / [ (m[k',l] - m[k,l] + k - k')
                    //                 (m[k',l] - m[k,l] + k - k' - 1) if k'≤l, k'≠k ]
                    let mkl = m.get(k, l);
                    let mut coef = Ratio::<BigInt>::from(BigInt::from(-1));
                    let mut skip = false;
                    for kp in 1..=(l + 1) {
                        let base = mkl + (kp as i64) - (k as i64);
                        let f1 = m.get(kp, l + 1) - base;
                        coef *= BigInt::from(f1);
                        if kp < l {
                            // reference: k' ≤ l-1 (so m[k', l-1] is in range)
                            let f2 = m.get(kp, l - 1) - base - 1;
                            coef *= BigInt::from(f2);
                        }
                        if coef.numer().is_zero() {
                            skip = true; // coef == 0: no entry
                            break;
                        }
                        if kp <= l && kp != k {
                            let g = m.get(kp, l) - base;
                            // Julia's Rational hits denominator 0 here and the
                            // reference skips; we detect the zero divisor and
                            // skip too (num_rational would panic on ÷0).
                            let den = g * (g - 1);
                            if den == 0 {
                                skip = true;
                                break;
                            }
                            coef /= BigInt::from(den);
                        }
                    }
                    if skip || coef.numer().is_zero() {
                        continue;
                    }
                    // m' = m with m[k,l] raised by 1. Look up its basis index.
                    //
                    // Why-not (the one exception to this crate's no-panic
                    // contract): a missing m' is proven unreachable, so we panic
                    // in every build rather than silently drop the entry.
                    // Invariant proof: raising m[k,l] zeroes a numerator factor
                    // (`m[k,l+1] - m[k,l]` or the lower-neighbour term) exactly
                    // when a GT betweenness constraint would break, so coef ≠ 0
                    // implies m' is a valid basis member. The panic is thus dead
                    // code by proof; a silent release drop would instead turn a
                    // future proof-breaking regression into a missing ladder
                    // entry — a silent-wrong-answer defect. Both reference
                    // implementations abort here (Julia `table[m']` throws
                    // KeyError in all builds; QSpace aborts on invariant
                    // violations).
                    let mut mp = m.clone();
                    mp.set(k, l, mkl + 1);
                    let &j = table.get(&mp).expect(
                        "GT invariant violated: coef != 0 implies the raised \
                         pattern is a valid basis member",
                    );
                    result[l - 1].push(LadderEntry {
                        row: j,
                        col: i,
                        value: signedroot(&coef),
                    });
                }
            }
        }
        result
    }

    /// The GT annihilation (lowering) matrices: the transpose of
    /// [`Irrep::creation`] (`annihilation(s) = [op' for op in creation(s)]`,
    /// `gtpatterns.jl`). Entries are real, so the transpose just swaps
    /// `row`/`col`.
    pub fn annihilation(&self) -> Vec<Vec<LadderEntry>> {
        self.creation()
            .into_iter()
            .map(|mat| {
                mat.into_iter()
                    .map(|e| LadderEntry {
                        row: e.col,
                        col: e.row,
                        value: e.value,
                    })
                    .collect()
            })
            .collect()
    }
}

/// A single nonzero entry of a GT ladder matrix: `value` at `(row, col)`, with
/// `row`/`col` 0-based indices into [`Irrep::patterns`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderEntry {
    /// Row index (0-based basis index of the raised pattern).
    pub row: usize,
    /// Column index (0-based basis index of the source pattern).
    pub col: usize,
    /// The exact matrix element `sign * sqrt(radicand)`.
    pub value: SignedSqrtRational,
}

/// A Gelfand–Tsetlin pattern of an SU(N) irrep.
///
/// Storage mirrors `gtpatterns.jl:GTPattern`: the triangular array flattened
/// top row first (`l = N`, then `l = N-1`, …, `l = 1`), each row `l` occupying
/// `l` contiguous entries. Access via [`GtPattern::get`] with 1-based
/// `(k, l)`, `1 ≤ k ≤ l ≤ N`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GtPattern {
    n: usize,
    data: Box<[i64]>,
}

impl GtPattern {
    /// The flat pattern data, top row (`l = N`) first (the reference
    /// `m.data` order).
    pub fn data(&self) -> &[i64] {
        &self.data
    }

    /// The rank `N`.
    pub fn rank(&self) -> usize {
        self.n
    }

    /// Entry `m[k, l]`, `1 ≤ k ≤ l ≤ N`. Panics only on an out-of-range index,
    /// which is an internal invariant of the enumeration/ladder code (never
    /// reachable from public input).
    #[inline]
    pub fn get(&self, k: usize, l: usize) -> i64 {
        self.data[Self::flat_index(self.n, k, l)]
    }

    #[inline]
    fn set(&mut self, k: usize, l: usize, v: i64) {
        let idx = Self::flat_index(self.n, k, l);
        self.data[idx] = v;
    }

    /// 0-based flat index of `m[k, l]`. The reference (1-based) index is
    /// `k + ((l+1+N)(N-l))>>1`; subtracting 1 gives the 0-based form.
    #[inline]
    fn flat_index(n: usize, k: usize, l: usize) -> usize {
        (k - 1) + (((l + 1 + n) * (n - l)) >> 1)
    }
}

/// Enumerate the flat data of every GT pattern with the given top row, in the
/// reference `GTPatternIterator` order.
///
/// The reference builds, for `N ≥ 2`, the product over the possible second rows
/// `I[i+1]:I[i]` (`reverse`d so the *last* sub-row entry varies fastest), and
/// for each recurses into `GTPatternIterator{N-1}`, with the recursion as the
/// inner (faster) loop. Concatenating `(toprow, subpattern)` reproduces both
/// the flat storage layout and the iteration order.
fn gt_enumerate(toprow: &[i64]) -> Vec<Vec<i64>> {
    let n = toprow.len();
    if n == 1 {
        return vec![vec![toprow[0]]];
    }
    let mut out = Vec::new();
    for subrow in subrow_order(toprow) {
        for sub in gt_enumerate(&subrow) {
            let mut data = toprow.to_vec();
            data.extend_from_slice(&sub);
            out.push(data);
        }
    }
    out
}

/// All admissible second rows for `toprow`, in the reference product order:
/// `sub[j] ∈ [toprow[j+1], toprow[j]]` (GT betweenness), with the *last* entry
/// `sub[N-2]` varying fastest and `sub[0]` slowest.
fn subrow_order(toprow: &[i64]) -> Vec<Vec<i64>> {
    let m = toprow.len() - 1;
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(m);
    subrow_rec(0, m, toprow, &mut cur, &mut out);
    out
}

fn subrow_rec(j: usize, m: usize, toprow: &[i64], cur: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
    if j == m {
        out.push(cur.clone());
        return;
    }
    // Ascending, matching Julia's `I[j+1]:I[j]` UnitRange iteration.
    for val in toprow[j + 1]..=toprow[j] {
        cur.push(val);
        subrow_rec(j + 1, m, toprow, cur, out);
        cur.pop();
    }
}

/// Littlewood–Richardson product decomposition: the fusion multiplicities of
/// `a ⊗ b`, keyed by the resulting irrep.
///
/// Ported from `gtpatterns.jl:directproduct`. Requires `rank(a) == rank(b)`
/// (both label the same SU(N)); a rank mismatch is an ill-posed input across
/// distinct groups and returns [`SunError::RankMismatch`] rather than a
/// zero-channel map — this signature is Layer 2's foundation. The reference
/// iterates the smaller-dimensional basis; we replicate the `dim` swap (the
/// result is independent of it, but the port stays faithful).
pub fn directproduct(a: &Irrep, b: &Irrep) -> Result<BTreeMap<Irrep, u32>, SunError> {
    if a.rank() != b.rank() {
        return Err(SunError::RankMismatch {
            a: a.rank(),
            b: b.rank(),
        });
    }
    if a.dim() > b.dim() {
        return directproduct(b, a);
    }
    let n = a.rank();
    let mut result: BTreeMap<Irrep, u32> = BTreeMap::new();
    for m in a.patterns() {
        // t starts as b's weight; each GT row of `a` shifts one component.
        let mut t: Vec<i64> = b.weight.to_vec();
        let mut bad = false;
        'scan: for k in 1..=n {
            for l in (k..=n).rev() {
                let mut bkl = m.get(k, l);
                if l > k {
                    // checkbounds(m, k, l-1): valid when l-1 ≥ k.
                    bkl -= m.get(k, l - 1);
                }
                t[l - 1] += bkl;
                if l > 1 && t[l - 2] < t[l - 1] {
                    bad = true;
                    break 'scan;
                }
            }
        }
        if !bad {
            // t is a valid (nonincreasing) weight; from_weight normalizes it.
            let s = Irrep::from_weight(&t).expect("GT descent yields a valid weight");
            *result.entry(s).or_insert(0) += 1;
        }
    }
    Ok(result)
}

/// `sign(coef) * sqrt(|coef|)` as an exact [`SignedSqrtRational`], matching
/// `RationalRoots.signedroot`. `signed_square()` of the result equals `coef`.
fn signedroot(coef: &Ratio<BigInt>) -> SignedSqrtRational {
    if coef.is_zero() {
        return SignedSqrtRational::zero();
    }
    let s = if coef.is_negative() {
        Ratio::from(BigInt::from(-1))
    } else {
        Ratio::from(BigInt::from(1))
    };
    SignedSqrtRational::from_prefactor_radical(s, coef.abs())
}

/// Opaque authority fingerprint of the generated SU(N) provider.
///
/// The bytes identify the *convention set*, generation pipeline, and
/// verification/tolerance policy under which every SU(N) Clebsch–Gordan
/// coefficient (and the F/R symbols contracted from it) is produced. Their sole
/// use is equality comparison: a consumer may persist the bytes next to data
/// derived from these coefficients and later compare them to decide whether that
/// derived data was produced under the same convention.
///
/// # Contract (binding)
///
/// > Equal fingerprints identify the same convention, generation pipeline, and
/// > tolerance policy. They do not imply byte-identical values or independently
/// > prove numerical agreement.
///
/// This is deliberately weaker than the base SU(2) fingerprint
/// ([`crate::su2_authority_fingerprint`]), whose exact big-rational surface lets
/// equal bytes mean equal values. The generated SU(N) family is a *two-layer*
/// contract (`docs/gauge.md` "value agreement within the oracle tolerance, not
/// cross-process bit-identity"): the gauge/sign/structure is a deterministic
/// function of the subspace, but the final dense linear-algebra stages run in
/// `f64` and the backend's parallel reductions are not bit-reproducible across
/// processes, so two builds may differ by a few ULPs. **Numerical agreement is
/// established by the generation-time verification gates** (`docs/gauge.md` §9:
/// multiplicity, orthonormality, ladder consistency — typed `SunError`, never
/// silent) **and the independent oracle suites** (`docs/gauge.md` §11: the SU(2)
/// embedding and the signed element-wise SUNRepresentations.jl v0.4.0
/// fixtures), **never by this fingerprint.**
///
/// # Consumer contract
///
/// - **Opaque.** Compare by equality only; never parse the tags or split on
///   `:` / `=`. The internal shape is not a stable interface.
/// - **Stable across patch and minor releases.** The value is not derived from
///   the crate version, source, docs, a pointer, or any process-local state.
/// - **Changes exactly with a value-affecting breaking release.** The trailing
///   `epoch` is bumped by hand — and only — when a change can alter a returned
///   coefficient value, its normalization, or the canonical convention it is
///   expressed in (the breaking-release event class of `docs/gauge.md`). The
///   compatibility-policy test (`tests/sun_fingerprint.rs`) pins the exact bytes,
///   so any such change is a mutation-visible review event.
/// - **Epoch is per-family and independent.** The SU(N) `epoch` moves
///   independently of the SU(2) and B/C/D epochs; an SU(N) gauge change never
///   invalidates SU(2)-derived or B/C/D-derived consumer state (and vice versa).
///   The base SU(2) surface is untouched by this fingerprint.
///
/// # Tags and the conventions they pin (each cites `docs/gauge.md`)
///
/// Every tag names a rule the gauge document already pins; nothing here invents
/// a convention. The backend identity is deliberately excluded — per-backend ULP
/// differences are inside the tolerance class this fingerprint's contract
/// disclaims, and a discrete gauge flip across backends is a defect, not a
/// tolerance event (`docs/gauge.md` §10).
///
/// - `ref=sunrep-0.4` — the port reference: SUNRepresentations.jl v0.4.0
///   (`docs/gauge.md`, header).
/// - `basis=gt-order` — the Gelfand–Tsetlin pattern basis order that indexes
///   `m1, m2, m3`, and the highest-weight-system coupling-pair enumeration that
///   follows from it (`docs/gauge.md` §1, §2).
/// - `gauge=qrpos-cref` — the gauge canonicalization
///   `gaugefix! = first ∘ qrpos! ∘ cref!`: the column-pivoted reduced echelon
///   pivot rule and the positive-diagonal QR sign fix (`docs/gauge.md` §4, 4a/4b).
/// - `descent=ladder-lstsq` — the lower-weight descent: reverse-lexicographic
///   weight order and the QR least-squares lowering solve (`docs/gauge.md` §5).
/// - `tol=sunrep-tol-tier` — the value-fixing tolerance tier (the reference
///   SUNRepresentations `TOL_*` constants: `TOL_NULLSPACE` rank cut, `TOL_GAUGE`
///   pivot, `TOL_PURGE`; `docs/gauge.md` §3, §4a, §6). The `TOL_ORTHO`/
///   `TOL_LADDER` gates are excluded: they cannot move a returned value, so
///   tightening them is not a breaking release (`docs/gauge.md` §9).
/// - `epoch=1` — the per-family manual epoch (see above).
///
/// # Stability
///
/// **Unstable: shape may change while the generated-provider contract is
/// negotiated.** Cargo features cannot express instability tiers; this label and
/// issue #47 are the ledger.
#[cfg(feature = "cgc-gen")]
pub fn sun_authority_fingerprint() -> &'static [u8] {
    // Manual per-family epoch: bump the trailing `epoch=N` (and the literal in
    // tests/sun_fingerprint.rs) only on a value-affecting breaking release.
    b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(dynkin: &[i64]) -> Irrep {
        Irrep::from_dynkin(dynkin).unwrap()
    }

    // ---- labels / normalization ----

    #[test]
    fn weight_normalizes_by_shift() {
        // (3,1,0) and (5,3,2) denote the same SU(3) irrep (Dynkin (2,1)).
        let a = Irrep::from_weight(&[3, 1, 0]).unwrap();
        let b = Irrep::from_weight(&[5, 3, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.weight(), &[3, 1, 0]);
        assert_eq!(a.dynkin(), vec![2, 1]);
    }

    #[test]
    fn dynkin_round_trip() {
        let s = irr(&[2, 0, 1]); // SU(4)
        assert_eq!(s.rank(), 4);
        let round = Irrep::from_dynkin(&s.dynkin()).unwrap();
        assert_eq!(s, round);
    }

    #[test]
    fn malformed_labels_are_typed_errors_not_panics() {
        assert_eq!(Irrep::from_weight(&[]), Err(SunError::EmptyLabel));
        assert!(matches!(
            Irrep::from_weight(&[1, 2, 0]),
            Err(SunError::NotNonincreasing { .. })
        ));
        assert!(matches!(
            Irrep::from_dynkin(&[1, -1]),
            Err(SunError::NegativeDynkin { .. })
        ));
        assert_eq!(Irrep::trivial(0), Err(SunError::EmptyLabel));
    }

    // ---- dim / dual ----

    #[test]
    fn weyl_dim_known_su3() {
        assert_eq!(irr(&[0, 0]).dim(), BigInt::from(1)); // trivial
        assert_eq!(irr(&[1, 0]).dim(), BigInt::from(3)); // fundamental
        assert_eq!(irr(&[0, 1]).dim(), BigInt::from(3)); // antifundamental
        assert_eq!(irr(&[1, 1]).dim(), BigInt::from(8)); // adjoint
        assert_eq!(irr(&[2, 0]).dim(), BigInt::from(6));
        assert_eq!(irr(&[3, 0]).dim(), BigInt::from(10));
    }

    #[test]
    fn dual_is_reverse_dynkin_and_involutive() {
        let s = irr(&[1, 0]); // SU(3) fundamental
        assert_eq!(s.dual().dynkin(), vec![0, 1]); // antifundamental
        for d in [vec![1, 0], vec![2, 1, 0], vec![1, 2, 0, 3]] {
            let x = irr(&d);
            assert_eq!(x.dual().dual(), x);
            assert_eq!(x.dim(), x.dual().dim()); // dual preserves dimension
        }
    }

    // ---- GT patterns ----

    #[test]
    fn patterns_count_equals_dim() {
        for d in [
            vec![1, 0],
            vec![1, 1],
            vec![2, 1],
            vec![1, 0, 1],
            vec![1, 1, 0, 1],
        ] {
            let s = irr(&d);
            assert_eq!(BigInt::from(s.patterns().len()), s.dim());
        }
    }

    #[test]
    fn su3_fundamental_pattern_order() {
        // Reference basis(SU3Irrep(1,0,0)) data order (verified against Julia):
        // (1,0,0, 0,0, 0), (1,0,0, 1,0, 0), (1,0,0, 1,0, 1).
        let s = Irrep::from_weight(&[1, 0, 0]).unwrap();
        let got: Vec<Vec<i64>> = s.patterns().iter().map(|p| p.data().to_vec()).collect();
        assert_eq!(
            got,
            vec![
                vec![1, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 1, 0, 0],
                vec![1, 0, 0, 1, 0, 1],
            ]
        );
    }

    #[test]
    fn pattern_get_matches_reference_layout() {
        let s = Irrep::from_weight(&[1, 0, 0]).unwrap();
        let p = &s.patterns()[2]; // data (1,0,0, 1,0, 1)
                                  // Top row l=3: m[1,3]=1, m[2,3]=0, m[3,3]=0.
        assert_eq!(p.get(1, 3), 1);
        assert_eq!(p.get(2, 3), 0);
        assert_eq!(p.get(3, 3), 0);
        // Row l=2: m[1,2]=1, m[2,2]=0. Row l=1: m[1,1]=1.
        assert_eq!(p.get(1, 2), 1);
        assert_eq!(p.get(2, 2), 0);
        assert_eq!(p.get(1, 1), 1);
    }

    // ---- directproduct ----

    #[test]
    fn su3_product_known() {
        // 3 ⊗ 3̄ = 8 ⊕ 1
        let dp = directproduct(&irr(&[1, 0]), &irr(&[0, 1])).unwrap();
        let mut got: Vec<(Vec<i64>, u32)> = dp.iter().map(|(k, &v)| (k.dynkin(), v)).collect();
        got.sort();
        assert_eq!(got, vec![(vec![0, 0], 1), (vec![1, 1], 1)]);

        // 3 ⊗ 3 = 6 ⊕ 3̄
        let dp = directproduct(&irr(&[1, 0]), &irr(&[1, 0])).unwrap();
        let mut got: Vec<(Vec<i64>, u32)> = dp.iter().map(|(k, &v)| (k.dynkin(), v)).collect();
        got.sort();
        assert_eq!(got, vec![(vec![0, 1], 1), (vec![2, 0], 1)]);
    }

    #[test]
    fn directproduct_rank_mismatch_is_typed_error() {
        // Distinct SU(N) groups: ill-posed input must be a typed error, never a
        // (well-formed-looking) empty decomposition.
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        assert_eq!(
            directproduct(&su3, &su4),
            Err(SunError::RankMismatch { a: 3, b: 4 })
        );
    }

    #[test]
    fn directproduct_dim_sum_rule() {
        // dim(a)·dim(b) == Σ_c N^c_ab dim(c)
        for (da, db) in [
            (vec![1, 1], vec![1, 1]),             // SU(3) 8⊗8
            (vec![2, 1], vec![1, 2]),             // SU(3)
            (vec![1, 0, 1], vec![1, 1, 0]),       // SU(4)
            (vec![1, 1, 0, 1], vec![0, 1, 1, 0]), // SU(5)
        ] {
            assert_dim_sum_rule(&irr(&da), &irr(&db));
        }
    }

    #[test]
    fn directproduct_commutes_and_dual_twist() {
        assert_commute_and_dual_twist(&irr(&[2, 1]), &irr(&[1, 1]));
    }

    fn assert_dim_sum_rule(a: &Irrep, b: &Irrep) {
        // dim(a)·dim(b) == Σ_c N^c_ab dim(c)
        let lhs = a.dim() * b.dim();
        let rhs: BigInt = directproduct(a, b)
            .unwrap()
            .iter()
            .map(|(c, &m)| c.dim() * BigInt::from(m))
            .sum();
        assert_eq!(
            lhs,
            rhs,
            "sum rule failed for {:?} ⊗ {:?}",
            a.dynkin(),
            b.dynkin()
        );
    }

    fn assert_commute_and_dual_twist(a: &Irrep, b: &Irrep) {
        // Commutativity (bosonic fusion) and N^c_ab == N^{c̄}_{ā b̄}.
        assert_eq!(directproduct(a, b), directproduct(b, a));
        let dp = directproduct(a, b).unwrap();
        let dpd = directproduct(&a.dual(), &b.dual()).unwrap();
        let twisted: BTreeMap<Irrep, u32> = dp.iter().map(|(c, &m)| (c.dual(), m)).collect();
        assert_eq!(dpd, twisted);
    }

    #[test]
    fn randomized_property_sweep() {
        // Seeded pair sweep so the in-crate properties are themselves
        // randomized (acceptance item 5): 50 random pairs per N in 2..=5.
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5150_4E37_0DEC_0DE5);
        // Bound the Dynkin range per rank so dims (hence directproduct basis
        // size) stay modest for higher N — the sweep is a property check, not a
        // large-irrep stress test.
        let max_dynkin = |n: usize| -> i64 {
            match n {
                2 => 4,
                3 => 3,
                4 => 2,
                _ => 1,
            }
        };
        let rand_irrep = |rng: &mut rand_chacha::ChaCha8Rng, n: usize| -> Irrep {
            let hi = max_dynkin(n);
            let dynkin: Vec<i64> = (0..n - 1).map(|_| rng.gen_range(0..=hi)).collect();
            Irrep::from_dynkin(&dynkin).unwrap()
        };
        for n in 2..=5usize {
            for _ in 0..50 {
                let a = rand_irrep(&mut rng, n);
                let b = rand_irrep(&mut rng, n);
                assert_dim_sum_rule(&a, &b);
                assert_commute_and_dual_twist(&a, &b);
                assert_eq!(a.dual().dual(), a); // dual involution
                let _ = a.creation(); // fires the P3 invariant guard in debug
            }
        }
    }

    // ---- ladder matrices ----

    #[test]
    fn su3_adjoint_creation_matches_reference() {
        // creation(SU3Irrep(2,1,0)) nonzero entries, signed_square (from Julia):
        // l=1: (i,j) with sq: (2,1)=1, (5,4)=2, (6,5)=2, (8,7)=1
        // l=2: (4,1)=1, (3,2)=3/2, (5,2)=1/2, (7,3)=3/2, (7,5)=1/2, (8,6)=1
        // (i,j) are 1-based; our row/col are 0-based.
        let s = Irrep::from_weight(&[2, 1, 0]).unwrap();
        let cr = s.creation();
        let key = |mat: &Vec<LadderEntry>| -> Vec<(usize, usize, (i64, i64))> {
            let mut v: Vec<_> = mat
                .iter()
                .map(|e| {
                    let sq = e.value.signed_square();
                    (
                        e.row + 1,
                        e.col + 1,
                        (
                            sq.numer().try_into().unwrap(),
                            sq.denom().try_into().unwrap(),
                        ),
                    )
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            key(&cr[0]),
            vec![
                (2, 1, (1, 1)),
                (5, 4, (2, 1)),
                (6, 5, (2, 1)),
                (8, 7, (1, 1)),
            ]
        );
        assert_eq!(
            key(&cr[1]),
            vec![
                (3, 2, (3, 2)),
                (4, 1, (1, 1)),
                (5, 2, (1, 2)),
                (7, 3, (3, 2)),
                (7, 5, (1, 2)),
                (8, 6, (1, 1)),
            ]
        );
    }

    #[test]
    fn annihilation_is_transpose_of_creation() {
        let s = Irrep::from_weight(&[2, 1, 0]).unwrap();
        let cr = s.creation();
        let an = s.annihilation();
        for (cm, am) in cr.iter().zip(an.iter()) {
            let mut a: Vec<_> = am.iter().map(|e| (e.row, e.col, e.value.clone())).collect();
            let mut ct: Vec<_> = cm.iter().map(|e| (e.col, e.row, e.value.clone())).collect();
            a.sort_by_key(|x| (x.0, x.1));
            ct.sort_by_key(|x| (x.0, x.1));
            assert_eq!(a, ct);
        }
    }

    // ---- SU(2) embedding cross-check (item 4) ----

    #[test]
    fn su2_dim_dual_fusion_match_closed_form() {
        for dj in 0..=6i64 {
            let j = irr(&[dj]); // SU(2), Dynkin (2j)
            assert_eq!(j.dim(), BigInt::from(dj + 1)); // 2j+1
            assert_eq!(j.dual(), j); // SU(2) is self-dual
        }
        // Fusion range |j1-j2|..j1+j2, each multiplicity 1 (doubled labels).
        for dj1 in 0..=4i64 {
            for dj2 in 0..=4i64 {
                let dp = directproduct(&irr(&[dj1]), &irr(&[dj2])).unwrap();
                let mut got: Vec<i64> = dp
                    .iter()
                    .map(|(c, &m)| {
                        assert_eq!(m, 1);
                        c.dynkin()[0]
                    })
                    .collect();
                got.sort();
                let want: Vec<i64> = ((dj1 - dj2).abs()..=(dj1 + dj2)).step_by(2).collect();
                assert_eq!(got, want, "SU(2) fusion {dj1}⊗{dj2}");
            }
        }
    }

    #[test]
    fn su2_creation_matches_closed_form() {
        // For SU(2) irrep 2j = dj, GT basis is x = m[1,1] = 0..dj (ascending),
        // and raising x→x+1 has matrix element sqrt((dj-x)(x+1)):
        // <j,m+1|J+|j,m> with m = x - j. signed_square = (dj-x)(x+1) > 0.
        for dj in 1..=6i64 {
            let s = irr(&[dj]);
            let cr = s.creation();
            assert_eq!(cr.len(), 1);
            let mut got: Vec<(usize, usize, BigInt)> = cr[0]
                .iter()
                .map(|e| {
                    assert_eq!(e.value.sign(), 1);
                    (e.row, e.col, e.value.signed_square().to_integer())
                })
                .collect();
            got.sort();
            let want: Vec<(usize, usize, BigInt)> = (0..dj as usize)
                .map(|x| (x + 1, x, BigInt::from((dj - x as i64) * (x as i64 + 1))))
                .collect();
            assert_eq!(got, want, "SU(2) creation dj={dj}");
        }
    }
}
