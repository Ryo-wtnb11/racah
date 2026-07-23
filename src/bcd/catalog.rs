//! The S3.3 `CanonicalCatalog`: the single, append-only owner of discovered
//! generator sets for one B/C/D family instance (series + rank fixed at
//! construction), providing on-demand, query-order-independent generator and
//! Clebsch–Gordan materialization on top of the S3.2 sweep.
//!
//! Design authority: issue #18 Ruling 2, spec: issue #25. The **canonical
//! parent rule** — which product `a ⊗ b` produces each irrep's generators, and
//! why the choice is a deterministic function of the exact S3.0 data rather than
//! of discovery order — is specified to re-derivation standard in
//! `docs/gauge_soN.md` §14 (it IS gauge: the parent's sweep fixes `c`'s basis).
//!
//! # What it owns (Ruling 2)
//!
//! Exactly one generator set [`Generators`] per irrep, **append-only** for the
//! catalog's lifetime (no per-entry eviction; the whole catalog may be
//! [`reset`](CanonicalCatalog::reset)). CGC/F/R **values do not live here** —
//! they are returned to the caller (and, for SU(N), go to the byte-bounded value
//! caches in `crate::cache`); the catalog owns only generator sets plus the
//! minimal bookkeeping (a byte counter) to bound them.
//!
//! # On-demand materialization
//!
//! [`generators`](CanonicalCatalog::generators) and
//! [`cgc`](CanonicalCatalog::cgc) recursively materialize an irrep's
//! canonical-parent chain as needed. The recursion is well-founded (§14 of the
//! gauge doc): each parent is strictly smaller than its child in a fixed
//! well-order over the exact irrep data, and the chain bottoms out at the two
//! base cases (the trivial and defining reps), which are seeded at construction.
//! QSpace's fixed-pass `dmax` enumeration (`clebsch.cc` bootstrap loop) is **not**
//! ported as semantics — see §14.
//!
//! # Atomic byte budget (Ruling 2)
//!
//! A request whose recursive materialization would exceed the byte budget fails
//! atomically with [`CatalogError::BudgetExceeded`], leaving **no** partial
//! state: the whole chain is assembled in a staging buffer, its cost checked
//! against the budget, and only then committed (compute-fully-then-commit).
//!
//! # Single-threaded
//!
//! The API is `&mut self`; there is no global state and no interior mutability.
//! Concurrency is a later, separately reviewed extension (issue #18 Ruling 2).

use std::collections::HashMap;

use num_bigint::BigInt;

use super::sweep::{decompose, Block, Generators, SweepError};
use super::{defining_seed, directproduct, BcdError, Irrep, Series};

/// Default byte budget for a catalog (256 MiB). Generator sets are dense `f64`
/// `D×D` blocks; a family exercised over modest ranks stays far below this,
/// while a runaway recursion (or a deliberately tiny budget in a test) trips
/// [`CatalogError::BudgetExceeded`] before committing.
const DEFAULT_MAX_BYTES: usize = 256 << 20;

// ---- typed errors (guard inventory, issue #15) -----------------------------

/// Failure of a [`CanonicalCatalog`] request. Every ill-posed input is a typed
/// error (never a panic, never a silent zero) — the PR #14 trivial-coupling
/// lesson applied verbatim: every `N^c_ab = 0` triple is
/// [`CatalogError::ZeroFusionChannel`], red-first.
///
/// Not `Eq`: [`CatalogError::Sweep`] carries a [`SweepError`], several of whose
/// variants hold an `f64` residual.
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogError {
    /// An irrep passed to the catalog belongs to a different family than the
    /// catalog owns (different series or rank). A catalog instance is fixed to
    /// one `(series, rank)` at construction; a foreign irrep is ill-posed.
    WrongGroup {
        /// The catalog's `(series, rank)`.
        catalog: (Series, usize),
        /// The offending irrep's `(series, rank)`.
        got: (Series, usize),
    },
    /// A malformed or out-of-scope label surfaced while constructing the family
    /// or an intermediate irrep (empty/negative/spinor/excluded-rank). Wraps the
    /// S3.0 [`BcdError`].
    Label(BcdError),
    /// [`cgc`](CanonicalCatalog::cgc) was asked for a triple with `N^c_ab = 0`
    /// (the coupled irrep `c` does not appear in `a ⊗ b`, per the exact S3.0
    /// [`directproduct`]). The reference sweep would simply never emit such a
    /// block; a query API must reject the ill-posed question loudly (issue #15
    /// guard class; PR #14 trivial-coupling P1). Carries the Dynkin labels.
    ZeroFusionChannel {
        /// Dynkin label of the left factor `a`.
        a: Vec<i64>,
        /// Dynkin label of the right factor `b`.
        b: Vec<i64>,
        /// Dynkin label of the requested coupled irrep `c`.
        c: Vec<i64>,
    },
    /// The recursive materialization of a request would push the catalog's
    /// retained generator bytes past its budget. Reported **before** any commit,
    /// so no partial chain is ever observable (Ruling 2 atomicity).
    BudgetExceeded {
        /// The byte budget.
        limit: usize,
        /// The bytes that would be retained after committing this request.
        needed: usize,
    },
    /// The S3.2 sweep (or a product-generator composition) failed while
    /// materializing a canonical-parent chain. Surfaced, not panicked: the
    /// floating-point stages are verification-gated (Ruling 1).
    Sweep(SweepError),
    /// A non-base irrep had **no** admissible canonical-parent pair (§14.4).
    /// This is **unreachable by the box-count-first existence theorem**
    /// (`(defining, c-minus-a-box)` is always admissible); it is surfaced as a
    /// typed error rather than an `unreachable!` panic as defense-in-depth while
    /// the corrected proof beds in — a wrong theorem should fail loudly and
    /// recoverably at the exact label, not abort the process. Carries the
    /// offending Dynkin label.
    NoCanonicalParent {
        /// The Dynkin label with no admissible parent pair.
        dynkin: Vec<i64>,
    },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::WrongGroup { catalog, got } => write!(
                f,
                "irrep of group {got:?} passed to a catalog owning group {catalog:?}"
            ),
            CatalogError::Label(e) => write!(f, "invalid label: {e}"),
            CatalogError::ZeroFusionChannel { a, b, c } => write!(
                f,
                "ill-posed coupling: irrep {c:?} does not appear in {a:?} ⊗ {b:?} (N^c_ab = 0)"
            ),
            CatalogError::BudgetExceeded { limit, needed } => write!(
                f,
                "byte budget exceeded: request needs {needed} bytes, budget is {limit}"
            ),
            CatalogError::Sweep(e) => write!(f, "sweep failed during materialization: {e}"),
            CatalogError::NoCanonicalParent { dynkin } => write!(
                f,
                "no admissible canonical parent for irrep {dynkin:?} \
                 (unreachable by the box-count-first existence theorem)"
            ),
        }
    }
}

impl std::error::Error for CatalogError {}

impl From<BcdError> for CatalogError {
    fn from(e: BcdError) -> Self {
        CatalogError::Label(e)
    }
}

impl From<SweepError> for CatalogError {
    fn from(e: SweepError) -> Self {
        CatalogError::Sweep(e)
    }
}

// ---- public CGC result -----------------------------------------------------

/// The Clebsch–Gordan isometry coupling `s1 ⊗ s2 → s3`, produced by decomposing
/// the queried product `s1 ⊗ s2` and selecting the `s3` blocks.
///
/// Layout mirrors the SU(N) surface (`crate::sun::Cgc`) but stays **dense**: the
/// sweep hands back each coupled multiplet as a dense isometry, so this holds the
/// concatenation of the outer-multiplicity copies, column-major, in
/// outer-multiplicity index order. `PartialEq` is bitwise over the coefficient
/// buffer, so the query-order-independence acceptance test can compare two
/// materializations directly.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogCgc {
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    /// `d1·d2` (rows of each copy's isometry).
    rows: usize,
    /// `d3 = dim(s3)` (columns of each copy's isometry).
    d3: usize,
    /// Outer multiplicity `N^{s3}_{s1 s2}` (number of copies).
    multiplicity: usize,
    /// Concatenated copies: copy `mu` occupies `cols[mu·rows·d3 .. (mu+1)·rows·d3]`,
    /// each a column-major `rows × d3` isometry.
    cols: Vec<f64>,
}

impl CatalogCgc {
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
    /// The outer multiplicity `N^{s3}_{s1 s2}` (number of copies).
    pub fn multiplicity(&self) -> usize {
        self.multiplicity
    }
    /// `(rows, cols)` of one copy's isometry: `(d1·d2, d3)`.
    pub fn copy_shape(&self) -> (usize, usize) {
        (self.rows, self.d3)
    }
    /// The isometry of outer-multiplicity copy `mu` (`< multiplicity`) as a flat
    /// column-major `d1·d2 × d3` buffer.
    pub fn copy(&self, mu: usize) -> &[f64] {
        let stride = self.rows * self.d3;
        &self.cols[mu * stride..(mu + 1) * stride]
    }
    /// The whole concatenated coefficient buffer (all copies, in order).
    pub fn data(&self) -> &[f64] {
        &self.cols
    }
}

// ---- the catalog ------------------------------------------------------------

/// Append-only owner of generator sets for one B/C/D family instance.
///
/// See the module docs for the ownership, canonical-parent, and atomicity
/// contracts; `docs/gauge_soN.md` §14 for the canonical-parent order and its
/// well-foundedness argument.
#[derive(Debug)]
pub struct CanonicalCatalog {
    series: Series,
    rank: usize,
    /// The discovered generator sets, keyed by irrep. Append-only; contains the
    /// two base cases (trivial, defining) from construction onward.
    store: HashMap<Irrep, Generators>,
    /// Retained generator bytes (conservative charge), the quantity bounded by
    /// `max_bytes`.
    bytes: usize,
    max_bytes: usize,
}

impl CanonicalCatalog {
    /// Build a catalog for `series` at rank `r` with the default byte budget.
    ///
    /// Seeds the two base cases — the trivial rep and the defining rep, the
    /// latter from the exact S3.1 [`defining_seed`]. Rejects the excluded
    /// low-rank isomorphisms (`SO(3)`, `Sp(2)`, `SO(4)`) with
    /// [`CatalogError::Label`] carrying [`BcdError::ExcludedRank`].
    pub fn new(series: Series, r: usize) -> Result<Self, CatalogError> {
        Self::with_budget(series, r, DEFAULT_MAX_BYTES)
    }

    /// Build a catalog with an explicit byte budget (see [`new`](Self::new)).
    pub fn with_budget(series: Series, r: usize, max_bytes: usize) -> Result<Self, CatalogError> {
        let mut cat = CanonicalCatalog {
            series,
            rank: r,
            store: HashMap::new(),
            bytes: 0,
            max_bytes,
        };
        cat.seed_base()?;
        Ok(cat)
    }

    /// The family series.
    pub fn series(&self) -> Series {
        self.series
    }
    /// The family rank.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Retained generator bytes (the quantity bounded by the budget).
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The byte budget.
    pub fn budget(&self) -> usize {
        self.max_bytes
    }
    /// Number of generator sets currently held (including the two base cases).
    pub fn len(&self) -> usize {
        self.store.len()
    }
    /// Whether the catalog holds no generator sets. Always `false` after a
    /// successful construction (the base cases are seeded), but kept for the
    /// `clippy::len_without_is_empty` contract.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Seed the two base cases: the trivial rep (a 1-dimensional carrier, all
    /// generators zero) and the defining rep (from the exact S3.1 seed). Both
    /// are `≺`-minimal (§14), so every canonical-parent chain bottoms out here.
    fn seed_base(&mut self) -> Result<(), CatalogError> {
        // Rank guard (excluded low-rank isomorphisms) fires here via the seed.
        let seed = defining_seed(self.series, self.rank)?;
        let defining = Generators::from_seed(&seed);
        let defining_irrep = self.defining_irrep()?;
        let trivial_irrep = Irrep::trivial(self.series, self.rank)?;
        let trivial = Generators::trivial(self.series, self.rank);
        self.commit_one(trivial_irrep, trivial);
        self.commit_one(defining_irrep, defining);
        Ok(())
    }

    /// The defining (vector / fundamental) irrep `(1, 0, …, 0)` of this family.
    fn defining_irrep(&self) -> Result<Irrep, CatalogError> {
        let mut dynkin = vec![0i64; self.rank];
        dynkin[0] = 1;
        Ok(Irrep::from_dynkin(self.series, &dynkin)?)
    }

    /// Validate that `c` belongs to this family (same series and rank).
    fn check_group(&self, c: &Irrep) -> Result<(), CatalogError> {
        if c.series() != self.series || c.rank() != self.rank {
            return Err(CatalogError::WrongGroup {
                catalog: (self.series, self.rank),
                got: (c.series(), c.rank()),
            });
        }
        Ok(())
    }

    /// The generator set of `c`, materializing its canonical-parent chain on
    /// demand (atomic under the byte budget). Idempotent: a second call is a map
    /// lookup.
    ///
    /// # Errors
    /// - [`CatalogError::WrongGroup`] if `c` is not of this family.
    /// - [`CatalogError::BudgetExceeded`] if materialization would exceed the
    ///   budget (no partial state is committed).
    /// - [`CatalogError::Sweep`] if a sweep in the chain fails a production gate.
    pub fn generators(&mut self, c: &Irrep) -> Result<&Generators, CatalogError> {
        self.check_group(c)?;
        self.ensure(c)?;
        Ok(self.store.get(c).expect("ensure guarantees presence"))
    }

    /// The Clebsch–Gordan isometry of `s1 ⊗ s2 → s3` (this crate's B/C/D public
    /// CGC surface, mirroring `crate::sun::cgc`).
    ///
    /// The coupling is computed from the **queried** product `s1 ⊗ s2` (not from
    /// `s3`'s canonical parent): the CGC is defined relative to the canonical
    /// bases of `s1`, `s2` (materialized from the catalog) and `s3`. Every
    /// ill-posed triple is a typed error.
    ///
    /// # Errors
    /// - [`CatalogError::WrongGroup`] if the three irreps are not one family.
    /// - [`CatalogError::ZeroFusionChannel`] if `N^{s3}_{s1 s2} = 0` (`s3` does
    ///   not appear in `s1 ⊗ s2`) — the red-first ill-posed-input guard.
    /// - [`CatalogError::BudgetExceeded`] / [`CatalogError::Sweep`] as for
    ///   [`generators`](Self::generators).
    pub fn cgc(&mut self, s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Result<CatalogCgc, CatalogError> {
        self.check_group(s1)?;
        self.check_group(s2)?;
        self.check_group(s3)?;

        // Ill-posed-input guard (PR #14 lesson): every N^c_ab = 0 is a typed
        // error, decided by the exact S3.0 decomposition before any float work.
        let expected = directproduct(s1, s2)?;
        if expected.get(s3).copied().unwrap_or(0) == 0 {
            return Err(CatalogError::ZeroFusionChannel {
                a: s1.dynkin(),
                b: s2.dynkin(),
                c: s3.dynkin(),
            });
        }

        // Canonical bases of the two factors (and s3, for the debug-assert).
        self.ensure(s1)?;
        self.ensure(s2)?;
        self.ensure(s3)?;

        let g1 = self.store.get(s1).expect("ensured").clone();
        let g2 = self.store.get(s2).expect("ensured").clone();
        let product = Generators::product(&g1, &g2)?;
        let decomp = decompose(&product, &expected)?;

        // Collect the s3 copies in outer-multiplicity index order.
        let mut copies: Vec<&Block> = decomp.blocks().iter().filter(|b| b.irrep() == s3).collect();
        copies.sort_by_key(|b| b.outer_multiplicity().0);

        let stored = self.store.get(s3).expect("ensured");
        let (rows, d3) = copies[0].cgc_shape();
        let mut cols = Vec::with_capacity(rows * d3 * copies.len());
        for b in &copies {
            debug_assert_cartan_matches(b, stored);
            cols.extend_from_slice(b.cgc());
        }

        Ok(CatalogCgc {
            s1: s1.clone(),
            s2: s2.clone(),
            s3: s3.clone(),
            rows,
            d3,
            multiplicity: copies.len(),
            cols,
        })
    }

    /// Drop every discovered generator set and re-seed the base cases, returning
    /// the catalog to its just-constructed state. Re-materialization afterward is
    /// bitwise identical (the canonical-parent chain is a deterministic function
    /// of the exact data).
    pub fn reset(&mut self) {
        self.store.clear();
        self.bytes = 0;
        // Base seeding cannot fail here: the rank was validated at construction.
        self.seed_base()
            .expect("base re-seed cannot fail after a valid construction");
    }

    // ---- materialization (compute-fully-then-commit) -----------------------

    /// Ensure `c`'s generators are committed, materializing its canonical-parent
    /// chain atomically: assemble every new set into a staging buffer, check the
    /// total against the budget, and commit only if it fits.
    fn ensure(&mut self, c: &Irrep) -> Result<(), CatalogError> {
        if self.store.contains_key(c) {
            return Ok(());
        }
        let mut staged: Vec<(Irrep, Generators)> = Vec::new();
        build_into(self.series, self.rank, &self.store, &mut staged, c)?;

        let add: usize = staged.iter().map(|(_, g)| gen_bytes(g)).sum();
        let needed = self.bytes + add;
        if needed > self.max_bytes {
            // Atomic failure: discard the staging buffer, commit nothing.
            return Err(CatalogError::BudgetExceeded {
                limit: self.max_bytes,
                needed,
            });
        }
        for (k, v) in staged {
            self.commit_one(k, v);
        }
        Ok(())
    }

    /// Commit one generator set, charging its bytes.
    fn commit_one(&mut self, irrep: Irrep, gens: Generators) {
        self.bytes += gen_bytes(&gens);
        self.store.insert(irrep, gens);
    }

    // ---- test / bench inspection -------------------------------------------

    /// Whether `c`'s generators are currently committed (no materialization).
    #[cfg(test)]
    pub(crate) fn is_materialized(&self, c: &Irrep) -> bool {
        self.store.contains_key(c)
    }

    /// The worst commutator residual of `c`'s stored generators (issue #18
    /// chain-depth error bench). `c` must already be materialized.
    #[cfg(test)]
    pub(crate) fn stored_commutator_residual(&self, c: &Irrep) -> Option<f64> {
        self.store.get(c).map(|g| g.max_commutator_residual())
    }
}

// ---- byte accounting -------------------------------------------------------

/// Conservative retained-byte charge for one generator set: the `r` dense
/// `D×D` raising operators plus the `r` length-`D` Cartan diagonals, over the
/// `f64` coefficient buffers, plus a fixed shell. Over-counts (never under),
/// so the budget stays a true ceiling.
fn gen_bytes(g: &Generators) -> usize {
    let d = g.dim();
    let r = g.rank();
    let f = std::mem::size_of::<f64>();
    r * (d * d + d) * f + std::mem::size_of::<Generators>()
}

// ---- the canonical parent rule (docs/gauge_soN.md §14) ---------------------

/// The number of boxes in an irrep's highest weight: `Σ_i |λ_i|` over the
/// ε-basis partition. Strictly monotone under adding/removing a box, and — unlike
/// `dim` — monotone in **every** coordinate including the D-series sign-carrying
/// last part (see §14.1). The primary `≺` component.
fn box_count(c: &Irrep) -> i64 {
    c.partition().iter().map(|x| x.abs()).sum()
}

/// The `≺` sort key of an irrep: `(box_count, dim, dynkin)` (§14.1). Box count
/// is the primary component so that removing a box always yields a strictly
/// smaller irrep — the fact the existence proof (§14.4) needs and that `dim`
/// alone fails for the D-series chirality pair (`dim` is not monotone in the last
/// partition coordinate: partition `(1,1,0)` has dim 15 > `(1,1,±1)` dim 10).
/// `≺` is a **well order**: box count is a non-negative integer and, at a fixed
/// box count and rank, only finitely many irreps exist (§14.1).
fn prec_key(c: &Irrep) -> (i64, BigInt, Vec<i64>) {
    (box_count(c), c.dim(), c.dynkin())
}

/// The canonical parent pair `(a, b)` of a non-base irrep `c` (§14): among all
/// pairs with `a ≺ c`, `b ≺ c`, and `c ∈ a ⊗ b` (exact S3.0), the minimum under
/// the pair order `(dim_a + dim_b, dim_a, dynkin_a, dynkin_b)`. Returns the pair
/// in canonical `a ⪯ b` form (the order's tie-break fixes which is `a`).
///
/// Existence is guaranteed for every non-base `c` (§14: the pair
/// `(defining, c-minus-a-box)` is always admissible), so the returned `Option`
/// is `None` only if called on a base case (trivial/defining), which the caller
/// never does — those are pre-seeded and short-circuited.
fn canonical_parent(series: Series, rank: usize, c: &Irrep) -> Option<(Irrep, Irrep)> {
    /// A candidate parent pair with its `key(a,b)` (§14.2).
    struct Cand {
        sum: BigInt,
        dim_a: BigInt,
        dynkin_a: Vec<i64>,
        dynkin_b: Vec<i64>,
        a: Irrep,
        b: Irrep,
    }
    impl Cand {
        /// The pair order `(dim_a + dim_b, dim_a, dynkin_a, dynkin_b)`.
        fn key(&self) -> (&BigInt, &BigInt, &Vec<i64>, &Vec<i64>) {
            (&self.sum, &self.dim_a, &self.dynkin_a, &self.dynkin_b)
        }
    }

    let key_c = prec_key(c);
    // All irreps strictly `≺ c` — the finite candidate set for `a` and `b`.
    let below = irreps_below(series, rank, c);

    let mut best: Option<Cand> = None;
    // Iterate `a` in ascending `≺` order; prune once `2·dim_a` exceeds the best
    // sum found (with `a ⪯ b`, `dim_a + dim_b ≥ 2·dim_a`; the minimum pair is
    // always reached via its smaller factor before this fires — §14.4).
    for a in &below {
        let dim_a = a.dim();
        if let Some(cur) = &best {
            if &dim_a * 2 > cur.sum {
                break; // `below` is sorted ascending by (dim, dynkin).
            }
        }
        // `c ∈ a ⊗ b`  ⟺  `b ∈ a* ⊗ c` (Frobenius reciprocity). Enumerate the
        // candidate `b` directly from that product rather than looping all irreps.
        let Ok(prod) = directproduct(&a.dual(), c) else {
            continue;
        };
        for b in prod.keys() {
            if prec_key(b) >= key_c {
                continue; // require b ≺ c
            }
            let cand = Cand {
                sum: &dim_a + b.dim(),
                dim_a: dim_a.clone(),
                dynkin_a: a.dynkin(),
                dynkin_b: b.dynkin(),
                a: a.clone(),
                b: b.clone(),
            };
            if best.as_ref().is_none_or(|cur| cand.key() < cur.key()) {
                best = Some(cand);
            }
        }
    }
    best.map(|c| (c.a, c.b))
}

/// All tensor irreps `x` of `(series, rank)` with `x ≺ c`, sorted ascending by
/// `(dim, dynkin)` — the order the pruning in [`canonical_parent`] relies on.
///
/// Enumerated by a depth-first walk over integer partitions `λ` (ε-basis,
/// nonincreasing, `≥ 0`; the D series additionally emits the `λ_r < 0` chiral
/// partner) bounded by **box count** `Σ|λ_i| ≤ box_count(c)`. Box count is
/// monotone in every coordinate (including the D-series last part, where `dim`
/// is not — the P1 fix), so the prune is exact for all three series. Every
/// `x ≺ c` has `box_count(x) ≤ box_count(c)`, so the walk is a complete
/// superset; the `retain` keeps exactly `{ x : x ≺ c }`. The set is finite
/// because a bounded box count bounds the partition.
fn irreps_below(series: Series, rank: usize, c: &Irrep) -> Vec<Irrep> {
    let max_boxes = box_count(c);
    let key_c = prec_key(c);
    let mut out: Vec<Irrep> = Vec::new();
    let mut cur = vec![0i64; rank];
    enum_partitions(series, rank, max_boxes, 0, 0, &mut cur, &mut out);
    out.retain(|x| prec_key(x) < key_c);
    out.sort_by_key(|x| (x.dim(), x.dynkin()));
    out
}

fn enum_partitions(
    series: Series,
    rank: usize,
    max_boxes: i64,
    pos: usize,
    used: i64,
    cur: &mut Vec<i64>,
    out: &mut Vec<Irrep>,
) {
    if pos == rank {
        push_partition_irrep(series, cur, out);
        return;
    }
    let upper = if pos == 0 { max_boxes } else { cur[pos - 1] };
    let mut v = 0i64;
    while v <= upper {
        // Prune on box count: monotone in v for every coordinate ⇒ safe break.
        if used + v > max_boxes {
            break;
        }
        cur[pos] = v;
        enum_partitions(series, rank, max_boxes, pos + 1, used + v, cur, out);
        v += 1;
    }
    cur[pos] = 0;
}

/// Emit the (non-negative) partition `cur` as an irrep, and — for the D series
/// with `λ_r > 0` — its chiral partner `λ_r ↦ -λ_r` (a distinct tensor irrep of
/// the same box count and dim). `irreps_below`'s `retain` applies the `≺` filter.
fn push_partition_irrep(series: Series, cur: &[i64], out: &mut Vec<Irrep>) {
    out.push(make_irrep(series, cur.to_vec()));
    if series == Series::D {
        let last = cur.len() - 1;
        if cur[last] > 0 {
            let mut w = cur.to_vec();
            w[last] = -w[last];
            out.push(make_irrep(series, w));
        }
    }
}

/// Construct an [`Irrep`] directly from an ε-basis partition `weight` (a
/// descendant module of `bcd` may build the private struct). The enumeration
/// only ever produces valid integer dominant weights, so no validation is
/// needed here.
fn make_irrep(series: Series, weight: Vec<i64>) -> Irrep {
    super::Irrep::from_weight(series, weight)
}

// ---- recursive build into the staging buffer -------------------------------

/// Look up `c`'s generators in the committed store or the staging buffer.
fn lookup<'a>(
    store: &'a HashMap<Irrep, Generators>,
    staged: &'a [(Irrep, Generators)],
    c: &Irrep,
) -> Option<&'a Generators> {
    store
        .get(c)
        .or_else(|| staged.iter().find(|(k, _)| k == c).map(|(_, g)| g))
}

/// Recursively assemble the generator sets `c`'s canonical-parent chain needs
/// but the store does not yet have, into `staged` (no commit, no budget check —
/// [`CanonicalCatalog::ensure`] does both once the whole chain is staged).
///
/// Harvest discipline (Ruling 2): decomposing the canonical parent yields blocks
/// for several irreps; a block's generators are staged **only** if that irrep has
/// no generators yet **and** this product is its canonical parent. A rediscovery
/// (the irrep already committed/staged) never writes — instead it debug-asserts
/// Cartan-spectrum agreement (the `clebsch.cc:6710-6718 @ dd2cc7e` cross-copy
/// `normDiff` check, replaced by-design; §14).
fn build_into(
    series: Series,
    rank: usize,
    store: &HashMap<Irrep, Generators>,
    staged: &mut Vec<(Irrep, Generators)>,
    c: &Irrep,
) -> Result<(), CatalogError> {
    if lookup(store, staged, c).is_some() {
        return Ok(()); // already committed or staged (includes the base cases)
    }

    // Non-base c: its canonical parent exists (§14.4 existence argument). The
    // error path is unreachable by that theorem; kept as defense-in-depth.
    let (a, b) = canonical_parent(series, rank, c)
        .ok_or_else(|| CatalogError::NoCanonicalParent { dynkin: c.dynkin() })?;
    build_into(series, rank, store, staged, &a)?;
    build_into(series, rank, store, staged, &b)?;

    let ga = lookup(store, staged, &a)
        .expect("staged by recursion")
        .clone();
    let gb = lookup(store, staged, &b)
        .expect("staged by recursion")
        .clone();
    let product = Generators::product(&ga, &gb)?;
    let expected = directproduct(&a, &b)?;
    let decomp = decompose(&product, &expected)?;

    for block in decomp.blocks() {
        let ci = block.irrep();
        if let Some(existing) = lookup(store, staged, ci) {
            // Rediscovery: never write; assert the intrinsic Cartan spectrum
            // agrees (deviation-by-design from QSpace's normDiff replacement).
            debug_assert_cartan_matches(block, existing);
            continue;
        }
        // Append only the outer-multiplicity-0 copy, and only when THIS product
        // is ci's canonical parent (query-order-independent gauge).
        if block.outer_multiplicity().0 != 0 {
            continue;
        }
        if canonical_parent(series, rank, ci).as_ref() == Some(&(a.clone(), b.clone())) {
            staged.push((ci.clone(), block.generators().clone()));
        }
    }

    debug_assert!(
        lookup(store, staged, c).is_some(),
        "the canonical parent of c must produce c's block"
    );
    Ok(())
}

/// Debug-assert that a rediscovered block's Cartan (weight) spectrum matches the
/// stored generator set's — the cheap, loud analogue of QSpace's `normDiff`
/// cross-copy check (`clebsch.cc:6712-6718 @ dd2cc7e`).
///
/// Compared as a **multiset** of per-state weight vectors, not state-by-state:
/// the weight *content* of an irrep is gauge-independent, but the state *order*
/// is not. A non-base entry (from a sweep) and a base-case entry (the S3.1 seed,
/// whose native basis is not the sweep's descending-weight order) can therefore
/// carry the same weights in a different order — a multiset check is the correct,
/// gauge-independent statement of "same irrep, same weight system". Weights are
/// integer Cartan eigenvalues (snapped in the sweep, §6), so they compare exactly
/// after rounding.
fn debug_assert_cartan_matches(block: &Block, stored: &Generators) {
    debug_assert_eq!(
        block.dim(),
        stored.dim(),
        "rediscovered block dim disagrees with stored generators"
    );
    if !cfg!(debug_assertions) {
        return;
    }
    let rank = stored.rank();
    let d = stored.dim();
    let round = |x: f64| x.round() as i64;
    let mut block_w: Vec<Vec<i64>> = (0..d)
        .map(|s| (0..rank).map(|j| round(block.weight(s, j))).collect())
        .collect();
    let mut stored_w: Vec<Vec<i64>> = (0..d)
        .map(|s| (0..rank).map(|j| stored.cartan_diag(j)[s] as i64).collect())
        .collect();
    block_w.sort_unstable();
    stored_w.sort_unstable();
    debug_assert_eq!(
        block_w, stored_w,
        "rediscovered block weight multiset disagrees with stored generators"
    );
}

#[cfg(test)]
mod tests;
