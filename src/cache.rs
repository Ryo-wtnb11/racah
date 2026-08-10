//! Bounded, thread-safe evaluation cache for 3j/6j symbols, keyed by canonical
//! Regge classes.
//!
//! In the tensor-network consumption pattern the same small labels recur many
//! thousands of times, so a warm hit should cost a hash lookup rather than a
//! fresh big-rational Racah sum. WignerSymbols.jl v2.0.0 caches transparently
//! inside `wigner3j`/`wigner6j` (per-kind `LRU` dicts keyed by canonical Regge
//! labels); racah follows that model but bounds the cache by policy — Julia's
//! caches are effectively unbounded in entries.
//!
//! # Why no gauge/version key component (in-process)
//!
//! A canonical Regge class names exactly one exact symbol value. The stored
//! [`SignedSqrtRational`] is that exact value — not a gauge- or
//! algorithm-dependent float — so within one process the canonical key is a
//! complete key: no gauge tag and no algorithm-version tag can change which
//! value a class maps to. (Contrast the `cgc-gen` coefficient caches, whose
//! floating values *are* gauge- and algorithm-dependent and are versioned.)
//!
//! # Why this cache must never be persisted to disk
//!
//! Persisting these entries across builds would reintroduce exactly the
//! versioning problem the in-process argument avoids: a future change to the
//! exact engine (a different but still-correct series arrangement, a widened
//! type, a bug fix) could alter the stored bytes for a class, and a persisted
//! store would then need an algorithm-version key to stay sound. Keeping the
//! cache process-local sidesteps that entirely — it is rebuilt from the engine
//! every run, so it can never disagree with the engine that filled it.
//!
//! # Base cache resource contract (static partition)
//!
//! The three base SU(2) tiers (3j, 6j, derived-F) are each bounded
//! independently by a per-tier entry and retained-charge cap; the documented
//! aggregate cap [`BASE_CACHE_MAX_BYTES`] is simply their sum. This is a **static
//! partition, not a dynamic shared pool**: a shared budget would couple
//! eviction across tiers whose entries differ wildly in size (big-rational
//! exact symbols vs `f64` scalars) and whose hit patterns are unrelated, for no
//! measured benefit — so it is deliberately rejected here and revisited only
//! with measurements. Because each per-tier charged-entry cap is enforced, the
//! aggregate charged-entry bound
//! holds as a corollary rather than needing global enforcement. The charge
//! covers entries currently owned by the cache. It excludes `HashMap`/
//! `VecDeque` retained capacity and scaffolding, allocator metadata and RSS,
//! transient or external clones, and values returned through public APIs.
//! Per-tier and total statistics are exposed via [`base_cache_stats`]; reset
//! ownership is on [`reset`]. (Design record: racah #43, PR-A.)

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock, RwLock};

use crate::exact::SignedSqrtRational;
use crate::su2::{FKey, Regge3j, Regge6j};

/// Default entry cap per kind (3j and 6j each). Matches the reference order of
/// magnitude (WignerSymbols.jl uses `10^6`); the retained-charge cap is the real
/// backstop.
const DEFAULT_MAX_ENTRIES: usize = 1 << 20;

/// Default retained-charge cap per kind. At the ~O(1)-limb sizes typical of
/// small-label TN work an entry charges well under a kilobyte, so 64 MiB holds
/// a large working set while bounding the conservative charge of cache-owned
/// entries, not allocator-live memory or RSS.
const DEFAULT_MAX_BYTES: usize = 64 << 20;

/// Process-local retained-charge limits for coefficient-cache tiers.
///
/// [`configure_cache_budgets`] accepts this value once, before the first cache
/// operation or cache-policy observation. Zero keeps computing values but
/// retains none for that tier. [`Default`] is the compiled policy and also the
/// maximum accepted policy; budgets can only shrink it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientCacheBudgets {
    /// Exact Wigner 3j retained-charge cap.
    three_j_bytes: usize,
    /// Exact Wigner 6j retained-charge cap.
    six_j_bytes: usize,
    /// Derived SU(2) F-symbol retained-charge cap.
    derived_f_bytes: usize,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) product retained-charge cap.
    sun_product_bytes: usize,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) CGC retained-charge cap.
    sun_cgc_bytes: usize,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) F-symbol retained-charge cap.
    sun_f_bytes: usize,
    #[cfg(feature = "cgc-gen")]
    /// B/C/D CGC retained-charge cap.
    bcd_cgc_bytes: usize,
    #[cfg(feature = "cgc-gen")]
    /// B/C/D F-symbol retained-charge cap.
    bcd_f_bytes: usize,
}

/// One existing coefficient-cache tier.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientCacheTier {
    /// Exact Wigner 3j tier.
    ThreeJ,
    /// Exact Wigner 6j tier.
    SixJ,
    /// Derived SU(2) F-symbol tier.
    DerivedF,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) product tier.
    SunProduct,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) CGC tier.
    SunCgc,
    #[cfg(feature = "cgc-gen")]
    /// SU(N) F-symbol tier.
    SunF,
    #[cfg(feature = "cgc-gen")]
    /// B/C/D CGC tier.
    BcdCgc,
    #[cfg(feature = "cgc-gen")]
    /// B/C/D F-symbol tier.
    BcdF,
}

impl std::fmt::Display for CoefficientCacheTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ThreeJ => "three_j",
            Self::SixJ => "six_j",
            Self::DerivedF => "derived_f",
            #[cfg(feature = "cgc-gen")]
            Self::SunProduct => "sun_product",
            #[cfg(feature = "cgc-gen")]
            Self::SunCgc => "sun_cgc",
            #[cfg(feature = "cgc-gen")]
            Self::SunF => "sun_f",
            #[cfg(feature = "cgc-gen")]
            Self::BcdCgc => "bcd_cgc",
            #[cfg(feature = "cgc-gen")]
            Self::BcdF => "bcd_f",
        })
    }
}

impl CoefficientCacheBudgets {
    /// Disable retention in every compiled tier while preserving evaluation.
    pub fn disabled() -> Self {
        let mut budgets = Self::default();
        for tier in [
            CoefficientCacheTier::ThreeJ,
            CoefficientCacheTier::SixJ,
            CoefficientCacheTier::DerivedF,
        ] {
            budgets = budgets.with_limit(tier, 0);
        }
        #[cfg(feature = "cgc-gen")]
        for tier in [
            CoefficientCacheTier::SunProduct,
            CoefficientCacheTier::SunCgc,
            CoefficientCacheTier::SunF,
            CoefficientCacheTier::BcdCgc,
            CoefficientCacheTier::BcdF,
        ] {
            budgets = budgets.with_limit(tier, 0);
        }
        budgets
    }

    /// Return this policy with one tier's retained-charge cap replaced.
    pub fn with_limit(mut self, tier: CoefficientCacheTier, bytes: usize) -> Self {
        match tier {
            CoefficientCacheTier::ThreeJ => self.three_j_bytes = bytes,
            CoefficientCacheTier::SixJ => self.six_j_bytes = bytes,
            CoefficientCacheTier::DerivedF => self.derived_f_bytes = bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunProduct => self.sun_product_bytes = bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes = bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunF => self.sun_f_bytes = bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes = bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes = bytes,
        }
        self
    }

    /// Return one tier's retained-charge cap.
    pub fn limit(&self, tier: CoefficientCacheTier) -> usize {
        match tier {
            CoefficientCacheTier::ThreeJ => self.three_j_bytes,
            CoefficientCacheTier::SixJ => self.six_j_bytes,
            CoefficientCacheTier::DerivedF => self.derived_f_bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunProduct => self.sun_product_bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::SunF => self.sun_f_bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes,
            #[cfg(feature = "cgc-gen")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes,
        }
    }
}

impl Default for CoefficientCacheBudgets {
    fn default() -> Self {
        Self {
            three_j_bytes: DEFAULT_MAX_BYTES,
            six_j_bytes: DEFAULT_MAX_BYTES,
            derived_f_bytes: DEFAULT_MAX_BYTES,
            #[cfg(feature = "cgc-gen")]
            sun_product_bytes: sun_product_cache::SUN_PRODUCT_MAX_BYTES,
            #[cfg(feature = "cgc-gen")]
            sun_cgc_bytes: cgc_cache::CGC_MAX_BYTES,
            #[cfg(feature = "cgc-gen")]
            sun_f_bytes: sun_f_cache::SUN_F_MAX_BYTES,
            #[cfg(feature = "cgc-gen")]
            bcd_cgc_bytes: bcd_cgc_cache::BCD_CGC_MAX_BYTES,
            #[cfg(feature = "cgc-gen")]
            bcd_f_bytes: bcd_f_cache::BCD_F_MAX_BYTES,
        }
    }
}

/// Failed coefficient-cache policy initialization.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheBudgetError {
    /// A cache operation or policy observation already fixed this process's policy.
    AlreadyInitialized,
    /// A requested tier limit exceeds its compiled maximum.
    ExceedsMaximum {
        /// Stable tier name for diagnostics.
        tier: CoefficientCacheTier,
        /// Rejected requested cap.
        requested: usize,
        /// Compiled maximum cap.
        maximum: usize,
    },
    /// Summing validated tier limits overflowed `usize`.
    AggregateOverflow,
}

impl std::fmt::Display for CacheBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInitialized => {
                f.write_str("coefficient-cache policy is already initialized")
            }
            Self::ExceedsMaximum {
                tier,
                requested,
                maximum,
            } => write!(
                f,
                "{tier} budget {requested} exceeds compiled maximum {maximum}"
            ),
            Self::AggregateOverflow => {
                f.write_str("coefficient-cache budget aggregate overflowed usize")
            }
        }
    }
}
impl std::error::Error for CacheBudgetError {}

static CACHE_BUDGETS: OnceLock<CoefficientCacheBudgets> = OnceLock::new();

fn effective_budgets() -> &'static CoefficientCacheBudgets {
    effective_budgets_in(&CACHE_BUDGETS)
}

fn effective_budgets_in(cell: &OnceLock<CoefficientCacheBudgets>) -> &CoefficientCacheBudgets {
    cell.get_or_init(CoefficientCacheBudgets::default)
}

/// Configure one shrink-only coefficient-cache policy for this process.
pub fn configure_cache_budgets(budgets: CoefficientCacheBudgets) -> Result<(), CacheBudgetError> {
    validate_budgets(budgets)?;
    configure_budgets_in(&CACHE_BUDGETS, budgets)
}

fn configure_budgets_in(
    cell: &OnceLock<CoefficientCacheBudgets>,
    budgets: CoefficientCacheBudgets,
) -> Result<(), CacheBudgetError> {
    cell.set(budgets)
        .map_err(|_| CacheBudgetError::AlreadyInitialized)
}

/// Return the effective policy, fixing the default if it was not configured.
pub fn cache_budgets() -> CoefficientCacheBudgets {
    *effective_budgets()
}

fn validate_budgets(b: CoefficientCacheBudgets) -> Result<(), CacheBudgetError> {
    let maximum = CoefficientCacheBudgets::default();
    let fields = [
        (
            CoefficientCacheTier::ThreeJ,
            b.three_j_bytes,
            maximum.three_j_bytes,
        ),
        (
            CoefficientCacheTier::SixJ,
            b.six_j_bytes,
            maximum.six_j_bytes,
        ),
        (
            CoefficientCacheTier::DerivedF,
            b.derived_f_bytes,
            maximum.derived_f_bytes,
        ),
        #[cfg(feature = "cgc-gen")]
        (
            CoefficientCacheTier::SunProduct,
            b.sun_product_bytes,
            maximum.sun_product_bytes,
        ),
        #[cfg(feature = "cgc-gen")]
        (
            CoefficientCacheTier::SunCgc,
            b.sun_cgc_bytes,
            maximum.sun_cgc_bytes,
        ),
        #[cfg(feature = "cgc-gen")]
        (
            CoefficientCacheTier::SunF,
            b.sun_f_bytes,
            maximum.sun_f_bytes,
        ),
        #[cfg(feature = "cgc-gen")]
        (
            CoefficientCacheTier::BcdCgc,
            b.bcd_cgc_bytes,
            maximum.bcd_cgc_bytes,
        ),
        #[cfg(feature = "cgc-gen")]
        (
            CoefficientCacheTier::BcdF,
            b.bcd_f_bytes,
            maximum.bcd_f_bytes,
        ),
    ];
    let mut total = 0usize;
    for (tier, requested, limit) in fields {
        if requested > limit {
            return Err(CacheBudgetError::ExceedsMaximum {
                tier,
                requested,
                maximum: limit,
            });
        }
        total = total
            .checked_add(requested)
            .ok_or(CacheBudgetError::AggregateOverflow)?;
    }
    Ok(())
}

/// Aggregate conservative retained-charge cap for the three base SU(2) tiers
/// (3j, 6j, derived-F), currently `192 MiB` = `3 × 64 MiB`.
///
/// This is a **documented static partition**, not a shared budget: each tier is
/// bounded independently by its own per-tier charged-entry cap
/// (`DEFAULT_MAX_BYTES`). The aggregate is therefore a provable corollary —
/// `Σ tier bytes ≤ Σ tier caps = BASE_CACHE_MAX_BYTES` — rather than an
/// enforced global limit. A dynamic shared pool (tiers competing for one
/// budget) is deliberately rejected: it would couple eviction across tiers with
/// very different entry sizes (big-rational vs `f64`) and hit patterns for no
/// measured benefit.
///
/// The `const` assertion below ties this constant to the per-tier cap so the
/// two cannot silently drift; all three base tiers (`CACHE_3J`, `CACHE_6J`,
/// `CACHE_F`) are constructed with the same `DEFAULT_MAX_BYTES`.
pub const BASE_CACHE_MAX_BYTES: usize = 192 << 20;

// Compile-time tie: if the per-tier retained-charge cap changes, BASE_CACHE_MAX_BYTES must
// be reconciled in the same edit or the crate stops building. (There is no
// compile-time way to read the tiers' runtime `max_bytes`; anchoring to the
// shared `DEFAULT_MAX_BYTES` they are all built from is the enforceable tie.)
const _: () = assert!(BASE_CACHE_MAX_BYTES == 3 * DEFAULT_MAX_BYTES);

/// Aggregate conservative retained-charge cap for the five generated `cgc-gen` tiers
/// (SU(N) product, SU(N) CGC, SU(N) F, B/C/D CGC, B/C/D F), currently
/// `640 MiB + 128 KiB` = `128 KiB + 256 MiB + 64 MiB + 256 MiB + 64 MiB`.
///
/// **Unstable: shape may change while the generated-provider contract is
/// negotiated** (racah #47; there is no Cargo-feature way to express an
/// instability tier, so this doc label plus that issue are the ledger).
///
/// # Two-layer aggregate story (why there is no single crate-wide constant)
///
/// Retained coefficient-cache entry charge is documented in two layers, not one
/// number:
///
/// - the base SU(2) tiers are bounded by [`BASE_CACHE_MAX_BYTES`] (a static
///   partition with a const-proved sum — see its docs);
/// - the generated tiers are bounded by this constant.
///
/// The whole-process coefficient-cache charged-entry cap is the **documented sum**
/// `BASE_CACHE_MAX_BYTES + GENERATED_CACHE_MAX_BYTES`. There is deliberately no
/// single cross-feature constant spanning both: this constant only exists under
/// `cgc-gen`, so a "one number" whole-crate cap would *change value with the
/// feature flag* and read as if the base cap shrank when `cgc-gen` is off —
/// misleading. Two feature-honest layers instead (racah #47 design record 2,
/// D4). Like the base cap this is a **static partition, not a shared pool**:
/// `Σ tier charged bytes ≤ Σ tier caps = GENERATED_CACHE_MAX_BYTES` holds as a
/// corollary without global enforcement.
///
/// Per-tier and total statistics are exposed via [`generated_cache_stats`]. The
/// `CanonicalCatalog` is *not* a value cache (generator state, its own byte
/// budget, `&mut` caller-owned lifecycle) and is intentionally excluded from
/// this budget and from these stats.
#[cfg(feature = "cgc-gen")]
pub const GENERATED_CACHE_MAX_BYTES: usize = (640 << 20) + (128 << 10);

// Compile-time tie: if any generated-tier retained-charge cap changes, this constant must
// be reconciled in the same edit or the crate stops building (the same drift
// guard the base tiers use). Each cap is `pub(super)` in its tier module.
#[cfg(feature = "cgc-gen")]
const _: () = assert!(
    GENERATED_CACHE_MAX_BYTES
        == cgc_cache::CGC_MAX_BYTES
            + sun_product_cache::SUN_PRODUCT_MAX_BYTES
            + sun_f_cache::SUN_F_MAX_BYTES
            + bcd_cgc_cache::BCD_CGC_MAX_BYTES
            + bcd_f_cache::BCD_F_MAX_BYTES
);

/// Snapshot of the aggregate cache counters (3j and 6j kinds summed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from a stored value.
    pub hits: u64,
    /// Lookups that had to compute the value. Under a concurrent-miss race the
    /// losing thread counts a miss without inserting (the winner already
    /// stored the entry), so `misses` can slightly exceed the number of stored
    /// entries.
    pub misses: u64,
    /// Entries currently retained across both kinds.
    pub entries: usize,
    /// Conservatively charged bytes currently retained across both kinds.
    pub bytes: usize,
}

/// Per-tier snapshot of one base SU(2) coefficient cache (3j, 6j, or derived-F).
///
/// The fields are consistent for the tier they describe (entries/bytes read
/// under the tier lock). See [`base_cache_stats`] and [`BaseCacheStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierStats {
    /// Entries currently retained in this tier.
    pub entries: usize,
    /// Conservatively charged bytes currently retained in this tier.
    pub bytes: usize,
    /// Lookups served from a stored value in this tier.
    pub hits: u64,
    /// Lookups that had to compute the value in this tier. Under a
    /// concurrent-miss race the losing thread counts a miss without inserting,
    /// so `misses` can slightly exceed the number of stored entries.
    pub misses: u64,
    /// Entries removed from this tier by eviction over its lifetime, including
    /// an entry larger than the retained-charge cap that is admitted then immediately
    /// evicted back out (it never fit, but it was charged, so it counts).
    pub evictions: u64,
}

/// Per-tier statistics for the three base SU(2) coefficient tiers.
///
/// Covers **only** the 3j, 6j, and derived-F tiers by definition — the base
/// SU(2) provider surface. (This is distinct from the aggregate [`stats`], which
/// under the `cgc-gen` feature also sums the generated SU(N)/B/C/D tiers.)
///
/// # Snapshot consistency
///
/// Each per-tier [`TierStats`] is internally consistent (taken under that tier's
/// read lock). [`total`](BaseCacheStats::total) is a field-wise sum of the three
/// per-tier snapshots, **not** a single global atomic snapshot: a concurrent
/// filler can interleave between the tier reads, so the total is only
/// eventually consistent. Racah does not take a global lock spanning the tiers —
/// that would serialize otherwise-independent lookups for no correctness gain
/// (the individual tier charged-entry caps already hold independently).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseCacheStats {
    /// The exact 3j tier.
    pub three_j: TierStats,
    /// The exact 6j tier.
    pub six_j: TierStats,
    /// The derived-f64 F-symbol tier.
    pub derived_f: TierStats,
}

impl BaseCacheStats {
    /// Field-wise sum of the three base tiers. See the type-level snapshot-
    /// consistency note: this is a sum of per-tier snapshots, not an atomic
    /// whole-cache snapshot.
    pub fn total(&self) -> TierStats {
        TierStats {
            entries: self.three_j.entries + self.six_j.entries + self.derived_f.entries,
            bytes: self.three_j.bytes + self.six_j.bytes + self.derived_f.bytes,
            hits: self.three_j.hits + self.six_j.hits + self.derived_f.hits,
            misses: self.three_j.misses + self.six_j.misses + self.derived_f.misses,
            evictions: self.three_j.evictions + self.six_j.evictions + self.derived_f.evictions,
        }
    }
}

/// Per-tier statistics for the five generated `cgc-gen` tiers (SU(N) product,
/// SU(N) CGC, SU(N) F, B/C/D CGC, B/C/D F).
///
/// **Unstable: shape may change while the generated-provider contract is
/// negotiated** (racah #47). The struct is `#[non_exhaustive]` — it is
/// constructed only inside the crate (by [`generated_cache_stats`]); consumers
/// read its fields or call [`total`](GeneratedCacheStats::total).
///
/// Reuses the base [`TierStats`] type (no new vocabulary). This covers **only**
/// the generated SU(N)/B/C/D tiers — the base SU(2) surface is
/// [`base_cache_stats`], and the aggregate [`stats`] sums both. Conservative
/// retained entry charge is capped by [`GENERATED_CACHE_MAX_BYTES`]
/// (`total().bytes ≤ GENERATED_CACHE_MAX_BYTES`).
///
/// # Snapshot consistency
///
/// Each per-tier [`TierStats`] is internally consistent (taken under that tier's
/// read lock). [`total`](GeneratedCacheStats::total) is a field-wise sum of the
/// five per-tier snapshots, **not** a single global atomic snapshot: a
/// concurrent filler can interleave between the tier reads, so the total is only
/// eventually consistent. Racah does not take a global lock spanning the tiers —
/// that would serialize otherwise-independent lookups for no correctness gain
/// (the individual tier charged-entry caps hold independently). Same contract as
/// [`BaseCacheStats`].
#[cfg(feature = "cgc-gen")]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeneratedCacheStats {
    /// The exact SU(N) tensor-product decomposition tier.
    pub sun_product: TierStats,
    /// The SU(N) CGC tier.
    pub sun_cgc: TierStats,
    /// The derived-f64 SU(N) F-symbol tier.
    pub sun_f: TierStats,
    /// The B/C/D CGC value tier.
    pub bcd_cgc: TierStats,
    /// The derived-f64 B/C/D F-symbol tier.
    pub bcd_f: TierStats,
}

#[cfg(feature = "cgc-gen")]
impl GeneratedCacheStats {
    /// Field-wise sum of the five generated tiers. See the type-level snapshot-
    /// consistency note: this is a sum of per-tier snapshots, not an atomic
    /// whole-cache snapshot.
    pub fn total(&self) -> TierStats {
        TierStats {
            entries: self.sun_product.entries
                + self.sun_cgc.entries
                + self.sun_f.entries
                + self.bcd_cgc.entries
                + self.bcd_f.entries,
            bytes: self.sun_product.bytes
                + self.sun_cgc.bytes
                + self.sun_f.bytes
                + self.bcd_cgc.bytes
                + self.bcd_f.bytes,
            hits: self.sun_product.hits
                + self.sun_cgc.hits
                + self.sun_f.hits
                + self.bcd_cgc.hits
                + self.bcd_f.hits,
            misses: self.sun_product.misses
                + self.sun_cgc.misses
                + self.sun_f.misses
                + self.bcd_cgc.misses
                + self.bcd_f.misses,
            evictions: self.sun_product.evictions
                + self.sun_cgc.evictions
                + self.sun_f.evictions
                + self.bcd_cgc.evictions
                + self.bcd_f.evictions,
        }
    }
}

/// Conservative retained-byte charge for a stored value, implemented per value
/// type as one component of the cache-owned entry charge.
///
/// The exact tier stores a [`SignedSqrtRational`] whose size is data-dependent
/// (big-integer limbs), so it must measure itself; the derived-f64 tier stores
/// a fixed-size scalar. Keys use the sibling [`CacheKeyCharge`] contract so
/// heap-backed generated labels cannot escape the same charged-entry cap.
pub(crate) trait CacheCharge {
    /// Bytes charged for one stored value (over-counts, never under-counts).
    fn value_bytes(&self) -> usize;
}

/// Conservative retained-byte charge for one stored key.
///
/// A FIFO entry owns two key clones (the hash map key and insertion-order key),
/// so heap-backed labels must report both their inline shell and owned backing.
pub(crate) trait CacheKeyCharge {
    /// Bytes retained by one key clone (over-counts, never under-counts).
    fn key_bytes(&self) -> usize;
}

macro_rules! fixed_key_charge {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CacheKeyCharge for $ty {
                fn key_bytes(&self) -> usize {
                    std::mem::size_of::<Self>()
                }
            }
        )+
    };
}

fixed_key_charge!(u32, Regge3j, Regge6j, FKey);

#[cfg(feature = "cgc-gen")]
impl<I: CacheKeyCharge> CacheKeyCharge for (I, I, I) {
    fn key_bytes(&self) -> usize {
        self.0
            .key_bytes()
            .saturating_add(self.1.key_bytes())
            .saturating_add(self.2.key_bytes())
    }
}

#[cfg(feature = "cgc-gen")]
impl<I: CacheKeyCharge> CacheKeyCharge for (I, I, I, I, I, I) {
    fn key_bytes(&self) -> usize {
        self.0
            .key_bytes()
            .saturating_add(self.1.key_bytes())
            .saturating_add(self.2.key_bytes())
            .saturating_add(self.3.key_bytes())
            .saturating_add(self.4.key_bytes())
            .saturating_add(self.5.key_bytes())
    }
}

impl CacheCharge for SignedSqrtRational {
    fn value_bytes(&self) -> usize {
        let r = self.radicand();
        let value_limbs = (r.numer().bits() + r.denom().bits()).div_ceil(8) as usize;
        // Two BigInt allocations (numer, denom) plus the SignedSqrtRational shell.
        const BIGINT_OVERHEAD: usize = 32;
        std::mem::size_of::<SignedSqrtRational>() + 2 * BIGINT_OVERHEAD + value_limbs
    }
}

impl CacheCharge for f64 {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<f64>()
    }
}

/// Conservative retained-byte charge for one stored entry keyed by `K`.
///
/// Counts the value (via [`CacheCharge`]) plus the key stored twice (once in
/// the map, once in the FIFO order queue). This is the conservative retained
/// charge of entries currently owned by the cache; it excludes container
/// retained capacity/scaffolding, allocator metadata/RSS, transient or external
/// clones, and returned public values.
fn entry_charge<K: CacheKeyCharge, V: CacheCharge>(key: &K, value: &V) -> usize {
    value
        .value_bytes()
        .saturating_add(2usize.saturating_mul(key.key_bytes()))
}

struct Inner<K, V> {
    map: HashMap<K, V>,
    /// Insertion order for FIFO eviction (front = oldest).
    order: VecDeque<K>,
    bytes: usize,
}

/// A bounded, thread-safe map from a canonical Regge key to its exact value.
///
/// Eviction policy: **FIFO**, not LRU. WignerSymbols.jl uses LRU, but LRU must
/// reorder recency on every hit, which forces a write lock on the hot read
/// path. FIFO lets a hit take only a read lock (the read-fast-path). In the
/// repeated-label regime the working set is small and fits the budget, so
/// eviction rarely fires; while it does not fire FIFO and LRU behave
/// identically, and when it does the exact value is recomputed on the next
/// miss — the choice never affects a returned value, only lock contention. So
/// FIFO is the cheaper policy for the same correctness.
pub(crate) struct FifoCache<K, V> {
    inner: RwLock<Inner<K, V>>,
    hits: AtomicU64,
    misses: AtomicU64,
    /// Entries removed by [`Self::evict`] over the cache's lifetime. Counts the
    /// oversize-entry immediate-eviction path (`src/cache.rs` `evict`) too: such
    /// an entry is admitted (charged, pushed) and then evicted back out on the
    /// same insert, so counting it keeps the byte-bound story honest — every
    /// admission that later leaves the map is one eviction.
    evictions: AtomicU64,
    max_entries: usize,
    max_bytes: usize,
}

impl<K: Clone + Eq + Hash + CacheKeyCharge, V: Clone + CacheCharge> FifoCache<K, V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        FifoCache {
            inner: RwLock::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            max_entries,
            max_bytes,
        }
    }

    /// Return the value for `key`, computing and storing it on a miss.
    ///
    /// Read-fast-path: a hit takes only a read lock and clones the stored
    /// value. A miss computes `compute()` *outside* any lock (the big-rational
    /// sum is the expensive part and must not serialize other readers), then
    /// takes the write lock to insert, re-checking in case a concurrent miss
    /// already stored it.
    pub(crate) fn get_or_compute(&self, key: K, compute: impl FnOnce() -> V) -> V {
        if let Some(v) = self.inner.read().unwrap().map.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return v.clone();
        }

        let value = compute();
        self.misses.fetch_add(1, Ordering::Relaxed);

        let mut inner = self.inner.write().unwrap();
        // A concurrent miss may have inserted between our read and this write.
        if let Some(v) = inner.map.get(&key) {
            return v.clone();
        }
        let charge = entry_charge(&key, &value);
        if !self.make_room(&mut inner, charge) {
            return value;
        }
        inner.bytes = inner
            .bytes
            .checked_add(charge)
            .expect("bounded cache charge");
        inner.order.push_back(key.clone());
        inner.map.insert(key, value.clone());
        value
    }

    /// Rejecting a new oversize/zero-cap entry does not discard older entries.
    fn make_room(&self, inner: &mut Inner<K, V>, charge: usize) -> bool {
        if self.max_entries == 0 || charge > self.max_bytes {
            self.evictions.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let bytes_before = self.max_bytes - charge;
        while inner.map.len() >= self.max_entries || inner.bytes > bytes_before {
            let Some(old) = inner.order.pop_front() else {
                break;
            };
            if let Some(value) = inner.map.remove(&old) {
                inner.bytes = inner
                    .bytes
                    .checked_sub(entry_charge(&old, &value))
                    .expect("cache charge accounting invariant");
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
        debug_assert!(inner.bytes <= bytes_before);
        true
    }

    /// Read-fast-path lookup: return a clone of the stored value on a hit
    /// (counted), `None` on a miss (not counted -- the caller decides whether to
    /// compute and [`Self::insert`]). Used by the fallible `cgc-gen` generation
    /// path, where a computation can error and errors must not be cached.
    #[cfg(feature = "cgc-gen")]
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let v = self.inner.read().unwrap().map.get(key).cloned();
        if v.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        v
    }

    /// Insert `value` for `key` (counting a miss) and return the value that
    /// ends up stored -- the existing one if a concurrent insert won the race,
    /// so all racers observe the same value.
    #[cfg(feature = "cgc-gen")]
    pub(crate) fn insert(&self, key: K, value: V) -> V {
        self.misses.fetch_add(1, Ordering::Relaxed);
        let mut inner = self.inner.write().unwrap();
        if let Some(v) = inner.map.get(&key) {
            return v.clone();
        }
        let charge = entry_charge(&key, &value);
        if !self.make_room(&mut inner, charge) {
            return value;
        }
        inner.bytes = inner
            .bytes
            .checked_add(charge)
            .expect("bounded cache charge");
        inner.order.push_back(key.clone());
        inner.map.insert(key, value.clone());
        value
    }

    fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.map.clear();
        inner.order.clear();
        inner.bytes = 0;
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Per-tier snapshot including the eviction counter. Entries/bytes are read
    /// under the tier read lock so they agree with each other; the atomic
    /// counters are `Relaxed` reads taken alongside. This snapshot is internally
    /// consistent for one tier — the cross-tier sum in [`BaseCacheStats::total`]
    /// is not a global atomic snapshot (see its docs).
    fn tier_stats(&self) -> TierStats {
        let inner = self.inner.read().unwrap();
        TierStats {
            entries: inner.map.len(),
            bytes: inner.bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    fn snapshot(&self) -> (u64, u64, usize, usize) {
        let inner = self.inner.read().unwrap();
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            inner.map.len(),
            inner.bytes,
        )
    }
}

static CACHE_3J: LazyLock<FifoCache<Regge3j, SignedSqrtRational>> =
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().three_j_bytes));
static CACHE_6J: LazyLock<FifoCache<Regge6j, SignedSqrtRational>> =
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().six_j_bytes));

pub(crate) fn cache_3j() -> &'static FifoCache<Regge3j, SignedSqrtRational> {
    &CACHE_3J
}

pub(crate) fn cache_6j() -> &'static FifoCache<Regge6j, SignedSqrtRational> {
    &CACHE_6J
}

/// Derived-f64 F-symbol tier (#7). Stores the rounded `f64` F-symbol so a warm
/// hit returns a `Copy` scalar without re-running the bigint `sqrt` in
/// [`SignedSqrtRational::to_f64`]. It is a *presentation* tier over the exact
/// 6j tier (the value authority), never an independent value source: its `f64`
/// is always derived from the exact value, so the two cannot disagree.
static CACHE_F: LazyLock<FifoCache<FKey, f64>> =
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().derived_f_bytes));

pub(crate) fn cache_f() -> &'static FifoCache<FKey, f64> {
    &CACHE_F
}

/// Bounded, byte-accounted SU(N) CGC cache (`cgc-gen`).
///
/// CGC tensors are large and expensive (a full SVD/QR/least-squares pipeline),
/// so unlike the exact 3j/6j tiers this cache is charged by *actual sparse
/// storage bytes* ([`crate::sun::Cgc`] entry vector + labels) and holds
/// `Arc<Cgc>` for cheap hit-path cloning. Keyed by the canonical
/// `(s1, s2, s3)` labels.
///
/// # Why in-memory only (no disk tier)
///
/// The reference persists CGCs to a scratch directory. This crate deliberately
/// does not: a persisted store would need an algorithm/gauge-version key to
/// stay sound, because the coefficient *values* are gauge- and
/// algorithm-dependent (unlike the exact 3j/6j tiers, whose bytes are the
/// canonical exact value). Keeping the cache process-local means it is rebuilt
/// from the generator every run and can never disagree with the generator that
/// filled it -- the same argument the exact tiers make for never persisting.
#[cfg(feature = "cgc-gen")]
mod cgc_cache {
    use super::{CacheCharge, FifoCache};
    use crate::sun::{Cgc, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Canonical cache key: the three irrep labels.
    pub(crate) type CgcKey = (Irrep, Irrep, Irrep);

    impl CacheCharge for Arc<Cgc> {
        fn value_bytes(&self) -> usize {
            self.storage_bytes()
        }
    }

    /// Entry cap for the CGC tier. The retained-charge cap is the real backstop.
    const CGC_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap for the CGC tier (256 MiB): CGC tensors are far larger
    /// than a scalar exact symbol, so this tier gets its own generous budget.
    ///
    /// `pub(super)` so the parent module can tie [`super::GENERATED_CACHE_MAX_BYTES`]
    /// to it in a compile-time assertion (the same drift guard the base tiers use).
    pub(super) const CGC_MAX_BYTES: usize = 256 << 20;

    pub(crate) static CACHE_CGC: LazyLock<FifoCache<CgcKey, Arc<Cgc>>> =
        LazyLock::new(|| FifoCache::new(CGC_MAX_ENTRIES, super::effective_budgets().sun_cgc_bytes));
}

#[cfg(feature = "cgc-gen")]
pub(crate) fn cache_cgc() -> &'static FifoCache<cgc_cache::CgcKey, std::sync::Arc<crate::sun::Cgc>>
{
    &cgc_cache::CACHE_CGC
}

/// Bounded exact SU(N) tensor-product tier (`cgc-gen`, issue #59).
///
/// The unordered irrep pair is a complete key after `sun::directproduct`
/// rejects rank mismatch. Values are sorted shared channel slices: Racah's
/// multiplicity/channel consumers avoid rebuilding a public `BTreeMap`, while
/// the public API preserves its owned-map contract by reconstructing one from
/// this exact value. The cache computes outside the lock and rechecks on write,
/// like the coefficient tiers; concurrent misses may duplicate exact work but
/// publish one value.
#[cfg(feature = "cgc-gen")]
mod sun_product_cache {
    use super::FifoCache;
    use crate::sun::{SunProduct, SunProductKey};
    use std::sync::LazyLock;

    // The mixed SU(3)+SU(4) structural collector retained 102 unique pairs
    // charging 38,600 bytes; the shared-process downstream Generic HomSpace/
    // topology sequence retained 85 rank-separated pairs charging 54,504 bytes.
    // These bounds give 2.51x entry and 2.40x retained-charge headroom. The byte
    // bound is a retained-charge backstop, not a live-memory ceiling.
    pub(super) const SUN_PRODUCT_MAX_ENTRIES: usize = 256;
    pub(super) const SUN_PRODUCT_MAX_BYTES: usize = 128 << 10;

    pub(crate) static CACHE_SUN_PRODUCT: LazyLock<FifoCache<SunProductKey, SunProduct>> =
        LazyLock::new(|| {
            FifoCache::new(
                SUN_PRODUCT_MAX_ENTRIES,
                super::effective_budgets().sun_product_bytes,
            )
        });
}

#[cfg(feature = "cgc-gen")]
pub(crate) fn cache_sun_product(
) -> &'static FifoCache<crate::sun::SunProductKey, crate::sun::SunProduct> {
    &sun_product_cache::CACHE_SUN_PRODUCT
}

/// Bounded, byte-accounted derived-f64 SU(N) F-symbol cache (`cgc-gen`,
/// Layer 3, issue #16).
///
/// An F block is the contraction of four CGC; even with warm CGC that is real
/// work, so the derived `[μ,ν,κ,λ]` block is cached. Keyed by the **plain
/// ordered six-label tuple** `(a,b,c,d,e,f)` — see the Why-comment in
/// `sun::fr::f_symbol` for why no Regge-style canonicalization exists for
/// GT-basis F blocks (the 6j symmetry group that lets the exact SU(2) F tier
/// key on a canonical class has no analogue here).
///
/// R needs no cache: it is a single sparse join of two CGC (no four-way
/// contraction), cheap enough that a cache slot would not pay for itself.
///
/// In-memory only, same argument as the CGC tier: the values are
/// gauge/algorithm-dependent, so a persisted store would need a version key;
/// keeping it process-local means it is always consistent with the generator.
#[cfg(feature = "cgc-gen")]
mod sun_f_cache {
    use super::{CacheCharge, FifoCache};
    use crate::sun::{FBlock, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Canonical cache key: the six irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type SunFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    impl CacheCharge for Arc<FBlock> {
        fn value_bytes(&self) -> usize {
            std::mem::size_of_val(self.data()) + std::mem::size_of::<FBlock>()
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const SUN_F_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB): F blocks are tiny (a few multiplicity
    /// indices), so this holds a very large working set. `pub(super)` for the
    /// [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const SUN_F_MAX_BYTES: usize = 64 << 20;

    pub(crate) static CACHE_SUN_F: LazyLock<FifoCache<SunFKey, Arc<FBlock>>> =
        LazyLock::new(|| FifoCache::new(SUN_F_MAX_ENTRIES, super::effective_budgets().sun_f_bytes));
}

#[cfg(feature = "cgc-gen")]
pub(crate) fn cache_sun_f(
) -> &'static FifoCache<sun_f_cache::SunFKey, std::sync::Arc<crate::sun::FBlock>> {
    &sun_f_cache::CACHE_SUN_F
}

/// Bounded, byte-accounted derived-f64 B/C/D F-symbol cache (`cgc-gen`, Stage 3
/// S3.4, issue #27).
///
/// The B/C/D analogue of [`cache_sun_f`]: an F block is the contraction of four
/// catalog-driven CGC, real work even with warm generators, so the derived
/// `[μ,ν,κ,λ]` block is cached. Same design as the SU(N) tier — the **plain
/// ordered six-label key** `(a,b,c,d,e,f)` (no Regge canonicalization exists for
/// GT/sweep-basis F blocks; see the Why-comment in `sun::fr::f_symbol`), the
/// shared `Arc<FBlock>` [`CacheCharge`] impl, in-memory only (values are
/// gauge/algorithm-dependent, so a persisted store would need a version key).
///
/// R needs no cache: it is a single sparse join of two CGC (issue #27, "R
/// uncached unless measured").
#[cfg(feature = "cgc-gen")]
mod bcd_f_cache {
    use super::FifoCache;
    use crate::bcd::Irrep;
    use crate::frcore::FBlock;
    use std::sync::{Arc, LazyLock};

    /// Canonical cache key: the six B/C/D irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type BcdFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_F_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB): F blocks are tiny (a few multiplicity indices).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_F_MAX_BYTES: usize = 64 << 20;

    pub(crate) static CACHE_BCD_F: LazyLock<FifoCache<BcdFKey, Arc<FBlock>>> =
        LazyLock::new(|| FifoCache::new(BCD_F_MAX_ENTRIES, super::effective_budgets().bcd_f_bytes));
}

#[cfg(feature = "cgc-gen")]
pub(crate) fn cache_bcd_f(
) -> &'static FifoCache<bcd_f_cache::BcdFKey, std::sync::Arc<crate::frcore::FBlock>> {
    &bcd_f_cache::CACHE_BCD_F
}

/// Bounded, byte-accounted B/C/D CGC value tier (`cgc-gen`, Stage 3 S3.4 P1
/// review, issue #27).
///
/// The B/C/D analogue of the SU(N) [`cache_cgc`] tier: a
/// [`CatalogCgc`](crate::bcd::CatalogCgc) is expensive (a full decomposition
/// sweep), and the F/R gates request the **same** `s1 ⊗ s2` product decomposed
/// to many different coupled `s3`. Without this tier every `bcd::f_symbol` /
/// gate call re-runs the whole sweep in `CanonicalCatalog::cgc`; with it, the
/// tier holds each channel's isometry so a warm request is a hash lookup, and
/// (populated all-channels-per-sweep from `CanonicalCatalog::cgc_product`) the
/// sweep runs once per **product**, not once per **triple**.
///
/// Keyed by the canonical `(s1, s2, s3)` labels — the complete value key, since
/// the CGC is a deterministic function of the labels and the canonical gauge
/// (Ruling 2), independent of which catalog instance produced it (exactly the
/// SU(N) `cache_cgc` argument). In-memory only: values are gauge/algorithm-
/// dependent, so a persisted store would need a version key.
///
/// # Catalog lifetime is orthogonal to this tier
///
/// The [`CanonicalCatalog`](crate::bcd::CanonicalCatalog) is caller-owned `&mut`
/// state, not process-global; dropping or rebuilding one does not invalidate the
/// entries it populated here. Cached values remain valid because catalog
/// instances implement the same canonical convention and tolerance contract, and
/// the complete family/rank/irrep labels determine the key. So [`reset`] and
/// catalog lifetime stay separate axes — no coupling is needed (issue #47, D5).
///
/// # Why a single (s1,s2,s3) tier and not a global tier plus a per-call memo
///
/// One process-global value tier serves both roles the P1 review split out — it
/// dedups the sweep across coupled channels (all channels of a product share the
/// one sweep that first populates any of them) *and* across calls. A separate
/// call-scoped memo would duplicate ownership of the same CGC across a global
/// and a local store, which the workspace cache policy warns against; the single
/// tier keeps ownership singular.
#[cfg(feature = "cgc-gen")]
mod bcd_cgc_cache {
    use super::{CacheCharge, FifoCache};
    use crate::bcd::{CatalogCgc, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Canonical cache key: the three B/C/D irrep labels.
    pub(crate) type BcdCgcKey = (Irrep, Irrep, Irrep);

    impl CacheCharge for Arc<CatalogCgc> {
        fn value_bytes(&self) -> usize {
            self.storage_bytes()
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_CGC_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (256 MiB): dense product isometries are far larger
    /// than an F block, so this tier gets its own generous budget (as the SU(N)
    /// CGC tier).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_CGC_MAX_BYTES: usize = 256 << 20;

    pub(crate) static CACHE_BCD_CGC: LazyLock<FifoCache<BcdCgcKey, Arc<CatalogCgc>>> =
        LazyLock::new(|| {
            FifoCache::new(
                BCD_CGC_MAX_ENTRIES,
                super::effective_budgets().bcd_cgc_bytes,
            )
        });
}

#[cfg(feature = "cgc-gen")]
pub(crate) fn cache_bcd_cgc(
) -> &'static FifoCache<bcd_cgc_cache::BcdCgcKey, std::sync::Arc<crate::bcd::CatalogCgc>> {
    &bcd_cgc_cache::CACHE_BCD_CGC
}

/// Clear the 3j, 6j, and derived-f64 F-symbol caches (and, under `cgc-gen`, the
/// SU(N) product and SU(N)/B/C/D CGC/F caches) and *all* their counters — entries, bytes,
/// hits, misses, and evictions all return to zero.
///
/// # Reset ownership (process-global, single-owner)
///
/// These tiers are `static`, so `reset()` acts on process-global state. It is a
/// **single-owner** operation: exactly one component in a consuming process
/// (for example an engine `Runtime` that owns the coefficient authority) may own
/// the reset policy. **A library must not call `reset()`** — doing so would
/// clear a cache another component is relying on, since there is one shared
/// coefficient-value authority per process (consumers must not keep a mirror).
pub fn reset() {
    CACHE_3J.reset();
    CACHE_6J.reset();
    CACHE_F.reset();
    #[cfg(feature = "cgc-gen")]
    {
        sun_product_cache::CACHE_SUN_PRODUCT.reset();
        cgc_cache::CACHE_CGC.reset();
        sun_f_cache::CACHE_SUN_F.reset();
        bcd_f_cache::CACHE_BCD_F.reset();
        bcd_cgc_cache::CACHE_BCD_CGC.reset();
    }
}

/// Aggregate hit/miss/entry/byte statistics across the 3j, 6j, and derived-f64
/// F-symbol caches.
pub fn stats() -> CacheStats {
    let (h3, m3, e3, b3) = CACHE_3J.snapshot();
    let (h6, m6, e6, b6) = CACHE_6J.snapshot();
    let (hf, mf, ef, bf) = CACHE_F.snapshot();
    #[cfg(feature = "cgc-gen")]
    let (hc, mc, ec, bc) = {
        let (hp, mp, ep, bp) = sun_product_cache::CACHE_SUN_PRODUCT.snapshot();
        let (h, m, e, b) = cgc_cache::CACHE_CGC.snapshot();
        let (h2, m2, e2, b2) = sun_f_cache::CACHE_SUN_F.snapshot();
        let (h3, m3, e3, b3) = bcd_f_cache::CACHE_BCD_F.snapshot();
        let (h4, m4, e4, b4) = bcd_cgc_cache::CACHE_BCD_CGC.snapshot();
        (
            hp + h + h2 + h3 + h4,
            mp + m + m2 + m3 + m4,
            ep + e + e2 + e3 + e4,
            bp + b + b2 + b3 + b4,
        )
    };
    #[cfg(not(feature = "cgc-gen"))]
    let (hc, mc, ec, bc) = (0u64, 0u64, 0usize, 0usize);
    CacheStats {
        hits: h3 + h6 + hf + hc,
        misses: m3 + m6 + mf + mc,
        entries: e3 + e6 + ef + ec,
        bytes: b3 + b6 + bf + bc,
    }
}

/// Per-tier and total statistics for the three base SU(2) coefficient tiers
/// (3j, 6j, derived-F).
///
/// Unlike the aggregate [`stats`] — which also sums the `cgc-gen` generated
/// tiers when that feature is on — this reports only the base SU(2) surface,
/// split per tier, and adds the eviction counter. Conservative retained entry charge is capped by
/// [`BASE_CACHE_MAX_BYTES`] (`total().bytes ≤ BASE_CACHE_MAX_BYTES`). See
/// [`BaseCacheStats`] for the snapshot-consistency contract of `total()`.
pub fn base_cache_stats() -> BaseCacheStats {
    BaseCacheStats {
        three_j: CACHE_3J.tier_stats(),
        six_j: CACHE_6J.tier_stats(),
        derived_f: CACHE_F.tier_stats(),
    }
}

/// Per-tier and total statistics for the five generated `cgc-gen` tiers
/// (SU(N) product, SU(N) CGC, SU(N) F, B/C/D CGC, B/C/D F).
///
/// The generated-family analogue of [`base_cache_stats`]: unlike the aggregate
/// [`stats`] (which sums base *and* generated tiers into one flat
/// [`CacheStats`]), this reports each generated tier separately, adds the
/// eviction counter, and exposes a field-wise [`total`](GeneratedCacheStats::total).
/// Conservative retained entry charge is capped by [`GENERATED_CACHE_MAX_BYTES`]
/// (`total().bytes ≤ GENERATED_CACHE_MAX_BYTES`). See [`GeneratedCacheStats`]
/// for the snapshot-consistency contract of `total()` and the stability caveat.
#[cfg(feature = "cgc-gen")]
pub fn generated_cache_stats() -> GeneratedCacheStats {
    GeneratedCacheStats {
        sun_product: sun_product_cache::CACHE_SUN_PRODUCT.tier_stats(),
        sun_cgc: cgc_cache::CACHE_CGC.tier_stats(),
        sun_f: sun_f_cache::CACHE_SUN_F.tier_stats(),
        bcd_cgc: bcd_cgc_cache::CACHE_BCD_CGC.tier_stats(),
        bcd_f: bcd_f_cache::CACHE_BCD_F.tier_stats(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use std::sync::Arc;

    fn val(n: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Ratio::from(BigInt::from(1)),
            Ratio::new(BigInt::from(n.unsigned_abs()), BigInt::from(1)),
        )
    }

    #[test]
    fn hit_returns_stored_and_counts() {
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(16, 1 << 20);
        let mut computed = 0;
        let a = c.get_or_compute(7, || {
            computed += 1;
            val(7)
        });
        let b = c.get_or_compute(7, || {
            computed += 1;
            val(999) // must not run
        });
        assert_eq!(a, b);
        assert_eq!(computed, 1, "second lookup must be a hit");
        let (hits, misses, entries, _) = c.snapshot();
        assert_eq!((hits, misses, entries), (1, 1, 1));
    }

    #[test]
    fn entry_bound_evicts_oldest() {
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(3, 1 << 30);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let (_, _, entries, _) = c.snapshot();
        assert!(entries <= 3, "entry bound violated: {entries}");
        // FIFO: the two oldest keys (0,1) were evicted, newest retained.
        assert!(c.inner.read().unwrap().map.contains_key(&4));
        assert!(!c.inner.read().unwrap().map.contains_key(&0));
    }

    #[test]
    fn byte_bound_evicts() {
        // Tiny byte budget: only a couple of entries fit at once.
        let per = entry_charge(&0u32, &val(1));
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1_000_000, per * 2 + per / 2);
        for k in 0..20u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let (_, _, _, bytes) = c.snapshot();
        assert!(
            bytes <= per * 2 + per / 2,
            "retained charge cap violated: {bytes}"
        );
    }

    #[test]
    fn eviction_thrash_never_changes_values() {
        // Budget of one entry, hammered with 200 distinct keys in a cycle:
        // every returned value must still equal its from-scratch computation.
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1, 1 << 30);
        for round in 0..3 {
            for k in 0..200u32 {
                let got = c.get_or_compute(k, || val(k as i64 * 3 + 1));
                assert_eq!(got, val(k as i64 * 3 + 1), "round {round} key {k}");
            }
        }
    }

    #[test]
    fn f64_tier_hit_skips_recompute() {
        // The derived-f64 F-symbol tier's contract: a warm hit returns the
        // stored scalar WITHOUT re-running the miss closure -- which is the sole
        // site of the bigint `sqrt` in SignedSqrtRational::to_f64 on the F path.
        // So the public su2_f_symbol hot path avoids bigint isqrt on a hit.
        let c: FifoCache<u32, f64> = FifoCache::new(16, 1 << 20);
        let mut rounded = 0;
        let a = c.get_or_compute(9, || {
            rounded += 1;
            val(9).to_f64() // stands in for f_symbol_exact(..).to_f64()
        });
        let b = c.get_or_compute(9, || {
            rounded += 1;
            val(999).to_f64() // must not run on the hit
        });
        assert_eq!(a, b);
        assert_eq!(rounded, 1, "a hit must not re-run the rounding closure");
        let (hits, misses, entries, _) = c.snapshot();
        assert_eq!((hits, misses, entries), (1, 1, 1));
    }

    #[test]
    fn f64_tier_charge_is_fixed() {
        // f64 values charge a fixed size (no data-dependent limbs), so the tier
        // is bounded by entry count in practice.
        assert_eq!((1.0f64).value_bytes(), std::mem::size_of::<f64>());
        assert_eq!((-3.5f64).value_bytes(), std::mem::size_of::<f64>());
        assert_eq!(
            entry_charge(&7u32, &1.0f64),
            std::mem::size_of::<f64>() + 2 * std::mem::size_of::<u32>(),
            "fixed-size key accounting must remain unchanged"
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn generated_keys_charge_owned_irrep_weights() {
        use super::bcd_f_cache::BcdFKey;
        use super::cgc_cache::CgcKey;
        use crate::bcd::{Irrep as BcdIrrep, Series};
        use crate::sun::Irrep as SunIrrep;

        let sun = SunIrrep::from_dynkin(&[1; 12]).unwrap();
        let sun_key: CgcKey = (sun.clone(), sun.clone(), sun);
        let sun_irrep_bytes = std::mem::size_of::<SunIrrep>() + 13 * std::mem::size_of::<i64>();
        assert_eq!(sun_key.key_bytes(), 3 * sun_irrep_bytes);

        let bcd = BcdIrrep::from_dynkin(Series::C, &[1; 12]).unwrap();
        let bcd_key: BcdFKey = (
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd,
        );
        let bcd_irrep_bytes = std::mem::size_of::<BcdIrrep>() + 12 * std::mem::size_of::<i64>();
        assert_eq!(bcd_key.key_bytes(), 6 * bcd_irrep_bytes);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn sun_product_tier_caps_charge_and_oversize_eviction_are_exact() {
        use super::sun_product_cache::{SUN_PRODUCT_MAX_BYTES, SUN_PRODUCT_MAX_ENTRIES};
        use crate::sun::{Irrep as SunIrrep, SunProduct, SunProductKey};
        use std::collections::BTreeMap;

        assert_eq!(SUN_PRODUCT_MAX_ENTRIES, 256);
        assert_eq!(SUN_PRODUCT_MAX_BYTES, 128 << 10);

        let a = SunIrrep::from_dynkin(&[2, 1]).unwrap();
        let b = SunIrrep::from_dynkin(&[1, 2]).unwrap();
        let key = SunProductKey::new(&a, &b);
        let key_bytes = 2 * std::mem::size_of::<SunIrrep>()
            + (a.rank() + b.rank()) * std::mem::size_of::<i64>();
        assert_eq!(key.key_bytes(), key_bytes);

        let trivial = SunIrrep::trivial(3).unwrap();
        let adjoint = SunIrrep::from_dynkin(&[1, 1]).unwrap();
        let product =
            SunProduct::from_map(BTreeMap::from([(trivial.clone(), 1), (adjoint.clone(), 2)]));
        let value_bytes = std::mem::size_of::<SunProduct>()
            + 2 * std::mem::size_of::<usize>()
            + 2 * std::mem::size_of::<(SunIrrep, u32)>()
            + (trivial.rank() + adjoint.rank()) * std::mem::size_of::<i64>();
        assert_eq!(product.value_bytes(), value_bytes);

        let charge = value_bytes + 2 * key_bytes;
        assert_eq!(entry_charge(&key, &product), charge);
        let cache: FifoCache<SunProductKey, SunProduct> = FifoCache::new(usize::MAX, charge - 1);
        assert_eq!(
            cache.get_or_compute(key, || product.clone()),
            product,
            "an oversize exact value is still returned"
        );
        assert_eq!(
            cache.tier_stats(),
            TierStats {
                misses: 1,
                evictions: 1,
                ..TierStats::default()
            }
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn concurrent_sun_product_misses_return_the_write_recheck_winner() {
        use crate::sun::{Irrep as SunIrrep, SunProduct, SunProductKey};
        use std::collections::BTreeMap;
        use std::sync::{Arc, Barrier};

        let a = SunIrrep::trivial(3).unwrap();
        let b = SunIrrep::from_dynkin(&[1, 0]).unwrap();
        let key = SunProductKey::new(&a, &b);
        let channel = b.clone();
        let cache = Arc::new(FifoCache::new(16, 1 << 20));
        let all_computing = Arc::new(Barrier::new(8));
        let products = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                let channel = channel.clone();
                let all_computing = Arc::clone(&all_computing);
                std::thread::spawn(move || {
                    cache.get_or_compute(key, || {
                        let product = SunProduct::from_map(BTreeMap::from([(channel, 1)]));
                        all_computing.wait();
                        product
                    })
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert!(products[1..]
            .iter()
            .all(|product| products[0].ptr_eq(product)));
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.tier_stats().misses, 8);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn deep_key_bytes_drive_eviction_oversize_and_reset() {
        use super::bcd_f_cache::BcdFKey;
        use super::cgc_cache::CgcKey;
        use crate::bcd::{Irrep as BcdIrrep, Series};
        use crate::sun::Irrep as SunIrrep;

        let sun = |label| SunIrrep::from_dynkin(&[label; 12]).unwrap();
        let first: CgcKey = (sun(1), sun(1), sun(1));
        let second: CgcKey = (sun(2), sun(2), sun(2));
        let one_entry_budget = entry_charge(&first, &1.0f64);
        let cache: FifoCache<CgcKey, f64> = FifoCache::new(usize::MAX, one_entry_budget);
        cache.insert(first.clone(), 1.0);
        cache.insert(second, 2.0);
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.tier_stats().evictions, 1);
        assert!(
            cache.get(&first).is_none(),
            "the oldest deep key is evicted"
        );
        cache.reset();
        assert_eq!(cache.tier_stats(), TierStats::default());

        let bcd = BcdIrrep::from_dynkin(Series::C, &[1; 12]).unwrap();
        let bcd_key: BcdFKey = (
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd,
        );
        let shallow_only = std::mem::size_of::<f64>() + 2 * std::mem::size_of::<BcdFKey>();
        assert!(entry_charge(&bcd_key, &1.0f64) > shallow_only);
        let oversize: FifoCache<BcdFKey, f64> = FifoCache::new(usize::MAX, shallow_only);
        oversize.insert(bcd_key, 1.0);
        assert_eq!(oversize.tier_stats().entries, 0);
        assert_eq!(oversize.tier_stats().bytes, 0);
        assert_eq!(oversize.tier_stats().evictions, 1);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn cgc_tier_charges_storage_bytes_and_evicts_by_bytes() {
        use super::cgc_cache::CgcKey;
        use crate::sun::{cgc, Cgc, Irrep};
        use std::sync::Arc;
        let irr = |d: &[i64]| Irrep::from_dynkin(d).unwrap();
        // Two real CGCs of different sizes.
        let a = Arc::new(cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap()); // 3⊗3̄→8
        let b = Arc::new(cgc(&irr(&[1, 1]), &irr(&[1, 1]), &irr(&[1, 1])).unwrap()); // 8⊗8→8, OM=2

        // The tier charge is exactly the sparse storage bytes (plus key copies).
        assert!(a.value_bytes() >= a.storage_bytes());
        assert_eq!(a.value_bytes(), a.storage_bytes());

        // A local CGC-typed cache with a retained-charge budget that fits only
        // one entry must evict the oldest when the second is inserted.
        let ka = (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1]));
        let kb = (irr(&[1, 1]), irr(&[1, 1]), irr(&[1, 1]));
        let budget = a.value_bytes().max(b.value_bytes()) + 2 * ka.key_bytes() + 8;
        let c: FifoCache<CgcKey, Arc<Cgc>> = FifoCache::new(1_000_000, budget);
        c.insert(ka.clone(), a);
        c.insert(kb, b);
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        // Oldest (ka) evicted.
        assert!(c.get(&ka).is_none());
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn sun_f_tier_charges_block_bytes_and_evicts_by_bytes() {
        use super::sun_f_cache::SunFKey;
        use crate::sun::{f_symbol, FBlock, Irrep};
        use std::sync::Arc;
        let irr = |d: &[i64]| Irrep::from_dynkin(d).unwrap();
        // A real SU(3) F block (8⊗8→8 family: the 2×2×2×2 OM=2 block).
        let e8 = irr(&[1, 1]);
        let a = Arc::new(f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap());
        // A multiplicity-free (smaller, 1⁴) block: a=1 forces e=3, f=d=6.
        let triv = Irrep::trivial(3).unwrap();
        let three = irr(&[1, 0]);
        let six = irr(&[2, 0]);
        let b = Arc::new(f_symbol(&triv, &three, &three, &six, &three, &six).unwrap());

        // Charge is the data bytes plus the block shell.
        assert_eq!(
            a.value_bytes(),
            std::mem::size_of_val(a.data()) + std::mem::size_of::<FBlock>()
        );
        assert!(a.value_bytes() > b.value_bytes(), "2⁴ block > 1⁴ block");

        // Budget for one entry: inserting the second evicts the oldest.
        let ka = (
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
        );
        let kb = (
            triv.clone(),
            three.clone(),
            three.clone(),
            six.clone(),
            three.clone(),
            six.clone(),
        );
        let budget = a.value_bytes() + 2 * ka.key_bytes() + 8;
        let c: FifoCache<SunFKey, Arc<FBlock>> = FifoCache::new(1_000_000, budget);
        c.insert(ka.clone(), a);
        c.insert(kb, b);
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.get(&ka).is_none(), "oldest not evicted");
    }

    #[test]
    fn reset_clears_entries_and_counters() {
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(16, 1 << 20);
        c.get_or_compute(1, || val(1));
        c.get_or_compute(1, || val(1));
        c.reset();
        let (hits, misses, entries, bytes) = c.snapshot();
        assert_eq!((hits, misses, entries, bytes), (0, 0, 0, 0));
    }

    #[test]
    fn evictions_counted_on_entry_bound() {
        // Cap 3 entries, insert 5 distinct keys: exactly 2 oldest evicted.
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(3, 1 << 30);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let ts = c.tier_stats();
        assert_eq!(ts.entries, 3);
        assert_eq!(ts.evictions, 2, "5 inserts over a cap of 3 evict exactly 2");
    }

    #[test]
    fn evictions_counted_on_byte_bound() {
        // Byte budget for ~2 entries: filling 20 forces many byte-driven evictions.
        let per = entry_charge(&0u32, &val(1));
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1_000_000, per * 2 + per / 2);
        for k in 0..20u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        assert!(
            c.tier_stats().evictions > 0,
            "retained-charge pressure must count evictions"
        );
    }

    #[test]
    fn oversize_entry_counts_as_eviction() {
        // Retained-charge cap smaller than any single entry: the entry is
        // admitted (charged, pushed) then immediately evicted back out.
        // Documented decision: it counts as an eviction, and nothing is retained.
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1_000_000, 1);
        c.get_or_compute(7, || val(7));
        let ts = c.tier_stats();
        assert_eq!(ts.entries, 0, "oversize entry is not retained");
        assert_eq!(ts.bytes, 0);
        assert_eq!(ts.evictions, 1, "an admitted-then-evicted entry counts");
    }

    #[test]
    fn reset_zeroes_evictions() {
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1, 1 << 30);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        assert!(c.tier_stats().evictions > 0, "precondition: some evictions");
        c.reset();
        assert_eq!(
            c.tier_stats(),
            TierStats::default(),
            "reset zeroes every field"
        );
    }

    #[test]
    fn concurrent_mixed_hit_miss_equals_sequential() {
        let c: Arc<FifoCache<u32, SignedSqrtRational>> = Arc::new(FifoCache::new(1 << 20, 1 << 30));
        let keys: Vec<u32> = (0..64).collect();
        // Reference: sequential fill.
        let seq: Vec<SignedSqrtRational> = keys.iter().map(|&k| val(k as i64 + 1)).collect();

        let mut handles = Vec::new();
        for t in 0..8u32 {
            let c = Arc::clone(&c);
            handles.push(std::thread::spawn(move || {
                let mut out = Vec::new();
                // Each thread walks all keys (mix of first-miss and later-hit),
                // offset so threads interleave differently.
                for i in 0..64u32 {
                    let k = (i + t) % 64;
                    out.push((k, c.get_or_compute(k, || val(k as i64 + 1))));
                }
                out
            }));
        }
        for h in handles {
            for (k, got) in h.join().unwrap() {
                assert_eq!(got, seq[k as usize], "thread value diverged at key {k}");
            }
        }
    }

    #[test]
    fn policy_observation_races_configuration_once() {
        let cell = Arc::new(OnceLock::new());
        let configured = CoefficientCacheBudgets::disabled();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let observer_cell = Arc::clone(&cell);
        let observer_barrier = Arc::clone(&barrier);
        let observer = std::thread::spawn(move || {
            observer_barrier.wait();
            *effective_budgets_in(&observer_cell)
        });
        barrier.wait();
        let configured_result = configure_budgets_in(&cell, configured);
        let observed = observer.join().unwrap();
        match configured_result {
            Ok(()) => assert_eq!(observed, configured),
            Err(CacheBudgetError::AlreadyInitialized) => {
                assert_eq!(observed, CoefficientCacheBudgets::default())
            }
            Err(error) => panic!("unexpected configuration result: {error}"),
        }
    }
}
