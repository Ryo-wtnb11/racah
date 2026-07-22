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

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, RwLock};

use crate::exact::SignedSqrtRational;
use crate::su2::{FKey, Regge3j, Regge6j};

/// Default entry cap per kind (3j and 6j each). Matches the reference order of
/// magnitude (WignerSymbols.jl uses `10^6`); the byte cap is the real backstop.
const DEFAULT_MAX_ENTRIES: usize = 1 << 20;

/// Default byte cap per kind. Conservative: at the ~O(1)-limb sizes typical of
/// small-label TN work an entry charges well under a kilobyte, so 64 MiB holds
/// a large working set while bounding worst-case retained memory.
const DEFAULT_MAX_BYTES: usize = 64 << 20;

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

/// Conservative retained-byte charge for a stored *value*, implemented per
/// value type so the FIFO byte bound stays a true ceiling whatever the tier
/// stores.
///
/// The exact tier stores a [`SignedSqrtRational`] whose size is data-dependent
/// (big-integer limbs), so it must measure itself; the derived-f64 tier stores
/// a fixed-size scalar. Keeping the charge on the value keeps [`entry_charge`]
/// generic over both without a size query the FIFO machinery could get wrong.
pub(crate) trait CacheCharge {
    /// Bytes charged for one stored value (over-counts, never under-counts).
    fn value_bytes(&self) -> usize;
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
/// the map, once in the FIFO order queue). It over-counts rather than
/// under-counts, so the byte bound is a true ceiling on live memory, never an
/// underestimate that could let the map grow past it.
fn entry_charge<K, V: CacheCharge>(v: &V) -> usize {
    v.value_bytes() + 2 * std::mem::size_of::<K>()
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
    max_entries: usize,
    max_bytes: usize,
}

impl<K: Clone + Eq + Hash, V: Clone + CacheCharge> FifoCache<K, V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        FifoCache {
            inner: RwLock::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
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
        let charge = entry_charge::<K, V>(&value);
        inner.bytes += charge;
        inner.order.push_back(key.clone());
        inner.map.insert(key, value.clone());
        self.evict(&mut inner);
        value
    }

    /// Evict from the front (oldest) until both bounds hold. A single entry
    /// larger than `max_bytes` is evicted back out (returned to the caller but
    /// not retained) rather than pinning the map over budget.
    fn evict(&self, inner: &mut Inner<K, V>) {
        while (inner.map.len() > self.max_entries || inner.bytes > self.max_bytes)
            && !inner.order.is_empty()
        {
            let Some(old) = inner.order.pop_front() else {
                break;
            };
            if let Some(v) = inner.map.remove(&old) {
                inner.bytes = inner.bytes.saturating_sub(entry_charge::<K, V>(&v));
            }
        }
    }

    fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.map.clear();
        inner.order.clear();
        inner.bytes = 0;
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
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
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES));
static CACHE_6J: LazyLock<FifoCache<Regge6j, SignedSqrtRational>> =
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES));

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
    LazyLock::new(|| FifoCache::new(DEFAULT_MAX_ENTRIES, DEFAULT_MAX_BYTES));

pub(crate) fn cache_f() -> &'static FifoCache<FKey, f64> {
    &CACHE_F
}

/// Clear the 3j, 6j, and derived-f64 F-symbol caches and their hit/miss
/// counters.
pub fn reset() {
    CACHE_3J.reset();
    CACHE_6J.reset();
    CACHE_F.reset();
}

/// Aggregate hit/miss/entry/byte statistics across the 3j, 6j, and derived-f64
/// F-symbol caches.
pub fn stats() -> CacheStats {
    let (h3, m3, e3, b3) = CACHE_3J.snapshot();
    let (h6, m6, e6, b6) = CACHE_6J.snapshot();
    let (hf, mf, ef, bf) = CACHE_F.snapshot();
    CacheStats {
        hits: h3 + h6 + hf,
        misses: m3 + m6 + mf,
        entries: e3 + e6 + ef,
        bytes: b3 + b6 + bf,
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
        let per = entry_charge::<u32, SignedSqrtRational>(&val(1));
        let c: FifoCache<u32, SignedSqrtRational> = FifoCache::new(1_000_000, per * 2 + per / 2);
        for k in 0..20u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let (_, _, _, bytes) = c.snapshot();
        assert!(bytes <= per * 2 + per / 2, "byte bound violated: {bytes}");
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
}
