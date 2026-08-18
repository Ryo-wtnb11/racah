# 7. Numerical behaviour and resources

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## Exact vs generated values

`racah` computes two different kinds of quantity, and it is worth knowing which
one you are holding.

| Quantity | Exactness |
|---|---|
| Labels, dimensions, duals, FS indicators, fusion multiplicities, weight multiplicities, basis orderings | **Exact** integer/rational arithmetic |
| SU(2) coefficients (3j, 6j, CG, F, R) | **Exact** big-rational, rounded once at the end |
| SU(N), SO(N), Sp(2N) coefficient values (CGC, F, R) | Verification-gated **floating point** |

Concretely: `Irrep::dim()` returns a `BigInt` and is never approximate;
`sun::cgc(...)` returns `f64` values produced by a nullspace solve and a
least-squares descent.

"Verification-gated" is the promise attached to those floats: orthogonality,
unitarity and pentagon/hexagon checks run *at generation time*, and a tolerance
violation becomes a typed error. You never receive a silently degraded
coefficient — you receive either a checked value or an error.

What is **not** promised: bit-identical values across machines or backend
versions. Round-off at the last few ULP is inside the disclaimed tolerance
class. What *is* promised is that no discrete choice moves: a sign flip, a
reordered multiplicity copy, or a different basis order is a bug, never a
tolerance event.

## Bounding cache memory

Coefficient generation is expensive, so `racah` caches. The caches are
process-global, bounded, and FIFO.

- **Base tiers** (SU(2): 3j, 6j, derived-F) are bounded by
  `cache::BASE_CACHE_MAX_BYTES` = 192 MiB, a static partition of 3 × 64 MiB.
- **Generated tiers** (`cgc-gen`: SU(N) product / CGC / F, B/C/D CGC / F) are
  bounded by `cache::GENERATED_CACHE_MAX_BYTES` = 640 MiB + 128 KiB.

Those are ceilings on *retained entry charge*, not RSS: they exclude container
capacity, allocator metadata, transient clones, and values you have taken out
through the public API.

If several hundred MiB is too much, set a smaller policy **once, before the
first coefficient call**:

```rust
use racah::cache::{configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier};

let budgets = CoefficientCacheBudgets::default()
    .with_limit(CoefficientCacheTier::SixJ, 1 << 20);   // 1 MiB of 6j
let _ = configure_cache_budgets(budgets);
```

The policy is **one-shot and shrink-only**: the compiled defaults are also the
maximum accepted caps, and configuration can only be applied before first use.
A zero cap evaluates normally but retains nothing;
`CoefficientCacheBudgets::disabled()` zeroes every tier. There are no presets,
environment variables, or runtime reconfiguration. `cache_budgets()` reports the
effective policy.

## Observing and releasing

```rust
use racah::cache::{base_cache_stats, reset, trim_to, CoefficientCacheTier};

let stats = base_cache_stats();
let _ = stats.six_j.entries;          // per-tier entries / bytes / hits / misses / evictions
let _ = stats.total();                // field-wise sum across the base tiers

// Release one tier down to a target charge, oldest entries first.
let report = trim_to(CoefficientCacheTier::SixJ, 0);
let _ = report.removed_entries;

reset();                               // all tiers: entries, bytes, counters to zero
```

With `cgc-gen`, `generated_cache_stats()` reports the five generated tiers the
same way.

**Ownership rule.** `reset` and `trim_to` act on process-global state, so
exactly one component in a process may own that policy — normally the
application. **A library must not call them**, because it cannot know what other
components depend on the cached values.

Per-tier snapshots are consistent under their own tier lock; `total()` sums
independent snapshots, so under concurrent fills it is only eventually
consistent.

## The B/C/D catalog is separate

A [`CanonicalCatalog`](clebsch-gordan.md#son-sp2n-you-need-a-catalog) is
caller-owned state with its own byte budget (`CanonicalCatalog::with_budget`),
holding discovered generator matrices. It is *not* one of the global cache
tiers. Dropping a catalog frees the generators but does not invalidate cached
coefficient values: those are keyed by the complete irrep labels and are valid
no matter which catalog instance produced them.

## Gauge and reproducibility

A coefficient has no meaning without the convention that fixes its basis, sign,
and ordering. `racah` publishes every coefficient in one **fixed canonical
gauge**, written down as a frozen normative specification. Most users never need
the details; what you do need is:

- **Same version, same numbers.** The gauge is not "whatever this build
  outputs". A change that moves a coefficient value is a bug unless it ships as
  an explicit specification correction with a fingerprint epoch bump and a
  CHANGELOG breaking-change entry.
- **You can checkpoint against it.** Each family exposes an opaque **authority
  fingerprint**: `su2_authority_fingerprint()`, `sun::sun_authority_fingerprint()`,
  `bcd::bcd_authority_fingerprint()`. Persist the bytes next to anything you
  derive from these coefficients, and on load compare for equality and reject
  the derived data on mismatch. Compare only — never parse them.

```rust
let fingerprint = racah::su2_authority_fingerprint();
assert!(!fingerprint.is_empty());
// Persist `fingerprint` alongside any table you derive from racah's SU(2) values.
```

The SU(2) fingerprint carries the strong contract: equal fingerprint ⇔ equal
values. The generated-family fingerprints are deliberately weaker — equal
fingerprints identify the same convention, generation pipeline, and tolerance
policy, but do not assert byte-identical values. Numerical agreement there is
established by the generation gates and the oracle suites, not by the
fingerprint. Each family's epoch is independent: an SU(N) specification
correction never invalidates SU(2)-derived state.

The specifications themselves: [`../gauge.md`](../gauge.md) (base SU(2) and
SU(N)) and [`../gauge_soN.md`](../gauge_soN.md) (SO(N)/Sp(2N)). They are
normative reference documents, not tutorials — read them when you are comparing
against another implementation or auditing a value, not to learn the API.

## Caveats

- The generated (`cgc-gen`) provider surface is marked **unstable**; its shape
  may change while the contract is negotiated. The base SU(2) surface is stable.
- There is deliberately no single cross-feature cache constant: one number
  spanning feature-gated tiers would change meaning with the `cgc-gen` flag.
- Selecting a linear-algebra backend is not public API today. All dense work
  routes through one seam, currently executed on a CPU backend.

## Related API

- [`racah::cache`]: `configure_cache_budgets`, `cache_budgets`,
  `base_cache_stats`, `generated_cache_stats`, `trim_to`, `reset`,
  `BASE_CACHE_MAX_BYTES`, `GENERATED_CACHE_MAX_BYTES`
- [`racah::su2_authority_fingerprint`], `sun::sun_authority_fingerprint`,
  `bcd::bcd_authority_fingerprint`
- Measurement evidence behind the defaults: [`../developer/`](../developer/README.md)

[`racah::cache`]: https://docs.rs/racah/latest/racah/cache/index.html
[`racah::su2_authority_fingerprint`]: https://docs.rs/racah/latest/racah/fn.su2_authority_fingerprint.html
