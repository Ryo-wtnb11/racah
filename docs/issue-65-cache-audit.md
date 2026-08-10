# Cache audit: issue #65

Revision `48bad0af875b4e045b4dd1b6d92a578c411d5d5a`; `cgc-gen`; crate-internal workload, so consumer revision is N/A. Measured with `rustc 1.96.0 (ac68faa2)`, macOS/aarch64, release test binary, and the test-only wrapper around `System`.

Run:

```text
RACAH_AUDIT_REVISION=48bad0af875b4e045b4dd1b6d92a578c411d5d5a cargo test --release --features cgc-gen --lib cache_audit::issue_65_cache_audit -- --ignored --nocapture
```

The harness uses the existing representative SU(2), SU(3), and SU(4) Wigner/CGC/F/product cases. It records every base and generated tier as `[entries, charged bytes, hits, misses, evictions]`; raw rows are in [issue-65-cache-audit.jsonl](issue-65-cache-audit.jsonl). `cold` is an empty-cache phase, `warm` reruns it, and `reset_before_each_query` is the production-faithful no-reuse control. It is not called “cache disabled”: reset has the normal production semantics.

| phase | cold ns, median ± MAD | warm ns, median ± MAD | reset-before-each ns, median ± MAD |
| --- | ---: | ---: | ---: |
| SU(2) | 128834 ± 11834 | 875 ± 125 | 46250 ± 5917 |
| SU(3) | 15378291 ± 96750 | 2917 ± 626 | 17900042 ± 100208 |
| SU(4) | 22406083 ± 216292 | 3042 ± 792 | 30254917 ± 29583 |

The forward and reverse fresh-reset sequential traces end at the same occupancy: SU(2) `3j/6j/F = 1/2/1` entries and `182/357/56` charged bytes; SU(N) product/CGC/F = `4/8/2` and `1352/6688/1184` bytes. All evictions were zero. The SU-only leaf intentionally does not exercise B/C/D CGC/F: their rows remain zero and are not a measurement of those tiers. The exact prime/factorial support tables grow to 122 rows, 30 primes, and 13240 conservative retained-capacity bytes.

`charged bytes` are cache-entry charge only. The System wrapper records requested live bytes routed through Rust `GlobalAlloc`, not allocator-live memory: it excludes C/library allocations and allocator metadata. Each timed phase resets its peak to its starting requested-live value; `transient_requested_live_lower_bound` is `peak - max(start, end)`, so retained cache/output growth is excluded. macOS samples current RSS with `ps -o rss= -p PID` outside timed sections; the representative trace records the samples. `/usr/bin/time -l` is recorded as external process provenance only, not a cache-only bound.

Five fresh test processes produced the median/MAD table above; their raw values are in [issue-65-cache-audit-timings.jsonl](issue-65-cache-audit-timings.jsonl). The checked-in JSONL is one complete representative trace (metadata, every phase, sequential intermediate, clone, and retention records), not an aggregate.

Public exact SU(2), owned `directproduct`, CGC, and F returns were each retained in a 1-versus-9 return slope trace. The harness also proves the one public shared-product `Arc` survives cache reset: cache reset reduces its strong count by exactly one and the returned channels remain readable. CGC and F public APIs return deep clones, so they cannot externally retain their internal cache `Arc`.

## Policy decision

1. Configurable small/default/HPC budgets: **defer** to #66; this leaf establishes only a tiny SU working set, not a budget choice.
2. Deterministic `trim_to`: **defer** to #67; no policy implementation belongs in this measurement leaf.
3. Aggregate scaling without shared LRU: **defer**; the tiny no-eviction trace gives no evidence to replace the current independent tiers or to introduce cross-tier competition. #66 needs broader pressure measurements.
4. Internal shared `Arc<Cgc>` / `Arc<FBlock>` consumption where ownership permits: **defer**; public clone slopes motivate investigation, while any consumer-visible shared accessor remains a separate ownership/API decision.
5. Prime/factorial table policy: **recommend** retaining the new private observation; defer any bound/reset policy until workloads above the observed 122-row support table are measured.

No persistent layer, R cache, cache tier, production instrumentation, policy/budget, or public API was added. Admission bypasses are not separately observable: generated admission either never reaches a tier (fallible input/generation) or is a normal insert followed by the existing immediate oversized eviction, already counted in `misses`/`evictions`.
