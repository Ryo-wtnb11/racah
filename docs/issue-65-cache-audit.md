# Cache audit: issue #65

Revision `65f923657c28f0e6fbc91658313927520527f43b`; `cgc-gen`; crate-internal workload, so consumer revision is N/A. Measured with `rustc 1.96.0 (ac68faa2)`, macOS/aarch64, release test binary, and the test-only wrapper around `System`.

Run:

```text
RACAH_AUDIT_REVISION=65f923657c28f0e6fbc91658313927520527f43b cargo test --release --features cgc-gen --lib cache_audit::issue_65_cache_audit -- --ignored --nocapture
```

The harness uses the existing representative SU(2), SU(3), and SU(4) Wigner/CGC/F/product cases. It records every base and generated tier as `[entries, charged bytes, hits, misses, evictions]`; raw rows are in [issue-65-cache-audit.jsonl](issue-65-cache-audit.jsonl). `cold` is an empty-cache phase, `warm` reruns it, and `reset_before_each_query` is the production-faithful no-reuse control. It is not called “cache disabled”: reset has the normal production semantics.

| phase | cold ns, median ± MAD | warm ns, median ± MAD | reset-before-each ns, median ± MAD |
| --- | ---: | ---: | ---: |
| SU(2) | 116000 ± 2292 | 708 ± 125 | 43041 ± 916 |
| SU(3) | 15262500 ± 291333 | 2375 ± 250 | 17835542 ± 30333 |
| SU(4) | 22222167 ± 119792 | 2875 ± 167 | 30205042 ± 233375 |

The forward and reverse fresh-reset sequential traces end at the same occupancy: SU(2) `3j/6j/F = 1/2/1` entries and `182/357/56` charged bytes; SU(N) product/CGC/F = `4/8/2` and `1352/6688/1184` bytes. All evictions were zero. The SU-only leaf intentionally does not exercise B/C/D CGC/F: their rows remain zero and are not a measurement of those tiers. The exact prime/factorial support tables grow to 122 rows, 30 primes, and 13240 conservative retained-capacity bytes.

`charged bytes` are cache-entry charge only. The System wrapper records requested live bytes routed through Rust `GlobalAlloc`, not allocator-live memory: it excludes C/library allocations and allocator metadata. Each timed phase resets its peak to its starting requested-live value; this is an approximate observed requested-live peak under backend concurrency. `transient_requested_live_lower_bound` is `peak - max(start, end)`, so retained cache/output growth is excluded. macOS samples current RSS with `ps -o rss= -p PID` outside timed sections; the representative trace records the samples.

Five fresh test processes produced the median/MAD table above; their raw values are in [issue-65-cache-audit-timings.jsonl](issue-65-cache-audit-timings.jsonl). The checked-in JSONL is one complete representative trace (metadata, every phase, sequential intermediate, clone, and retention records), not an aggregate.

Public return allocation is measured by monotonic successful allocation requests, not noisy live deltas. The representative 1/9 retained calls request respectively exact SU(2) `48/1008` B, owned `directproduct` `376/3576` B, public CGC `408/4504` B, and public F `344/3544` B; these include public-call overhead, not payload-only attribution. The selected warm CGC is asserted nonempty. The harness also proves the one public shared-product `Arc` survives cache reset: cache reset reduces its strong count by exactly one and the returned channels remain readable. CGC and F public APIs return deep clones, so they cannot externally retain their internal cache `Arc`.

## Policy decision

1. Configurable small/default/HPC budgets: **defer** to #66; this leaf establishes only a tiny SU working set, not a budget choice.
2. Deterministic `trim_to`: **defer** to #67; no policy implementation belongs in this measurement leaf.
3. Aggregate scaling without shared LRU: **defer**; the tiny no-eviction trace gives no evidence to replace the current independent tiers or to introduce cross-tier competition. #66 needs broader pressure measurements.
4. Internal shared `Arc<Cgc>` / `Arc<FBlock>` consumption where ownership permits: **defer**; public clone slopes motivate investigation, while any consumer-visible shared accessor remains a separate ownership/API decision.
5. Prime/factorial table policy: **recommend** retaining the new private observation; defer any bound/reset policy until workloads above the observed 122-row support table are measured.

No persistent layer, R cache, cache tier, production instrumentation, policy/budget, or public API was added. Admission bypasses are not separately observable: SU(2) invalid/overflow requests bypass a tier before lookup; generated fallible input/generation also may not reach a tier; an oversized admitted entry follows the existing immediate-eviction path already counted in `misses`/`evictions`.
