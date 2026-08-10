# Coefficient-cache budget tradeoff

This release-only, fresh-process pressure trace measures revision
`728dc8b8dddca660a7df0c663435a9119ddbdad1` on macOS/aarch64 with rustc
1.96.0. Command (one fresh process per mode):

```text
cargo test --release --features cgc-gen --lib cache_budget_pressure_<mode> -- --ignored --nocapture
```

The workload in `src/cache_budget_pressure.rs::workload` evaluates three SU(2)
3j/6j labels and SU(3) fundamental/adjoint plus SU(4) fundamental products.
The raw JSONL is [coefficient-cache-budget-pressure.jsonl](coefficient-cache-budget-pressure.jsonl).
Each tier tuple is `[entries, charged_bytes, hits, misses, evictions]`.

| mode | elapsed ns | retained charged bytes | evictions | RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| default | 552000 | 2252 | 0 | 3489792 |
| constrained (1 B in active 3j/6j/SU(N)-product tiers) | 373292 | 0 | 9 | 3424256 |
| disabled (all tiers zero) | 352792 | 0 | 9 | 3424256 |

`requested_live_*` in the artifact is only Rust `System` requested-live memory;
it includes non-cache work and excludes allocator metadata/backend allocations.
RSS is sampled separately with macOS `ps`. Neither is a charged-entry ceiling.
The trace is one sample per fresh mode, intended to demonstrate policy effects
and actual eviction, not to establish a throughput ranking.

The evidence supports independent shrink-only per-tier caps and the explicit
all-zero configuration. It does not justify presets, larger maxima, aggregate
redistribution, a shared LRU, or runtime reconfiguration.
