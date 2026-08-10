# Coefficient-cache budget tradeoff

This release-only, fresh-process pressure trace measures revision
`52d3732b1263e40b7bd095c08e0bfa214c5d127f` on macOS/aarch64 with rustc
1.96.0. Five fresh processes were recorded per mode:

```text
for mode in default constrained disabled; do for sample in 1 2 3 4 5; do COEFFICIENT_CACHE_BUDGET_SAMPLE=$sample RACAH_BUDGET_REVISION=52d3732b1263e40b7bd095c08e0bfa214c5d127f CARGO_TARGET_DIR=/private/tmp/racah-budget-pressure-52d3732 cargo test --release --features cgc-gen --lib cache_budget_pressure_$mode -- --ignored --nocapture; done; done
```

`RACAH_BUDGET_REVISION` is embedded by `option_env!` at compilation, not read
by the test binary at runtime. The distinct target directory above therefore
prevents a direct reuse of a release binary compiled with a different revision;
when changing that variable in an existing target directory, rebuild first
(for example, `cargo clean -p racah`) rather than invoking an old binary.

The workload in `src/cache_budget_pressure.rs::workload` evaluates three SU(2)
3j/6j labels and SU(3) fundamental/adjoint plus SU(4) fundamental products.
The raw JSONL is [coefficient-cache-budget-pressure.jsonl](coefficient-cache-budget-pressure.jsonl).
`budgets` records every effective compiled tier limit. Each tier tuple is
`[entries, charged_bytes, hits, misses, evictions]`.

| mode | elapsed ns, median ± MAD | retained charged bytes | evictions | RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| default | 390583 ± 5708 | 2252 | 0 | raw in artifact |
| constrained (400 B in active 3j/6j/SU(N)-product tiers) | 381292 ± 2459 | 789 | 6 | raw in artifact |
| disabled (all tiers zero) | 376583 ± 7916 | 0 | 9 | raw in artifact |

`requested_live_*` in the artifact is only Rust `System` requested-live memory;
it includes non-cache work and excludes allocator metadata/backend allocations.
RSS is sampled separately with macOS `ps`; this test-only collector intentionally
has no Linux RSS path. Neither is a charged-entry ceiling.
The artifact has 15 rows (`schema=1`, revision, mode, unique sample, effective
per-tier budgets, tier stats, requested-live, and RSS). Median/MAD is
`median(|x - median(x)|)`. It demonstrates policy effects and actual eviction,
not a throughput ranking.

The evidence supports independent shrink-only per-tier caps and the explicit
all-zero configuration. It does not justify presets, larger maxima, aggregate
redistribution, a shared LRU, or runtime reconfiguration.
