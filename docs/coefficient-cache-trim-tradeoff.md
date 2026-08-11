# Coefficient-cache trim semantics

Fresh release processes measured revision `78491855d1023545121020454989a6dee6859c6e`
on macOS/aarch64 with rustc 1.96.0. Five samples per mode used:

```text
for mode in retained partial zero; do for sample in 1 2 3 4 5; do RACAH_TRIM_REVISION=78491855d1023545121020454989a6dee6859c6e COEFFICIENT_CACHE_TRIM_SAMPLE=$sample CARGO_TARGET_DIR=/private/tmp/racah-67-trim-release cargo test --release --features cgc-gen --lib coefficient_cache_trim_$mode -- --ignored --nocapture; done; done
```

The SU(2), SU(3), and SU(4) workload first fills 3j, 6j, and SU(N)-product tiers,
then retains, partially trims, or zero-trims them before a second workload.
Elapsed time therefore measures warm hits, mixed regeneration, or full regeneration.

| mode | elapsed ns, median plus/minus MAD |
| --- | ---: |
| retained | 1417 plus/minus 42 |
| partial | 71583 plus/minus 1249 |
| zero | 194125 plus/minus 8625 |

The JSONL artifact records targets, trim reports, post-rerun tier statistics,
requested-live allocation accounting, and RSS. Charged-entry bytes are distinct
from requested-live memory and RSS. RSS comes from macOS `ps`; this test-only
collector has no Linux RSS implementation and makes no RSS-release claim.
