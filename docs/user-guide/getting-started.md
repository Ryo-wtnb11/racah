# 1. Getting started

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## Install

```toml
[dependencies]
racah = "0.1.1"
```

That is the **base** build: exact SU(2) only, three small numeric dependencies,
no linear-algebra stack. To compute for SU(N), SO(N), Spin(N) or Sp(2N), turn on
the `cgc-gen` feature:

```toml
[dependencies]
racah = { version = "0.1.1", features = ["cgc-gen"] }
```

There is no fixed MSRV; `racah` builds and is tested on current stable Rust.

### Which feature flag do I need?

| Feature | Gives you | Cost |
|---|---|---|
| *(default)* | SU(2): 3j, 6j, Clebsch–Gordan, F, R, Frobenius–Schur — closed-form, big-rational exact | `num-bigint`, `num-rational`, `num-traits` |
| `cgc-gen` | SU(N) (`racah::sun`), SO(N)/Spin(N)/Sp(2N) (`racah::bcd`), and the `racah::group` global forms applied to them | pulls the Tenferro dense-linear-algebra backend |

The split is mathematical, not organizational: SU(2) has closed-form
coefficients and needs no matrix computation; every other family must be
*constructed* numerically. If you only need SU(2), you never pull the backend.

## First calculation: SU(2) (no features)

```rust
use racah::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

**Spins are doubled.** Every SU(2) label in this crate is `dj = 2j`, so `2`
means spin 1 and `1` means spin ½. The same holds for projections: `dm = 2m`.
This is not a stylistic choice — it keeps half-integer spins as exact integers,
so no label is ever a float. See
[Representations § doubled spins](representations.md#su2-doubled-spins).

The returned [`SignedSqrtRational`] is an *exact* value of the form
`sign · sqrt(p/q)`; call `.to_f64()` when you want a number.

## First calculation: SU(3) (`cgc-gen`)

```rust
use racah::sun::Irrep;

let fund = Irrep::from_dynkin(&[1, 0]).unwrap();   // the 3 of SU(3)
assert_eq!(fund.dim(), 3u32.into());
assert_eq!(fund.dual().dynkin(), vec![0, 1]);      // the 3-bar
```

An SU(N) irrep is named by its `N-1` Dynkin labels. See
[Representations](representations.md).

## How errors work

`racah` has two error styles, and the difference is deliberate.

**The infallible SU(2) functions return exact zero for a forbidden coupling.**
`wigner_6j`, `wigner_3j`, `clebsch_gordan`, `su2_f_symbol`, `su2_r_symbol` never
fail: a label set that violates a triangle or parity rule returns `0`, matching
the physics convention where a forbidden channel simply contributes nothing.

**The checked SU(2) functions distinguish "forbidden" from "zero".** An
*admissible* 6j can still be accidentally zero. When you need to tell the two
apart, use the `*_checked` twins:

```rust
use racah::su2::{wigner_6j_checked, Su2Error};

// Admissible, and genuinely nonzero.
assert!(wigner_6j_checked(2, 2, 2, 2, 2, 2).is_ok());

// Triangle violation: not a real zero, a forbidden label set.
assert!(matches!(
    wigner_6j_checked(2, 2, 20, 2, 2, 2),
    Err(Su2Error::NotAdmissible(_))
));
```

Both paths share one admissibility predicate, so they can never disagree.

**The generated families are fallible throughout.** `racah::sun` and
`racah::bcd` return `Result`, because construction can fail for reasons that are
not "the answer is zero": a label the group does not admit, mismatched ranks, an
empty fusion channel, an exhausted memory budget, or a violated verification
gate. The error types name the reason —
[`SunError`](https://docs.rs/racah/latest/racah/sun/enum.SunError.html),
[`BcdError`](https://docs.rs/racah/latest/racah/bcd/enum.BcdError.html),
[`CatalogError`](https://docs.rs/racah/latest/racah/bcd/enum.CatalogError.html),
[`FrError`](https://docs.rs/racah/latest/racah/bcd/enum.FrError.html).

A verification-gate failure is always an error, never a silently degraded
number: `racah` will not hand you a coefficient that failed its orthogonality,
unitarity, or pentagon/hexagon check.

## Caveats

- With `cgc-gen` on, the coefficient caches may retain several hundred MiB by
  default. If that matters, call `racah::cache::configure_cache_budgets` once
  before first use — see [Resources](resources.md#bounding-cache-memory).
- The generated (`cgc-gen`) surface is marked **unstable**: its shape may still
  change. The base SU(2) surface is stable.

## Related API

- [`racah::wigner_6j`](https://docs.rs/racah/latest/racah/fn.wigner_6j.html),
  [`racah::su2`](https://docs.rs/racah/latest/racah/su2/index.html)
- [`racah::sun`](https://docs.rs/racah/latest/racah/sun/index.html),
  [`racah::bcd`](https://docs.rs/racah/latest/racah/bcd/index.html)
- [`racah::cache`](https://docs.rs/racah/latest/racah/cache/index.html)

[`SignedSqrtRational`]: https://docs.rs/racah/latest/racah/struct.SignedSqrtRational.html
