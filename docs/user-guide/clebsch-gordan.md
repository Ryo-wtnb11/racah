# 5. Clebsch–Gordan coefficients

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

A Clebsch–Gordan coefficient is one entry of the intertwiner that maps the
product basis of `a ⊗ b` onto the basis of a coupled irrep `c`. As a tensor it
has four indices:

```text
C[m1, m2, m3, μ]      m1 ∈ basis(a), m2 ∈ basis(b), m3 ∈ basis(c),
                      μ ∈ 0 .. N^c_ab   (the outer-multiplicity axis)
```

The fourth axis exists because `c` can appear in `a ⊗ b` more than once
([Fusion](fusion.md#multiplicity-greater-than-one)); `μ` selects the copy. For a
multiplicity-free family such as SU(2) it always has length 1.

## When do I need it?

When you need the actual change of basis — projecting a product state onto a
coupled sector, building symmetric tensors, computing matrix elements. If you
only need recoupling *between* coupling orders, you want
[F- and R-symbols](recoupling.md) instead: those already contract the CGC away
and are cheaper to consume.

## SU(2)

Exact, closed-form, no feature flag. Both irrep and projection labels are
doubled (`dj = 2j`, `dm = 2m`).

```rust
use racah::su2::clebsch_gordan;

// ⟨½ +½, ½ −½ | 0 0⟩ = 1/√2, in doubled labels.
let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
```

The argument order is `(dj1, dm1, dj2, dm2, dj3, dm3)` — factor label and its
projection, interleaved. The return is a [`SignedSqrtRational`]: the exact value
`sign · sqrt(p/q)`. A forbidden coupling returns exact zero; use
`clebsch_gordan_checked` if you must distinguish that from an accidental zero.

Related exact objects on the same surface: `wigner_3j` (the 3j symbol,
`(dj1 dj2 dj3; dm1 dm2 dm3)`) and `wigner_6j` (the 6j / recoupling symbol,
`{dj1 dj2 dj3; dj4 dj5 dj6}`), from which the CG and F-symbol are composed.

## SU(N)

`sun::cgc(s1, s2, s3)` generates the whole block at once — all multiplicity
copies together, since they share one nullspace.

```rust
use racah::sun::{cgc, Irrep};

let three = Irrep::from_dynkin(&[1, 0]).unwrap();
let anti = three.dual();
let eight = Irrep::from_dynkin(&[1, 1]).unwrap();

let c = cgc(&three, &anti, &eight).unwrap();
assert_eq!(c.dims(), [3, 3, 8, 1]);   // [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]
assert_eq!(c.multiplicity(), 1);
// Only nonzero entries are stored, sorted by (m1, m2, m3, mu).
assert!(c.entries().iter().all(|e| e.value != 0.0));
```

### What the result means

- **Sparse storage.** `Cgc::entries()` is a slice of `CgcEntry { m1, m2, m3, mu,
  value }`, holding the nonzeros only (after a tolerance purge), sorted by
  `(m1, m2, m3, mu)`. `Cgc::dims()` gives the logical shape
  `[dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]`; `Cgc::nnz()` the stored count.
- **What `m1`/`m2`/`m3` index.** 0-based positions in the **Gelfand–Tsetlin
  basis** of `s1`/`s2`/`s3`, in the order `Irrep::patterns()` returns. That
  ordering is part of the frozen gauge specification, so it is stable across
  releases; if you need to interpret an individual index, `patterns()` gives you
  the GT pattern at that position.
- **Values are real** `f64`, in the crate's canonical gauge.

### Multiplicity greater than one

```rust
use racah::sun::{cgc, Irrep};

let eight = Irrep::from_dynkin(&[1, 1]).unwrap();
let c = cgc(&eight, &eight, &eight).unwrap();
assert_eq!(c.multiplicity(), 2);       // the 8 occurs twice in 8 ⊗ 8
assert_eq!(c.dims(), [8, 8, 8, 2]);
assert!(c.entries().iter().any(|e| e.mu == 1));
```

The two copies `μ = 0, 1` are mutually orthonormal and are fixed by the gauge
specification, not by an arbitrary choice at run time — the same input always
yields the same two copies, in the same order.

## SO(N) / Sp(2N): you need a catalog

The B/C/D generation is a bootstrap: to build a CGC for `s1 ⊗ s2 → s3` it needs
the explicit generator matrices of the irreps involved, which it discovers and
retains. That retained state is a [`CanonicalCatalog`], one per `(series,
rank)`, and you own it.

```rust
use racah::bcd::{CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();   // SO(5) / Spin(5)
let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();      // the 5
let adj = Irrep::from_dynkin(Series::B, &[0, 2]).unwrap();    // the 10

let c = cat.cgc(&v, &v, &adj).unwrap();
assert_eq!(c.multiplicity(), 1);
assert_eq!(c.copy_shape(), (25, 10));    // (dim(s1)·dim(s2), dim(s3))
```

### What the result means

[`CatalogCgc`] is **dense**, not sparse, and laid out per multiplicity copy:

- `copy_shape()` is `(d1·d2, d3)` — the shape of one copy's isometry.
- `copy(mu)` is that copy as a flat **column-major** buffer: element
  `(row, m3)` lives at `copy[m3 * rows + row]`.
- The product-basis row index is **first factor fast**:
  `row = m1 + d1·m2`, so `m1 = row % d1` and `m2 = row / d1` with
  `d1 = dim(s1)`.
- `data()` is all copies concatenated in `μ` order.

Keep a catalog alive for as long as you are working in one `(series, rank)`:
rebuilding it discards the discovered generator sets and makes the next query
re-derive them. Dropping it does **not** invalidate cached coefficient values —
the value cache is keyed on the irrep labels, not on the catalog instance.

## Caveats

- **Generation cost.** CGC generation is the expensive step; F- and R-symbols
  are contractions of already-generated CGC. Results are cached, so repeated
  queries are cheap, but a cold first query on a large irrep is not.
- **Empty channels are errors, not zeros.** Asking for a CGC with
  `N^{s3}_{s1 s2} = 0` returns a typed error. Check with
  [`directproduct`](fusion.md) first if you are unsure.
- **Values are floating point.** Labels, dimensions and multiplicities are
  exact; generated coefficient *values* are verification-gated `f64`. See
  [Resources § exact vs generated](resources.md#exact-vs-generated-values).
- **Index conventions are frozen, not incidental.** Basis order, sign, and
  multiplicity-copy order are specified in [`../gauge.md`](../gauge.md) (SU(N))
  and [`../gauge_soN.md`](../gauge_soN.md) (B/C/D). Most users need not read
  them; consult them if you are comparing against another implementation.

## Related API

- [`su2::clebsch_gordan`], `clebsch_gordan_checked`, `wigner_3j`, `wigner_6j`
- [`sun::cgc`], [`sun::Cgc`], [`sun::CgcEntry`]
- [`bcd::CanonicalCatalog::cgc`], [`CatalogCgc`]

[`SignedSqrtRational`]: https://docs.rs/racah/latest/racah/struct.SignedSqrtRational.html
[`su2::clebsch_gordan`]: https://docs.rs/racah/latest/racah/fn.clebsch_gordan.html
[`sun::cgc`]: https://docs.rs/racah/latest/racah/sun/fn.cgc.html
[`sun::Cgc`]: https://docs.rs/racah/latest/racah/sun/struct.Cgc.html
[`sun::CgcEntry`]: https://docs.rs/racah/latest/racah/sun/struct.CgcEntry.html
[`CanonicalCatalog`]: https://docs.rs/racah/latest/racah/bcd/struct.CanonicalCatalog.html
[`bcd::CanonicalCatalog::cgc`]: https://docs.rs/racah/latest/racah/bcd/struct.CanonicalCatalog.html#method.cgc
[`CatalogCgc`]: https://docs.rs/racah/latest/racah/bcd/struct.CatalogCgc.html
