# 4. Tensor products and fusion

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

The tensor product of two irreps decomposes into a direct sum of irreps:

```text
a ⊗ b  ≅  ⨁_c  N^c_ab · c
```

The non-negative integer `N^c_ab` is the **fusion multiplicity** (also: outer
multiplicity): how many independent copies of `c` sit inside `a ⊗ b`. `racah`
computes it exactly, in integer arithmetic, for every family.

## When do I need it?

Before anything else. `N^c_ab` tells you which coupling channels exist at all,
and it is the length of the multiplicity axis on every CGC, F and R block you
will later ask for. If `N^c_ab = 0` there is no coefficient to compute, and the
coefficient APIs say so with a typed error instead of returning zeros.

## SU(2)

The classic triangle rule. `Su2Irrep::fusion` returns an allocation-free
iterator over the coupled doubled spins `|dj1 − dj2| ..= dj1 + dj2` in steps of
2, each with multiplicity 1 (SU(2) is multiplicity-free).

```rust
use racah::su2::Su2Irrep;

let half = Su2Irrep::new(1);
let channels: Vec<u32> = half.fusion(half).unwrap().map(|s| s.dj()).collect();
assert_eq!(channels, vec![0, 2]);   // ½ ⊗ ½ = 0 ⊕ 1
```

The only failure is `Su2Error::LabelOverflow`, when `dj1 + dj2` exceeds `u32`.

## SU(N)

`sun::directproduct` returns a `BTreeMap<Irrep, u32>` — every `c` with
`N^c_ab > 0`, mapped to its multiplicity. The computation is
Littlewood–Richardson, exact.

```rust
use racah::sun::{directproduct, Irrep};

let three = Irrep::from_dynkin(&[1, 0]).unwrap();
let anti = three.dual();
let out = directproduct(&three, &anti).unwrap();

// 3 ⊗ 3-bar = 1 ⊕ 8, each once.
assert_eq!(out.len(), 2);
assert_eq!(out[&Irrep::trivial(3).unwrap()], 1);
assert_eq!(out[&Irrep::from_dynkin(&[1, 1]).unwrap()], 1);
```

### Multiplicity greater than one

`N^c_ab > 1` is the normal case for `N ≥ 3`, and it is what makes the
coefficient blocks arrays instead of scalars.

```rust
use racah::sun::{directproduct, Irrep};

let eight = Irrep::from_dynkin(&[1, 1]).unwrap();       // SU(3) adjoint
let out = directproduct(&eight, &eight).unwrap();
assert_eq!(out[&eight], 2);     // the 8 appears twice in 8 ⊗ 8
```

There is a second entry point, `sun::shared_directproduct`, which returns a
cheaply cloneable [`SunProduct`] backed by the process-global product cache. Use
it when you decompose the same pair repeatedly; use `directproduct` when you
want a plain owned map.

## SO(N) / Sp(2N) / Spin(N)

`bcd::directproduct` has the same shape and the same exactness. It uses the
Brauer–Klimyk / Racah–Speiser character rule and needs no
[`CanonicalCatalog`](clebsch-gordan.md#son-sp2n-you-need-a-catalog) — you can
decompose products without ever generating a coefficient.

```rust
use racah::bcd::{directproduct, Irrep, Series};

// SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
let out = directproduct(&v, &v).unwrap();
assert_eq!(out.len(), 3);
for (irrep, mult) in &out {
    assert_eq!(*mult, 1);
    let _ = irrep.dim();
}
```

## What does the result mean?

- The map's **keys** are the irreps `c` that occur; irreps with `N^c_ab = 0` are
  absent, not present with value `0`.
- The **values** are `N^c_ab`. Summing `N^c_ab · dim(c)` over the map reproduces
  `dim(a) · dim(b)` — a cheap sanity check.
- The map is a `BTreeMap`, so iteration order is the deterministic `Ord` order
  of the label type. That order is stable, but it is *not* a physics ordering
  (not by dimension, not by highest weight height).

## Caveats

- Both `directproduct` functions require the two irreps to belong to the same
  family and rank; a mismatch is a typed error, not a panic.
- Multiplicity is `u32`. The B/C/D weight-multiplicity accumulation is `i128`
  internally and reports overflow as a typed error rather than wrapping.
- Fusion respects global forms automatically: if `a` and `b` are admissible for
  a form, every `c` in the decomposition is too, so no re-filtering is needed.

## Related API

- [`su2::Su2Irrep::fusion`], [`su2::Su2Fusion`]
- [`sun::directproduct`], [`sun::shared_directproduct`], [`SunProduct`]
- [`bcd::directproduct`]

[`su2::Su2Irrep::fusion`]: https://docs.rs/racah/latest/racah/su2/struct.Su2Irrep.html#method.fusion
[`su2::Su2Fusion`]: https://docs.rs/racah/latest/racah/su2/struct.Su2Fusion.html
[`sun::directproduct`]: https://docs.rs/racah/latest/racah/sun/fn.directproduct.html
[`sun::shared_directproduct`]: https://docs.rs/racah/latest/racah/sun/fn.shared_directproduct.html
[`SunProduct`]: https://docs.rs/racah/latest/racah/sun/struct.SunProduct.html
[`bcd::directproduct`]: https://docs.rs/racah/latest/racah/bcd/fn.directproduct.html
