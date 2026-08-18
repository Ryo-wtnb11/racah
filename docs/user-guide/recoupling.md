# 6. Recoupling: F- and R-symbols

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

Coupling three irreps `a, b, c` to a total `d` can be done in two orders, and
both give a complete basis of the same space:

```text
(a ⊗ b) ⊗ c → d      via an intermediate e         ("left" tree)
a ⊗ (b ⊗ c) → d      via an intermediate f         ("right" tree)
```

The **F-symbol** `F^{abc}_d[e, f]` is the change of basis between them. The
**R-symbol** `R^{ab}_c` is the change of basis under swapping two factors,
`a ⊗ b → c` versus `b ⊗ a → c`.

Both are contractions of Clebsch–Gordan tensors over all magnetic indices; what
survives are the *multiplicity* indices, one per coupling vertex.

For SU(2) these reduce to the classical closed forms: the F-symbol is the 6j
symbol times a dimension factor and a phase, and R is a sign.

## When do I need them?

When you work in a coupled basis and need to reassociate or braid without ever
touching magnetic indices — the standard situation in tensor networks with
non-abelian symmetry, in recoupling algebra, and in fusion-tree manipulations.
If you need explicit product-basis vectors instead, you want
[CGC](clebsch-gordan.md).

## SU(2): exact scalars

```rust
use racah::{su2_f_symbol, su2_r_symbol, wigner_6j};

// {1 1 1; 1 1 1} = 1/6 in doubled labels (all dj = 2, i.e. spin 1).
assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);

// F and R are multiplicity-free here: plain f64 scalars.
let f = su2_f_symbol(1, 1, 1, 1, 0, 0);   // F^{½½½}_{½}[0, 0]
assert!((f + 0.5).abs() < 1e-14);

let r = su2_r_symbol(1, 1, 0);            // (-1)^(j1+j2-j3) = -1 for ½ ⊗ ½ → 0
assert_eq!(r, -1.0);
```

Argument order is `su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6)` for
`F^{dj1 dj2 dj3}_{dj4}[dj5, dj6]`, and `su2_r_symbol(dj1, dj2, dj3)` for
`R^{dj1 dj2}_{dj3}`. Non-admissible labels give exact `0.0`; the `*_checked`
twins return `Err(Su2Error::NotAdmissible)` instead.

## SU(N): blocks over multiplicity indices

```rust
use racah::sun::{f_symbol, r_symbol, Irrep};

let three = Irrep::from_dynkin(&[1, 0]).unwrap();
let anti = three.dual();
let eight = Irrep::from_dynkin(&[1, 1]).unwrap();

// F^{3 3bar 3}_{3}[8, 8]
let block = f_symbol(&three, &anti, &three, &three, &eight, &eight).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-12);

// R^{3 3bar}_{8}
let r = r_symbol(&three, &anti, &eight).unwrap();
assert_eq!(r.dim(), 1);
assert!((r.at(0, 0).abs() - 1.0).abs() < 1e-12);
```

The argument order is `f_symbol(a, b, c, d, e, f)` for `F^{abc}_d[e, f]` and
`r_symbol(a, b, c)` for `R^{ab}_c`.

## SO(N) / Sp(2N)

Identical, with a [`CanonicalCatalog`](clebsch-gordan.md#son-sp2n-you-need-a-catalog)
threaded through as the first argument:

```rust
use racah::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();   // Sp(4) = C_2
let triv = Irrep::trivial(Series::C, 2).unwrap();
let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap();   // the 5
let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap();    // the adjoint 10

let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

(With `a` trivial the F-move is the identity, so the single entry is 1.)

## What does the result mean?

### `FBlock` — the F-symbol

A dense rank-4 array over the four **multiplicity** indices `[μ, ν, κ, λ]`,
stored row-major. One index per vertex of the recoupling move:

| Axis | Index | Vertex | Length |
|---|---|---|---|
| 0 | `μ` | `a ⊗ b → e` | `N^e_ab` |
| 1 | `ν` | `e ⊗ c → d` | `N^d_ec` |
| 2 | `κ` | `b ⊗ c → f` | `N^f_bc` |
| 3 | `λ` | `a ⊗ f → d` | `N^d_af` |

`FBlock::dims()` returns exactly those four lengths;
`FBlock::at(mu, nu, kappa, lambda)` reads one element; `FBlock::data()` is the
flat row-major buffer. In a multiplicity-free situation all four lengths are 1
and the block holds one scalar at `at(0, 0, 0, 0)`.

This axis order matches the `GenericFusion` convention of TensorKitSectors, so
blocks can be handed to a consumer expecting that layout without a permutation.

### `RBlock` — the R-symbol

A dense `N^c_ab × N^c_ba` matrix. `RBlock::dim()` is `N^c_ab`,
`RBlock::at(mu, nu)` reads an element, `RBlock::data()` is the flat row-major
buffer. Multiplicity-free means a `1×1` block holding a phase.

## Self-checks you can call

The identities that gate generation are also public API, per family, so you can
use them as oracles on your own labels:

```rust
use racah::sun::{check_f_unitarity, check_hexagon, check_pentagon, Irrep};

let three = Irrep::from_dynkin(&[1, 0]).unwrap();
let anti = three.dual();
check_f_unitarity(&three, &anti, &three, &three).unwrap();
check_pentagon(&three, &anti, &three, &anti).unwrap();
check_hexagon(&three, &anti, &three).unwrap();
```

`racah::bcd` has the same three (taking `&mut CanonicalCatalog` first) plus
`check_commutators` for the seed algebra. CGC orthonormality and R-orthogonality
run as generation gates only — they are not callable, and a violation surfaces
as a typed error.

## Caveats

- **An empty vertex is an error.** If any of the four F-symbol vertices has zero
  fusion multiplicity, the call returns a typed error rather than a zero block.
  Check the channels with [`directproduct`](fusion.md) first if in doubt.
- **Cost.** The first F-symbol on a set of labels generates the underlying CGC;
  after that both the CGC and the F block are cached. R is a single sparse join
  of two CGC and is not separately cached.
- **The multiplicity axes are not interchangeable.** `μ` and `ν` belong to the
  left tree, `κ` and `λ` to the right one. Transposing them silently produces a
  different (wrong) recoupling.
- **Gauge.** F and R values depend on the CGC gauge. `racah` publishes them in
  one frozen canonical gauge; see [Resources § gauge](resources.md#gauge-and-reproducibility).

## Related API

- [`racah::su2_f_symbol`], [`racah::su2_r_symbol`], [`racah::wigner_6j`],
  and their `*_checked` twins
- [`sun::f_symbol`], [`sun::r_symbol`], [`sun::check_f_unitarity`],
  `check_pentagon`, `check_hexagon`
- [`bcd::f_symbol`], [`bcd::r_symbol`], and the same three checks
- [`FBlock`], [`RBlock`]

[`racah::su2_f_symbol`]: https://docs.rs/racah/latest/racah/fn.su2_f_symbol.html
[`racah::su2_r_symbol`]: https://docs.rs/racah/latest/racah/fn.su2_r_symbol.html
[`racah::wigner_6j`]: https://docs.rs/racah/latest/racah/fn.wigner_6j.html
[`sun::f_symbol`]: https://docs.rs/racah/latest/racah/sun/fn.f_symbol.html
[`sun::r_symbol`]: https://docs.rs/racah/latest/racah/sun/fn.r_symbol.html
[`sun::check_f_unitarity`]: https://docs.rs/racah/latest/racah/sun/fn.check_f_unitarity.html
[`bcd::f_symbol`]: https://docs.rs/racah/latest/racah/bcd/fn.f_symbol.html
[`bcd::r_symbol`]: https://docs.rs/racah/latest/racah/bcd/fn.r_symbol.html
[`FBlock`]: https://docs.rs/racah/latest/racah/sun/struct.FBlock.html
[`RBlock`]: https://docs.rs/racah/latest/racah/sun/struct.RBlock.html
