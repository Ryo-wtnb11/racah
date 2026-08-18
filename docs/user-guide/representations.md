# 2. Representations

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is an irrep here?

An **irrep** (irreducible representation) is, in `racah`, just a *label*: a
dominant integral highest weight. Nothing about the representation is stored —
no matrices, no basis — only the weight that identifies it. Everything else
(dimension, dual, basis size, coefficients) is computed from that label on
demand.

Each family has its own label type, because each family has its own natural
encoding:

| Family | Type | Label |
|---|---|---|
| SU(2) | [`su2::Su2Irrep`] | one `u32`, the **doubled spin** `dj = 2j` |
| SU(N) | [`sun::Irrep`] | `N-1` Dynkin labels (stored internally as a normalized weight) |
| SO(N), Spin(N), Sp(2N) | [`bcd::Irrep`] | `r` Dynkin labels plus the Cartan series `B`/`C`/`D` |

## When do I need this?

Always — every coefficient function takes irrep labels. Getting the label
convention right is the single most common source of confusion, so this chapter
is worth reading once before anything else.

## SU(2): doubled spins

```rust
use racah::su2::{su2_frobenius_schur, Su2Irrep};

let s = Su2Irrep::new(3);        // dj = 3, i.e. spin 3/2
assert_eq!(s.dj(), 3);
assert_eq!(s.dim(), 4);          // 2j + 1
assert_eq!(s.dual(), s);         // every SU(2) irrep is self-dual
assert_eq!(su2_frobenius_schur(3), -1.0); // half-integer j: symplectic self-duality
```

Every `u32` is a valid doubled spin, so `Su2Irrep::new` cannot fail and neither
can `dj`, `dim` or `dual`. Projections are doubled too: `dm = 2m`, an `i32`.

**Why doubled.** Spin ½ is not an integer, and a float label would make
admissibility tests approximate. `dj = 2j` keeps every label exact and turns the
triangle and parity rules into integer arithmetic. The B/C/D families use the
same trick for the same reason — see [doubled weights](#son-sp2n-dynkin-labels-and-doubled-weights).

## SU(N): Dynkin labels

An SU(N) irrep is built from its `N-1` Dynkin labels `(a₁,…,a_{N-1})`, all
non-negative integers, in Bourbaki numbering (`a₁` is attached to the first node
of the `A_{N-1}` chain).

```rust
use racah::sun::Irrep;

let three = Irrep::from_dynkin(&[1, 0]).unwrap();   // SU(3) fundamental
let eight = Irrep::from_dynkin(&[1, 1]).unwrap();   // SU(3) adjoint
let singlet = Irrep::trivial(3).unwrap();           // == from_dynkin(&[0, 0])

assert_eq!(three.dim(), 3u32.into());
assert_eq!(eight.dim(), 8u32.into());
assert_eq!(three.dual().dynkin(), vec![0, 1]);      // 3-bar
assert_eq!(eight.dual(), eight);                    // the adjoint is self-dual
```

`N` is implied by the label length: `&[1, 0]` has two labels, so it is SU(3).
`Irrep::rank()` returns `N` (not `N-1`).

**Dimensions are exact and unbounded.** `dim()` returns a `BigInt`, computed by
the Weyl dimension formula in exact integer arithmetic. There is no label cut;
`3u32.into()` above is just the conversion for the comparison.

**Weights vs Dynkin labels.** Internally an `Irrep` stores the highest weight as
a non-increasing `N`-tuple `λ` with `λ_N = 0`; the Dynkin labels are
`aᵢ = λᵢ − λᵢ₊₁`. `Irrep::from_weight` accepts any shift-representative of `λ`
and normalizes it; `Irrep::weight()` and `Irrep::dynkin()` read the two forms
back. Use Dynkin labels unless you have a reason not to — they are the form the
rest of the documentation quotes.

## SO(N), Sp(2N): Dynkin labels and doubled weights

`racah::bcd` covers the three orthogonal/symplectic Cartan series in one module,
selected by [`bcd::Series`]:

| Series | Group (simply connected form) | Historically published form |
|---|---|---|
| `Series::B`, rank `r` | `Spin(2r+1)` | `SO(2r+1)` |
| `Series::C`, rank `r` | `Sp(2r)` | `Sp(2r)` (already simply connected) |
| `Series::D`, rank `r` | `Spin(2r)` | `SO(2r)` |

The module is called `bcd` rather than `son` because `C = Sp` is not an
orthogonal series; only the Cartan letters name all three honestly.

```rust
use racah::bcd::{Irrep, Series};

// SO(5) = B_2. Dynkin [1, 0] is the 5-dimensional vector representation.
let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
assert_eq!(v.dim(), 5u32.into());
assert_eq!(v.dual(), v);
assert_eq!(v.frobenius_schur(), 1);       // real (orthogonal) self-duality
assert_eq!(v.partition(), Some(vec![1, 0])); // λ = (1, 0)
```

**Weights are doubled.** `bcd::Irrep` stores `2λ` in the orthonormal ε-basis, so
`two_partition()` is always exact and `partition()` returns `Some(λ)` only for a
tensor irrep (integer `λ`) and `None` for a spinor (half-integer `λ`). Same
motivation as SU(2)'s `dj = 2j`.

**Excluded low ranks.** `B_1`, `C_1` and `D_2` are rejected with
`BcdError::ExcludedRank`, because they are SU(2) statements in disguise
(`Spin(3) ≅ SU(2)`, `Sp(2) ≅ SU(2)`, `Spin(4) ≅ SU(2)×SU(2)`). The error names
the SU(2) redirect for the form you asked about.

## Duals and Frobenius–Schur indicators

The **dual** `ā` of an irrep `a` is the conjugate representation; `a ⊗ ā` always
contains the singlet exactly once. The **Frobenius–Schur indicator** is only
meaningful for a *self-dual* irrep (`a == ā`), where it says *how* the irrep is
self-dual:

| FS value | Meaning |
|---|---|
| `+1` | real / orthogonal self-duality (a symmetric invariant bilinear form) |
| `−1` | pseudo-real / symplectic self-duality (an antisymmetric one) |
| `0` | not self-dual (complex) |

```rust
use racah::bcd::{Irrep, Series};

// Sp(4) = C_2. Its defining 4 is pseudo-real; the 5 is real.
let four = Irrep::from_dynkin(Series::C, &[1, 0]).unwrap();
let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap();
assert_eq!(four.dim(), 4u32.into());
assert_eq!(four.frobenius_schur(), -1);
assert_eq!(five.dim(), 5u32.into());
assert_eq!(five.frobenius_schur(), 1);
```

For SU(2), `su2_frobenius_schur(dj)` is `(-1)^dj`: `+1` for integer spin, `−1`
for half-integer. For SU(N) the indicator is derivable from `dual()` (an irrep
with `a != a.dual()` is complex, indicator `0`); `bcd::Irrep::frobenius_schur()`
returns it directly.

## Caveats

- Dynkin label *length* determines the rank, and the rank determines the group.
  A three-element slice passed to `sun::Irrep::from_dynkin` is SU(4), not SU(3)
  with a typo — there is no way for the library to tell.
- `sun::Irrep::rank()` is `N`, whereas `bcd::Irrep::rank()` is the Cartan rank
  `r`. Both match their family's conventional meaning; neither is the number of
  Dynkin labels for SU(N) (that is `N - 1`).
- Negative Dynkin labels are rejected everywhere. In the `D` series a *negative
  last partition entry* `λ_r` is legal (it is the chirality sign) but that is
  the partition, not the Dynkin label.

## Related API

- [`su2::Su2Irrep`], [`racah::su2_frobenius_schur`]
- [`sun::Irrep`]: `from_dynkin`, `from_dynkin_in`, `from_weight`, `trivial`,
  `dim`, `dual`, `dynkin`, `weight`, `rank`, `patterns`
- [`bcd::Irrep`]: `from_dynkin`, `from_dynkin_in`, `trivial`, `dim`, `dual`,
  `dynkin`, `partition`, `two_partition`, `is_spinor`, `frobenius_schur`,
  `weight_multiplicities`

[`su2::Su2Irrep`]: https://docs.rs/racah/latest/racah/su2/struct.Su2Irrep.html
[`racah::su2_frobenius_schur`]: https://docs.rs/racah/latest/racah/fn.su2_frobenius_schur.html
[`sun::Irrep`]: https://docs.rs/racah/latest/racah/sun/struct.Irrep.html
[`bcd::Irrep`]: https://docs.rs/racah/latest/racah/bcd/struct.Irrep.html
[`bcd::Series`]: https://docs.rs/racah/latest/racah/bcd/enum.Series.html
