# 3. Choosing a group

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

Three different things get called "the group", and `racah` keeps them apart:

- the **Lie algebra / root system** (`A_{N-1}`, `B_r`, `C_r`, `D_r`) — fixes the
  weight lattice and all the combinatorics;
- the **compact group** whose representations you want (`SU(N)`, `Spin(N)`,
  `Sp(2r)`, …) — several groups share one root system;
- the **global form** — which of those groups: `SU(N)` vs `PSU(N)`, `Spin(N)` vs
  `SO(N)`.

The load-bearing fact is small: **a global form changes only which highest
weights are allowed, never the value of any coefficient of a surviving irrep.**
Dimensions, duals, fusion rules, CGC, F and R are all shared. So picking a
global form is a construction-time filter, not a different computation.

## When do I need this?

Choosing the *family* (SU(2)/SU(N)/B/C/D): always, at the first line.
Choosing the *global form*: only if you need spinors, or need to exclude irreps
that your physical system does not carry. Basics do not require it.

## SU(2)

Use `racah::su2` when your symmetry is spin. Labels are doubled spins, values
are exact closed forms, and no feature flag is needed.

```rust
use racah::su2::Su2Irrep;

let one = Su2Irrep::new(2);   // spin 1
let channels: Vec<u32> = one.fusion(one).unwrap().map(|s| s.dj()).collect();
assert_eq!(channels, vec![0, 2, 4]); // 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2
```

SU(2) is also reachable as SU(N) with `N = 2` (`racah::sun`, `cgc-gen`), and the
two agree. Prefer `su2`: it is exact, dependency-light, and much faster.

## SU(N), N ≥ 3

Use `racah::sun` (`cgc-gen`). Coefficients are built by the Gelfand–Tsetlin
construction: GT patterns label the basis of an irrep uniquely, so ladder
operators have exact closed-form matrix elements and the CGC follow from a
highest-weight nullspace solve.

```rust
use racah::sun::{directproduct, Irrep};

let eight = Irrep::from_dynkin(&[1, 1]).unwrap();          // SU(3) adjoint
let decomposition = directproduct(&eight, &eight).unwrap();
// 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10-bar ⊕ 27 — note the 8 appears twice.
assert_eq!(decomposition[&eight], 2);
```

## SO(N) and Sp(2N): the B/C/D series

Use `racah::bcd` (`cgc-gen`). These families have no practical GT-type basis, so
`racah` builds them by a **generator bootstrap**: seed the defining
representation's generators explicitly, take tensor products, decompose them
numerically, harvest the new generator sets, recurse.

```rust
use racah::bcd::{directproduct, Irrep, Series};

// SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
let v = Irrep::from_dynkin(Series::B, &[1, 0]).unwrap();
let out = directproduct(&v, &v).unwrap();
let mut dims: Vec<String> = out.keys().map(|s| s.dim().to_string()).collect();
dims.sort();
assert_eq!(dims, vec!["1", "10", "14"]);
```

The label combinatorics here — dimensions, duals, FS indicators, weight
multiplicities, the decomposition `N^c_ab` — are exact integer arithmetic and
need no catalog. Only CGC and F/R generation needs a
[`CanonicalCatalog`](clebsch-gordan.md#son-sp2n-you-need-a-catalog).

## Spin(N): when you need spinors

`Spin(N)` is the simply connected double cover of `SO(N)`. Use it when your
problem carries spinor representations — the ones whose ε-basis highest weights
are half-integers. `SO(N)` does not have them: a spinor is a representation of
the cover, not of `SO(N)` itself, which is why asking for a spinor label as an
`SO(N)` irrep is an error rather than a rounding.

```rust
use racah::bcd::Irrep;
use racah::group::GroupId;

let spin5 = GroupId::spin(5).unwrap();
let s = Irrep::from_dynkin_in(&spin5, &[0, 1]).unwrap();
assert_eq!(s.dim(), 4u32.into());        // the Spin(5) Dirac spinor
assert!(s.is_spinor());
assert_eq!(s.two_partition(), &[1, 1]);  // λ = (½, ½)

// The same label is not a representation of SO(5).
let so5 = GroupId::so(5).unwrap();
assert!(Irrep::from_dynkin_in(&so5, &[0, 1]).is_err());
```

`Irrep::from_dynkin(Series::B, …)` (no group argument) is the historically
published `SO(2r+1)` / `SO(2r)`: tensor irreps only. To get spinors you must
name the cover with `from_dynkin_in`. Their coefficients come from the same
bootstrap, started from a second base case (the Clifford/Fock seeds).

`Spin(2r)` has two spinor chiralities (the two half-spin irreps `ω_{r-1}`,
`ω_r`); `Spin(2r+1)` has one.

## Sp(2N): naming, rank, and the defining dimension

This is the easiest place to be off by a factor of two, so it is spelled out.

- `Series::C` at **rank `r`** is the group `Sp(2r)`, of Cartan type `C_r`.
- Its **defining representation has dimension `2r`**, Dynkin label
  `[1, 0, …, 0]`.
- So `Sp(4)` is `Series::C` with `r = 2`, and its defining rep is the **4**.

```rust
use racah::bcd::{directproduct, Irrep, Series};

// Sp(4) = C_2. The defining 4, and 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
let four = Irrep::from_dynkin(Series::C, &[1, 0]).unwrap();
assert_eq!(four.dim(), 4u32.into());
let out = directproduct(&four, &four).unwrap();
let mut dims: Vec<String> = out.keys().map(|s| s.dim().to_string()).collect();
dims.sort();
assert_eq!(dims, vec!["1", "10", "5"]); // string sort
```

`Sp(2r)` is simply connected, so the `C` series has no spinor sector and no
second global form to choose. (`PSp(2r) = Sp(2r)/Z₂` exists and is available via
`GroupId::psp`, but there is no cover above `Sp(2r)`.)

## Global forms and admissibility

Every connected compact group with a given root system is `G_sc/Γ` for a
subgroup `Γ` of the center of the simply connected form `G_sc`. Its irreps are
exactly those highest weights whose central character is trivial on `Γ`. That
predicate is [`GroupId::admits`], and it is the *whole* content of a global
form here.

```rust
use racah::group::GroupId;
use racah::sun::Irrep;

let psu3 = GroupId::psu(3).unwrap();       // SU(3)/Z₃

// The adjoint 8 has zero triality, so it is a genuine PSU(3) representation.
assert!(Irrep::from_dynkin_in(&psu3, &[1, 1]).is_ok());

// The fundamental 3 has triality 1: PSU(3) has no such representation.
assert!(Irrep::from_dynkin_in(&psu3, &[1, 0]).is_err());
```

Available forms: `GroupId::su`, `su_quotient(n, k)`, `psu`, `sp`, `psp`, `spin`,
`so`, `pso`, `half_spin_plus`, `half_spin_minus`.

**What "admissible" means, precisely.** Two different things in this crate are
called admissibility and they are unrelated:

1. *Global-form admissibility* (this section): is this weight a representation
   of this group at all? Enforced once, at `Irrep` construction, by
   `from_dynkin_in`. Failure is `SunError::NotAdmissible` /
   `BcdError::NotAdmissible`.
2. *Coupling admissibility* (SU(2)): does this label tuple satisfy the triangle
   and parity rules of a 3j/6j? Failure is an exact zero, or
   `Su2Error::NotAdmissible` on the `*_checked` functions.

Because the central-character class map is a group homomorphism, admissibility
is closed under fusion and duality: a product of admissible irreps decomposes
into admissible irreps only, and never needs re-checking. That is also why the
coefficient caches are form-free — two forms that share an irrep share its
basis, gauge, and cached values.

## Caveats

- `racah` covers **connected compact** groups only. `O(N)` and `Pin(N)` are not
  central quotients of a simply connected group and are out of scope.
- Choosing a global form never changes a number. If you were hoping `SO(N)`
  coefficients would differ from `Spin(N)` ones on shared irreps: they do not,
  by construction.
- The `G2`–`E8` variants of [`RootSystem`] exist for type completeness but carry
  no coefficient engine.

## Related API

- [`racah::group`]: `GroupId`, `RootSystem`, `GlobalForm`, `CenterSubgroup`,
  `GroupId::admits`
- [`sun::Irrep::from_dynkin_in`], [`bcd::Irrep::from_dynkin_in`]
- Deeper background: [`../theory.pdf`](../theory.pdf) §3 (the Killing–Cartan
  classification and the Dynkin diagrams), §4 (each family's compact group,
  its centre, and the global forms as centre quotients), §5 (the
  classification-to-code correspondence table), §8 (why each family gets its
  algorithm), and the `racah::group` module docs for the lattice arithmetic.

[`GroupId::admits`]: https://docs.rs/racah/latest/racah/group/struct.GroupId.html#method.admits
[`RootSystem`]: https://docs.rs/racah/latest/racah/group/enum.RootSystem.html
[`racah::group`]: https://docs.rs/racah/latest/racah/group/index.html
[`sun::Irrep::from_dynkin_in`]: https://docs.rs/racah/latest/racah/sun/struct.Irrep.html#method.from_dynkin_in
[`bcd::Irrep::from_dynkin_in`]: https://docs.rs/racah/latest/racah/bcd/struct.Irrep.html#method.from_dynkin_in
