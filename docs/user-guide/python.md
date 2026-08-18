# 8. Python bindings

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

[`racah-py`](../../racah-py/README.md) is the PyO3/maturin binding of the
crate's SU(N) surface. Import name `racah`, distribution `racah-py`. The
semantics are the crate's — same values, same gauge, same axis conventions as
the Rust API — and CGC/F/R arrays come back as C-contiguous NumPy `float64`
arrays. Wheels are always built with `cgc-gen` on, so the generated
coefficients work out of the box.

## Installation

```sh
pip install racah-py
```

Prebuilt abi3 wheels cover CPython ≥ 3.12 on linux-x86_64 and macos-arm64;
anywhere else, `pip` builds from source (needs a Rust stable toolchain). The
package ships type stubs ([`racah.pyi`](../../racah-py/racah.pyi)) and a
`py.typed` marker, so `mypy`/`ty`/IDEs resolve the full typed surface — the
stubs double as the Python API reference, with per-function docstrings
(also available at runtime via `help(racah.f_symbol)` etc.).

## Quick start

```python
import racah

three = racah.Irrep([1, 0])            # SU(3) fundamental, from its Dynkin label
assert three.dim() == 3                # exact integer, arbitrary precision
eight = racah.Irrep([1, 1])            # the adjoint

# Fusion: 3 x 3bar = 1 + 8, and the adjoint appears twice in 8 x 8.
assert racah.fusion(three, three.dual()) == [
    (racah.Irrep([0, 0]), 1),
    (eight, 1),
]
assert racah.fusion_multiplicity(eight, eight, eight) == 2  # N^8_88

# m-basis Clebsch-Gordan tensor: [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}].
cgc = racah.clebsch_gordan(three, three.dual(), eight)
assert cgc.shape == (3, 3, 8, 1)

# F-symbol block over the four multiplicity indices [mu, nu, kappa, lambda].
f = racah.f_symbol(eight, eight, eight, eight, eight, eight)
assert f.shape == (2, 2, 2, 2)

# R-symbol multiplicity matrix, N^c_ab x N^c_ba.
r = racah.r_symbol(three, three.dual(), eight)
assert r.shape == (1, 1)

# The verification gates raise RuntimeError on violation.
racah.check_pentagon(three, three.dual(), three, three.dual())

# The gauge fingerprint a consumer pins next to persisted coefficients.
print(racah.sun_authority_fingerprint())
```

The rest of the surface: `Irrep.from_weight` / `Irrep.trivial` constructors,
the `dynkin` / `weight` / `rank` properties, `check_f_unitarity` and
`check_hexagon` alongside `check_pentagon`, and `su2_frobenius_schur(two_j)`.

## Worked example: the CGC multiplicity axis

The Python twin of [Clebsch–Gordan § SU(N)](clebsch-gordan.md). The tensor for
`s1 ⊗ s2 → s3` has shape `(dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2})`: axes
0–2 index the Gelfand–Tsetlin m-basis states of `s1`, `s2`, `s3`; the trailing
axis `mu` indexes the outer-multiplicity copies of `s3`. Each `mu` slice is an
orthonormal isometry:

```python
import numpy as np
import racah

eight = racah.Irrep([1, 1])                      # SU(3) adjoint
n = racah.fusion_multiplicity(eight, eight, eight)
assert n == 2                                    # 8 appears twice in 8 x 8

cgc = racah.clebsch_gordan(eight, eight, eight)
assert cgc.shape == (8, 8, 8, 2)                 # trailing axis = mu

# Orthonormality across multiplicity copies: contracting the m-indices of two
# copies gives delta_{mu mu'} x identity on m3.
for mu in range(n):
    for nu in range(n):
        overlap = np.einsum("abm,abn->mn", cgc[:, :, :, mu], cgc[:, :, :, nu])
        expected = np.eye(8) if mu == nu else np.zeros((8, 8))
        assert np.allclose(overlap, expected, atol=1e-12)
```

An impossible channel is a typed error, not a zero array:

```python
import racah

three = racah.Irrep([1, 0])
try:
    racah.f_symbol(three, three, three, three, three, three)  # 3x3 has no 3
except ValueError:
    pass
```

## Worked example: the F-symbol's four multiplicity axes

The Python twin of [Recoupling § F](recoupling.md). `f_symbol(a, b, c, d, e, f)`
returns the block `F^{abc}_d[e, f]` with the magnetic indices contracted away,
shape `(N^e_ab, N^d_ec, N^f_bc, N^d_af)` — one axis per fusion vertex:

| axis | index | vertex |
|---|---|---|
| 0 | `mu` | `a ⊗ b → e` |
| 1 | `nu` | `e ⊗ c → d` |
| 2 | `kappa` | `b ⊗ c → f` |
| 3 | `lambda` | `a ⊗ f → d` |

Row-major (C-contiguous), so the flat index of `block[mu, nu, kappa, lam]` is
`((mu * N2 + nu) * N3 + kappa) * N4 + lam` — the TensorKitSectors
`GenericFusion` axis order, no permutation needed.

```python
import numpy as np
import racah

three = racah.Irrep([1, 0])
anti = three.dual()
eight = racah.Irrep([1, 1])

# Multiplicity-free: every vertex has N = 1, the block is one scalar.
f = racah.f_symbol(three, anti, three, three, eight, eight)
assert f.shape == (1, 1, 1, 1)
assert np.isclose(f[0, 0, 0, 0], 1.0 / 3.0)

# With multiplicity: each 8 x 8 -> 8 vertex has N = 2, so 2x2x2x2.
f8 = racah.f_symbol(eight, eight, eight, eight, eight, eight)
assert f8.shape == (
    racah.fusion_multiplicity(eight, eight, eight),  # mu:     a x b -> e
    racah.fusion_multiplicity(eight, eight, eight),  # nu:     e x c -> d
    racah.fusion_multiplicity(eight, eight, eight),  # kappa:  b x c -> f
    racah.fusion_multiplicity(eight, eight, eight),  # lambda: a x f -> d
)

# Assembled over all (e, f), the F-move is unitary; the gate checks that.
racah.check_f_unitarity(eight, eight, eight, eight)
```

## The fingerprint contract for Python consumers

F/R/CGC values depend on the CGC gauge; `racah` publishes them in one frozen
canonical gauge ([`../gauge.md`](../gauge.md)).
`racah.sun_authority_fingerprint()` returns the opaque authority string
identifying that convention, generation pipeline and tolerance policy.

The contract, identical to the Rust one:

- **Pin it.** Store the fingerprint next to any coefficients you persist and
  refuse a mismatch on load. TeNeT-py, the primary Python consumer, pins it
  exactly this way in its SU(N) symmetry provider.
- **Compare by equality only.** The string is opaque; do not parse it.
- **Epoch bumps are breaking.** A change that moves coefficient values bumps
  the fingerprint epoch and is recorded in the
  [CHANGELOG](../../CHANGELOG.md).
- **Same fingerprint ≠ bit-identity.** It means same convention and value
  agreement within the oracle tolerance, not cross-process bit-identical
  floats.

## Errors

Ill-posed input (bad Dynkin label, mixed rank, empty fusion vertex) raises
`ValueError`; a tripped numerical gate (orthonormality, F-unitarity, pentagon,
hexagon, factorization failure) raises `RuntimeError`. The gate checkers
(`check_f_unitarity`, `check_pentagon`, `check_hexagon`) return `None` on
success and raise on violation.

## API reference

The type stubs [`racah-py/racah.pyi`](../../racah-py/racah.pyi) are the Python
API reference: every function and method with its precise types, array shapes,
axis semantics and raised exceptions. The same information is in each object's
`__doc__` (`help(racah.clebsch_gordan)`). For the underlying per-item
semantics, [docs.rs/racah](https://docs.rs/racah).
