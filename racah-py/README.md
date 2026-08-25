# racah-py

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

Python bindings (PyO3 + maturin) for the [`racah`](../README.md) crate. `racah`
is the coefficient authority behind TeNeT-py's symmetry providers; this package
is how that consumer (and any other Python code) reaches it.

Two surfaces:

- **SU(N)**, taking `Irrep` labels and running the generated Gelfand–Tsetlin
  pipeline — irreps from Dynkin labels, fusion with outer multiplicities, dense
  m-basis Clebsch–Gordan tensors, F/R symbols with their multiplicity axes, and
  the verification gates.
- **exact SU(2)**, taking doubled integer spins and returning scalars from
  closed-form big-rational arithmetic — `wigner_3j`, `wigner_6j`,
  `su2_clebsch_gordan`, `su2_f_symbol`, `su2_r_symbol`, `su2_frobenius_schur`.

SU(2) is reachable through either (`Irrep([2j])` is a rank-1 label), and they
are separate authorities with separate fingerprints. They agree on F and R and
**differ on the Clebsch–Gordan coefficients by one sign per fusion channel** —
see "Versioning and the gauge fingerprint" below.

Import name is `racah`; the distribution is `racah-py`. Wheels are always built
with the crate's `cgc-gen` feature on, so the generated coefficients are
available out of the box. CGC/F/R arrays come back as NumPy `float64` arrays.

## Installation

From [PyPI](https://pypi.org/project/racah-py/) (Python ≥ 3.12; prebuilt
abi3 wheels for linux-x86_64 and macos-arm64):

```sh
pip install racah-py
```

Or from a checkout (needs a Rust stable toolchain):

```sh
git clone https://github.com/Ryo-wtnb11/racah
cd racah/racah-py
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release          # builds the extension into the active venv
```

or build a wheel and install it anywhere:

```sh
maturin build --release --manifest-path racah-py/Cargo.toml --out dist
pip install --no-index --find-links dist racah-py
```

CI ([`wheels.yml`](../.github/workflows/wheels.yml)) builds `abi3-py312` wheels
for linux-x86_64 and macos-arm64; `py-v*` tags/releases publish them to PyPI.
One wheel per platform covers CPython ≥ 3.12.

## Quick start

Verified against the built extension; every call below is the actual API
(see [`racah.pyi`](racah.pyi) for the full typed surface).

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

The exact SU(2) surface takes **doubled** labels (`dj = 2j`, `dm = 2m`), so
every label stays an exact integer:

```python
import racah

assert abs(racah.wigner_6j(2, 2, 2, 2, 2, 2) - 1 / 6) < 1e-14
assert racah.su2_r_symbol(1, 1, 0) == -1.0        # (-1)^(1/2 + 1/2 - 0)
assert racah.su2_r_symbol(1, 1, 4) == 0.0         # inadmissible is exact zero,
                                                  # not an error
print(racah.su2_authority_fingerprint())
```

Reaching the same R-symbol through `r_symbol(Irrep([1]), Irrep([1]), Irrep([0]))`
builds a Clebsch–Gordan tensor first; `su2_r_symbol` is a sign.

The rest of the surface: `Irrep.from_weight` / `Irrep.trivial` constructors,
the `dynkin` / `weight` / `rank` properties, and `check_f_unitarity` and
`check_hexagon` alongside `check_pentagon`.

Ill-posed input (bad label, mixed rank, empty fusion vertex) raises
`ValueError`; a tripped numerical gate (orthonormality, F-unitarity, pentagon,
hexagon, factorization failure) raises `RuntimeError`.

## Versioning and the gauge fingerprint

F/R/CGC values depend on the CGC gauge; `racah` publishes them in one frozen
canonical gauge. `racah.sun_authority_fingerprint()` returns the opaque
authority string identifying that convention, generation pipeline, and
tolerance policy — TeNeT-py pins it next to persisted coefficients and refuses
a mismatch on load. Compare it by equality only; a breaking coefficient change
bumps the fingerprint epoch and is recorded in the
[CHANGELOG](../CHANGELOG.md). Same fingerprint means same convention and value
agreement within the oracle tolerance, not cross-process bit-identity
([docs.rs: `sun_authority_fingerprint`](https://docs.rs/racah/latest/racah/sun/fn.sun_authority_fingerprint.html)).

`racah.su2_authority_fingerprint()` is its twin for the exact SU(2) surface, and
the two are different strings on purpose. Numerical agreement is not authority
identity: F-symbols and R-symbols do agree between the tiers to round-off, but
their **Clebsch–Gordan coefficients differ by exactly one sign per fusion
channel**, uniform in the magnetic indices and equal to `su2_r_symbol` itself.
F and R are gauge-invariant combinations in which that phase cancels; CGC are
gauge data. Mixing the two tiers' CGC without the factor gives wrong signs and
no error, so record the fingerprint of whichever surface produced what you
persisted.

## Documentation

The semantics are the crate's; the bindings add nothing:

- [User Guide](../docs/user-guide/README.md) — task-oriented chapters; shapes
  and axis conventions match the arrays returned here.
- [`docs/theory.pdf`](../docs/theory.pdf) — the mathematics behind the objects.
- [`docs/gauge.md`](../docs/gauge.md) — the frozen SU(N) gauge specification
  the fingerprint names.
- [docs.rs/racah](https://docs.rs/racah) — exact per-item semantics and errors.
- [`docs/README.md`](../docs/README.md) — the full documentation index.

## Development

```sh
uv venv && uv pip install maturin pytest numpy
cd racah-py && maturin develop --release
pytest tests
```
