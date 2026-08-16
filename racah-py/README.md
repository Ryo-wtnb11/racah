# racah-py

Python bindings (PyO3 + maturin) for the [`racah`](../README.md) crate's SU(N)
surface. Wheels are always built with the crate's `cgc-gen` feature on, so the
generated Clebsch-Gordan / F / R coefficients are available out of the box.

Import name is `racah`; the distribution is `racah-py`.

## Surface

```python
import racah

three = racah.Irrep([1, 0])          # SU(3) fundamental, from Dynkin labels
three.dim()                          # 3 (exact, arbitrary precision)
three.dual()                         # Irrep([0, 1])
racah.fusion(three, three.dual())    # [(Irrep([0, 0]), 1), (Irrep([1, 1]), 1)]
racah.fusion_multiplicity(a, b, c)   # N^c_ab

racah.clebsch_gordan(s1, s2, s3)     # [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]
racah.f_symbol(a, b, c, d, e, f)     # [mu, nu, kappa, lambda] multiplicity block
racah.r_symbol(a, b, c)              # [N^c_ab, N^c_ba]

racah.check_pentagon(a, b, c, d)     # raise on violation
racah.check_hexagon(a, b, c)
racah.check_f_unitarity(a, b, c, d)

racah.sun_authority_fingerprint()    # gauge/authority string to persist alongside coefficients
racah.su2_frobenius_schur(two_j)     # SU(2) Frobenius-Schur indicator
```

Ill-posed input (bad label, mixed rank, empty fusion vertex) raises
`ValueError`; a tripped numerical gate (orthonormality, F-unitarity, pentagon,
hexagon, factorization failure) raises `RuntimeError`.

## Development

```sh
uv venv && uv pip install maturin pytest numpy
maturin develop --release --manifest-path racah-py/Cargo.toml
pytest racah-py/tests
```
