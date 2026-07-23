# QSpace CGC oracle fixtures

Role (see `tools/README.md` matrix): license-gated hardening strand — the
reference implementation's own CGC numbers, for gauge-alignment cross-checks
(issue #29). GroupMath remains the primary value oracle.

## Provenance

- Source: QSpace v4-pub @ `dd2cc7e` (bitbucket.org/qspace4u/qspace-v4-pub),
  built from source on macOS ARM (maca64) with MATLAB R2026a (30-day trial)
  because upstream ships only Intel (mexmaci64) binaries.
- `getCG_maca64_sync.patch`: the shipped `Source/getCG.cc` entry wrapper
  predates the current clebsch library API (`getCData` was removed upstream;
  still stale at public HEAD `fb36f4e`, 2026-07-14). The patch (a) re-routes
  enumeration through `getQfinal_v(..., QF_LOADC)`, (b) fixes a stale rank-1-only
  argument check, (c) returns hand-built double structs (qset/size/idx/data)
  since the `CData<double>` template closure is broken outside MPFR mode.
  Build must keep MPFR enabled (`QS_SKIP_MPFR` breaks `SetupSym` at runtime).
- `gen_qspace_cgc.m`: generator producing `tests/fixtures/qspace_cgc.txt`.
  Requires the patched `getCG.mexmaca64` on the MATLAB path and `RC_STORE` set
  to a writable directory (bootstrap of new symmetries takes minutes; the
  store persists and reruns resume).
- `getRC_maca64_sync.patch`: analogous sync for the `getRC.cc` entry wrapper,
  which returns an irrep's **generator matrices** (weights `Z`, ladder `Sp{k}`,
  Cartan `Sz{k}`). The stock wrapper's `toMx()` emits MPFR decimal-string blobs
  that are useless to an external consumer; the patch hand-builds a plain-double
  dense struct and handles both `wbsparray` flat-dense and diagonal storage.
- `gen_qspace_generators.m`: generator producing `tests/fixtures/qspace_generators.txt`
  (the six irreps SO5 `[1 0]`/`[0 2]`, Sp4 `[1 0]`/`[2 0]`, SO6 `[1 0 0]`/`[0 1 1]`).
  Requires the patched `getRC.mexmaca64` on the MATLAB path. These generators let
  the S3.5 anchor derive the factor-basis dictionary as a **verified intertwiner**
  (residual `≤ 1e-10`) between racah's and QSpace's generator sets, instead of
  fitting the dictionary against the same projector it is asserted on. The Sp4
  `[2 0]` (dim-10 adjoint) block is currently unused by any test — the CGC fixture
  has no `[2 0]²` channel — and is exported for completeness and future Sp4
  adjoint² work should that channel be added.

## Conventions observed (verified against SU2)

- CGC normalized to **unit Frobenius norm per (channel, OM slice)** —
  e.g. the SU2 triplet block is the standard Wigner table divided by sqrt(3).
- Outer multiplicity appears as a trailing 4th tensor axis
  (e.g. SO6 adjoint x adjoint -> 15 has size `[15 15 15 2]`), matching racah's
  trailing-OM convention.
- `qset` is the concatenation `[J1 J2 J]` of Dynkin-type QSpace labels.
- Indices in the fixture are 0-based, column-major order as emitted.
