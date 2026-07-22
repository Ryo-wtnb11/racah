# racah Agent Policy

Design authority: the architecture, acceptance criteria, and reference-role
map recorded in the upstream design discussion. Read them before any
non-trivial change.

## Boundaries

- Pure representation mathematics only. No fusion-category trait vocabulary,
  no sector identity types, no tensor-network concepts, and no dependency on
  any tensor-network engine crate.
- Base crate: exact SU(2) only; dependencies stay minimal (big-integer
  arithmetic at most).
- `cgc-gen` feature: all runtime generation (SU(N)/SO(N)/Sp(2N)) and every
  dense-kernel dependency live behind this feature. Factorizations
  (nullspace/QR/least-squares) and CGC contractions route through the
  selected dense backend; hand-rolled numeric kernels are not accepted.
- CGC are a legitimate public output of this crate; recoupling (F/R)
  derivation and its caches live here, next to the gauge contract.

## Acceptance criteria (every coefficient-affecting change)

1. Combinatorial structure exact (integer/rational; no floats in
   enumeration, multiplicities, or weights).
2. Discrete data exact (duals, Frobenius–Schur phases, signs, basis order).
3. Gauge fixing deterministic: pivot rules and sign conventions are part of
   the specification; discrete gauge flips across runs/backends are defects.
4. Algorithm/gauge changes bump the version; values are part of the semver
   contract.
5. Floating-point stages verification-gated: orthogonality, unitarity,
   pentagon/hexagon checks run at generation time; violations are typed
   errors, never silently degraded coefficients.

## Guard inventory (every port PR)

Reference guards that live *outside* the expression being ported are the
recurring defect class tracked in issue #15 (dropped `@assert`s, type
constraints, and unreachable-by-construction assumptions). Every port PR must:

1. Enumerate, in the PR body, each guard on the reference path being ported —
   `@assert`/`@check`, type or domain constraints, and invariants the reference
   relies on as unreachable-by-construction.
2. Map each to exactly one of: a typed error, a documented loud-panic invariant,
   or an explicit N/A with justification. An unmapped guard blocks merge.
3. Ensure every fast-path / special-case branch runs the same verification
   gates as the general path, or prove those gates vacuous for that branch.

## tenferro-rs dependency (cgc-gen)

The `cgc-gen` feature depends on `tenferro-{linalg,cpu,runtime}` via a **git
dependency pinned to an exact revision** (`Cargo.toml`), so a standalone CI
checkout with no sibling `tenferro-rs` on disk resolves. Bumping the pinned
`rev` is an ordinary reviewed commit (it can change coefficient values — see the
gauge semver contract).

To develop against an in-progress local `tenferro-rs` checkout, use a
**developer-local, never-committed** override: a gitignored `.cargo/config.toml`
with

```toml
[patch."https://github.com/tensor4all/tenferro-rs"]
tenferro-linalg  = { path = "../tenferro-rs/crates/tenferro-linalg" }
tenferro-cpu     = { path = "../tenferro-rs/crates/tenferro-cpu" }
tenferro-runtime = { path = "../tenferro-rs/crates/tenferro-runtime" }
```

`.cargo/` is gitignored. CI and every reviewed build use the pinned git rev, not
the patch.

## Verification

- Oracles are independent: reference-implementation outputs (WignerSymbols,
  SUNRepresentations, QSpace after gauge alignment), checked-in fixtures
  with provenance, and self-consistency identities. Values derived from the
  code under test are not oracles.
- `cargo fmt` and `cargo test` (all feature combinations touched) before
  every commit; fine-grained commits, each building and testing green.
