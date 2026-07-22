# SU(N) Clebsch–Gordan gauge specification

This document specifies the **gauge** of the SU(N) Clebsch–Gordan coefficients
produced by `racah::sun::cgc` — the deterministic rule that fixes the otherwise
free basis of each coupled subspace. The gauge is part of this crate's
**semantic-versioning contract**: any change that can alter a returned
coefficient *value* (a different pivot rule, sign convention, tolerance that
moves a rank cut, descent order, or multiplicity-column order) is a **breaking
release**, so consumers may key persisted data on the crate version.

The construction is a port of **SUNRepresentations.jl v0.4.0**
(`~/.julia/packages/SUNRepresentations/BM32Z/src`). Every choice below cites the
reference `file:symbol`. A reader with this document and the reference source
can re-derive the gauge without reading the Rust implementation.

Coefficient *values* are `f64` (as in the reference, which is `Float64`
end-to-end after the exact ladder matrices). What is exact and gauge-fixing is
the *procedure*: the combinatorial basis order, the pivot/sign rules, and the
descent order are discrete facts; only the final linear-algebra solve is
floating point, and it is verification-gated.

---

## 0. Notation

- SU(N) irrep `s` has a normalized highest weight `λ = (λ₁ ≥ … ≥ λ_N)`, `λ_N =
  0`, Dynkin labels `aᵢ = λᵢ − λᵢ₊₁`.
- `d(s) = dim(s)` is the Weyl dimension (`sector.jl:dim`).
- A coupling is `s1 ⊗ s2 → s3` with outer multiplicity
  `N = N^{s3}_{s1 s2}` (`gtpatterns.jl:directproduct`).
- The CGC is a sparse tensor `C[m1, m2, m3, μ]`, `m1 ∈ [0,d1)`, `m2 ∈ [0,d2)`,
  `m3 ∈ [0,d3)`, `μ ∈ [0,N)`. Indices `m` are 0-based positions in the GT basis
  order (§1); `μ` is the outer-multiplicity (trailing) axis (§8).

---

## 1. GT basis order (the load-bearing basis)

The magnetic indices `m1, m2, m3` index the **Gelfand–Tsetlin pattern basis** in
the reference iteration order.

- Reference: `gtpatterns.jl:GTPatternIterator{N}` (`basis(s) =
  GTPatternIterator{N}(weight(s))`). For `N ≥ 2` the iterator loops over the
  admissible second rows `I[i+1]:I[i]` with the **last** sub-row entry varying
  fastest, recursing into `GTPatternIterator{N-1}` as the inner (faster) loop;
  pattern data is stored top row (`l = N`) first.
- Port: `sun::Irrep::patterns` (`sun.rs`), pinned index-for-index by the Layer 1
  fixtures (`tests/sun_oracle.rs`) and re-verified here by the signed CGC oracle
  (`tests/sun_cgc_fixtures.rs`, §11).
- The highest-weight pattern (all rows equal to the top-row prefix) is the
  **last** basis index `d3 − 1`; this is where the highest-weight block is
  stored (`clebschgordan.jl:highest_weight_CGC`, `CGC[m1m2, d3, α]`).

The pattern **weight** used throughout is `gtpatterns.jl:weight(m)`: the
`N`-tuple with component `l` (1-based) equal to `rowsum(l) − rowsum(l−1)`,
`rowsum(l) = Σ_{k=1..l} m[k,l]`, `rowsum(0) = 0`. Port: `cgc.rs:pattern_weight`.

The weight offset `wshift = ⌊(Σλ(s1) + Σλ(s2) − Σλ(s3)) / N⌋` maps an `s1`
weight to the matching `s2` weight at fixed total: `w2 = w3 − w1 + wshift`
(`clebschgordan.jl:highest_weight_CGC`; port `cgc.rs:Ctx::new`).

---

## 2. Highest-weight system

Reference: `clebschgordan.jl:highest_weight_CGC`. Over the coupling pairs
`(m1, m2)` whose weights sum to `s3`'s highest weight, build the sparse linear
system expressing that every simple raising operator annihilates the coupled
highest-weight state:

```
(J⁺_l(s1) ⊗ 𝟙 + 𝟙 ⊗ J⁺_l(s2)) |m1, m2⟩ = 0,   l = 1 … N−1
```

The raising matrices are the exact GT ladder matrices
(`gtpatterns.jl:creation`, `sun::Irrep::creation`), entries `signedroot(coef)`.
Rows are the distinct raised targets `(l, m1′, m2′)` (sorted, deduplicated);
columns are the coupling pairs `(m1, m2)` in first-seen order. Port:
`cgc.rs:highest_weight_cgc`.

---

## 3. Nullspace: tolerance and rank rule

Reference: `clebschgordan.jl:_nullspace!`, called with `atol = TOL_NULLSPACE`.

```
const TOL_NULLSPACE = 1.0e-13
SVD = svd!(A; full = true)
tol = max(atol, S[1] * rtol),   rtol = (min(size(A)) * eps) * iszero(atol)
indstart = #{ i : S[i] > tol } + 1        # = rank + 1
nullspace = copy(SVD.Vt[indstart:end, :]')  # trailing right-singular vectors
```

- Because `atol = 1e-13 > 0`, the relative term vanishes (`iszero(atol) = 0`),
  so the cut is **purely** `Sᵢ > 1e-13`; `rank = #{ Sᵢ > 1e-13 }` and the
  nullspace dimension is `n − rank`.
- **Full** SVD is required: the nullspace is the trailing `n − rank` **rows of
  the full `Vh` (n×n)**, which a thin SVD would discard whenever the system is
  wide (`m < n`, e.g. the minimal SU(2) singlet ½⊗½→0, a 1×2 system).
- Empty system (`m = 0` or `n = 0`): the whole space is the nullspace — return
  the `n×n` identity (`_nullspace!`'s `(m==0 || n==0)` guard).
- Port: `linalg.rs:nullspace` via `tenferro_linalg::svd_full` (§10).

**Multiplicity gate.** `#{nullspace vectors} == directproduct(s1, s2)[s3]` must
hold (`clebschgordan.jl` `@assert N123 == directproduct(s1, s2)[s3]`); a
mismatch is the typed error `SunError::NullspaceDimMismatch` (never silent).

---

## 4. Gauge canonicalization: `gaugefix! = first ∘ qrpos! ∘ cref!`

Reference: `clebschgordan.jl:gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))`.
The nullspace basis `A` (shape `n × N`, columns spanning the coupled subspace)
is canonicalized in two steps. Both steps preserve the **column space** (the
subspace), so the result depends only on the subspace, not on which nullspace
basis the SVD happened to return — which is why an independent SVD/QR
implementation reproduces the reference gauge (verified in §11).

### 4a. `cref!` — column-pivoted reduced echelon (THE pivot rule)

Reference: `clebschgordan.jl:cref!` with `ɛ = TOL_GAUGE = 1.0e-11` (deliberately
looser than `TOL_NULLSPACE`, per the reference comment). Port: `cgc.rs:cref`,
ported statement-for-statement.

Walk pivot rows `i = 1, 2, …` and pivot columns `j = 1, 2, …`:

1. **Pivot column selection.** Among the not-yet-pinned columns `j … nc`, pick
   the column with the **largest `|A[i, j']|`** in the current row `i`. This is
   `findabsmax(view(A, i, j:nc))`.
2. **Tie behavior.** `findabsmax` updates its running maximum only on a **strict**
   `>` (`abs(v) > m`), so on a tie the **leftmost** (smallest column index)
   candidate wins. This tie rule is part of the gauge.
3. **Dead row.** If that maximum is `≤ ɛ`, the row is set to zero over `j:nc`
   (since `ɛ > 0`) and skipped (`i += 1`, `j` unchanged).
4. **Eliminate.** Otherwise swap the pivot column into position `j`, scale
   column `j` so `A[i, j] = 1`, and subtract multiples of column `j` from every
   other column to clear row `i`. Advance `i += 1, j += 1`.

The result is a canonical reduced **column**-echelon representative of the
subspace; the pivot rule fixes which representative.

### 4b. `qrpos!` — positive-diagonal QR sign fix

Reference: `clebschgordan.jl:qrpos!`:

```
q, r = qr!(C)
d = diag(r);  d .= (d == 0 ? 1 : sign(d))   # zero diagonal → +1 (no flip)
Q = q * Diagonal(d);  R = Diagonal(d) \ r     # so every R[i,i] ≥ 0
```

`gaugefix!` keeps **`Q`** (`first(...)`), the orthonormal basis with the sign
convention "each `R` diagonal entry is non-negative; an exactly-zero diagonal is
left unflipped." Port: `linalg.rs:qr_positive_q` via
`tenferro_linalg::QrGauge::PositiveDiagonal`, whose contract ("make each `R`
diagonal entry positive-real, compensating `Q`") is exactly `qrpos!`.

The gauge-fixed highest-weight block is scattered into `C[·, ·, d3−1, μ]`.

---

## 5. Lower-weight descent: order and solve

Reference: `clebschgordan.jl:lower_weight_CGC!`. Port: `cgc.rs:lower_weight_cgc`.

- **Descent order.** Weights of `s3` are visited in **reverse lexicographic**
  order (`w3list = sort(keys(map3); rev = true)`), skipping the first (the
  highest weight, already solved). Reverse-lex guarantees every parent weight
  `w3′` (one raising step up) is solved before its children — the descent never
  reads an unfilled coefficient.
- **Per-weight system.** For each remaining weight `w3` and each multiplicity
  column `α`, apply the lowering intertwiner
  `J⁻₃ |m3⟩ = (J⁻₁ ⊗ 𝟙 + 𝟙 ⊗ J⁻₂) |m1,m2⟩`. The left-hand `eqs[i,j] =
  J⁻₃[m3, m3′]` (over parent states `m3′`, one block per level `l`) and the
  right-hand `rhs` accumulates `J⁻[·]·C[parent]` from the already-solved parents.
  Lowering matrices are `sun::Irrep::annihilation` (transpose of `creation`,
  `gtpatterns.jl`).
- **Solve.** `sols = ldiv!(qr!(eqs), rhs)` — a QR **least-squares** solve of the
  (tall or square, full-column-rank) system. Port: `linalg.rs:lstsq` via
  `tenferro_linalg::lstsq` (§10). Contributions accumulate into
  `C[·, ·, m3, α]`.

---

## 6. Purge

Reference: `clebschgordan.jl:purge!`, `atol = TOL_PURGE = 1.0e-14`: drop every
stored coefficient with `|v| ≤ 1e-14`. Port: `cgc.rs:purge`.

---

## 7. Trivial couplings

Reference: `clebschgordan.jl:trivial_CGC`. `1 ⊗ s → s` gives `C[0, m, m, 0] = 1`;
`s ⊗ 1 → s` gives `C[m, 0, m, 0] = 1` (identity embeddings, no linear algebra).
Port: `cgc.rs:trivial_cgc`.

---

## 8. Outer-multiplicity axis

- The `N` multiplicity columns share **one** nullspace and are gauge-fixed
  **together as a block** by a single `cref! ∘ qrpos!` (§4). Their order on the
  trailing axis `μ` is therefore the column order that block produces — it is
  *not* an independent convention and cannot be chosen per column.
- This is the same ordering SUNRepresentations produces (its 4th CGC index).
  The signed oracle (§11) checks OM ≥ 2 channels **including the `μ` order**, so
  a divergent column order would fail the oracle. (Umbrella #9 pins the
  consumer-facing multiplicity order to TensorKit `[μ,ν,κ,λ]`; that is a
  downstream adapter concern, outside this crate.)

---

## 9. Generation gates (typed, never silent)

Floating-point stages are verification-gated (`AGENTS.md` acceptance 5). A
violation is a typed `SunError`, never a silently degraded coefficient.

- **Multiplicity** (§3): `SunError::NullspaceDimMismatch`.
- **Orthonormality**: the CGC reshaped as `M[(m1,m2),(m3,μ)]` is an isometry,
  `Σ_{m1,m2} C[··,m3,α] C[··,m3′,β] = δ_{m3 m3′} δ_{αβ}` (contracted over the
  coupling indices per output column, **not** summed over `m3`). Worst residual
  `> TOL_ORTHO` → `SunError::NotOrthonormal`.
- **Ladder consistency**: the level-1 lowering intertwiner evaluated at the
  highest-weight parent must reproduce the descended coefficients; residual
  `> TOL_LADDER` → `SunError::LadderInconsistent`.

`TOL_ORTHO = TOL_LADDER = 1e-9` are **not** reference constants; they are sized
well above the f64 SVD/QR/descent round-off floor (`~√dim · eps ≈ 1e-14`) and
far below any coefficient of interest, so a genuine gauge/algebra defect trips
them while faithful round-off does not. Tightening them is not a gauge change
(it cannot alter a returned value), so it is not a breaking release.

Proven-unreachable invariant violations (e.g. a missing raised GT pattern when
its ladder coefficient is nonzero) `panic` in every build (`sun.rs:creation`),
per the crate's error discipline — those are not tolerance events.

---

## 10. Numerical seams (backend)

All dense factorizations route through **tenferro-linalg public APIs only** (no
hand-rolled kernels); the CPU **faer** provider is the one that implements
full-matrices SVD and is pinned by the `cgc-gen` feature.

| Stage | Reference | tenferro-linalg API (`linalg.rs`) |
|---|---|---|
| nullspace | `_nullspace!` (`svd!(A; full=true)`) | `svd_full` → trailing `Vh` rows |
| gauge sign | `qrpos!` (`qr!` + `sign(diag R)`) | `qr_with_options(QrGauge::PositiveDiagonal)` → `Q` |
| descent | `ldiv!(qr!(eqs), rhs)` | `lstsq(eqs, rhs)` |

Build-time tenferro-rs revision is recorded in the PR body. `cref!` is **not** a
factorization kernel — it is the gauge algorithm itself and is ported directly
in `cgc.rs` (§4a).

---

## 11. Verification (independent oracles)

- **SU(2) embedding** (`tests/su2_embedding.rs`): N = 2 CGC vs the crate's exact
  `clebsch_gordan` (big-rational Racah sums, rounded once) over a randomized
  sweep — signed, exact up to the single per-channel highest-weight sign.
- **Gauge continuity** (`tests/sun_cgc_fixtures.rs`): **signed, element-wise**
  agreement with SUNRepresentations.jl v0.4.0 fixtures
  (`tools/gen_sun_cgc_fixtures.jl`, provenance header) across N ∈ {2,3,4}
  including OM ≥ 2 channels and the `μ`-axis order. Observed worst
  `|Δ| ≈ 2.4e-15`.

A change that moves any observed value beyond these oracles' tolerances is, by
definition of this document, a breaking release.
