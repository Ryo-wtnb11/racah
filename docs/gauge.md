# Coefficient gauge specification (base SU(2) and SU(N))

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository,
> and its prose may contain errors. Its **gauge values are not commentary**:
> they are the frozen normative specification of racah's conventions, verified
> by the golden tests. Code follows this document — where the two disagree, the
> code is the bug.

This document specifies the **gauge** of the coefficients this crate returns —
the deterministic rules that fix the otherwise free basis, sign, and ordering of
every coefficient. It covers the base SU(2) family (§12) and the SU(N) family
(§1–§11, the bulk of the document); SO(N)/Sp(2N) is specified in
[`gauge_soN.md`](gauge_soN.md) and is normative on the same terms.

## Status: FROZEN NORMATIVE SPECIFICATION

**This document is the authority. The code is an implementation of it.**

That direction is the whole point, so it is worth stating without hedging.
A gauge that means "whatever the current code outputs" gives a consumer nothing:
every internal refactor — a rewritten loop with a different iteration order, a
swapped-in factorization kernel with a different pivot, a "tidied"
multiplicity-column order — silently redefines the gauge, and a consumer's
checkpointed coefficients quietly stop matching what the crate now produces,
with no signal anywhere. Under this freeze the arrow points the other way:

- The rules below **define** the coefficient values. A build that returns a
  value contradicting a rule here has a **bug** — by definition, not by
  judgement. The right fix is to restore the specified value, not to redefine
  the gauge to match the new output.
- The rules below are **complete for value-fixing purposes**. Every discrete
  choice that can move a returned value (basis order, column order, pivot rule,
  tie-break, sign convention, rank cut, descent order, multiplicity-axis order,
  the value-fixing tolerance tier) is stated here. A choice the implementation
  makes that is *not* stated here is not gauge, and is free to change.
- Several rules are **implicit in the implementation** — fixed by an inherited
  iteration order or by a comparison operator rather than by anything that looks
  like a convention. Those are called out at the point of use with the phrase
  *implicit in the code, normative here*. Implicitness is exactly what makes them
  easy to break by accident, so being written down is the only protection they
  have.
- The **authority fingerprints are the version of this specification**, not a
  label for the current build's output. `sun_authority_fingerprint()`,
  `bcd_authority_fingerprint()`, and `su2_authority_fingerprint()` change only
  when the specification they cite is corrected.

### Operational rule for changing a value

A change to a returned coefficient value, its normalization, or the canonical
convention it is expressed in is not an ordinary change. It is a **specification
correction**, and it requires, in one PR:

1. **The spec edit.** The changed rule edited here (or in `gauge_soN.md`), with
   the defect it corrects stated — what the old rule got wrong, not merely what
   the new rule says.
2. **A fingerprint bump.** The affected family's `epoch=N` tag incremented in
   `su2.rs:su2_authority_fingerprint` / `sun.rs:sun_authority_fingerprint` /
   `bcd.rs:bcd_authority_fingerprint`, and the pinned literal in the matching
   `tests/*_fingerprint.rs` updated in the same PR. Epochs are per-family and
   independent: correcting the SU(N) spec never invalidates SU(2)- or
   B/C/D-derived consumer state.
3. **A CHANGELOG breaking-change entry** naming the spec correction and the new
   epoch, so a consumer reading only the changelog still learns their persisted
   coefficients are stale.
4. **Regenerated golden values** (`tests/gauge_golden.rs`) in the same commit.

Anything that moves a value without those four is a defect. `tests/gauge_golden.rs`
is the tripwire that makes it fail loudly rather than ship: a small committed
table of coefficient values asserted at `1e-12`, running in the default
`cgc-gen` test run with no reference toolchain in the loop.

**What is *not* a specification correction**, and so needs none of the above:
anything that cannot move a returned value. Tightening the `TOL_ORTHO` /
`TOL_LADDER` verification gates (§9), reorganizing code, changing the dense
backend (§10), changing cache behaviour, and editing prose are ordinary changes.

The contract is *value agreement within the oracle tolerance*, not cross-process
bit-identity — see the next paragraph. Two builds under the same specification
version may differ by a few ULPs; that is not a spec deviation.

---

The contract is *value agreement within the oracle tolerance*, not cross-process
bit-identity: the dense backend's parallel reductions are not bit-reproducible,
so two independent generations of the same coupling can differ by a few ULPs
(within a single process the cache serializes all readers to one winner value).

**§1–§11 specify the SU(N) family** — the coefficients produced by
`racah::sun::cgc` and the F/R symbols contracted from them. The construction is
a port of **SUNRepresentations.jl v0.4.0** (`src/`). Every choice below cites the
reference `file:symbol` and its implementing function in this crate. A reader
with this document and the reference source can re-derive the gauge without
reading the Rust implementation.

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
- *Implicit in the code, normative here*: nothing in `patterns` announces itself
  as a convention — the order is whatever the recursive enumeration emits. It
  nonetheless indexes every returned coefficient, so a rewritten enumeration that
  emits the same *set* of patterns in a different order permutes `m1, m2, m3` and
  is a specification deviation, not a refactor.
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

$$ (J^+_l(s_1) \otimes \mathbb{1} + \mathbb{1} \otimes J^+_l(s_2))\, |m_1, m_2\rangle = 0, \quad l = 1 \dots N-1 $$

The raising matrices are the exact GT ladder matrices
(`gtpatterns.jl:creation`, `sun::Irrep::creation`), entries `signedroot(coef)`.

**Column (coupling-pair) order — gauge-relevant.** The columns of the system
are the coupling pairs `(m1, m2)` enumerated in this exact order
(`clebschgordan.jl:highest_weight_CGC`, port `cgc.rs:highest_weight_cgc`): the
**outer** loop is `m1` ascending over `basis(s1)` (the GT basis order of §1);
the **inner** loop is `m2` ascending over the members of the *matching weight
class* `map2[w2]`, `w2 = w3_top − weight(m1) + wshift`, where `map2[w]` lists
the `s2` basis indices of weight `w` **in `basis(s2)` order**
(`clebschgordan.jl:weightmap` preserves basis order). This lexicographic
`(m1, then matching m2)` order is the "first-seen" column order the nullspace
and the gauge consume, so it is part of the gauge. Rows are the distinct raised
targets `(l, m1′, m2′)`, sorted and deduplicated (their order does not affect
the nullspace).

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
   candidate wins. This tie rule is part of the gauge specification. It is,
   however, **value-neutral in `cref`'s output**: reduced column echelon form is
   unique, so a different tie rule cannot change any returned coefficient. No
   coefficient fixture can therefore catch a change to it; the rule is pinned
   instead by a unit test at the selection site
   (`cgc.rs:findabsmax` / `findabsmax_breaks_ties_leftmost`).
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
  **together as a block** by a single `qrpos! ∘ cref!` (§4: `cref!` first, then
  `qrpos!`). Their order on the
  trailing axis `μ` is therefore the column order that block produces — it is
  *not* an independent convention and cannot be chosen per column.
  *Implicit in the code, normative here*: no line assigns `μ` indices; they fall
  out of `cref`'s pivot walk. Sorting, reversing, or otherwise "normalizing" the
  multiplicity columns is a specification deviation.
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

A change that moves any observed value beyond these oracles' tolerances is a
deviation from this specification: a defect unless it ships as a specification
correction under the four-step rule in **Status** above.

The in-repo drift tripwire is `tests/gauge_golden.rs` (§Status). It is not an
oracle — its numbers come from this crate — and it does not replace the two
suites above; it is what fires when a refactor moves a value with no reference
toolchain present.

---

## 12. The base SU(2) family

The base (no-feature) SU(2) path has no free basis to fix: every coefficient is a
closed-form expression in exact big-rational arithmetic, so its "gauge" is the
set of **phase, normalization, and argument-order conventions** the formulas are
written in. They are normative on the same terms as §1–§11, and are the
conventions the `su2_authority_fingerprint` tags name.

- **Evaluation model — `model=bigrational-round-once`.** Every value is computed
  as a big-rational Racah sum carried as `exact.rs:SignedSqrtRational` (a signed
  rational under a square root) with a **single final rounding** to `f64`
  (`su2.rs` module docs). No intermediate rounding: the dimension factors fold
  into the radicand (`times_sqrt_int`) and phases into the sign. Where the
  presented value is `f64` (`su2.rs:su2_f_symbol`) it is a *presentation* of that
  same exact value, never an independent computation. This is value-fixing: a
  differently-associated float evaluation is a different (worse) value.
- **3j — `3j=condon-shortley`.** `su2.rs:wigner_3j` (`wigner_3j_uncached`): the
  Condon–Shortley phase convention, evaluated by the prime-factorized Racah
  single sum (`primefactor.rs`). Non-admissible labels return exact zero, never
  an error and never a panic.
- **6j — `6j=racah-single-sum`.** `su2.rs:wigner_6j` (`wigner_6j_uncached`): the
  Racah single-sum closed form, with the triangle coefficients from
  `su2.rs:delta_sq_pf`.
- **Clebsch–Gordan — `cg=condon-shortley`.** `su2.rs:clebsch_gordan`, composed
  from the 3j as
  $\sqrt{dj_3+1}\,(-1)^{(dj_2-dj_1-dm_3)/2}\, \begin{pmatrix} dj_1 & dj_2 & dj_3 \\ dm_1 & dm_2 & -dm_3\end{pmatrix}$.
  Both the $\sqrt{dj_3+1}$ normalization and the sign are part of the gauge:
  this is its own convention, not inherited from the 3j.
- **F symbol — `f=tks-su2irrep`.** `su2.rs:f_symbol_exact` is the value
  authority:
  $F = (-1)^{j_1+j_2+j_3+j_4}\sqrt{(dj_5+1)(dj_6+1)}\; \{dj_1\,dj_2\,dj_5 / dj_3\,dj_4\,dj_6\}$,
  matching TensorKitSectors `su2irrep.jl:Fsymbol`. **The 6j argument order is
  gauge**, not an implementation detail — a permuted argument order is a
  different F convention with the same 6j.
- **R symbol — `r=tks-su2irrep`.** `su2.rs:su2_r_symbol`: $(-1)^{j_1+j_2-j_3}$ on
  an admissible triangle, exact `0.0` otherwise (TensorKitSectors
  `su2irrep.jl:Rsymbol`). The zero on a non-admissible triple mirrors
  `Nsymbol == 0` and is normative: it is what stops a caller multiplying a
  spurious sign into a forbidden channel.
- **Frobenius–Schur — `fs=tks-su2irrep`.** `su2.rs:su2_frobenius_schur`:
  $(-1)^{dj}$, i.e. `+1` for integer `j` and `-1` for half-integer. Tagged
  separately from `r` because it is a distinct formula, not the R convention.

**Deliberately not gauge** (so changing it needs no epoch bump):

- The **doubled-spin label encoding** `dj = 2j`. It is input addressing / API
  shape; changing it would be an API break that leaves every value unchanged.
- The **canonical Regge keys** (`su2.rs:canonical_regge_3j`,
  `canonical_regge_6j`, `canonicalize3j`). These canonicalize a symbol onto a
  representative of its symmetry class *for caching*, carrying the compensating
  phase so the returned value is identical either way. They are value-neutral by
  construction, and pinned by the symmetry tests in `tests/properties.rs`.

## 13. SO(N)/Sp(2N)

Specified in [`gauge_soN.md`](gauge_soN.md): the defining-representation
generator seeds, the Kronecker convention, the decomposition sweep and its
Gram–Schmidt/QR gauge, the descending-weight sort and its tie-break, the
first-significant-entry sign convention, the outer-multiplicity assignment, the
canonical-parent rule, and the intertwiner alignment. That document is normative
under this same freeze, and `bcd_authority_fingerprint`'s `epoch` is its
specification version.
