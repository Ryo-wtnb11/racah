# SO(N)/Sp(2N) decomposition-sweep gauge specification

This document specifies the **gauge** of the SO(N)/Sp(2N) Clebsch–Gordan
isometries and projected generators produced by `racah::bcd`'s decomposition
sweep (`src/bcd/sweep.rs`, Layer S3.2) — the deterministic rules that fix the
otherwise-free basis of each coupled multiplet. The gauge is part of this
crate's **semantic-versioning contract**: any change that can alter a returned
coefficient *value* (a different seed rule, Gram–Schmidt order, QR sign
convention, weight-sort tie-break, sign convention, or a tolerance that moves a
rank cut) is a **breaking release**, so consumers may key persisted data on the
crate version.

The construction is a port of **QSpace v4** (Weichselbaum), revision `dd2cc7e`,
`Source/clebsch_aux.cc:getSymmetryStates` (`:53-348`) and `findMaxWeight`
(`:957-1045`), with the product-generator composition from
`Source/clebsch.cc:6649-6656`. Every choice below cites the reference
`file:line @ dd2cc7e`. A reader with this document and the reference source can
re-derive the gauge without reading the Rust implementation.

Coefficient *values* are `f64` (as in QSpace's non-MPFR tier). What is exact and
gauge-fixing is the *procedure*: the seed order, the operator-application order,
the Gram–Schmidt order, the descending-weight sort and its tie-break, and the
sign conventions are discrete facts; only the QR/matmul stages are floating
point, and they are verification-gated (§10).

Two deliberate racah deviations from QSpace, called out once here and again at
their sites:

1. **PositiveDiagonal QR gauge** (§4a): the column orthonormalization uses
   `tenferro-linalg`'s **PositiveDiagonal** QR gauge, a *tightening* of QSpace's
   unspecified `OrthoNormalizeColsQR` sign, fixing the sign of each retained
   orthonormal direction deterministically.
2. **Unconditional block sign convention** (§8): racah applies
   `rangeSignConvention` to every block; QSpace applies it only when the
   weight-sort permutation is nontrivial (the `:304` call sits inside
   `if (!P.isIdentityPerm())`, `clebsch_aux.cc:297-305 @ dd2cc7e`).

Both are value-affecting gauge choices and therefore part of this contract.

---

## 0. Notation

- A B/C/D group is one of `B_r = SO(2r+1)`, `C_r = Sp(2r)`, `D_r = SO(2r)`; `r`
  is the rank. It carries `r` simple-root **raising** operators `Sp[i]`
  (`i = 0…r-1`) and `r` mutually Frobenius-orthogonal integer **Cartan**
  operators `Sz[j]` (`j = 0…r-1`), in QSpace's basis (not Chevalley) — see
  `src/bcd/seeds.rs` and its gauge notes.
- `Sp[i]†` (conjugate transpose; real here, so plain transpose) is the
  corresponding **lowering** operator.
- An irrep `c` is labelled by its integer Dynkin label `a = (a_0,…,a_{r-1})`,
  matching `bcd::Irrep::from_dynkin` (whose label↔partition maps are pinned in
  `src/bcd.rs`).
- The CGC of a coupled multiplet is the isometry `V`, a real `d1·d2 × d3`
  column-major matrix; each column is one coupled state.

---

## 1. Input and the Kronecker convention (gauge)

The sweep takes the **product generators** of `a ⊗ b`, built by the caller from
the two irreps' generator sets via `Generators::product` (QSpace
`clebsch.cc:6649-6656 @ dd2cc7e`):

```
Sp_prod[i] = Sp_a[i] ⊗ 1_b + 1_a ⊗ Sp_b[i]      (i = 0…r-1)
Sz_prod[j] = Sz_a[j] ⊗ 1_b + 1_a ⊗ Sz_b[j]
```

**Kronecker convention — gauge, pinned here.** The product basis index of
`|m_a, m_b⟩` is

```
composite(m_a, m_b) = m_a + d_a · m_b
```

i.e. the **first** factor (`a`) is the *fast* (least-significant) index and the
second (`b`) the slow index. This matches QSpace's
`wbsparray::setRec_kron` (`q[i] = i1[i] + SIZE_a·i2[i]`, `wbsparray.cc:3210 @
dd2cc7e`), which is the **reverse** of the textbook `kron(A,B)` (first factor
slow). A different convention permutes the CGC rows and is a different gauge.

The product `Sz_prod[j]` stays diagonal (a sum of a diagonal ⊗ identity terms);
its diagonal is `Sz_a[j][m_a] + Sz_b[j][m_b]` at the composite index.

Generator-count guards (QSpace `clebsch_aux.cc:85,90 @ dd2cc7e`): `nz ≠ 0`,
`np ≤ nz`, and `np = nz = r` — else `SweepError::InvalidGeneratorCounts`.

---

## 2. Seed selection (per multiplet)

Reference: `clebsch_aux.cc:105-116 @ dd2cc7e`. The sweep tiles the `D = d1·d2`
product space with multiplets `it = 0, 1, …`. The seed index `i0` is
**persistent across multiplets** (it never resets): the seed is the **lowest
product-basis index `i0` not already in the span of the accumulated basis `U`**.
Concretely, walking `i0` upward, `e_{i0}` is skipped iff `‖Uᵀ e_{i0}‖ = 1`
(within `EPS_SWEEP`), i.e. it lies in `span(U)` (`U` has orthonormal columns);
the first `i0` failing that test is the seed. The very first seed (`it = 0`,
`i0 = 0`) is `e_0` unconditionally.

For `it > 0` the seed is orthogonalized against `U` by **two** Gram–Schmidt
passes with renormalization (`clebsch_aux.cc:118-123 @ dd2cc7e`):
`v ← v − U(Uᵀv)`, normalize; repeat once.

The seed must be a simultaneous `Sz`-eigenvector (a definite-weight vector); if
some `Sz[j] v` is not parallel to `v` (and non-negligible), the labels are
ambiguous — `SweepError::SeedNotWeightVector` (QSpace `sameUptoFac` guard,
`clebsch_aux.cc:126-129 @ dd2cc7e`; QSpace's `sameUptoFac` returns 0 when the
vectors *are* parallel, so the abort fires when they are **not**).

---

## 3. Raise to the maximum weight

Reference: `clebsch_aux.cc:134-142 @ dd2cc7e`. Repeatedly apply the raising
operators in **ascending index order** `Sp[0], Sp[1], …`; whenever
`Sp[i] v` is non-negligible (`‖·‖ > EPS_SWEEP`), replace `v` by the normalized
result. Iterate until a full pass raises nothing. The result is the multiplet's
highest-weight (MW) vector.

---

## 4. Sweep down (the descent)

Reference: `clebsch_aux.cc:147-224 @ dd2cc7e`. From the MW vector, generate the
whole multiplet by lowering, level by level (constant weight-height per level).

Notation for one multiplet: `V` = states found so far (this multiplet, all
completed levels, MW first), `Vi` = new states of the **current** level, `U` =
globally accumulated basis of all previous multiplets. The lowering **frontier**
is the previous level (initially the MW).

For each level, loop the lowering operators in **ascending index order**
`Sp[i]†`; for each `i`:

1. `vi ← Sp[i]† · frontier` (lower the whole frontier block at once). Skip this
   operator if the block's per-column RMS `√(‖vi‖²/cols)` is `< EPS_SWEEP`
   (QSpace `sqrt(vi2/SIZE[1]) < eps`, `:154-158`).
2. **Gram–Schmidt pass 1, order self → pass → global** (`:161-196`):
   - project out `Vi` (this level): `vi ← vi − Vi(Viᵀ vi)`;
   - guard: the residual overlap with `V` must be `≤ EPS_SWEEP`
     (`SweepError::OverlapWithVspace`), then project out `V`;
   - guard: the residual max-overlap with `U` must be `≤ EPS_SWEEP`
     (`SweepError::OverlapWithUspace`), then project out `U`.
   The `V`/`U` guards are near-zero for a correct sweep (lowering stays within
   the multiplet, orthogonal to higher levels and to complete earlier
   multiplets); a large overlap signals a defective generator set.
3. Drop columns with norm `< EPS_SWEEP` (QSpace `SkipTinyCols`, `:198`), then
   **QR-orthonormalize** the survivors (`OrthoNormalizeColsQR(FL, CG_EPS1)`,
   `:200`) — see §4a.
4. **Gram–Schmidt pass 2, order global → pass → self** (`:206-217`): project the
   QR result out of `U`, then `V`, then `Vi`, and **QR again**. (Note the order
   is the reverse of pass 1; this is a faithful port of the QSpace ordering, not
   a symmetrization.)
5. Append the result to `Vi`.

After the operator loop, if any new states were found, append `Vi` to `V`, set
the frontier to `Vi`, and continue to the next level; otherwise the multiplet is
complete. The multiplet is then appended to `U`. Bounds `V,Vi,U ≤ D` are guarded
(`SweepError::SpaceOutOfBounds`, `:210,215,235`). The sweep ends when `U` fills
`D`; failing to fill `D` is `SweepError::IncompleteDecomposition` (`:236`).

### 4a. QR gauge — PositiveDiagonal, rank-revealing by R-row

The orthonormalization is `tenferro-linalg`'s QR with
`QrGauge::PositiveDiagonal` (each retained `R` diagonal made ≥ 0 by folding the
sign into `Q`) — the **deliberate racah tightening** of QSpace's unspecified
`OrthoNormalizeColsQR` sign convention.

Rank is read from `R = QᵀA` by **row norm**, not the diagonal: the backend's QR
is un-pivoted, so a zero or dependent *leading* column shifts the pivots off the
diagonal (a rank-2 input can produce an all-zero `R` diagonal). Row `i` of `R`
is (with the guarantees stated below) the row of `R` that certifies `Q[:,i]`
participates in representing `A`; the retained columns are exactly
`{ i : ‖R[i,·]‖ > CG_EPS1 }`, giving an orthonormal basis of the column space.
(`src/bcd/linalg.rs:qr_positive_q`.)

What the row-norm test guarantees (it is *not* the theorem "row `i` nonzero iff
`Q[:,i] ∈ col(A)`", which can fail mid-matrix — a Householder reflector with
`τ = 0` can leave a nonzero `R` row for a `Q` column outside `col(A)`):

- **No genuine direction is lost.** With `R_k` the retained rows and `Q_k` the
  retained columns, `A = Q_k R_k` up to `k·CG_EPS1`, so every column of `A` — and
  thus its whole column space — is reproduced by the retained orthonormal basis.
- **A spurious retained column is impossible for a trailing dependency** (the
  case that actually arises in the descent, where the zero/dependent columns are
  the later lowering images), and if one ever slipped through elsewhere it is
  caught **loudly** downstream: it would inflate a block beyond its irrep
  dimension and break the Cartan-diagonality (§6) or the dimension-bookkeeping /
  `U†U` (§5) gates, never a silent wrong answer. QSpace's `OrthoNormalizeColsQR`
  R-staircase check is therefore not ported — it would be redundant with those
  gates (a few lines that duplicate existing loud coverage; skipped per the
  fewest-moving-parts rule).

**Round-off-neutrality note (Gram–Schmidt order and the second pass).** Within
pass 1, `U`, `V`, and the current level `Vi` are mutually orthonormal subspaces,
so the three `x ← x − Q(Qᵀx)` projections **commute**: their order changes the
result only at the floating-point round-off floor (`~1e-13` here), not the
gauge. Likewise the **second** orthonormalization (pass 2 + its QR) is a
numerical-cleanup no-op once pass 1 has converged — it re-projects vectors that
are already orthogonal, shifting values by `~1e-13`. Both are faithful ports of
QSpace and are kept for numerical robustness, but neither is value-affecting:
they are round-off-neutral gauge choices (the analogue of `docs/gauge.md` §4a's
value-neutral reduced-column-echelon tie rule), so no CGC value oracle can — or
should — distinguish them. The *order* is nonetheless documented above so the
procedure is fully specified.

---

## 5. Global orthogonality gate

Reference: `clebsch_aux.cc:251-257 @ dd2cc7e`. After the sweep, `UᵀU` must equal
the identity within `EPS_VERIFY` (QSpace `isIdentityMatrix(eps2)`); else
`SweepError::NotOrthonormal`.

---

## 6. Generator projection and Cartan snapping

Reference: `clebsch_aux.cc:264-283 @ dd2cc7e`. For each multiplet with isometry
`V`, project the product generators:

```
R.Sp[i] = Vᵀ (Sp_prod[i] V),      R.Sz[j] = Vᵀ (Sz_prod[j] V).
```

Each `R.Sz[j]` must be **diagonal** within `EPS_VERIFY` (QSpace
`isDiagMatrix(eps2)`, `:274`) — else `SweepError::NonDiagonalCartan`. Its
diagonal entries are the states' Cartan eigenvalues; they are **integers** (each
column of `V` is a definite-weight state), and are snapped to the nearest
integer (**FixRational**, integer-target only; QSpace
`FixRational(...,4)`, `:282`). A value farther than `FIXRATIONAL_TOL` from an
integer is `SweepError::NonIntegerWeight`. The `d3 × r` matrix of snapped
eigenvalues is the block's weight table `Z` (row = state, column = Cartan op).

The projected generators are gated by the S3.1 commutator relations, evaluated
in `f64` (the projected `Sp` are generally irrational, so the exact integer
`check_commutators` does not apply directly): `[Sz_j, Sp_i] = d_{i,j} Sp_i` and
`[Sp_i, Sp_i†] ∈ span(Sz)`, worst residual `≤ EPS_SWEEP` — else
`SweepError::CommutatorResidual`.

---

## 7. findMaxWeight: sort, tie-break, Dynkin conversion, uniqueness

Reference: `clebsch_aux.cc:957-1045 @ dd2cc7e`.

**Descending-weight sort (gauge).** The states are permuted into descending
weight order. The comparison is lexicographic on the Cartan columns read in
**reversed** order (column `r-1` first, …, column `0` last), descending (QSpace
`z2.FlipCols(); z2.sortRecs_float(P,-1)`, `:969-970`). **Tie-break:** states with
identical weight rows keep ascending original basis-index order (a stable,
deterministic total order). The max-weight state is the first after sorting.

The whole block — `V` columns, `Z` rows, and each `R.Sp`/`R.Sz` — is reordered by
this permutation (`clebsch_aux.cc:295-301 @ dd2cc7e`).

**Max-weight uniqueness (gate).** The top two sorted rows must differ:
`‖Z[k0,·] − Z[k1,·]‖² > EPS_MW_UNIQUE` — else `SweepError::MaxWeightNotUnique`
(QSpace `recDiff2(0,i) > 1e-8`, `:1035-1039`).

**Dynkin conversion (per series).** The max-weight state's Cartan eigenvalues
`qm` (in QSpace's `Sz` basis) map to the Dynkin label as follows
(`clebsch_aux.cc:977-1031 @ dd2cc7e`; each division must yield an integer within
`FIXRATIONAL_TOL`, else `SweepError::NonIntegerWeight`):

- **`C_r` (SpN, `:990-996`):** `a_i = (qm_i − qm_{i-1})/(i+1)` for
  `i = r-1…1`, and `a_0 = qm_0`.
- **`B_r` (SON, `:1001-1013`):** with `x = 2·qm_0`, set `a_{i-1} = qm_i − qm_{i-1}`
  for `i = 1…r-1`, then `a_{r-1} = x`, then reverse-swap `a_i ↔ a_{r-2-i}` for
  `i = 0…⌊(r-1)/2⌋-1`.
- **`D_r` (SEN, `:1018-1031`):** as `B_r` but with `x = qm_0 + qm_1`.

The resulting Dynkin label constructs `bcd::Irrep::from_dynkin`; an invalid label
is `SweepError::InvalidDiscoveredLabel` (unreachable for a faithful sweep).

QSpace's `r > 9` upper-rank guard (`:983,993,1004,1021`) is a fixed-buffer build
artifact, **not** a mathematical constraint — **N/A** here (same disposition as
`src/bcd/seeds.rs`). The low-rank `r < 2` (`r ≤ 2` for `D`) redirect is inherited
from the seed layer (`BcdError::ExcludedRank`), so the sweep never receives such
generators.

---

## 8. Sign convention

Reference: `signFirstVal`/`rangeSignConvention` (`clebsch_aux.cc:26-51 @
dd2cc7e`). The **whole block's** sign is fixed so that the **first significant**
CGC entry (first `|·| > CG_EPS1` scanning the flattened, sorted `V` in storage
order) is **positive**; if it is negative, negate the whole block. This is
`rangeSignConvention` on `V.D`. The CGC entries are then integer-snapped where
they land on integers (FixRational, `:307`); genuinely irrational entries are
left as-is.

**Deliberate racah deviation #2 (unconditional block sign).** QSpace applies
`rangeSignConvention` only when the descending-weight sort actually permuted the
block: the `:304` call sits inside `if (!P.isIdentityPerm())`
(`clebsch_aux.cc:297-305 @ dd2cc7e`), so a block already in descending-weight
order (identity permutation) keeps whatever whole-block sign the QR/descent
happened to produce. racah instead applies the sign convention to **every**
block, unconditionally. Rationale: a uniform, permutation-independent gauge is
more predictable, and the QSpace conditionality appears accidental (the sign of
a coupled block should not depend on whether its states happened to be generated
already sorted). Consequence for the S3.5 QSpace-CGC-fixture harness: on
identity-permutation blocks the two implementations may differ by a whole-block
sign; those are expected and are absorbed by the harness's signed-permutation
alignment (§13 of the S3.5 plan), not treated as defects.

---

## 9. Outer-multiplicity assignment

Reference: `clebsch_aux.cc:331-345 @ dd2cc7e`. Blocks that share the same highest
weight (same discovered irrep) receive `(index, size)`: `index = 0, 1, …` in
**discovery (sweep) order**, and `size` = the number of such blocks. A
multiplicity-free irrep gets `(0, 1)`.

---

## 10. Exact-multiplicity production gate (Ruling 1)

For every discovered irrep `c`, the sweep multiplicity `M^c_sweep` (the number of
blocks labelled `c`) must equal the exact fusion multiplicity `N^c_ab` from
`bcd::directproduct` (S3.0), **and** the discovered support must equal the exact
support — **both directions**. A missing block (`found = 0 < N`) is as fatal as
a spurious one (`found > 0 = N`). Any mismatch is
`SweepError::MultiplicityMismatch`. This gate is on the production path (Ruling
1); it is not optional or test-only. `decompose(product, expected)` takes the
exact decomposition `expected` and enforces it; `decompose_defining_product`
computes `expected` from the defining label via `directproduct`.

This is racah's addition over QSpace, which has no exact Layer-1 gate (it only
warns on discovered outer multiplicity, `:341`).

---

## 11. Tolerance tier (QSpace CG_EPS ladder)

Named constants (`src/bcd/sweep.rs`), with provenance:

| Constant | Value | Source (`@ dd2cc7e`) | Role |
|---|---|---|---|
| `EPS_SWEEP` | `1e-8` | `getSymmetryStates` `eps` (`clebsch_aux.cc:76`) | significant-vector / overlap threshold in the sweep and commutator gate |
| `EPS_VERIFY` | `1e-10` | `getSymmetryStates` `eps2` (`clebsch_aux.cc:76`) | `UᵀU = I` and Cartan-diagonality checks (`isIdentityMatrix`/`isDiagMatrix(eps2)`) |
| `CG_EPS1` | `1e-10` | `clebsch.hh:244` (non-MPFR tier) | QR orthonormalization / rank-reveal (`OrthoNormalizeColsQR`); sign-scan threshold |
| `EPS_MW_UNIQUE` | `1e-8` | `clebsch_aux.cc:1035` (`recDiff2 > 1e-8`) | max-weight uniqueness |
| `FIXRATIONAL_TOL` | `1e-6` | see below | integer-snap tolerance for Cartan eigenvalues and Dynkin conversions |

The QSpace non-MPFR ladder is `CG_EPS1 = 1e-10`, `CG_EPS2 = 1e-12`,
`CG_SKIP_DEPS1 = 1e-12`, `CG_SKIP_DEPS2 = 1e-14` (`clebsch.hh:224-246`). QSpace's
`FixRational` snaps within `CG_SKIP_DEPS1 = 1e-12` at its working precision;
here, in plain `f64` with a round-off floor `~1e-13`, `FIXRATIONAL_TOL = 1e-6` is
sized well above the floor and far below the minimum integer gap (1) of a Cartan
eigenvalue, so it snaps every genuine integer while a real non-integer (a defect)
stays outside it. It is **integer-target only** — applied to Cartan eigenvalues
and the (integer) Dynkin-conversion quotients, never to general CGC entries
(which are irrational). Tightening any tolerance that cannot move a returned
value is *not* a breaking release; loosening one that can, is.

**Reference tolerance-regime deviations.** Three racah tolerance choices differ
in *regime* (not just value) from QSpace; none moves a discovered label or
multiplicity, but each is recorded for the S3.5 fixture harness:

1. **Descent overlap guards are stricter/absolute.** QSpace's `U`-overlap abort
   fires at `x·x > eps` with `x = aMax(Uᵀvi)`, i.e. a **max-overlap `> 1e-4`**;
   its `V`-overlap abort is **norm-relative** (`√(‖Vᵀvi‖²/‖vi‖²) > eps`,
   `clebsch_aux.cc:176-194 @ dd2cc7e`). racah uses a single **absolute**
   threshold `EPS_SWEEP = 1e-8` on the max-overlap for both guards — strictly
   tighter. For a faithful sweep both residuals are `~0`, so the choice only
   affects *how loudly a defective generator set fails*, never a valid result.
2. **"Skip if already inside `Vi`" is realized differently.** QSpace early-skips
   a lowering operator whose image already lies in the current level `Vi`
   (`|1 − √(‖Viᵀvi‖²/‖vi‖²)| < eps`, `clebsch_aux.cc:165-169`). racah has no
   early test; it projects out `Vi` unconditionally and lets `skip_tiny_cols`
   drop the now-zero columns — **equivalent outcome, different mechanism** (a
   fully-redundant block becomes zero and is dropped either way).
3. **FixRational is integer-only vs denominator-bounded rational.** QSpace's
   `FixRational` is continued-fraction **rational** snapping (mode `'r'`,
   denominator `≤ 1024`, tol `~1e-12`; `clebsch_aux.cc:410-425`), applied to the
   CGC entries too. racah snaps **integers only** (§6), leaving irrational CGC
   entries untouched. Consequence: an exact half-integer (or other small-rational)
   CGC entry that QSpace would snap may differ from racah's value by `≤ 1e-12` —
   below every gate tolerance, absorbed by the S3.5 element-wise comparison.

---

## 12. Numerical seams (backend)

`CoeffScalar = f64`. Two operations reach `tenferro-linalg`
(`src/bcd/linalg.rs`, no hand-rolled factorization kernels):

| Stage | Reference (`@ dd2cc7e`) | tenferro API |
|---|---|---|
| column orthonormalization (§4a) | `OrthoNormalizeColsQR` | `qr_with_options(QrGauge::PositiveDiagonal)` → `Q`, rank-revealed by `R`-row |
| block CGC contractions & `UᵀU` (§5,6) | `Wb::MatProd` | `TracedTensor::matmul` |

The Gram–Schmidt sweep arithmetic (vector scaling, subtraction, norms, and the
sparse `Sp`/diagonal-`Sz` applications) is the **gauge algorithm itself** — the
analogue of `sun::cgc`'s `cref`, which `docs/gauge.md` §10 likewise carries in
plain code rather than routing through a factorization kernel. The build-time
`tenferro-rs` revision is recorded in the PR body.

The contract is *value agreement within the verification tolerances*, and (on a
single-threaded run) **bitwise reproducibility** across runs — the sweep is
deterministic end to end, pinned by the determinism test. Cross-process bit
identity is not promised (the backend's reductions are not bit-reproducible).

---

## 13. Verification (independent oracles)

- **Exact decomposition** (`bcd::sweep::tests`): for `defining ⊗ defining` across
  `B2/B3/B4/C2/C3/C4/D3/D4`, the discovered labels and multiplicities equal
  S3.0's exact `directproduct` (itself cross-checked against Sage/OSCAR
  fixtures — code racah did not write); block dimensions sum to `d1·d2`; each
  CGC is an isometry within tier; and each block's Cartan-eigenvalue rows
  reproduce the irrep's exact weight system (distinct-weight count and total from
  Freudenthal + Weyl orbits, S3.0).
- **Outer multiplicity ≥ 2**: `D3 (0,1,1) ⊗ (0,1,1)` — the exact layer predicts
  `N = 2` and the sweep reproduces two `(0,1,1)` blocks with OM indices `(0,2)`
  and `(1,2)`.
- **Determinism**: two runs produce bitwise-identical CGC bytes.
- **Sign convention**: each block's first significant CGC entry is positive.
- **Gates**: the multiplicity gate is exercised in both directions (spurious and
  missing expectation); ill-posed generator products are typed errors.

A change that moves any observed value beyond these oracles' tolerances is, by
definition of this document, a breaking release.

---

## 14. The canonical-parent rule (S3.3 `CanonicalCatalog`)

Sections 1–13 fix the gauge of **one** decomposition of **one** product `a ⊗ b`.
But an irrep `c` appears in *many* products, and its projected generators
`R.Sp`/`R.Sz` (§6) — which fix `c`'s basis — are read off from whichever product
was decomposed. If the catalog stored `c`'s generators from "whichever product
was decomposed first", the gauge would depend on **query order**. This section
specifies the rule that removes that dependence: each non-base irrep's generators
come from **one** canonical parent product, chosen by a deterministic total order
over the exact S3.0 data. This rule IS gauge — it is part of the semantic-version
contract in the same way §§1–13 are.

Design authority: issue #18 Ruling 2, issue #25. Implementation:
`src/bcd/catalog.rs`. A reader with this section and the S3.0 `directproduct`
(§0, `src/bcd.rs`) can re-derive every catalog gauge choice without reading the
Rust.

### 14.1 The well-order `≺` on irreps

Define the total order `≺` on the tensor irreps of a fixed `(series, rank)`:

```
c₁ ≺ c₂   ⟺   ( dim(c₁), dynkin(c₁) )  <lex  ( dim(c₂), dynkin(c₂) )
```

i.e. compare exact Weyl dimensions first (a positive integer, §0), then break
ties by the integer Dynkin label read left to right. Both components are exact
S3.0 data; no float, no discovery order.

**`≺` is a well-order** (no infinite strictly-descending chain). Proof: along any
descending chain the dimensions are non-increasing positive integers, so they are
eventually constant at some `D`. At a fixed dimension `D` there are only
**finitely many** irreps (a dominant weight of bounded dimension has bounded
entries — `dim` is strictly monotone in each partition part, §14.4), and the
Dynkin-lexicographic order is a strict total order on that finite set, which has
no infinite descending chain. ∎

Well-foundedness is what makes the on-demand recursion (§14.3) terminate.

### 14.2 The canonical parent

The two **base cases** are the `≺`-minimal irreps and carry generators directly,
not from any product:

- the **trivial** rep (dim 1): all generators zero on a 1-dimensional carrier;
- the **defining** rep `(1,0,…,0)`: the exact S3.1 seed (`src/bcd/seeds.rs`).

They are seeded into the catalog at construction. Every other (`non-base`) irrep
`c` has a **canonical parent pair** `(a, b)`, defined as the minimum, under the
**pair order**

```
key(a,b) = ( dim(a) + dim(b),  dim(a),  dynkin(a),  dynkin(b) )   (lex)
```

over all pairs satisfying

1. `a ≺ c` and `b ≺ c`  (both factors strictly smaller in `≺`), and
2. `N^c_{ab} ≥ 1`, i.e. `c` appears in the exact decomposition `a ⊗ b` (S3.0
   `directproduct`).

The pair order's `dim(a)` tie-break (second component) makes the winner unique
and puts it in canonical `a ⪯ b` form: of the two orderings `(a,b)`/`(b,a)` — which
share the first component `dim(a)+dim(b)` — the one with the smaller-dimensional
factor first wins. `c`'s generators are then the projected generators (§6) of the
`c`-block produced by decomposing `a ⊗ b` (the outer-multiplicity-0 copy when
`N^c_{ab} > 1`; the copies share the same intrinsic generators, §14.5).

**Deliberate refinement of the issue-#25 sketch.** Issue #25 sketches minimizing
`(dim_a + dim_b, dim_a, label_a, label_b)` over admissible pairs. That is exactly
`key(a,b)` above; the refinement this document commits to is the **admissibility
condition (1)**, `a ≺ c ∧ b ≺ c`, as the precise, computable-from-S3.0 statement
of "already catalog-reachable", together with the proof (§14.4) that it is both
non-empty and well-founded. The alternative anchoring "always take `a` = defining"
was rejected: it forces `depth ~ |λ|` (one box per level, §14.6), maximizing the
chain-depth error accumulation that issue #18 flags as a watch item, whereas the
`dim_a+dim_b` minimizer favors **balanced** splits and shallow chains (measured:
C2 `Sym⁶`, dim 84, has depth 3, not 5).

### 14.3 On-demand recursion (materialization)

`generators(c)` (and the factor bases inside `cgc(a,b,c)`) materialize `c`'s
canonical-parent chain:

```
materialize(c):
    if c already has generators: return
    if c is a base case: it is already seeded; return
    (a, b) = canonical_parent(c)          # §14.2, pure over S3.0
    materialize(a); materialize(b)         # a ≺ c, b ≺ c
    decompose (a ⊗ b)                      # the §1–§13 sweep
    harvest the resulting blocks           # §14.5
```

Because each recursive call is on a **strictly `≺`-smaller** irrep and `≺` is a
well-order, the recursion terminates — it cannot descend forever, and it bottoms
out at the base cases (the `≺`-minimal irreps). This replaces QSpace's fixed-pass
`dmax` enumeration (the bootstrap loop around `getSymmetryStates`,
`clebsch.cc:6649-6773 @ dd2cc7e`): racah does **not** enumerate a fixed sequence
of product passes up to a dimension cap; it materializes exactly the chain a query
needs, in an order fixed by `≺` rather than by a pass counter — so the gauge is
query-order-independent **structurally**, not procedurally.

### 14.4 Existence (every non-base irrep has a canonical parent)

The candidate set (pairs meeting conditions 1–2 of §14.2) is **non-empty** for
every non-base `c`, so the minimum exists. Argument: every tensor irrep of
`SO(N)`/`Sp(2N)` lives in a tensor power of the defining (vector) rep `V`
(this is precisely why the object is the tensor irreps and spinors are excluded,
§0 / Ruling 3). For non-trivial `c`, the product `V ⊗ c` contains a component `b`
obtained by **removing one box** from `c`'s highest weight (lowering one part of
the partition), which has strictly smaller dimension — `dim` is strictly monotone
in each partition part — hence `b ≺ c`. The defining rep is the unique smallest
**non-trivial** tensor irrep (the next dimension after the trivial rep's 1 is
`dim(V) = 2r` or `2r+1`; no tensor irrep lies strictly between), so for any
non-base `c` we have `V ≺ c` as well. By Frobenius reciprocity
`N^c_{V,b} = N^b_{V*,c} = N^b_{V,c} ≥ 1` (`V` is self-dual), so `c ∈ V ⊗ b`.
Thus `(V, b)` satisfies conditions 1–2, and the candidate set is non-empty. The
actual canonical parent is the `key`-minimum over this non-empty finite set, which
may be a more balanced pair than `(V, b)`. ∎

The search is finite and computable from S3.0 alone: enumerate the finite set
`{ x : x ≺ c }` (all irreps with `dim ≤ dim(c)`, obtained by a depth-first walk
over integer partitions with `dim`-monotone pruning — §14.1's finiteness), and for
each candidate `a`, read the admissible `b` directly from `directproduct(a*, c)`
(the reciprocity above). Pruning: iterating `a` in ascending `≺`, stop once
`2·dim(a)` exceeds the best sum found — safe because the `key`-minimum pair `{x,y}`
with `dim(x) ≤ dim(y)` is always reached via its smaller factor `x`
(`2·dim(x) ≤ dim(x)+dim(y) ≤` best sum).

### 14.5 Harvest discipline (append-only, canonical-gated)

Decomposing `a ⊗ b` yields blocks for several irreps at once. The catalog appends
a block's generators **iff**:

- the block's irrep `c'` has **no** generators yet (append-only; never overwrite,
  never evict — Ruling 2), **and**
- `(a, b)` is exactly `c'`'s canonical parent (§14.2), taking the
  outer-multiplicity-0 copy.

Blocks whose canonical parent is a *different* product are **not** written here;
they are materialized later from their own canonical parent (or are base cases).
This is what makes the stored gauge independent of which product happened to be
decomposed — a second product that rediscovers `c'` finds it already present and
**never** writes.

**Deviation-by-design from QSpace's cross-copy check
(`clebsch.cc:6710-6718 @ dd2cc7e`).** QSpace, lacking a canonical-parent rule,
reaches the same coupled irrep `J` from multiple products and reconciles the
copies at runtime: it compares the freshly projected generators `G` against the
stored `G0` with `normDiff(G,G0,1e-10)` and aborts on disagreement, and it may
**replace** the stored copy when the new one has a smaller commutator error
(`e2 ≤ 0.5·e0 ⇒ saveR=2`, the "replacing RStore" branch). racah's canonical-parent
rule makes a second *written* copy **structurally impossible**, so there is
nothing to reconcile and nothing to replace — the replacement branch has no
analogue. Where a non-canonical product *rediscovers* an already-stored `c'`, racah
keeps only a **debug-assert-class** check: the rediscovered block's Cartan (weight)
spectrum must match the stored generators' (cheap, loud, `debug_assert`; the
weights are gauge-independent so this is a fast structural sanity check, not the
full `normDiff`). QSpace's error-driven replacement is deliberately dropped: the
canonical parent — not "whichever copy is numerically cleaner this run" — fixes the
gauge, so the stored copy must never depend on runtime residuals.

### 14.6 Atomicity of the byte budget (Ruling 2)

The catalog is byte-bounded. A materialization that would exceed the budget fails
with a typed error **before any commit**, leaving no partial chain. The mechanism
is **compute-fully-then-commit**: the whole chain's new generator sets are
assembled in a staging buffer (reading committed entries and the staging buffer,
appending only what is missing), the staging buffer's total byte charge is added
to the current retained bytes, and only if the sum is within budget are the staged
entries committed; otherwise the staging buffer is discarded untouched. Because
the chain is a deterministic function of `c` (§14.2–14.5), the failed request is
reproducible and the committed state is exactly what it was before the call. This
was chosen over incremental commit-as-you-go (which could leave a half-registered
chain on the entry that overflows) because atomicity is a Ruling-2 requirement.

### 14.7 What the catalog does NOT own

CGC/F/R **values** are not stored in the catalog (Ruling 2): `cgc(a,b,c)` returns
the isometry to the caller and does not cache it here. The catalog owns only
generator **sets** and a byte counter. The generators are `f64` (§12); a chain of
depth `d` accumulates round-off across `d` sweeps, but each sweep re-gates the
commutator relations at `≤ EPS_SWEEP` (§6), so a stored set's residual is bounded.
Measured accumulation stays deep below the gate (C2 symmetric powers: depth-3 sets
have worst commutator residual `~10⁻¹⁴`), which is the issue-#18 chain-depth watch
item's evidence.
