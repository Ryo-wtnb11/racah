# Theory primer

A short review of exactly the representation-theory objects `racah` computes,
written for a physics/math reader who wants to know *what the coefficients mean*
before reaching for the API. It is not a textbook; each section ends with
pointers into the crate API and into [`docs/references.md`](references.md) for
the source literature (cited below as `[n]`, numbered in that file's
bibliography).

Notation: $G$ is a compact Lie group — here $SU(2)$, $SU(N)$, $SO(N)$, or
$Sp(2N)$. Irreducible representations ("irreps") are written $a, b, c, \dots$;
their dimensions $d_a$. Math is written GitHub-math compatible (`$...$`).

## 1. Irreps, labels, dimensions, duals, Frobenius–Schur

Every finite-dimensional representation of a compact group is a direct sum of
irreps, and each irrep is fixed up to isomorphism by its **highest weight** — a
dominant weight of the Lie algebra. `racah` labels an irrep two equivalent ways:

- **Dynkin labels** $a = (a_1, \dots, a_r)$, non-negative integers, one per
  simple root ($r$ = rank). This is the primary constructor input.
- **Highest weight / partition** in an orthonormal ($\varepsilon$) basis — the
  form the internal combinatorics use.

The **dimension** $d_a$ is given in closed form by the Weyl dimension formula,
$d_a = \prod_{\alpha > 0} \frac{\langle \lambda + \rho, \alpha\rangle}{\langle
\rho, \alpha\rangle}$, the product running over positive roots with $\rho$ the
half-sum of positive roots. It is a ratio of integers and is computed exactly.

The **dual** (conjugate) irrep $\bar a$ carries the complex-conjugate
representation; $a$ is **self-dual** when $\bar a = a$. For a self-dual irrep the
invariant bilinear form on the representation space is either symmetric or
antisymmetric, and the **Frobenius–Schur indicator** $\varkappa_a \in \{+1, -1,
0\}$ records which: $+1$ real/orthogonal, $-1$ pseudoreal/symplectic, $0$
complex (non-self-dual). These are discrete, combinatorial facts — never
numerical results.

- API: `su2_frobenius_schur`; `sun::Irrep::{dim, dual, from_dynkin}`;
  `bcd::Irrep::{dim, dual, frobenius_schur, from_dynkin}`.
- References: Weyl dimension / root data $[7]$; series-specific label maps in
  [`docs/references.md`](references.md).

## 2. Tensor products, fusion multiplicities, outer multiplicity

The tensor product of two irreps decomposes into irreps,
$$ a \otimes b \;\cong\; \bigoplus_c N^c_{ab}\, c , $$
where the **fusion multiplicity** $N^c_{ab}$ is a non-negative integer counting
how many independent copies of $c$ appear. For $SU(2)$ every $N^c_{ab}$ is $0$
or $1$ (the coupling is *multiplicity-free*), but for $SU(N \ge 3)$, $SO(N)$,
and $Sp(2N)$ a given $c$ can occur several times: $N^c_{ab} > 1$.

That repetition is why an extra **outer-multiplicity index** $\mu = 1, \dots,
N^c_{ab}$ appears throughout the API: a single label triple $(a, b, c)$ does not
name a unique coupling channel; the pair $(c, \mu)$ does. $N^c_{ab}$ itself is
pure combinatorics (Littlewood–Richardson for $SU(N)$; Brauer–Klimyk /
Racah–Speiser over Weyl characters for $SO(N)/Sp(2N)$) and is computed in exact
integer arithmetic.

- API: `sun::Irrep` product decomposition; `bcd` `N^c_ab` decomposition;
  outer multiplicity surfaces as the trailing CGC index and the $[\mu,\nu,\kappa,
  \lambda]$ axes of F/R blocks.
- References: Littlewood–Richardson / Brauer–Klimyk background $[7]$, character
  sign rule $[8]$; the port rows in [`docs/references.md`](references.md).

## 3. Clebsch–Gordan coefficients and gauge freedom

The **Clebsch–Gordan coefficients** (CGC) are the entries of the intertwiner
that realizes the decomposition of Section 2 concretely. For a channel $(a, b
\to c, \mu)$ they express each coupled basis vector $|c, m_c; \mu\rangle$ in the
product ("magnetic") basis,
$$ |c, m_c; \mu\rangle \;=\; \sum_{m_a, m_b} C^{\,c\,\mu}_{a\,m_a\,;\,b\,m_b}\,
|a, m_a\rangle \otimes |b, m_b\rangle . $$

CGC are **basis-dependent**: they depend on an arbitrary choice of orthonormal
basis inside each irrep and, when $N^c_{ab} > 1$, on how the $\mu$-copies are
oriented within the isotypic component. This freedom is the **gauge**. Two valid
CGC sets related by a unitary change of basis on each leg (and a unitary mixing
of the $\mu$-copies) describe the same physics.

What is gauge-*invariant* is the orthogonal projector onto the isotypic
component,
$$ P^c_{ab} \;=\; \sum_{\mu} C^{\,c\,\mu}\, C^{\,c\,\mu\,\dagger} , $$
which is independent of the basis choice. `racah` therefore fixes a
**deterministic** gauge (a specified function of the ordered basis and the
nullspace it solves) so that coefficient *values* are reproducible across runs,
builds, and backends, while cross-checks against another convention go through
the gauge-invariant projector (or an explicit gauge-transformation harness).

- API: `clebsch_gordan` (SU(2)); `sun::cgc`; `bcd` CGC via `CanonicalCatalog`.
- Gauge specifications: [`docs/gauge.md`](gauge.md) (SU(N)),
  [`docs/gauge_soN.md`](gauge_soN.md) (SO(N)/Sp(2N)).

## 4. Recoupling: 6j, F-symbols, R-symbols

Coupling three or more irreps can be bracketed in different orders, and the
change of basis between bracketings is the **recoupling** data.

- The **6j symbol** $\{ \begin{smallmatrix} a & b & e \\ c & d & f
  \end{smallmatrix} \}$ (SU(2)) relates $(a \otimes b) \otimes c$ coupled
  through $e$ to $a \otimes (b \otimes c)$ coupled through $f$. It has a
  closed-form single-sum (Racah) expression $[5]$.
- The **F-symbol** $[F^{abc}_d]_{(e,\mu\nu),(f,\kappa\lambda)}$ generalizes the
  6j to arbitrary $G$: it is the **associator**, the unitary relating the two
  ways of fusing $a, b, c$ into $d$, now carrying the four outer-multiplicity
  indices $\mu, \nu, \kappa, \lambda$ of Section 2. `racah` builds it by
  contracting four CGC over all magnetic indices, leaving only the multiplicity
  axes.
- The **R-symbol** $[R^{ab}_c]_{\mu\nu}$ is the **braiding** — the phase (matrix,
  with multiplicity) picked up when the order of two fused irreps is exchanged.

Recoupling data is not free: it must satisfy the categorical consistency laws.
The **pentagon equation** expresses associativity of four-fold fusion (a
condition on $F$ alone), and the **hexagon equations** relate braiding to
fusion (conditions on $R$ and $F$ together). `racah` ships these as public
self-checks and runs them as generation gates: a violation beyond tolerance is a
typed error, never a silently returned coefficient.

- API: `wigner_6j`, `su2_f_symbol`, `su2_r_symbol` (SU(2), closed form);
  `sun::{f_symbol, r_symbol}`, `bcd::{f_symbol, r_symbol}`; the pentagon/hexagon
  and orthogonality/unitarity self-checks are public.
- References: Racah recoupling $[5]$; the contraction wiring and pentagon/hexagon
  provenance rows in [`docs/references.md`](references.md).

## 5. The two constructions, and why each family gets the one it does

For $SU(2)$ the recoupling coefficients have closed forms (Racah), so `racah`
evaluates them directly in exact big-rational arithmetic — nothing is
generated. For the larger families no such closed forms are available, and the
CGC must be *constructed*. `racah` uses two different constructions, and the
choice is forced by the branching structure of each family, not by convenience.

### Gelfand–Tsetlin (GT) — used for $SU(N)$

The unitary subgroup chain
$$ U(N) \supset U(N-1) \supset \cdots \supset U(1) $$
is **multiplicity-free**: at every step an irrep of $U(k)$ restricts to a
*direct sum of distinct* irreps of $U(k-1)$, each appearing at most once (Weyl
branching, the highest weights interlacing $\lambda_1 \ge \mu_1 \ge \lambda_2
\ge \cdots \ge \mu_{k-1} \ge \lambda_k$). Iterating the chain therefore labels
every basis vector of an $SU(N)$ irrep *uniquely* by the tower of intermediate
labels — a **Gelfand–Tsetlin pattern**. (The $SU$ chain alone is not enough:
$SU(k) \supset SU(k-1)$ *does* have multiplicities — the adjoint $\mathbf{8}$ of
$SU(3)$ restricts to $SU(2)$ as $\mathbf{2} \oplus \mathbf{2} \oplus \mathbf{3}
\oplus \mathbf{1}$ — and it is the intermediate $U(1)$ charge at each step that
separates the recurring copies.)
Because the labelling is unique, the ladder (raising/lowering) operators have
**exact closed-form matrix elements** in this basis $[1]$. That is what makes a
direct, exact CGC construction possible, and it is specific to $SU(N)$: enumerate
GT patterns, build the exact rational ladder matrices, solve the highest-weight
nullspace, fix the gauge, and descend by the ladder. See
[`docs/gauge.md`](gauge.md).

### Generator bootstrap — used for $SO(N)$ and $Sp(2N)$

The symplectic reduction chain $Sp(2r) \supset Sp(2r-2)$ is **not**
multiplicity-free: intermediate irreps recur, so there is no GT-type pattern
that labels states uniquely, and hence no practical closed-form ladder matrix
elements. (The orthogonal chains $SO(n) \supset SO(n-1)$ *are* multiplicity-free,
and explicit GT-type matrix elements for them do exist $[4, 12]$, but they are
substantially more involved and no production implementation exists, so `racah`
follows the generator bootstrap for the whole $B/C/D$ set.) So for
the whole $B/C/D$ set `racah` uses a **generator bootstrap** that needs almost
no family-specific structure:

1. **seed** the defining representation of each series explicitly (simple-root
   raising operators + Cartan generators — writable by hand per series);
2. form **tensor products** of already-known irreps;
3. **decompose** numerically by finding highest-weight vectors (a nullspace
   sweep) and orthonormalizing;
4. **harvest** the new irreps' generators and **recurse**.

The price of this generality is that the resulting basis — and therefore the
gauge — is defined *procedurally*: it is whatever the deterministic sweep
produces, not a formula. [`docs/gauge_soN.md`](gauge_soN.md) pins that
procedural determinism down. A reader can now answer "why doesn't `racah` use GT
for $Sp(4)$?" — because the $Sp$ chain is not multiplicity-free, so the GT
labelling and its closed-form ladder elements simply do not exist there.

- API: `sun` (GT construction), `bcd` (generator bootstrap).
- References: SU(N) GT algorithm $[1]$; generator-bootstrap discipline $[2]$,
  $[3]$; the per-family rows and rationale in [`docs/references.md`](references.md).

## 6. The exactness contract, in theory terms

`racah` separates what is *combinatorial* (and therefore exact) from what is
*numeric* (floating point, but verification-gated):

- **Exact / combinatorial**: irrep labels and dominance, dimensions, duals,
  Frobenius–Schur signs, fusion multiplicities $N^c_{ab}$, weight systems, GT
  pattern enumeration and basis ordering, and — for $SU(2)$ — the full 3j / 6j /
  CGC / F / R values, carried as signed square-rooted rationals until a single
  final rounding.
- **Numeric / verification-gated**: for the generated families the CGC (and the
  F/R contracted from them) are computed by a nullspace solve and are floating
  point. Their *values* are finite-precision, but the *gauge* fixing them is a
  deterministic function of the exact basis, and every generation runs
  orthogonality, unitarity, and pentagon/hexagon checks before returning.

So "exact" here is a statement about **structure, gauge determinism, and
verification**, not about symbolic algebraic-number coefficient values.

- API: the self-check functions (CGC orthogonality, F-unitarity,
  R-orthogonality, pentagon, hexagon) are public and double as oracle harnesses.
- References: the exactness-contract discussion in the crate `README` and the
  gauge specifications [`docs/gauge.md`](gauge.md), [`docs/gauge_soN.md`](gauge_soN.md).
