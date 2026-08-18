# Theory primer

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

A short review of exactly the representation-theory objects `racah` computes,
written for a physics/math reader who wants to know *what the coefficients mean*
before reaching for the API. It is not a textbook; each section ends with
pointers into the crate API and into [`docs/references.md`](references.md) for
the `file:symbol`-level port provenance.

**Self-containedness.** Every symbol this document uses is defined in it, and
every citation `[n]` is reproduced in the [Bibliography](#bibliography) at the
end (numbering identical to [`docs/references.md`](references.md), which carries
the same list plus the `file:symbol`-level port provenance). What this document
deliberately does *not* restate is the **normative gauge values** — the frozen
basis, phase, ordering and normalization conventions live once, in
[`docs/gauge.md`](gauge.md) and [`docs/gauge_soN.md`](gauge_soN.md), verified by
the golden tests. Those two files are the authority; this one explains what the
objects they pin down *are*. Math is written GitHub-math compatible (`$...$`).

## 0. Symbols

| Symbol | Meaning |
| --- | --- |
| $G$ | a connected compact simple Lie group — here $SU(2)$, $SU(N)$, $Spin(N)$/$SO(N)$, $Sp(2N)$ and their central quotients |
| $\mathfrak{g}$, $r$ | its Lie algebra, and the rank of $\mathfrak{g}$ (= number of Dynkin labels) |
| $a, b, c, \dots$ | irreducible representations ("irreps") of $G$ |
| $d_a$ | the dimension of the irrep $a$ |
| $\bar a$ | the dual (complex-conjugate) irrep of $a$ |
| $\lambda$ | a highest weight; $\lambda_i$ its components in the orthonormal ($\varepsilon$) basis |
| $a_i = \langle \lambda, \alpha_i^\vee \rangle$ | the $i$-th Dynkin label of $\lambda$, in Bourbaki numbering |
| $\alpha$, $\alpha^\vee$, $\rho$ | a root, its coroot, and the half-sum of the positive roots |
| $P$, $Q$, $P^+$ | the weight lattice, the root lattice, and the dominant integral weights |
| $Z(G)$, $\Gamma$ | the center of $G$, and a subgroup of it that is quotiented out |
| $m_a$ | a basis ("magnetic") index inside the irrep $a$, $1 \le m_a \le d_a$ |
| $N^c_{ab}$ | the fusion multiplicity: how many copies of $c$ occur in $a \otimes b$ |
| $\mu, \nu, \kappa, \lambda$ | outer-multiplicity indices, each running $1, \dots, N^{\bullet}_{\bullet\bullet}$ — these are the four F/R block axes, and the name $\lambda$ is reused from the highest weight only here (context disambiguates, and it is what the API calls them) |
| $C^{\,c\,\mu}_{a\,m_a;\,b\,m_b}$ | a Clebsch–Gordan coefficient for the channel $(a, b \to c, \mu)$ |
| $P^c_{ab}$ | the orthogonal projector onto the $c$-isotypic component of $a \otimes b$ |
| $[F^{abc}_d]_{(e,\mu\nu),(f,\kappa\lambda)}$ | the F-symbol (associator) for fusing $a, b, c$ into $d$ |
| $[R^{ab}_c]_{\mu\nu}$ | the R-symbol (braiding) exchanging $a$ and $b$ inside $c$ |
| $\varkappa_a$ | the Frobenius–Schur indicator of $a$ |

## 1. Irreps, labels, dimensions, duals, Frobenius–Schur

Every finite-dimensional representation of a compact group is a direct sum of
irreps, and each irrep is fixed up to isomorphism by its **highest weight** — a
dominant weight of the Lie algebra. `racah` labels an irrep two equivalent ways:

- **Dynkin labels** $a = (a_1, \dots, a_r)$, non-negative integers, one per
  simple root ($r$ = rank). This is the primary constructor input.
- **Highest weight / partition** in an orthonormal ($\varepsilon$) basis — the
  form the internal combinatorics use.

The **dimension** $d_a$ is given in closed form by the Weyl dimension formula,
$d_a = \prod_{\alpha > 0} \frac{\langle \lambda + \rho, \alpha\rangle}{\langle \rho, \alpha\rangle}$,
the product running over positive roots with $\rho$ the half-sum of positive
roots. It is a ratio of integers and is computed exactly.

The **dual** (conjugate) irrep $\bar a$ carries the complex-conjugate
representation; $a$ is **self-dual** when $\bar a = a$. For a self-dual irrep
the invariant bilinear form on the representation space is either symmetric or
antisymmetric, and the **Frobenius–Schur indicator**
$\varkappa_a \in \{+1, -1, 0\}$ records which: $+1$ real/orthogonal, $-1$
pseudoreal/symplectic, $0$ complex (non-self-dual). These are discrete,
combinatorial facts — never numerical results.

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

That repetition is why an extra **outer-multiplicity index**
$\mu = 1, \dots, N^c_{ab}$ appears throughout the API: a single label triple
$(a, b, c)$ does not name a unique coupling channel; the pair $(c, \mu)$ does.
$N^c_{ab}$ itself is pure combinatorics (Littlewood–Richardson for $SU(N)$;
Brauer–Klimyk / Racah–Speiser over Weyl characters for $SO(N)/Sp(2N)$) and is
computed in exact integer arithmetic.

- API: `sun::Irrep` product decomposition; `bcd` `N^c_ab` decomposition; outer
  multiplicity surfaces as the trailing CGC index and the
  $[\mu,\nu,\kappa, \lambda]$ axes of F/R blocks.
- References: Littlewood–Richardson / Brauer–Klimyk background $[7]$, character
  sign rule $[8]$; the port rows in [`docs/references.md`](references.md).

## 3. Clebsch–Gordan coefficients and gauge freedom

The **Clebsch–Gordan coefficients** (CGC) are the entries of the intertwiner
that realizes the decomposition of Section 2 concretely. For a channel
$(a, b \to c, \mu)$ they express each coupled basis vector $|c, m_c; \mu\rangle$
in the product ("magnetic") basis,
$$ |c, m_c; \mu\rangle \;=\; \sum_{m_a, m_b} C^{\,c\,\mu}_{a\,m_a\,;\,b\,m_b}\,
|a, m_a\rangle \otimes |b, m_b\rangle . $$

The coefficients are not free: for every channel the map is an isometry, and
across channels the images are orthogonal. Writing the two relations out (they
are what `racah` enforces as generation gates, Fulton–Harris $[7]$ §1 for the
underlying Schur orthogonality):

$$ \sum_{m_a, m_b} \overline{C^{\,c\,\mu}_{a\,m_a;\,b\,m_b}}\; C^{\,c'\,\mu'}_{a\,m_a;\,b\,m_b} \;=\; \delta_{c c'}\,\delta_{\mu \mu'}\,\delta_{m_c m_{c'}} , $$

$$ \sum_{c, \mu, m_c} C^{\,c\,\mu}_{a\,m_a;\,b\,m_b}\; \overline{C^{\,c\,\mu}_{a\,m_a';\,b\,m_b'}} \;=\; \delta_{m_a m_a'}\,\delta_{m_b m_b'} , \qquad \sum_{c} N^c_{ab}\, d_c \;=\; d_a d_b . $$

CGC are **basis-dependent**: they depend on an arbitrary choice of orthonormal
basis inside each irrep and, when $N^c_{ab} > 1$, on how the $\mu$-copies are
oriented within the isotypic component. This freedom is the **gauge**. Concretely,
for unitaries $U^a$ on each leg and a unitary $V^{c}_{ab}$ mixing the copies, the
transformed coefficients

$$ \tilde C^{\,c\,\mu}_{a\,m_a;\,b\,m_b} \;=\; \sum_{\mu', n_a, n_b, n_c} V^{c}_{ab,\,\mu\mu'}\; U^{a}_{m_a n_a}\, U^{b}_{m_b n_b}\, \overline{U^{c}_{m_c n_c}}\; C^{\,c\,\mu'}_{a\,n_a;\,b\,n_b} $$

are an equally valid CGC set: two such sets describe the same physics.

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

- The **6j symbol**
  $\{ \begin{smallmatrix} a & b & e \\ c & d & f \end{smallmatrix} \}$ (SU(2))
  relates $(a \otimes b) \otimes c$ coupled through $e$ to
  $a \otimes (b \otimes c)$ coupled through $f$. It has a closed-form single-sum
  (Racah) expression $[5]$.
- The **F-symbol** $[F^{abc}_d]_{(e,\mu\nu),(f,\kappa\lambda)}$ generalizes the
  6j to arbitrary $G$: it is the **associator**, the unitary relating the two
  ways of fusing $a, b, c$ into $d$, now carrying the four outer-multiplicity
  indices $\mu, \nu, \kappa, \lambda$ of Section 2.
- The **R-symbol** $[R^{ab}_c]_{\mu\nu}$ is the **braiding** — the phase (matrix,
  with multiplicity) picked up when the order of two fused irreps is exchanged.

`racah` builds $F$ by contracting four CGC over every magnetic index, so that
only the multiplicity axes survive — schematically, with $e$ the intermediate of
the left bracketing and $f$ that of the right,

$$ [F^{abc}_{d}]_{(e,\mu\nu),(f,\kappa\lambda)} \;=\; \sum_{m_a, m_b, m_c, m_e, m_f} \overline{C^{\,d\,\nu}_{e\,m_e;\,c\,m_c}}\; \overline{C^{\,e\,\mu}_{a\,m_a;\,b\,m_b}}\; C^{\,f\,\kappa}_{b\,m_b;\,c\,m_c}\; C^{\,d\,\lambda}_{a\,m_a;\,f\,m_f} . $$

Recoupling data is not free: it must satisfy the categorical consistency laws.
With the multiplicity indices suppressed (the multiplicity-carrying form, and
its normative index placement, is specified in
[`docs/gauge.md`](gauge.md) — this crate follows the TensorKitSectors
conventions cited in [`docs/references.md`](references.md)), the **pentagon
equation** expresses associativity of four-fold fusion and is a condition on $F$
alone — here $e, f, g, h, k, l$ are irrep labels, not multiplicity indices,

$$ [F^{fcd}_{e}]_{gl}\,[F^{abl}_{e}]_{fk} \;=\; \sum_{h} [F^{abc}_{g}]_{fh}\; [F^{ahd}_{e}]_{gk}\; [F^{bcd}_{k}]_{hl} , $$

and the two **hexagon equations** relate braiding to fusion, a condition on $R$
and $F$ together,

$$ R^{ca}_{e}\,[F^{acb}_{d}]_{eg}\,R^{cb}_{g} \;=\; \sum_{f} [F^{cab}_{d}]_{ef}\; R^{cf}_{d}\; [F^{abc}_{d}]_{fg} , $$

with the second obtained by replacing every $R$ with $R^{-1}$. `racah` ships
these as public self-checks and runs them as generation gates: a violation
beyond tolerance is a typed error, never a silently returned coefficient.

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
branching, the highest weights interlacing
$\lambda_1 \ge \mu_1 \ge \lambda_2 \ge \cdots \ge \mu_{k-1} \ge \lambda_k$).
Iterating the chain therefore labels every basis vector of an $SU(N)$ irrep
*uniquely* by the tower of intermediate labels — a **Gelfand–Tsetlin pattern**.
(The $SU$ chain alone is not enough: $SU(k) \supset SU(k-1)$ *does* have
multiplicities — the adjoint $\mathbf{8}$ of $SU(3)$ restricts to $SU(2)$ as
$\mathbf{2} \oplus \mathbf{2} \oplus \mathbf{3} \oplus \mathbf{1}$ — and it is
the intermediate $U(1)$ charge at each step that separates the recurring
copies.) Because the labelling is unique, the ladder (raising/lowering)
operators have **exact closed-form matrix elements** in this basis $[1]$. That
is what makes a direct, exact CGC construction possible, and it is specific to
$SU(N)$: enumerate GT patterns, build the exact rational ladder matrices, solve
the highest-weight nullspace, fix the gauge, and descend by the ladder. See
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

## 7. Global form: which group, not which algebra

Naming the Lie **algebra** does not name the **group**, and the coefficients
above are coefficients *of a group*. $\mathfrak{so}(N)$ is the algebra of both
$Spin(N)$ and $SO(N) = Spin(N)/\mathbb{Z}_2$; $\mathfrak{su}(N)$ is the algebra
of $SU(N)$, of $SU(N)/\mathbb{Z}_k$ for every $k \mid N$, and of
$PSU(N) = SU(N)/\mathbb{Z}_N$. These groups differ in exactly one respect:
**which dominant integral highest weights are genuine representations.**

Let $G_{sc}$ be the simply connected compact group with a given root system and
$Z(G_{sc}) \cong P/Q$ its center. Every connected compact group with that root
system is $G_\Gamma = G_{sc}/\Gamma$ for a subgroup $\Gamma \subseteq Z(G_{sc})$,
and

$$ \mathrm{Irr}(G_\Gamma) \;=\; \{\, \lambda \in P^+ \;:\; \chi_\lambda|_\Gamma = 1 \,\} , $$

where the central character $\chi_\lambda$ depends on $\lambda$ only through its
class $[\lambda] \in P/Q$ (Fulton–Harris $[7]$ §23; congruency classes tabulated
by Slansky $[13]$ §5). For $A_{N-1}$ that class is the **$N$-ality**

$$ t(\lambda) \;\equiv\; \sum_{i=1}^{N-1} i \, a_i \pmod N , $$

so $SU(N)/\mathbb{Z}_k$ admits exactly the $\lambda$ with
$t(\lambda) \equiv 0 \pmod k$ — $PSU(3)$ takes the adjoint $\mathbf{8}$ and
rejects the fundamental $\mathbf{3}$. For $B_r$ the condition is $a_r$ even
(the spinor labels are the ones $SO(2r+1)$ lacks and $Spin(2r+1)$ has), for
$C_r$ it is $a_1 + a_3 + \cdots$ even, and for $D_r$ the
$\mathbb{Z}_2 \times \mathbb{Z}_2$ (or $\mathbb{Z}_4$) center gives $SO(2r)$,
$PSO(2r)$ and the two half-spin forms.

Two consequences make this cheap, and they are the reason `racah` has one
coefficient engine per family rather than one per group:

1. **Admissibility is a per-irrep predicate at construction time.** The class map
   $P^+ \to P/Q$ is a group homomorphism, so $[\lambda + \mu] = [\lambda] + [\mu]$:
   the admissible set is closed under fusion and duality and never needs
   re-checking on a fusion output.
2. **A global form deletes irreps; it never changes a coefficient.** For a
   $\lambda$ admissible in both a cover and a quotient, the CGC, $F$, $R$ and
   $\varkappa$ are *the same numbers in the same basis* — they are computed from
   the representation data, which the choice of $\Gamma$ does not touch. The
   coefficient caches are therefore keyed on irrep labels with no global form in
   the key: two forms sharing an irrep **must** share the entry.

- API: `racah::group` — `RootSystem`, `GlobalForm`, `CenterSubgroup`, `GroupId`
  and the predicate `GroupId::admits`; the form-aware constructors
  `sun::Irrep::from_dynkin_in` and `bcd::Irrep::from_dynkin_in`. The plain
  `from_dynkin` constructors keep their historically published groups.
- Scope: connected compact groups only. $O(N)$ and $Pin(N)$ are not central
  quotients of a simply connected group and are out of scope.
- References: root data and central characters $[7]$; congruency classes and the
  $SO(8)$/$SO(10)$ table anchors $[13]$; the fixed-convention rows in
  [`docs/references.md`](references.md).

## Bibliography

The numbering is identical to [`docs/references.md`](references.md); only the
entries cited above are reproduced here, so this document resolves its own
citations.

1. A. Alex, M. Kalus, A. Huckleberry, J. von Delft, "A numerical algorithm for
   the explicit calculation of SU(N) and SL(N,C) Clebsch–Gordan coefficients,"
   *J. Math. Phys.* **52**, 023507 (2011).
   DOI: [10.1063/1.3521562](https://doi.org/10.1063/1.3521562).
2. A. Weichselbaum, "Non-abelian symmetries in tensor networks: A quantum
   symmetry space approach," *Ann. Phys.* **327**, 2972–3047 (2012).
   DOI: [10.1016/j.aop.2012.07.009](https://doi.org/10.1016/j.aop.2012.07.009).
3. A. Weichselbaum, "QSpace — An open-source tensor library for Abelian and
   non-Abelian symmetries," *SciPost Phys. Codebases* **40** (2024).
   DOI: [10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40).
4. I. M. Gelfand, M. L. Tsetlin, "Finite-dimensional representations of the
   group of unimodular matrices," *Dokl. Akad. Nauk SSSR* **71**, 825–828 (1950).
5. G. Racah, "Theory of Complex Spectra. II," *Phys. Rev.* **62**, 438–462
   (1942). DOI: [10.1103/PhysRev.62.438](https://doi.org/10.1103/PhysRev.62.438).
7. W. Fulton, J. Harris, *Representation Theory: A First Course*, GTM **129**,
   Springer (1991).
   DOI: [10.1007/978-1-4612-0979-9](https://doi.org/10.1007/978-1-4612-0979-9).
8. J. E. Humphreys, *Introduction to Lie Algebras and Representation Theory*,
   GTM **9**, Springer (1972).
   DOI: [10.1007/978-1-4612-6398-2](https://doi.org/10.1007/978-1-4612-6398-2).
12. A. I. Molev, "Gelfand-Tsetlin bases for classical Lie algebras," in
    *Handbook of Algebra* **4**, Elsevier (2006), pp. 109–170.
    DOI: [10.1016/S1570-7954(06)80006-9](https://doi.org/10.1016/S1570-7954(06)80006-9).
13. R. Slansky, "Group theory for unified model building," *Phys. Rep.* **79**,
    1–128 (1981).
    DOI: [10.1016/0370-1573(81)90092-2](https://doi.org/10.1016/0370-1573(81)90092-2).

The F/R contraction wiring and the pentagon/hexagon index conventions follow
TensorKitSectors, cited at `file:symbol` level in
[`docs/references.md`](references.md).
