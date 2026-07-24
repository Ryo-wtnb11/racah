# Porting provenance and references

`racah` is a faithful port: nearly every coefficient it computes traces to a
named production implementation or a standard result. This page collects that
provenance in one place — what was taken, from which reference at which version,
*why that machinery is the right (or only) one for the family*, and where
`racah` deviates. The module and item rustdoc keep the load-bearing contracts
next to the code; this page is the map from `racah` symbol to source.

Citations of the form `file:symbol` name a symbol in the cited project's source
tree (path relative to that project's root), not a local checkout path. Numbered
citations `[n]` refer to the [bibliography](#bibliography) at the end.

## SU(2) — base, exact closed form

**Why this construction.** For $SU(2)$ the 3j/6j/CGC/F/R all have closed-form
expressions (Racah single sums, $[5]$), so nothing is generated: `racah`
evaluates the closed forms in exact big-rational arithmetic and rounds once. The
factorial-heavy sums are carried in prime-factorized form to avoid big-integer
blow-up.

| Implementation area (`file::symbol`) | Reference (project, version, `file:symbol`) | What was taken | Why this reference / algorithm | Deviations |
|---|---|---|---|---|
| `su2.rs::wigner_6j`, `racah_6j` engine | WignerSymbols.jl v2.0.0, `src/WignerSymbols.jl:_wigner6j`, `compute6jseries` $[10]$ | Racah single-sum 6j evaluation | Closed-form 6j exists for SU(2); exact rational sum, single final rounding | Keyed by canonical Regge class with a bounded FIFO publication cache |
| `su2.rs::wigner_3j` | WignerSymbols.jl v2.0.0, `src/WignerSymbols.jl:_wigner3j` $[10]$ | Racah 3j sum | Same closed-form rationale | Keys on the canonical Regge labels directly, not the `(β,α)` reparametrization the reference uses (`su2.rs`, `canonical_regge_3j` doc) |
| `su2.rs::canonical_regge_3j`, `ReggePhase` | WignerSymbols.jl v2.0.0, `reorder3j` $[10]$ | Regge canonicalization + net compensation phase | Regge symmetry collapses the 3j orbit to one representative | `racah` completes the canonicalization, breaking every tie deterministically |
| `su2.rs::su2_f_symbol`, `su2_r_symbol` | TensorKitSectors `src/irreps/su2irrep.jl:Fsymbol`/`Rsymbol`; WignerSymbols.jl v2.0.0 `racahW` $[10]$ | F from `sqrtdim·racahW`; R sign rule | SU(2) F/R reduce to the closed-form 6j | — |
| `exact.rs::SignedSqrtRational` | WignerSymbols.jl (signed-√-rational form of the Racah expressions) $[10]$ | Exact value carried as signed square-rooted rational | Keeps triangle/dimension products exact under the radical until the final `f64` | — |
| `primefactor.rs` (whole module) | WignerSymbols.jl v2.0.0, `src/primefactorization.jl` $[10]$ | Prime-factorized factorial arithmetic (`mul!`, `divexact!`, `splitsquare`, `primefactorial`, `commondenominator!`, `sumlist!`) | Factorial-heavy Racah sums blow up as big integers; prime factorization keeps them small — a measured-need upgrade | Same growing-global-table design |
| `cache.rs` (whole module) | WignerSymbols.jl v2.0.0, transparent LRU inside `wigner3j`/`wigner6j` $[10]$ | Per-kind cache keyed by canonical Regge labels | A Regge class names exactly one exact value, so caching is pure memoization | **FIFO**, not LRU, and bounded by a byte policy (`cache.rs::FifoCache` doc) |

## SU(N) — `cgc-gen`, Gelfand–Tsetlin construction

**Why this construction.** The chain $SU(N) \supset SU(N-1) \supset \cdots
\supset SU(1)$ is multiplicity-free, so states are labelled uniquely by
Gelfand–Tsetlin patterns and the ladder operators have exact closed-form matrix
elements $[1]$. That closed form is what makes a direct, exact CGC construction
possible, and it is $SU(N)$-specific (see [`docs/theory.md`](theory.md) §5).

| Implementation area (`file::symbol`) | Reference (project, version, `file:symbol`) | What was taken | Why this reference / algorithm | Deviations |
|---|---|---|---|---|
| `sun.rs::Irrep` (`dim`, `dual`, `patterns`, `creation`) | SUNRepresentations.jl v0.4.0, `src/sunirrep.jl`, `src/gtpatterns.jl` (`GTPatternIterator`, `creation`), `src/sector.jl` $[9]$, algorithm $[1]$ | GT pattern enumeration, Weyl dimension, dual, exact ladder (creation) matrices | GT chain is multiplicity-free → unique labels + closed-form ladder elements | `Irrep::patterns` basis order reproduces `gtpatterns.jl:GTPatternIterator` **index-for-index** and is pinned by checked-in fixtures — load-bearing, because the Layer-2 gauge depends on it |
| `sun.rs` product (`directproduct`) | SUNRepresentations.jl v0.4.0, `src/gtpatterns.jl:directproduct` $[9]$ | Littlewood–Richardson fusion multiplicities $N^c_{ab}$ | Exact integer combinatorics | — |
| `sun/cgc.rs::cgc` (`highest_weight_CGC`, `lower_weight_CGC!`, `cref!`, `purge!`, `weightmap`, `weight`) | SUNRepresentations.jl v0.4.0, `src/clebschgordan.jl:_CGC` and helpers $[9]$ | Highest-weight nullspace → gauge canonicalization → weight-ladder descent | The exact ladder elements let the CGC be solved subspace by subspace | Tolerances `TOL_NULLSPACE = 1e-13`, `TOL_GAUGE = 1e-11`, `TOL_PURGE = 1e-14` ported verbatim; gauge specified in [`docs/gauge.md`](gauge.md) |
| `sun/linalg.rs` | SUNRepresentations.jl v0.4.0, `src/clebschgordan.jl:_nullspace!` (`svd!(A; full=true)`), `qrpos!`, `lower_weight_CGC!` least squares $[9]$ | Nullspace, positive-diagonal QR, least squares | The three dense factorizations the CGC solve needs | Routed through the selectable dense backend; no hand-rolled kernels |
| `sun/fr.rs::f_symbol`, `r_symbol` | SUNRepresentations.jl v0.4.0, `src/sector.jl:_Fsymbol` (`:58-89`), `_Rsymbol` (`:91-110`); contraction wiring/axis order TensorKitSectors `src/sectors.jl:Fsymbol_from_fusiontensor` $[9]$ | F = four-CGC contraction over magnetic indices; R = braiding matrix | Standard categorical definition of F/R from CGC | No Regge-style canonicalization — no tetrahedral/Regge symmetry is implemented for the GT-basis SU(N) F blocks (`sun/fr.rs`, `cache.rs::cache_f` doc) |

## SO(N) / Sp(2N) — `cgc-gen`, generator bootstrap

**Why this construction.** The symplectic chain $Sp(2r) \supset Sp(2r-2)$ has
branching multiplicities, so no GT-type basis with practical closed-form matrix
elements exists; the orthogonal chains $SO(n) \supset SO(n-1)$ are
multiplicity-free but their closed forms are not production-viable either. So the
whole $B/C/D$ set is built by a generator bootstrap — defining-rep seeds, tensor
products, numeric highest-weight decomposition, harvest, recurse — which needs
almost no family-specific structure. Its price, a procedurally-defined gauge, is
pinned by [`docs/gauge_soN.md`](gauge_soN.md). See [`docs/theory.md`](theory.md) §5.

| Implementation area (`file::symbol`) | Reference (project, version, `file:symbol`) | What was taken | Why this reference / algorithm | Deviations |
|---|---|---|---|---|
| `bcd.rs::Irrep` (`dim`, `dual`, `frobenius_schur`, weight multiplicities, `N^c_ab`) | Fulton–Harris §§18, 24 $[7]$; Humphreys §13.4 (Freudenthal), §24 (character sign rule) $[8]$; QSpace v4 (rev `dd2cc7e`) `Source/clebsch_aux.cc` `wdim_B/C/D` (`:458/486/524`), `findMaxWeight` (`:957–1045`) $[3]$ | Weyl dimension, Freudenthal recursion, Brauer–Klimyk / Racah–Speiser product decomposition, ε↔Dynkin label maps | Textbook root-system data for exact combinatorics; QSpace as the numerical dimension oracle it reproduces | Low ranks `B_1/C_1/D_2` rejected and redirected (guard inventory, issue #15) |
| `bcd/seeds.rs` (`Seed`, `Setup_*`, `check_commutators`) | QSpace v4 (rev `dd2cc7e`), `Source/clebsch.cc:Setup_SpN` (`:7145-7244`), `Setup_SON` (`:7246-7348`), `Setup_SEN` (`:7350-7457`); `initCommRel`/`checkCommRel` (`:5971-5987`) $[3]$ | Per-family defining-rep seed matrices (`Sp` raising ops, `Sz` Cartan diagonals), commutator self-check | Seeds are the only family-specific input the bootstrap needs; explicitly writable per series | Kept in QSpace's (non-Chevalley) integer basis; the `Sz[i]` are integer Frobenius-projected generators, not Chevalley coroots (module docs) |
| `bcd/sweep.rs` (`getSymmetryStates` port, `BcdError`) | QSpace v4 (rev `dd2cc7e`), `Source/clebsch_aux.cc:getSymmetryStates` and error sites (`:76`, `:186/194/214/236/251/274/1036`) $[3]$ | Raising-op seed → Gram–Schmidt sweep → column QR decomposition loop; the error taxonomy | Family-generic numeric decomposition — the machinery that replaces GT for $B/C/D$ | `CG_EPS` tolerance tiers ported; each `BcdError` maps a QSpace `wblog(...ERR...)` abort |
| `bcd/linalg.rs` | QSpace v4 (rev `dd2cc7e`), `OrthoNormalizeColsQR`, `Wb::MatProd` $[3]$ | Positive-diagonal QR, matmul | The two dense primitives the sweep needs | `QrGauge::PositiveDiagonal` tightens QSpace's unspecified QR sign (documented in [`docs/gauge_soN.md`](gauge_soN.md)); via the selectable backend |
| `bcd/catalog.rs::CanonicalCatalog` | QSpace v4 (rev `dd2cc7e`), `Source/clebsch.cc` cross-copy `normDiff` check (`:6710-6718`) $[3]$ | Per-(series, rank) catalog caching the aligned CGC; cross-copy coherence check | Aligns and reuses the generator/CGC data across couplings | QSpace's fixed-pass `dmax` enumeration is **not** ported (`catalog.rs` module doc); coherence is a loud debug assert, deviation-by-design from `normDiff` replacement |
| `bcd/fr.rs` | QSpace v4 (rev `dd2cc7e`), `wbsparray::setRec_kron` (product packing) $[3]$; shared `frcore` | Kronecker product index packing; F/R contraction | Consistent product-basis packing between generation and contraction | — |
| `bcd/qspace_oracle_tests.rs` (oracle, not production) | QSpace v4 (rev `dd2cc7e`), `getCG` and generator fixtures ($Sp_k$, $Sz_k$) $[3]$ | Projector battery via verified factor-basis dictionaries | Independent external oracle: `racah`'s aligned CGC vs QSpace to round-off | Compares gauge-invariant projectors through solved intertwiner dictionaries |

## Family-generic F/R core and gates

| Implementation area (`file::symbol`) | Reference (project, version, `file:symbol`) | What was taken | Why this reference / algorithm | Deviations |
|---|---|---|---|---|
| `frcore.rs` (F/R contraction) | SUNRepresentations.jl v0.4.0, `src/sector.jl:_Fsymbol`/`_Rsymbol`; TensorKitSectors `src/sectors.jl:Fsymbol_from_fusiontensor` $[9]$ | Shared F/R contraction wiring and axis order for `sun::fr` and `bcd::fr` | One correct contraction, reused across families | Private core; only `FBlock`/`RBlock` re-exported |
| `frcore.rs` pentagon / hexagon gates | TensorKitSectors `src/sectors.jl:pentagon_equation` (`:786-819`), `hexagon_equation` (`:834-871`), GenericFusion branch $[2]$ background | Pentagon (associativity) and hexagon (braiding) consistency checks | The categorical consistency laws every F/R must satisfy | Shipped as public self-checks / generation gates |

## Verification oracles

The oracles are independent of the code under test:

- **SU(2)**: exhaustive agreement with the Rust `wigner-symbols` crate v0.5.1
  $[11]$ over its label domain, plus reference fixtures beyond it.
- **SU(N)**: signed element-wise table regeneration against SUNRepresentations.jl
  v0.4.0 $[9]$; products/multiplicities cross-checked against GroupMath v1.1.3
  fixtures $[6]$.
- **SO(N)/Sp(2N)**: the QSpace v4 CGC projector battery $[3]$ through verified
  factor-basis dictionaries (`bcd/qspace_oracle_tests.rs`). Fixture provenance:
  [`tools/README.md`](../tools/README.md).

## Bibliography

Every identifier below was verified against the publisher/preprint record.

1. A. Alex, M. Kalus, A. Huckleberry, J. von Delft, "A numerical algorithm for
   the explicit calculation of SU(N) and SL(N,C) Clebsch–Gordan coefficients,"
   *J. Math. Phys.* **52**, 023507 (2011).
   DOI: [10.1063/1.3521562](https://doi.org/10.1063/1.3521562);
   arXiv: [1009.0437](https://arxiv.org/abs/1009.0437).
2. A. Weichselbaum, "Non-abelian symmetries in tensor networks: A quantum
   symmetry space approach," *Ann. Phys.* **327**, 2972–3047 (2012).
   DOI: [10.1016/j.aop.2012.07.009](https://doi.org/10.1016/j.aop.2012.07.009);
   arXiv: [1202.5664](https://arxiv.org/abs/1202.5664).
3. A. Weichselbaum, "QSpace — An open-source tensor library for Abelian and
   non-Abelian symmetries," *SciPost Phys. Codebases* **40** (2024).
   DOI: [10.21468/SciPostPhysCodeb.40](https://doi.org/10.21468/SciPostPhysCodeb.40);
   arXiv: [2405.06632](https://arxiv.org/abs/2405.06632). `racah` cites QSpace
   v4 source at revision `dd2cc7e`.
4. I. M. Gelfand, M. L. Tsetlin, "Finite-dimensional representations of the
   group of unimodular matrices," *Dokl. Akad. Nauk SSSR* **71**, 825–828
   (1950) (Russian). No DOI (predates DOI registration).
5. G. Racah, "Theory of Complex Spectra. II," *Phys. Rev.* **62**, 438–462
   (1942). DOI: [10.1103/PhysRev.62.438](https://doi.org/10.1103/PhysRev.62.438).
6. R. M. Fonseca, "GroupMath: A Mathematica package for group theory
   calculations," *Comput. Phys. Commun.* **267**, 108085 (2021).
   DOI: [10.1016/j.cpc.2021.108085](https://doi.org/10.1016/j.cpc.2021.108085);
   arXiv: [2011.01764](https://arxiv.org/abs/2011.01764). `racah` fixtures use
   GroupMath v1.1.3.
7. W. Fulton, J. Harris, *Representation Theory: A First Course*, Graduate Texts
   in Mathematics **129**, Springer (1991). ISBN 0-387-97495-4.
   DOI: [10.1007/978-1-4612-0979-9](https://doi.org/10.1007/978-1-4612-0979-9).
8. J. E. Humphreys, *Introduction to Lie Algebras and Representation Theory*,
   Graduate Texts in Mathematics **9**, Springer (1972). ISBN 978-0-387-90053-7.
   DOI: [10.1007/978-1-4612-6398-2](https://doi.org/10.1007/978-1-4612-6398-2).
9. SUNRepresentations.jl (v0.4.0), a Julia implementation of $[1]$.
   <https://github.com/QuantumKitHub/SUNRepresentations.jl>.
10. WignerSymbols.jl (v2.0.0), J. Haegeman.
    <https://github.com/Jutho/WignerSymbols.jl>.
11. `wigner-symbols` Rust crate (v0.5.1), P. Ruffwind.
    <https://crates.io/crates/wigner-symbols>.

The port also follows the public conventions and F/R contraction wiring of
TensorKitSectors (the categorical-symmetry layer of TensorKit), cited inline
above as `sectors.jl:symbol`.
