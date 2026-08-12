#!/usr/bin/env wolframscript
(* ::Package:: *)

(*
  GroupMath value-oracle fixtures for racah: tensor-product decompositions
  (channels + fusion multiplicities N^c_ab) across SU(N) and the B/C/D series.

  Run (GroupMath 1.1.3, free download arXiv:2011.01764):
    export GROUPMATH_DIR=/path/to/parent/of/GroupMath   # dir CONTAINING GroupMath/
    /Applications/Wolfram.app/Contents/MacOS/wolframscript \
      -file tools/groupmath/gen_groupmath_products.wls > tests/fixtures/groupmath_products.txt

  Role (tools/README.md matrix): GroupMath is the primary EXTERNAL value oracle
  for CGC/F/R across SU(N) and B/C/D. This fixture commits only the part that is
  deterministically comparable to racah WITHOUT a gauge-alignment step: the
  channel list and integer multiplicities. The CGC-value / invariant-projector
  comparison needs the #29 alignment harness and is out of scope here.

  ------------------------------------------------------------------------------
  API RECONNAISSANCE (done before any comparison test; issue #32 requirement)
  ------------------------------------------------------------------------------
  Two GroupMath entry points were surveyed:

  1. ReduceRepProduct[group, {r1, r2, ...}]  ->  { {dynkin, mult}, ... }
     The product-decomposition oracle used by THIS fixture. `dynkin` is the
     GroupMath Dynkin label; `mult` is the outer multiplicity N^c_{r1 r2}.
     VERIFIED: GroupMath's Dynkin node ordering is identical to racah's for
     every group used here (SU(N) matches by construction; and for B/C/D,
     DimR[SO5,{1,0}]=5 vector / {0,1}=4 spinor, DimR[Sp4,{1,0}]=4 / {0,1}=5,
     DimR[SO6,{1,0,0}]=6 vector, DimR[SO7,{1,0,0}]=7 vector -- all matching
     racah::bcd node 0 = vector). So multiplicities compare with NO relabeling.
     This data is gauge-free (integers), so it is exact and license-independent.

  2. Invariants[group, {r1, r2, ...}, TensorForm->True]  ->  the CGC entry point.
     Returns a basis of the invariant subspace of the product as explicit
     tensors (SparseArray). Example: Invariants[SU3, {{1,0},{0,1}}] = {a.b}, and
     with TensorForm->True the 3x3 identity delta_ij -- i.e. GroupMath reorders
     conjugate-rep weights so the two-rep invariant is Sum_i a[i] b[i]
     (LieAlgebras.m:454). The number of invariants of {r1, r2, dual(r3)} equals
     N^{r3}_{r1 r2}; verified nInv(SU2 {2},{2},{2}) = 1.
     NORMALIZATION / BASIS: Invariants applies NormalizeAndSymmetrize by default
     (orthonormal invariant basis, permutation-symmetry-adapted). This basis is
     GroupMath's own gauge; it is NOT the racah GT/QSpace gauge, and there is no
     documented deterministic map between them. Per the issue's rule ("do not
     fake agreement"), the CGC VALUES are therefore NOT committed here. The
     gauge-invariant projector P = C C^T onto each isotypic block is the robust
     comparable object, but building it on the racah side is exactly the #29
     alignment harness -- deferred there, not duplicated here.
  ------------------------------------------------------------------------------
*)

groupmathDir = Environment["GROUPMATH_DIR"];
If[groupmathDir === $Failed,
  (* No session-scratchpad default: a fresh checkout has no such path, and a
     silent fallback into a nonexistent directory yields a confusing GroupMath
     load failure downstream. Fail loudly with the fix instead. *)
  (Print["ERROR: set GROUPMATH_DIR to the directory containing the GroupMath` package"];
   Exit[1])];
AppendTo[$Path, groupmathDir];
(* Suppress GroupMath's load-time banner (a Print) so stdout is pure fixture. *)
Block[{Print = (Null &)}, Quiet[Get["GroupMath`"]]];

(* Curated battery (no RNG): the channels named in issue #32 -- SO5/Sp4 vector
   products, SO6 adjoint^2 (incl. the 84 and the OM=2 adjoint channel) -- plus
   SU(N) samples that independently audit the SUNRepresentations lineage. Only
   tensor (non-spinor) B/C/D labels appear, since racah::bcd rejects spinors. *)
cases = {
  (* SU(N): fundamental x antifundamental, adjoint^2, and a mixed product. *)
  {"SU2", SU2, {2}, {2}},
  {"SU2", SU2, {1}, {1}},
  {"SU2", SU2, {2}, {3}},
  {"SU3", SU3, {1,0}, {0,1}},
  {"SU3", SU3, {1,1}, {1,1}},
  {"SU3", SU3, {2,0}, {1,1}},
  {"SU4", SU4, {1,0,0}, {0,0,1}},
  {"SU4", SU4, {1,0,1}, {1,0,1}},
  {"SU5", SU5, {1,0,0,0}, {0,0,0,1}},
  {"SU5", SU5, {1,0,0,1}, {1,0,0,1}},
  (* SO(5) = B2: vector^2 and vector x higher tensor. *)
  {"SO5", SO5, {1,0}, {1,0}},
  {"SO5", SO5, {1,0}, {0,2}},
  {"SO5", SO5, {2,0}, {2,0}},
  (* Sp(4) = C2: vector^2 and mixed. *)
  {"Sp4", Sp4, {1,0}, {1,0}},
  {"Sp4", Sp4, {1,0}, {0,1}},
  {"Sp4", Sp4, {2,0}, {1,1}},
  (* SO(6) = D3: adjoint^2 (the named battery: 84 + OM=2) and vector^2. *)
  {"SO6", SO6, {0,1,1}, {0,1,1}},
  {"SO6", SO6, {1,0,0}, {1,0,0}},
  (* SO(7) = B3 and Sp(6) = C3 vector products. *)
  {"SO7", SO7, {1,0,0}, {1,0,0}},
  {"SO7", SO7, {0,1,0}, {0,1,0}},
  {"Sp6", Sp6, {1,0,0}, {1,0,0}}
};

csv[l_] := StringRiffle[ToString /@ l, ","];
channelStr[dec_] := StringRiffle[(csv[#[[1]]] <> ":" <> ToString[#[[2]]]) & /@ dec, " "];

sha = Hash[ReadString[$InputFileName], "SHA256", "HexString"];
Print["--- racah GroupMath product fixtures (external value oracle)"];
Print["--- tool: GroupMath 1.1.3 (arXiv:2011.01764)"];
Print["--- script: gen_groupmath_products.wls sha256=" <> ToLowerCase[sha]];
Print["--- entry point: ReduceRepProduct (gauge-free multiplicities; CGC via Invariants deferred to #29)"];
Print["--- format: GROUP | a | b | dim_a dim_b | c:n c:n ..."];

Do[
  Module[{tag = c[[1]], grp = c[[2]], a = c[[3]], b = c[[4]], dec, da, db},
    dec = ReduceRepProduct[grp, {a, b}];
    da = DimR[grp, a]; db = DimR[grp, b];
    Print[tag <> " | " <> csv[a] <> " | " <> csv[b] <> " | " <>
      ToString[da] <> " " <> ToString[db] <> " | " <> channelStr[dec]];
  ],
  {c, cases}];
