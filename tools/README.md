# Fixture generators: taxonomy and rules

Every oracle fixture in `tests/fixtures/` is produced by a script in this
directory and carries a provenance header (tool + version, script sha256,
seed). Scripts are permanent test assets: deleting one breaks the
provenance chain and the regeneration recipe. Data is what tests read;
scripts never link into the build or CI.

## The claim decides the generator

A fixture asserts a specific claim, and only the implementation that OWNS
that claim may generate it:

| Claim of the fixture | Generator (fixed) | Why it cannot move |
|---|---|---|
| "matches the TensorKitSectors convention" (SU(2) F/R) | TensorKitSectors (Julia) | the convention's owner is the only source of truth |
| "gauge-continuous with SUNRepresentations" (SU(N) CGC, SU(3) F/R table) | SUNRepresentations (Julia) | the gauge's owner; also the table-deletion authorization gate |
| "agrees with exact rational values" (SU(2) 6j beyond the u8-key domain) | WignerSymbols.jl (exact arithmetic) | the exactness claim needs an exact-arithmetic tool |
| "an independent implementation gets the same exact decomposition" (B/C/D N-symbols, dims) | Sage WeylCharacterRing or OSCAR (free) | independence + free reproducibility |
| "the values are correct" (B/C/D CGC/F/R, gauge-agnostic, compared through an alignment harness) | GroupMath (Wolfram) | value-level check with no convention claim |

## Free-first rule

racah is a public crate; provenance must be re-verifiable by contributors
without commercial licenses. Therefore:

- Claims that a free tool can serve keep a free-tool generator (Julia /
  Sage / OSCAR / Python), even where a Wolfram equivalent exists.
- GroupMath (Wolfram, campus/commercial license) is the preferred
  generator for NEW generic value/count fixtures where it is the most
  capable tool — as an addition, not a replacement.
- Existing committed fixtures are never replaced by a different tool;
  toolchain diversity is verification strength (two independent
  implementations agreeing is stronger than one, and has already caught
  real issues).

## Conventions

- One script → one or two fixture files, paired by name
  (`gen_X…` → `X…_fixtures`).
- Scripts self-hash: the provenance header's sha256 must match the
  committed script byte-for-byte (regenerate the data when the script
  changes).
- Wolfram scripts load GroupMath by explicit path (no system install
  required); the header records the GroupMath version.
- Heavy-fixture or heavy-gate runtime claims carry count × per-op
  arithmetic; the count is computable from the exact layer before running
  anything.
- If this directory grows past easy scanning, split by semantics
  (per group family or per oracle kind), not by language.

## Reference-coverage roles (who verifies what)

Each verification strand has one role; strands do not substitute for each
other:

| Strand | Verifies | Does NOT verify |
|---|---|---|
| TensorKitSectors fixtures | racah reproduces the TKS **convention** (SU(2) F/R) | general value correctness elsewhere |
| SUNRepresentations fixtures | racah reproduces the SUNRep **gauge** (SU(N) CGC/F/R; authorizes downstream table deletion) | B/C/D anything |
| WignerSymbols.jl fixtures | **exact** SU(2) values beyond the reference crate's domain | conventions of other packages |
| Sage / OSCAR fixtures | exact **combinatorics** (N-symbols, dims) against an independent implementation | coefficient values, gauges |
| GroupMath fixtures | gauge-agnostic **values** — the primary external value oracle for CGC/F/R across **SU(N) and B/C/D** (through the alignment harness; for SU(N) this independently audits the SUNRep lineage itself) | any convention/gauge claim |
| Isomorphism cross-checks (SO(4)≅SU(2)², SO(6)≅SU(4), Sp(4)≅SO(5)) | B/C/D pipeline consistency against the independently-oracled SU(N) pipeline | labels outside the isomorphic images |
| Self-consistency gates (orthogonality, unitarity, pentagon/hexagon, exact-multiplicity, coherence guard) | internal coherence of every generated family, always on | agreement with any external convention |
| QSpace (optional, license-gated) | the ported pipeline against the **reference implementation's own numbers** (after gauge alignment) | nothing exclusively — hardening only, since GroupMath covers value claims |

Rule: when adding a test, first name the claim, then pick the strand that
owns it. A test whose claim spans two strands is two tests.
