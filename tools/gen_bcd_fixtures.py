#!/usr/bin/env sage
# -*- coding: python -*-
"""Independent-implementation oracle fixtures for racah's B/C/D combinatorics.

Runs under SageMath (`sage tools/gen_bcd_fixtures.py`). It uses
`WeylCharacterRing` in the "coroots" style, so irreps are addressed by their
integer Dynkin (coroot) labels — the same convention as `racah::bcd`. For
seeded-random tensor irrep pairs across B/C/D ranks {2,3,4} it emits the exact
dimension and full tensor-product decomposition.

Output goes to tests/fixtures/bcd_fixtures.json (a line-oriented text format,
NOT JSON — the name is kept for the test path; parsing is a few `split`s, so no
serde dependency is pulled into the crate). Each record is

    SERIES RANK | dynkin_a | dynkin_b | dim_a dim_b | c1:n1 c2:n2 ...

with Dynkin labels comma-separated. A provenance header records the Sage
version and this script's sha256 so a checked-in fixture is auditable.

Only TENSOR irreps are generated (racah's published object): for B_r the last
Dynkin label is forced even; for D_r the last two are forced equal-parity;
C_r accepts every non-negative label. Spinor labels are never emitted.
"""

import hashlib
import os
import random
import sys

# Cartan types per series/rank in scope; racah excludes B1/C1/D2.
CASES = [
    ("B", 2), ("B", 3), ("B", 4),
    ("C", 2), ("C", 3), ("C", 4),
    ("D", 3), ("D", 4),
]
PAIRS_PER_CASE = 8
MAX_LABEL = {2: 3, 3: 2, 4: 1}  # keep weight systems (hence products) modest
SEED = 0x0BCD_0019


def tensorize(series, dynkin):
    """Force a Dynkin label onto the tensor (integer-weight) sublattice."""
    a = list(dynkin)
    r = len(a)
    if series == "B":
        a[r - 1] -= a[r - 1] % 2            # last label even
    elif series == "D":
        if (a[r - 2] + a[r - 1]) % 2 != 0:  # last two equal parity
            a[r - 1] += 1 if a[r - 1] < MAX_LABEL[r] else -1
    return tuple(a)


def rand_label(rng, series, r):
    hi = MAX_LABEL[r]
    return tensorize(series, [rng.randint(0, hi) for _ in range(r)])


def dynkin_of(R, weight):
    """Integer Dynkin (coroot) labels of an ambient-space weight."""
    L = R.space()
    return tuple(int(weight.inner_product(L.simple_coroots()[i])) for i in R.index_set())


def main():
    rng = random.Random(SEED)
    with open(__file__, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()

    out = []
    out.append("--- racah B/C/D fixtures (independent oracle)")
    out.append("--- tool: SageMath %s" % version())
    out.append("--- script: gen_bcd_fixtures.py sha256=%s" % sha)
    out.append("--- seed: 0x%X  format: SERIES RANK | a | b | dim_a dim_b | c:n ...")

    for series, r in CASES:
        R = WeylCharacterRing("%s%d" % (series, r), style="coroots")
        for _ in range(PAIRS_PER_CASE):
            da = rand_label(rng, series, r)
            db = rand_label(rng, series, r)
            A = R(*da)
            B = R(*db)
            prod = A * B
            terms = []
            for wt, mult in prod.monomial_coefficients().items():
                dc = dynkin_of(R, wt)
                terms.append("%s:%d" % (",".join(map(str, dc)), int(mult)))
            terms.sort()
            # A.degree() is the Weyl-dimension FORMULA (no module is
            # constructed), so it stays fast even for rank-4 labels — unlike
            # Oscar, where building the h.w. module to read its dim is the slow
            # path (see gen_bcd_fixtures.jl).
            out.append("%s %d | %s | %s | %d %d | %s" % (
                series, r,
                ",".join(map(str, da)),
                ",".join(map(str, db)),
                int(A.degree()), int(B.degree()),
                " ".join(terms),
            ))

    dest = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "bcd_fixtures.json")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    sys.stderr.write("wrote %s (%d records)\n" % (path, len(out) - 4))


if __name__ == "__main__":
    main()
