"""The exact SU(2) surface (issue #107).

Two questions, and the second is the reason this file exists. First, do the exact
functions return the textbook values? Second, do they agree with the *generated* SU(N)
surface at rank 1 — because SU(2) is reachable through both, and a consumer choosing
between them on cost alone needs that agreement to be a test rather than a measurement
someone once took.
"""

import math

import numpy as np
import pytest

import racah

# Doubled spins: dj = 2j, dm = 2m.
SPINS = range(0, 5)


def admissible(dj1, dj2, dj3):
    return (
        abs(dj1 - dj2) <= dj3 <= dj1 + dj2 and (dj1 + dj2 + dj3) % 2 == 0
    )


# --- the exact surface against closed forms ----------------------------------------


def test_wigner_3j_known_values():
    """(1/2 1/2 0; 1/2 -1/2 0) = 1/sqrt(2), and the all-ones 3j."""
    assert racah.wigner_3j(1, 1, 0, 1, -1, 0) == pytest.approx(1 / math.sqrt(2), abs=1e-15)
    assert racah.wigner_3j(2, 2, 2, 0, 0, 0) == pytest.approx(0.0, abs=1e-15)


def test_wigner_6j_known_value():
    """{1 1 1; 1 1 1} = 1/6 — the crate's own quick-start example, through Python."""
    assert racah.wigner_6j(2, 2, 2, 2, 2, 2) == pytest.approx(1 / 6, abs=1e-14)


def test_inadmissible_is_exact_zero_not_an_error():
    """The documented contract: these engines do not raise, they return exact zero.

    A consumer that already guards with a triangle test wants this; a consumer that
    wants the distinction wants the crate's ``*_checked`` twins, which are deliberately
    not bound yet.
    """
    assert racah.wigner_3j(1, 1, 4, 1, -1, 0) == 0.0
    assert racah.wigner_6j(1, 1, 4, 1, 1, 1) == 0.0
    assert racah.su2_r_symbol(1, 1, 4) == 0.0
    assert racah.su2_clebsch_gordan(1, 1, 1, 1, 0, 0) == 0.0


def test_r_symbol_is_the_closed_form_sign():
    """``R^{ab}_c = (-1)^(j_a + j_b - j_c)``, exactly, on every admissible triangle."""
    for dj1 in SPINS:
        for dj2 in SPINS:
            for dj3 in SPINS:
                if not admissible(dj1, dj2, dj3):
                    continue
                want = (-1.0) ** ((dj1 + dj2 - dj3) // 2)
                assert racah.su2_r_symbol(dj1, dj2, dj3) == want


def test_r_symbol_is_its_own_inverse():
    """SU(2) braiding is symmetric; this is the property ``transpose`` consumers gate on."""
    for dj1 in SPINS:
        for dj2 in SPINS:
            for dj3 in SPINS:
                if admissible(dj1, dj2, dj3):
                    r = racah.su2_r_symbol(dj1, dj2, dj3)
                    assert r * racah.su2_r_symbol(dj2, dj1, dj3) == pytest.approx(1.0)


def test_clebsch_gordan_rows_are_orthonormal():
    """Sum over m1 of <j1 m1; j2 m2 | j3 m3>^2 = 1 for a fully stretched coupling."""
    dj1 = dj2 = 1
    for dj3 in (0, 2):
        total = 0.0
        for dm1 in (-1, 1):
            dm2 = -dm1
            total += racah.su2_clebsch_gordan(dj1, dm1, dj2, dm2, dj3, 0) ** 2
        assert total == pytest.approx(1.0, abs=1e-14)


def test_frobenius_schur_is_the_parity_of_the_doubled_spin():
    for dj in SPINS:
        assert racah.su2_frobenius_schur(dj) == (1.0 if dj % 2 == 0 else -1.0)


# --- the two surfaces against each other -------------------------------------------


def su2_irrep(dj):
    """SU(2) as a rank-1 SU(N) label: the Dynkin label of spin ``dj/2`` is ``[dj]``."""
    return racah.Irrep([dj])


def test_r_symbol_agrees_with_the_generated_surface():
    """The claim issue #107 rests on, as a test rather than a one-off measurement.

    Every admissible triangle up to ``2j = 4``. SU(2) has no outer multiplicity, so the
    generated block is 1x1 and its single entry is the scalar.
    """
    compared = 0
    for dj1 in SPINS:
        for dj2 in SPINS:
            for dj3 in SPINS:
                if not admissible(dj1, dj2, dj3):
                    continue
                block = racah.r_symbol(su2_irrep(dj1), su2_irrep(dj2), su2_irrep(dj3))
                assert block.shape == (1, 1)
                assert block[0, 0] == pytest.approx(
                    racah.su2_r_symbol(dj1, dj2, dj3), abs=1e-12
                )
                compared += 1
    assert compared > 20, "the admissibility filter rejected almost everything"


def test_f_symbol_agrees_with_the_generated_surface():
    """The same for F, over the label box the comparison in issue #107 used.

    Smaller than the R box because the generated F-symbol is the expensive one: each
    entry costs a CGC construction plus contractions, so this is a few seconds and the
    R test above is milliseconds.
    """
    compared = 0
    for dj1 in range(3):
        for dj2 in range(3):
            for dj3 in range(3):
                for dj4 in range(4):
                    for dj5 in range(4):
                        for dj6 in range(4):
                            try:
                                block = racah.f_symbol(
                                    *(su2_irrep(d) for d in (dj1, dj2, dj3, dj4, dj5, dj6))
                                )
                            except ValueError:
                                continue  # an empty vertex: nothing to compare
                            assert block.shape == (1, 1, 1, 1)
                            assert block[0, 0, 0, 0] == pytest.approx(
                                racah.su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6), abs=1e-10
                            )
                            compared += 1
    assert compared > 20, "every labelling was rejected; the comparison ran on nothing"


def test_cgc_differs_from_the_generated_tensor_by_exactly_the_r_symbol_phase():
    """The two surfaces' CGC do **not** agree, and the discrepancy is exactly one sign.

    F and R agree between the tiers (the two tests above) because they are gauge-invariant
    combinations in which a per-channel CGC phase cancels. The CGC themselves are gauge
    *data*, and the two tiers fix that gauge differently: the generated tier's
    Gelfand-Tsetlin construction against the exact tier's Condon-Shortley convention.

    Measured over every channel up to ``2j = 3``, the ratio is uniform in the magnetic
    indices and equals ``(-1)^(j1 + j2 - j3)`` -- which is to say, it *is*
    :func:`racah.su2_r_symbol`. That makes the conversion one multiplication, and it is
    pinned here because a consumer mixing the two tiers' CGC without it gets silently
    wrong signs rather than an error.

    The m-basis order is asserted alongside: a rank-1 GT basis is the magnetic basis
    ascending in m, which is what makes the two comparable entry by entry at all.
    """
    compared = 0
    for dj1 in range(4):
        for dj2 in range(4):
            for dj3 in range(abs(dj1 - dj2), dj1 + dj2 + 1, 2):
                dense = racah.clebsch_gordan(su2_irrep(dj1), su2_irrep(dj2), su2_irrep(dj3))
                assert dense.shape == (dj1 + 1, dj2 + 1, dj3 + 1, 1)
                phase = racah.su2_r_symbol(dj1, dj2, dj3)
                for i1, dm1 in enumerate(range(-dj1, dj1 + 1, 2)):
                    for i2, dm2 in enumerate(range(-dj2, dj2 + 1, 2)):
                        for i3, dm3 in enumerate(range(-dj3, dj3 + 1, 2)):
                            scalar = racah.su2_clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3)
                            assert dense[i1, i2, i3, 0] == pytest.approx(
                                phase * scalar, abs=1e-12
                            )
                            compared += 1
    assert compared > 100


def test_the_cgc_phase_is_not_the_identity_so_the_test_above_has_content():
    """At least one channel really does flip, or the relation above is vacuous."""
    assert racah.su2_r_symbol(1, 1, 0) == -1.0


# --- the fingerprint ----------------------------------------------------------------


def test_the_two_fingerprints_are_different_strings():
    """Numerical agreement is not authority identity, and the API must not blur them.

    A consumer persisting coefficients records the fingerprint of the surface that
    produced them; if these two ever compared equal, a file written from one tier would
    silently validate against the other.
    """
    su2 = racah.su2_authority_fingerprint()
    sun = racah.sun_authority_fingerprint()
    assert isinstance(su2, str) and su2
    assert su2 != sun
    assert su2.startswith("racah:su2-exact:")
    assert sun.startswith("racah:sun-gt:")


def test_the_exact_surface_is_much_cheaper_than_the_generated_one():
    """Not a benchmark — a floor under the reason this surface was bound at all.

    Cold, the generated path measured milliseconds per R-symbol against a sign. Asserting
    a 10x margin on a warm-cache comparison is far inside that and still fails loudly if
    ``su2_r_symbol`` ever starts routing through the generator.
    """
    import time

    labels = [
        (a, b, c) for a in SPINS for b in SPINS for c in SPINS if admissible(a, b, c)
    ]
    irreps = [tuple(su2_irrep(d) for d in t) for t in labels]
    for a, b, c in irreps:  # warm the generated tier so the comparison is fair
        racah.r_symbol(a, b, c)

    t0 = time.perf_counter()
    for a, b, c in labels:
        racah.su2_r_symbol(a, b, c)
    exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    for a, b, c in irreps:
        racah.r_symbol(a, b, c)
    generated = time.perf_counter() - t0

    assert exact * 10 < generated, f"exact {exact:.4f}s vs warm generated {generated:.4f}s"


def test_stubs_cover_every_new_name():
    """``racah.pyi`` ships in the wheel, so a name without a stub is a typing hole."""
    from pathlib import Path

    stub = Path(racah.__file__).with_name("__init__.pyi").read_text()
    for name in (
        "wigner_3j",
        "wigner_6j",
        "su2_clebsch_gordan",
        "su2_f_symbol",
        "su2_r_symbol",
        "su2_authority_fingerprint",
    ):
        assert hasattr(racah, name), f"{name} is not bound"
        assert f"def {name}(" in stub, f"{name} has no stub in racah.pyi"


def test_numpy_is_not_needed_for_the_exact_surface():
    """The scalars are plain floats, not zero-dimensional arrays."""
    assert type(racah.su2_r_symbol(1, 1, 0)) is float
    assert type(racah.wigner_6j(2, 2, 2, 2, 2, 2)) is float
    assert not isinstance(racah.su2_f_symbol(1, 1, 1, 1, 0, 2), np.ndarray)
