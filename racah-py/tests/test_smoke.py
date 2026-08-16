"""Smoke tests for the racah Python bindings (issue #81)."""

import numpy as np
import pytest

import racah

HALF = racah.Irrep([1])  # SU(2) spin-1/2
SINGLET2 = racah.Irrep([0])  # SU(2) spin-0
TRIPLET = racah.Irrep([2])  # SU(2) spin-1

FUND = racah.Irrep([1, 0])  # SU(3) 3
ANTIFUND = racah.Irrep([0, 1])  # SU(3) 3bar
ADJOINT = racah.Irrep([1, 1])  # SU(3) 8
SINGLET3 = racah.Irrep([0, 0])  # SU(3) 1


def test_irrep_labels():
    assert FUND.dynkin == [1, 0]
    assert FUND.rank == 3
    assert FUND.dim() == 3
    assert ADJOINT.dim() == 8
    assert FUND.dual() == ANTIFUND
    assert ADJOINT.dual() == ADJOINT
    assert racah.Irrep.trivial(3) == SINGLET3
    with pytest.raises(ValueError):
        racah.Irrep([-1, 0])


def test_su3_fusion():
    # 3 (x) 3bar = 1 + 8, both multiplicity-free.
    assert racah.fusion(FUND, ANTIFUND) == [(SINGLET3, 1), (ADJOINT, 1)]
    # The classic multiplicity: 8 (x) 8 contains 8 twice.
    assert racah.fusion_multiplicity(ADJOINT, ADJOINT, ADJOINT) == 2
    with pytest.raises(ValueError):
        racah.fusion(FUND, HALF)  # SU(3) against SU(2)


def test_su2_clebsch_gordan_known_values():
    # 1/2 (x) 1/2 -> 0: the antisymmetric singlet, entries +-1/sqrt(2).
    singlet = racah.clebsch_gordan(HALF, HALF, SINGLET2)
    assert singlet.shape == (2, 2, 1, 1)
    block = singlet[:, :, 0, 0]
    assert block[0, 0] == 0.0 and block[1, 1] == 0.0
    assert np.isclose(abs(block[0, 1]), 1.0 / np.sqrt(2.0))
    assert np.isclose(block[0, 1], -block[1, 0])

    # 1/2 (x) 1/2 -> 1: stretched states have |C| = 1, the m = 0 state 1/sqrt(2).
    triplet = racah.clebsch_gordan(HALF, HALF, TRIPLET)
    assert triplet.shape == (2, 2, 3, 1)
    magnitudes = np.sort(np.abs(triplet.reshape(-1)))
    expected = np.sort(
        np.array([1.0, 1.0, 1 / np.sqrt(2), 1 / np.sqrt(2)] + [0.0] * 8)
    )
    assert np.allclose(magnitudes, expected)
    # Columns over the coupled index are orthonormal.
    gram = np.einsum("abcm,abdm->cd", triplet, triplet)
    assert np.allclose(gram, np.eye(3))


def test_su3_clebsch_gordan_shape():
    cg = racah.clebsch_gordan(FUND, ANTIFUND, ADJOINT)
    assert cg.shape == (3, 3, 8, 1)
    gram = np.einsum("abcm,abdm->cd", cg, cg)
    assert np.allclose(gram, np.eye(8))


def test_f_and_r_symbol_shapes():
    # 3 (x) 3 = 6 + 3bar, all vertices multiplicity-free.
    six = racah.Irrep([2, 0])
    ten = racah.Irrep([3, 0])
    # a=b=c=3, e=f=6, d=10: every vertex (3x3->6, 6x3->10) exists and is simple.
    f = racah.f_symbol(FUND, FUND, FUND, ten, six, six)
    assert f.shape == (1, 1, 1, 1)
    # The multiplicity axes are kept: every vertex of this block is 8 (x) 8 -> 8.
    f_adjoint = racah.f_symbol(*([ADJOINT] * 6))
    assert f_adjoint.shape == (2, 2, 2, 2)

    r = racah.r_symbol(FUND, FUND, six)
    assert r.shape == (1, 1)
    assert np.isclose(abs(r[0, 0]), 1.0)
    with pytest.raises(ValueError):
        racah.r_symbol(FUND, FUND, ADJOINT)  # empty fusion vertex


def test_checks_pass():
    racah.check_pentagon(FUND, FUND, FUND, FUND)
    racah.check_hexagon(FUND, FUND, FUND)
    racah.check_f_unitarity(FUND, FUND, FUND, FUND)


def test_fingerprint():
    fingerprint = racah.sun_authority_fingerprint()
    assert isinstance(fingerprint, str)
    assert fingerprint.startswith("racah:")


def test_su2_frobenius_schur():
    assert racah.su2_frobenius_schur(0) == 1.0
    assert racah.su2_frobenius_schur(1) == -1.0
