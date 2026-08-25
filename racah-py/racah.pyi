"""Type stubs for `racah` — Python bindings for the racah crate.

**Two surfaces.** The SU(N) one takes :class:`Irrep` labels and runs the generated
Gelfand-Tsetlin pipeline: fusion with outer multiplicities, dense m-basis
Clebsch-Gordan tensors, F/R symbols with their multiplicity axes, and the
verification gates. All its coefficient arrays are C-contiguous (row-major)
``numpy.float64`` arrays. The exact SU(2) one takes doubled integer spins and
returns scalars from closed-form big-rational arithmetic: :func:`wigner_3j`,
:func:`wigner_6j`, :func:`su2_clebsch_gordan`, :func:`su2_f_symbol`,
:func:`su2_r_symbol`, :func:`su2_frobenius_schur`.

SU(2) is reachable through either — ``Irrep([2j])`` is a rank-1 SU(N) label — and the
two agree numerically. They are still **separate authorities**:
:func:`sun_authority_fingerprint` and :func:`su2_authority_fingerprint` are different
strings, and whichever produced a coefficient is what you must record beside it.

They are not close in cost. The generated path builds a CGC tensor by SVD nullspace,
least-squares ladder descent and QR gauge fixing; ``su2_r_symbol`` is a sign.

Errors: on the SU(N) surface, ill-posed input (bad label, mixed rank, empty fusion
vertex) raises ``ValueError`` and a tripped generation/verification gate
(orthonormality, F-unitarity, pentagon, hexagon, factorization) raises
``RuntimeError``. The exact SU(2) functions do not raise: an inadmissible label set
returns exactly ``0.0``.

The values live in the crate's frozen canonical gauges; pin the matching fingerprint
next to anything you persist.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

class Irrep:
    """An irreducible representation of SU(N), labelled by its Dynkin label.

    Frozen, hashable and comparable by value: usable as a dict key.

    Parameters
    ----------
    dynkin : Sequence[int]
        The Dynkin label, length ``N - 1``, entries nonnegative. ``[1, 0]`` is
        the SU(3) fundamental **3**, ``[1, 1]`` the adjoint **8**; for SU(2)
        the single entry is the doubled spin ``2j``.

    Raises
    ------
    ValueError
        If the label is empty or has a negative entry.
    """

    def __init__(self, dynkin: Sequence[int]) -> None: ...
    @classmethod
    def from_weight(cls, weight: Sequence[int]) -> Irrep:
        """Build the irrep with (unnormalized) highest weight `weight`.

        Parameters
        ----------
        weight : Sequence[int]
            Length ``N``, nonincreasing. Normalized internally by subtracting
            the last entry (SU(N) weights are defined up to a uniform shift).

        Returns
        -------
        Irrep

        Raises
        ------
        ValueError
            If the weight is empty or not nonincreasing.
        """

    @classmethod
    def trivial(cls, n: int) -> Irrep:
        """The SU(`n`) singlet (all-zero Dynkin label of length ``n - 1``).

        Raises
        ------
        ValueError
            If ``n < 2``.
        """

    @property
    def dynkin(self) -> list[int]:
        """The Dynkin label (length ``N - 1``)."""

    @property
    def weight(self) -> list[int]:
        """The normalized highest weight (length ``N``, nonincreasing, last entry 0)."""

    @property
    def rank(self) -> int:
        """``N`` of the SU(N) this irrep belongs to."""

    def dim(self) -> int:
        """The Weyl dimension, as an exact arbitrary-precision ``int``."""

    def dual(self) -> Irrep:
        """The dual (conjugate) irrep."""

    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def fusion(a: Irrep, b: Irrep) -> list[tuple[Irrep, int]]:
    """Decompose the tensor product ``a ⊗ b`` into irreps.

    Parameters
    ----------
    a, b : Irrep
        SU(N) irreps of the same ``N``.

    Returns
    -------
    list[tuple[Irrep, int]]
        ``[(c, N^c_ab), ...]`` — each irrep ``c`` appearing in ``a ⊗ b`` with
        its outer multiplicity ``N^c_ab >= 1``, in the crate's deterministic
        irrep order (stable across calls and processes).

    Raises
    ------
    ValueError
        If ``a`` and ``b`` are not both SU(N) for one ``N``.
    """

def fusion_multiplicity(a: Irrep, b: Irrep, c: Irrep) -> int:
    """The outer multiplicity ``N^c_ab``: how many copies of ``c`` sit in ``a ⊗ b``.

    Returns 0 if ``c`` does not appear. This is the length of the multiplicity
    axis on the CGC, F and R blocks for that vertex.

    Raises
    ------
    ValueError
        If the three irreps are not all SU(N) for one ``N``.
    """

def clebsch_gordan(s1: Irrep, s2: Irrep, s3: Irrep) -> NDArray[np.float64]:
    """The dense m-basis Clebsch-Gordan tensor for ``s1 ⊗ s2 → s3``.

    Parameters
    ----------
    s1, s2, s3 : Irrep
        SU(N) irreps of the same ``N``.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``float64`` array of shape
        ``(dim(s1), dim(s2), dim(s3), N)`` with ``N = N^{s3}_{s1 s2}``:
        axes 0-2 index the Gelfand-Tsetlin m-basis states ``m1``, ``m2``,
        ``m3`` of ``s1``, ``s2``, ``s3``; the trailing axis ``mu`` indexes the
        outer-multiplicity copies of ``s3``. Entry
        ``cgc[m1, m2, m3, mu] = <s3 m3; mu | s1 m1; s2 m2>`` in the crate's
        frozen gauge (SUNRepresentations.jl v0.4.0 convention; the
        multiplicity columns are orthonormal and gauge-fixed as a block).
        If ``N^{s3}_{s1 s2} = 0`` the trailing axis has length 0.

    Raises
    ------
    ValueError
        If the irreps are not all SU(N) for one ``N``.
    RuntimeError
        If a generation gate trips (nullspace-dimension mismatch,
        orthonormality, ladder consistency, dense factorization failure).
    """

def f_symbol(a: Irrep, b: Irrep, c: Irrep, d: Irrep, e: Irrep, f: Irrep) -> NDArray[np.float64]:
    """The F-symbol ``F^{abc}_d[e, f]`` as its ``[mu, nu, kappa, lambda]`` multiplicity block.

    Relates the two fusion orders of ``a ⊗ b ⊗ c → d``: via intermediate ``e``
    (``(a b) c``) versus via ``f`` (``a (b c)``), with the magnetic (m-basis)
    indices contracted away.

    Parameters
    ----------
    a, b, c, d, e, f : Irrep
        SU(N) irreps of the same ``N``. ``e`` and ``f`` are the intermediate
        fusion channels.

    Returns
    -------
    numpy.ndarray
        C-contiguous (row-major) ``float64`` array of shape
        ``(N^e_ab, N^d_ec, N^f_bc, N^d_af)`` — one axis per fusion vertex:

        ========  ================  ==================
        axis      index             vertex
        ========  ================  ==================
        0         ``mu``            ``a ⊗ b → e``
        1         ``nu``            ``e ⊗ c → d``
        2         ``kappa``         ``b ⊗ c → f``
        3         ``lambda``        ``a ⊗ f → d``
        ========  ================  ==================

        The flat (row-major) index of ``block[mu, nu, kappa, lambda]`` is
        ``((mu * N2 + nu) * N3 + kappa) * N4 + lambda`` with
        ``(N1, N2, N3, N4) = block.shape``. The axis order matches the
        TensorKitSectors ``GenericFusion`` convention. In a multiplicity-free
        situation the shape is ``(1, 1, 1, 1)`` and the block holds a single
        scalar.

    Raises
    ------
    ValueError
        If the six irreps are not all SU(N) for one ``N``, or any of the four
        vertices has fusion multiplicity 0 (no all-zero blocks are returned).
    RuntimeError
        If an underlying CGC generation gate trips.
    """

def r_symbol(a: Irrep, b: Irrep, c: Irrep) -> NDArray[np.float64]:
    """The R-symbol ``R^{ab}_c`` as its multiplicity matrix.

    Relates the fusion ``a ⊗ b → c`` to the braided ``b ⊗ a → c``.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``float64`` matrix of shape ``(N^c_ab, N^c_ba)`` (the two
        multiplicities are equal): row ``mu`` indexes the ``a ⊗ b → c``
        vertex copy, column ``nu`` the ``b ⊗ a → c`` copy. Multiplicity-free
        means a ``(1, 1)`` matrix holding the braiding phase.

    Raises
    ------
    ValueError
        If the irreps are not all SU(N) for one ``N``, or ``a ⊗ b → c`` is
        empty.
    RuntimeError
        If an underlying CGC generation gate trips.
    """

def check_f_unitarity(a: Irrep, b: Irrep, c: Irrep, d: Irrep) -> None:
    """Verify the F-move unitarity gate for ``(a, b, c, d)`` over all ``(e, f)``.

    Raises
    ------
    ValueError
        If the irreps are not all SU(N) for one ``N``.
    RuntimeError
        If the assembled F-move is not unitary within tolerance.
    """

def check_pentagon(a: Irrep, b: Irrep, c: Irrep, d: Irrep) -> None:
    """Verify the pentagon identity for ``(a, b, c, d)``.

    Raises
    ------
    ValueError
        If the irreps are not all SU(N) for one ``N``.
    RuntimeError
        If the pentagon identity is violated within tolerance.
    """

def check_hexagon(a: Irrep, b: Irrep, c: Irrep) -> None:
    """Verify both hexagon identities for ``(a, b, c)``.

    Raises
    ------
    ValueError
        If the irreps are not all SU(N) for one ``N``.
    RuntimeError
        If either hexagon identity is violated within tolerance.
    """

def sun_authority_fingerprint() -> str:
    """The SU(N) gauge/authority fingerprint string.

    Opaque identifier of the frozen CGC gauge, generation pipeline and
    tolerance policy behind every SU(N) coefficient this module returns.
    Embed it next to any persisted coefficients and refuse a mismatch on
    load; compare by equality only. A breaking coefficient change bumps the
    fingerprint epoch (recorded in the CHANGELOG). Same fingerprint means
    same convention and value agreement within the oracle tolerance, not
    cross-process bit-identity.
    """

def su2_frobenius_schur(two_j: int) -> float:
    """The SU(2) Frobenius-Schur indicator of the irrep with doubled spin ``two_j``.

    Returns ``1.0`` for integer spin (``two_j`` even), ``-1.0`` for
    half-integer spin (``two_j`` odd).
    """

# --- the exact SU(2) surface (issue #107) -------------------------------------------
#
# Doubled labels throughout: ``dj = 2j`` and ``dm = 2m``, so ``dj1=1`` is spin 1/2 and
# ``dm1=-1`` is m = -1/2. Doubling is what keeps every label an exact integer, which is
# what lets the engine behind these be exact.

def wigner_3j(dj1: int, dj2: int, dj3: int, dm1: int, dm2: int, dm3: int) -> float:
    """The Wigner 3j symbol ``(j1 j2 j3; m1 m2 m3)``, spins and projections doubled.

    Exact big-rational arithmetic, rounded once on return. Does not raise: an
    inadmissible label set — m-sum nonzero, a projection out of range, or a violated
    triangle condition — is exactly ``0.0``.
    """

def wigner_6j(dj1: int, dj2: int, dj3: int, dj4: int, dj5: int, dj6: int) -> float:
    """The Wigner 6j symbol ``{j1 j2 j3; j4 j5 j6}``, spins doubled.

    Exact big-rational arithmetic, rounded once on return. Does not raise: a label set
    violating any of the four triangle conditions is exactly ``0.0``.
    """

def su2_clebsch_gordan(dj1: int, dm1: int, dj2: int, dm2: int, dj3: int, dm3: int) -> float:
    """``<j1 m1; j2 m2 | j3 m3>`` in the Condon-Shortley phase, doubled labels.

    Note the argument order: each spin is followed by its own projection. This is the
    *scalar* SU(2) coefficient; :func:`clebsch_gordan` is the dense SU(N) m-basis
    tensor and a different object. Does not raise; inadmissible is ``0.0``.

    **The two tiers' CGC differ by one sign per channel.** :func:`f_symbol` and
    :func:`r_symbol` agree with their exact twins because a per-channel CGC phase
    cancels in a gauge-invariant combination; CGC are gauge data and do not. The ratio
    is uniform in the magnetic indices and equals ``(-1)**(j1 + j2 - j3)``, which is
    exactly :func:`su2_r_symbol`::

        dense = clebsch_gordan(Irrep([dj1]), Irrep([dj2]), Irrep([dj3]))
        dense[i1, i2, i3, 0] == su2_r_symbol(dj1, dj2, dj3) * su2_clebsch_gordan(
            dj1, dm1, dj2, dm2, dj3, dm3
        )

    reading the rank-1 Gelfand-Tsetlin basis as the magnetic basis ascending in ``m``.
    Mixing the two tiers' CGC without that factor gives wrong signs and no error.
    """

def su2_f_symbol(dj1: int, dj2: int, dj3: int, dj4: int, dj5: int, dj6: int) -> float:
    """The SU(2) F-symbol ``F^{j1 j2 j3}_{j4}[j5, j6]``, spins doubled.

    Scalar: SU(2) has no outer multiplicity, so the four vertex axes of the SU(N)
    :func:`f_symbol` block are all length 1 and this is that single entry.
    """

def su2_r_symbol(dj1: int, dj2: int, dj3: int) -> float:
    """The SU(2) R-symbol ``R^{j1 j2}_{j3} = (-1)^(j1 + j2 - j3)``, spins doubled.

    Exactly ``+1.0`` or ``-1.0``, and ``0.0`` when the triangle condition fails. SU(2)
    braiding is symmetric, so this is its own inverse.
    """

def su2_authority_fingerprint() -> str:
    """The exact-SU(2) gauge/authority fingerprint string.

    The twin of :func:`sun_authority_fingerprint`, for the functions above. Compare by
    equality only. A consumer that persists SU(2) coefficients records *this* one; a
    consumer that reached SU(2) through :class:`Irrep` records the SU(N) one. They are
    different strings on purpose — the two surfaces are separate authorities even where
    they agree numerically.
    """
