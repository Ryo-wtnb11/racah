//! Python bindings for the `racah` surfaces (issues #81, #107).
//!
//! Deliberately thin: irrep labels, fusion with outer multiplicities, the
//! m-basis Clebsch-Gordan tensor, F/R symbols with their multiplicity axes, the
//! verification gates, and the gauge fingerprint. Anything a consumer can write
//! in ten lines of Python stays in Python.
//!
//! **Two surfaces, two authorities.** The SU(N) functions (`clebsch_gordan`,
//! `f_symbol`, `r_symbol`) run the generated Gelfand-Tsetlin pipeline and carry
//! [`sun_authority_fingerprint`]. The `su2_*` / `wigner_*` functions are the exact
//! closed-form SU(2) engine -- big-rational arithmetic rounded once -- and carry
//! [`su2_authority_fingerprint`]. SU(2) is reachable through *either*: an `Irrep([2j])`
//! is a rank-1 SU(N) label, so the generated path answers for it too. They are not
//! interchangeable bookkeeping: they agree numerically (issue #107 records the
//! comparison) but are separate authorities with separate fingerprints, and a consumer
//! that persists coefficients must record which one produced them.
//!
//! They are very far apart in cost. The generated path builds a CGC tensor by SVD
//! nullspace, least-squares ladder descent and QR gauge fixing; `su2_r_symbol` is a
//! sign. Issue #107 measured 3.8 ms against ~0 for the same value.

use numpy::ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray4};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use racah_core::sun::{self, SunError};

/// Ill-posed input becomes `ValueError`; a tripped generation/verification gate
/// (orthonormality, unitarity, pentagon, hexagon, factorization) becomes
/// `RuntimeError` — those are numerical failures, not bad arguments.
fn to_py_err(error: SunError) -> PyErr {
    let message = error.to_string();
    match error {
        SunError::EmptyLabel
        | SunError::NotNonincreasing { .. }
        | SunError::NegativeDynkin { .. }
        | SunError::RankMismatch { .. }
        | SunError::ZeroFusionChannel { .. } => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// An irreducible representation of SU(N), labelled by its Dynkin label.
#[pyclass(frozen, from_py_object, name = "Irrep", module = "racah")]
#[derive(Clone)]
struct PyIrrep {
    inner: sun::Irrep,
}

#[pymethods]
impl PyIrrep {
    /// `Irrep(dynkin)` — the SU(N) irrep with Dynkin label `dynkin` (length
    /// `N - 1`, nonnegative).
    #[new]
    fn new(dynkin: Vec<i64>) -> PyResult<Self> {
        sun::Irrep::from_dynkin(&dynkin)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    /// `Irrep.from_weight(weight)` — the irrep with (unnormalized) highest
    /// weight `weight`, length `N`.
    #[classmethod]
    fn from_weight(_cls: &Bound<'_, PyType>, weight: Vec<i64>) -> PyResult<Self> {
        sun::Irrep::from_weight(&weight)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    /// `Irrep.trivial(n)` — the SU(`n`) singlet.
    #[classmethod]
    fn trivial(_cls: &Bound<'_, PyType>, n: usize) -> PyResult<Self> {
        sun::Irrep::trivial(n)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }

    /// The Dynkin label (length `N - 1`).
    #[getter]
    fn dynkin(&self) -> Vec<i64> {
        self.inner.dynkin()
    }

    /// The normalized highest weight (length `N`, nonincreasing, last entry 0).
    #[getter]
    fn weight(&self) -> Vec<i64> {
        self.inner.weight().to_vec()
    }

    /// `N` of the SU(N) this irrep belongs to.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// The Weyl dimension (exact, arbitrary precision).
    fn dim(&self) -> num_bigint::BigInt {
        self.inner.dim()
    }

    /// The dual (conjugate) irrep.
    fn dual(&self) -> Self {
        Self {
            inner: self.inner.dual(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Irrep({:?})", self.inner.dynkin())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// The tensor-product decomposition of `a` and `b` as `[(irrep, multiplicity)]`
/// in the crate's deterministic irrep order.
#[pyfunction]
fn fusion(a: &PyIrrep, b: &PyIrrep) -> PyResult<Vec<(PyIrrep, u32)>> {
    let product = sun::directproduct(&a.inner, &b.inner).map_err(to_py_err)?;
    Ok(product
        .into_iter()
        .map(|(inner, multiplicity)| (PyIrrep { inner }, multiplicity))
        .collect())
}

/// The outer multiplicity `N^c_{ab}`.
#[pyfunction]
fn fusion_multiplicity(a: &PyIrrep, b: &PyIrrep, c: &PyIrrep) -> PyResult<u32> {
    let product = sun::shared_directproduct(&a.inner, &b.inner).map_err(to_py_err)?;
    Ok(product.multiplicity(&c.inner))
}

/// The m-basis Clebsch-Gordan tensor for `s1 ⊗ s2 → s3`, dense, with shape
/// `[dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]`: axes 0–2 index the
/// Gelfand-Tsetlin m-basis states `m1`, `m2`, `m3`; the trailing axis `μ`
/// indexes the outer-multiplicity copies of `s3`.
#[pyfunction]
fn clebsch_gordan<'py>(
    py: Python<'py>,
    s1: &PyIrrep,
    s2: &PyIrrep,
    s3: &PyIrrep,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let cgc = sun::cgc(&s1.inner, &s2.inner, &s3.inner).map_err(to_py_err)?;
    let dims = cgc.dims();
    let mut dense = Array4::<f64>::zeros((dims[0], dims[1], dims[2], dims[3]));
    for entry in cgc.entries() {
        dense[[
            entry.m1 as usize,
            entry.m2 as usize,
            entry.m3 as usize,
            entry.mu as usize,
        ]] = entry.value;
    }
    Ok(dense.into_pyarray(py))
}

/// The F-symbol `F^{abc}_d[e, f]` as a `[μ, ν, κ, λ]` multiplicity block
/// (magnetic indices contracted), shape `[N^e_ab, N^d_ec, N^f_bc, N^d_af]` —
/// one axis per vertex: μ: `a ⊗ b → e`, ν: `e ⊗ c → d`, κ: `b ⊗ c → f`,
/// λ: `a ⊗ f → d` (TensorKitSectors `GenericFusion` axis order).
#[pyfunction]
fn f_symbol<'py>(
    py: Python<'py>,
    a: &PyIrrep,
    b: &PyIrrep,
    c: &PyIrrep,
    d: &PyIrrep,
    e: &PyIrrep,
    f: &PyIrrep,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let block = sun::f_symbol(&a.inner, &b.inner, &c.inner, &d.inner, &e.inner, &f.inner)
        .map_err(to_py_err)?;
    let dims = block.dims();
    let array = Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), block.data().to_vec())
        .expect("FBlock data length matches its dims");
    Ok(array.into_pyarray(py))
}

/// The R-symbol `R^{ab}_c` as an `N^c_{ab} × N^c_{ba}` multiplicity matrix
/// (row: `a ⊗ b → c` vertex copy, column: `b ⊗ a → c` copy).
#[pyfunction]
fn r_symbol<'py>(
    py: Python<'py>,
    a: &PyIrrep,
    b: &PyIrrep,
    c: &PyIrrep,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let block = sun::r_symbol(&a.inner, &b.inner, &c.inner).map_err(to_py_err)?;
    let n = block.dim();
    let array = Array2::from_shape_vec((n, n), block.data().to_vec())
        .expect("RBlock data length matches its dim");
    Ok(array.into_pyarray(py))
}

/// Verify the F-move unitarity gate for `(a, b, c, d)`; raises on violation.
#[pyfunction]
fn check_f_unitarity(a: &PyIrrep, b: &PyIrrep, c: &PyIrrep, d: &PyIrrep) -> PyResult<()> {
    sun::check_f_unitarity(&a.inner, &b.inner, &c.inner, &d.inner).map_err(to_py_err)
}

/// Verify the pentagon identity for `(a, b, c, d)`; raises on violation.
#[pyfunction]
fn check_pentagon(a: &PyIrrep, b: &PyIrrep, c: &PyIrrep, d: &PyIrrep) -> PyResult<()> {
    sun::check_pentagon(&a.inner, &b.inner, &c.inner, &d.inner).map_err(to_py_err)
}

/// Verify both hexagon identities for `(a, b, c)`; raises on violation.
#[pyfunction]
fn check_hexagon(a: &PyIrrep, b: &PyIrrep, c: &PyIrrep) -> PyResult<()> {
    sun::check_hexagon(&a.inner, &b.inner, &c.inner).map_err(to_py_err)
}

/// The SU(N) gauge/authority fingerprint. Embed it next to any persisted
/// coefficients and refuse a mismatch on load.
#[pyfunction]
fn sun_authority_fingerprint() -> String {
    String::from_utf8(sun::sun_authority_fingerprint().to_vec())
        .expect("the fingerprint is an ASCII literal")
}

/// The SU(2) Frobenius-Schur indicator of the irrep with doubled spin `two_j`.
#[pyfunction]
fn su2_frobenius_schur(two_j: u32) -> f64 {
    racah_core::su2_frobenius_schur(two_j)
}

// --- the exact SU(2) surface (issue #107) ------------------------------------------
//
// Spins are doubled throughout (`dj = 2j`, `dm = 2m`), as in the crate: it keeps every
// label an exact integer, which is the whole reason the engine below can be exact.
//
// These are the *infallible* engines. An inadmissible label set is exact zero, not an
// error -- the crate's documented contract, and what a consumer that already guards with
// a triangle test wants. The crate's `*_checked` twins are deliberately not bound yet:
// nothing has asked to distinguish "zero because forbidden" from "zero because zero",
// and each would add a stub, a test and an error mapping. Bind them when something does.

/// The Wigner 3j symbol `(dj1 dj2 dj3; dm1 dm2 dm3)`, spins and projections doubled.
///
/// Exact big-rational arithmetic, rounded once to `float` on return. An inadmissible
/// label set (m-sum, projection bound, or triangle) is exactly `0.0`.
#[pyfunction]
fn wigner_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> f64 {
    racah_core::wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3).to_f64()
}

/// The Wigner 6j symbol `{dj1 dj2 dj3; dj4 dj5 dj6}`, spins doubled.
///
/// Exact big-rational arithmetic, rounded once to `float` on return. A label set
/// violating any of the four triangle conditions is exactly `0.0`.
#[pyfunction]
fn wigner_6j(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> f64 {
    racah_core::wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6).to_f64()
}

/// The SU(2) Clebsch-Gordan coefficient `<dj1 dm1; dj2 dm2 | dj3 dm3>`, doubled labels.
///
/// Condon-Shortley phase, exact big-rational arithmetic rounded once. Note the argument
/// order interleaves each spin with its projection, as the crate's does, and that this is
/// the *scalar* SU(2) coefficient -- [`clebsch_gordan`] is the dense SU(N) m-basis tensor
/// and a different object.
///
/// **The two tiers' CGC differ, and by exactly one sign.** F and R agree between them
/// because a per-channel CGC phase cancels in any gauge-invariant combination; CGC are
/// gauge *data* and do not. Measured over every channel up to `2j = 3`, the ratio is
/// uniform in the magnetic indices and equals `(-1)^(j1 + j2 - j3)`, i.e. exactly
/// [`su2_r_symbol`]:
///
/// ```text
/// clebsch_gordan(Irrep([dj1]), Irrep([dj2]), Irrep([dj3]))[m1, m2, m3, 0]
///     == su2_r_symbol(dj1, dj2, dj3) * su2_clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3)
/// ```
///
/// (with the rank-1 GT basis read as the magnetic basis ascending in m). Mixing the two
/// tiers' CGC without that factor gives wrong signs and no error, so the relation is
/// pinned by a test rather than left to this comment.
#[pyfunction]
fn su2_clebsch_gordan(dj1: u32, dm1: i32, dj2: u32, dm2: i32, dj3: u32, dm3: i32) -> f64 {
    racah_core::clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3).to_f64()
}

/// The SU(2) F-symbol `F^{dj1 dj2 dj3}_{dj4}[dj5, dj6]`, doubled spins.
///
/// Scalar: SU(2) has no outer multiplicity, so the four vertex axes of the SU(N)
/// [`f_symbol`] block are all length 1 and this returns the single entry directly.
#[pyfunction]
fn su2_f_symbol(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> f64 {
    racah_core::su2_f_symbol(dj1, dj2, dj3, dj4, dj5, dj6)
}

/// The SU(2) R-symbol `R^{dj1 dj2}_{dj3} = (-1)^{(j1 + j2 - j3)}`, doubled spins.
///
/// Scalar and exactly `+-1.0`; `0.0` when the triangle condition fails. SU(2) braiding is
/// symmetric, so this is its own inverse.
#[pyfunction]
fn su2_r_symbol(dj1: u32, dj2: u32, dj3: u32) -> f64 {
    racah_core::su2_r_symbol(dj1, dj2, dj3)
}

/// The exact-SU(2) gauge/authority fingerprint -- the twin of
/// [`sun_authority_fingerprint`] for the functions above. A consumer that persists SU(2)
/// coefficients records *this* one, and a consumer that reaches SU(2) through `Irrep`
/// records the SU(N) one; they are different strings on purpose.
#[pyfunction]
fn su2_authority_fingerprint() -> String {
    String::from_utf8(racah_core::su2_authority_fingerprint().to_vec())
        .expect("the fingerprint is an ASCII literal")
}

#[pymodule]
fn racah(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIrrep>()?;
    m.add_function(wrap_pyfunction!(fusion, m)?)?;
    m.add_function(wrap_pyfunction!(fusion_multiplicity, m)?)?;
    m.add_function(wrap_pyfunction!(clebsch_gordan, m)?)?;
    m.add_function(wrap_pyfunction!(f_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(r_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(check_f_unitarity, m)?)?;
    m.add_function(wrap_pyfunction!(check_pentagon, m)?)?;
    m.add_function(wrap_pyfunction!(check_hexagon, m)?)?;
    m.add_function(wrap_pyfunction!(sun_authority_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(su2_frobenius_schur, m)?)?;
    // the exact SU(2) surface (#107)
    m.add_function(wrap_pyfunction!(wigner_3j, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_6j, m)?)?;
    m.add_function(wrap_pyfunction!(su2_clebsch_gordan, m)?)?;
    m.add_function(wrap_pyfunction!(su2_f_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(su2_r_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(su2_authority_fingerprint, m)?)?;
    Ok(())
}
