//! Python bindings for the `racah` SU(N) surface (issue #81).
//!
//! Deliberately thin: irrep labels, fusion with outer multiplicities, the
//! m-basis Clebsch-Gordan tensor, F/R symbols with their multiplicity axes, the
//! verification gates, and the gauge fingerprint. Anything a consumer can write
//! in ten lines of Python stays in Python.

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
    Ok(())
}
