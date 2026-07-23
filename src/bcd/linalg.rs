//! Dense factorization / contraction seams for the S3.2 decomposition sweep,
//! routed through `tenferro-linalg`'s public traced surface (no hand-rolled
//! numeric kernels — `AGENTS.md`).
//!
//! Two operations reach the backend:
//!
//! - **positive-diagonal QR** (QSpace `OrthoNormalizeColsQR`,
//!   `wbsparray.cc`): column orthonormalization via
//!   [`tenferro_linalg::QrGauge::PositiveDiagonal`] — the deliberate racah
//!   tightening of QSpace's unspecified QR sign (documented in
//!   `docs/gauge_soN.md`). Returns the sign-fixed `Q`.
//! - **matmul** (QSpace `Wb::MatProd`): plain `A·B` via
//!   [`tenferro_runtime::TracedTensor::matmul`], used for the block-level CGC
//!   contractions that build the projected generators and the `U†U` isometry
//!   check.
//!
//! The mirror of `sun::linalg`, at the B/C/D layer and with [`SweepError`]. The
//! two seams stay separate (different error type, different layer); unifying the
//! ~40 lines of tenferro plumbing is a later refactor, not this PR's scope.
//! `ponytail:` shared tenferro-plumbing helper, extract when a third consumer
//! appears.

use tenferro_cpu::CpuBackend;
use tenferro_linalg::{QrGauge, QrOptions, TracedTensorLinalgExt};
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

use super::sweep::SweepError;

fn linalg_err(context: &str, e: impl std::fmt::Display) -> SweepError {
    SweepError::Linalg(format!("{context}: {e}"))
}

/// A column-major dense `rows × cols` matrix of `f64` — the plain buffer the
/// rest of the sweep manipulates so no tenferro type escapes this module.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dense {
    pub rows: usize,
    pub cols: usize,
    /// Column-major: element `(i, j)` at `data[i + j * rows]`.
    pub data: Vec<f64>,
}

impl Dense {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Dense {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.rows]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i + j * self.rows] = v;
    }

    /// Column `j` as a slice.
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// The `k`-th column of the identity, as a `rows × 1` matrix.
    pub fn unit(rows: usize, k: usize) -> Self {
        let mut m = Dense::zeros(rows, 1);
        m.data[k] = 1.0;
        m
    }

    /// Conjugate (real: plain) transpose.
    pub fn transpose(&self) -> Dense {
        let mut t = Dense::zeros(self.cols, self.rows);
        for j in 0..self.cols {
            for i in 0..self.rows {
                t.data[j + i * self.cols] = self.data[i + j * self.rows];
            }
        }
        t
    }

    /// Append the columns of `other` (same `rows`) to the right.
    pub fn cat_cols(&mut self, other: &Dense) {
        debug_assert_eq!(self.rows, other.rows, "cat_cols row mismatch");
        self.data.extend_from_slice(&other.data);
        self.cols += other.cols;
    }

    /// A view of `self` keeping only the columns in `keep` (in order).
    pub fn select_cols(&self, keep: &[usize]) -> Dense {
        let mut out = Dense::zeros(self.rows, keep.len());
        for (jo, &j) in keep.iter().enumerate() {
            out.data[jo * self.rows..(jo + 1) * self.rows].copy_from_slice(self.col(j));
        }
        out
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

fn traced(m: &Dense) -> Result<TracedTensor, SweepError> {
    TracedTensor::from_vec_col_major(vec![m.rows, m.cols], m.data.clone())
        .map_err(|e| linalg_err("build input", e))
}

fn run(outputs: &[&TracedTensor]) -> Result<Vec<Tensor>, SweepError> {
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(outputs)
        .map_err(|e| linalg_err("compile", e))?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .map_err(|e| linalg_err("register", e))?;
    executor
        .run_many(&program)
        .map_err(|e| linalg_err("run", e))
}

fn f64_out(t: &Tensor) -> Result<Vec<f64>, SweepError> {
    Ok(t.as_slice::<f64>()
        .map_err(|e| linalg_err("read", e))?
        .to_vec())
}

/// `A · B` (plain matmul), routed through the tenferro traced surface.
pub(crate) fn matmul(a: &Dense, b: &Dense) -> Result<Dense, SweepError> {
    debug_assert_eq!(a.cols, b.rows, "matmul inner-dim mismatch");
    if a.rows == 0 || b.cols == 0 || a.cols == 0 {
        return Ok(Dense::zeros(a.rows, b.cols));
    }
    let ta = traced(a)?;
    let tb = traced(b)?;
    let tc = ta.matmul(&tb).map_err(|e| linalg_err("matmul", e))?;
    let out = run(&[&tc])?;
    Ok(Dense {
        rows: a.rows,
        cols: b.cols,
        data: f64_out(&out[0])?,
    })
}

/// `Aᵀ · B` (`A†B` for real `A`), the projection contraction. One matmul.
pub(crate) fn tmatmul(a: &Dense, b: &Dense) -> Result<Dense, SweepError> {
    matmul(&a.transpose(), b)
}

/// Thin SVD `A = U · diag(s) · Vᵀ`, routed through the tenferro traced surface.
///
/// Returns `(U, s, Vt)` with `U` (`rows × k`), `s` (length `k`), `Vt` (`k × cols`),
/// `k = min(rows, cols)`. The only consumer is the small per-weight-space
/// orthogonal Procrustes solve in intertwiner alignment (issue #29): for a square
/// `M`, `W = U · Vt` is the nearest orthogonal matrix to `M`. No hand-rolled
/// kernel — the decision to route factorizations through the backend is `AGENTS.md`.
pub(crate) fn svd(a: &Dense) -> Result<(Dense, Vec<f64>, Dense), SweepError> {
    let k = a.rows.min(a.cols);
    if k == 0 {
        return Ok((Dense::zeros(a.rows, 0), Vec::new(), Dense::zeros(0, a.cols)));
    }
    let ta = traced(a)?;
    let (u, s, vt) = ta.svd().map_err(|e| linalg_err("svd", e))?;
    let out = run(&[&u, &s, &vt])?;
    let u = Dense {
        rows: a.rows,
        cols: k,
        data: f64_out(&out[0])?,
    };
    let s = f64_out(&out[1])?;
    let vt = Dense {
        rows: k,
        cols: a.cols,
        data: f64_out(&out[2])?,
    };
    Ok((u, s, vt))
}

/// Rank-revealing orthonormalization of the columns of `a` (QSpace
/// `OrthoNormalizeColsQR(FL, tol)`): an orthonormal basis of `a`'s column space,
/// dropping linearly dependent/zero columns so a rank-deficient input never
/// contributes a spurious vector. Gauge:
/// [`tenferro_linalg::QrGauge::PositiveDiagonal`] (the documented racah
/// tightening).
///
/// Rank is read from `R = QᵀA` by **row** norm, not the diagonal: tenferro's QR
/// is *un-pivoted*, so a zero/dependent leading column shifts the pivots off the
/// diagonal (e.g. a rank-2 input can yield an all-zero `R` diagonal with the
/// content on the superdiagonal).
///
/// Guarantees (this is *not* the theorem "row `i` nonzero iff `Q[:,i] ∈ col(A)`",
/// which a `τ = 0` Householder reflector can violate mid-matrix):
/// - keeping `{ i : ‖R[i,·]‖ > tol }` loses **no genuine direction** —
///   `A = Q[:,keep]·R[keep,:]` up to `keep.len()·tol`, so the whole column space
///   is spanned;
/// - a spurious retained column cannot arise for a **trailing** dependency (the
///   only case the descent produces — the dependent columns are the later
///   lowering images); were one ever retained elsewhere, it inflates the block
///   past its irrep dimension and is caught **loudly** by the caller's Cartan
///   diagonality / `U†U` / dimension gates, never silently. QSpace's R-staircase
///   check is therefore not ported (it would duplicate that loud coverage).
pub(crate) fn qr_positive_q(a: &Dense, tol: f64) -> Result<Dense, SweepError> {
    let ta = traced(a)?;
    let (q, rr) = ta
        .qr_with_options(QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .map_err(|e| linalg_err("qr", e))?;
    let out = run(&[&q, &rr])?;
    let qdata = f64_out(&out[0])?;
    let rdata = f64_out(&out[1])?;
    let k = a.rows.min(a.cols); // Q is rows × k, R is k × a.cols
    let q = Dense {
        rows: a.rows,
        cols: k,
        data: qdata,
    };
    // R is column-major k × a.cols; row i norm = Σ_c R[i,c]² over c.
    let keep: Vec<usize> = (0..k)
        .filter(|&i| {
            (0..a.cols)
                .map(|c| rdata[i + c * k].powi(2))
                .sum::<f64>()
                .sqrt()
                > tol
        })
        .collect();
    Ok(q.select_cols(&keep))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_and_procrustes_recovers_rotation() {
        // A known 2×2, plus: the nearest-orthogonal of a rotation IS that rotation.
        let m = Dense {
            rows: 2,
            cols: 2,
            data: vec![2.0, 0.0, 0.0, 3.0], // diag(2,3), column-major
        };
        let (u, s, vt) = svd(&m).unwrap();
        // reconstruct U·diag(s)·Vt.
        let mut sd = Dense::zeros(2, 2);
        for (i, &sv) in s.iter().enumerate() {
            sd.set(i, i, sv);
        }
        let recon = matmul(&matmul(&u, &sd).unwrap(), &vt).unwrap();
        for i in 0..4 {
            assert!((recon.data[i] - m.data[i]).abs() < 1e-12);
        }
        // Procrustes on a pure rotation returns the rotation itself (s ≈ [1,1]).
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let rot = Dense {
            rows: 2,
            cols: 2,
            data: vec![c, c, -c, c], // [[c,-c],[c,c]] column-major
        };
        let (u, _s, vt) = svd(&rot).unwrap();
        let w = matmul(&u, &vt).unwrap();
        for i in 0..4 {
            assert!((w.data[i] - rot.data[i]).abs() < 1e-12);
        }
    }
}
