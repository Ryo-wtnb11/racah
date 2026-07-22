//! Dense factorization seams for the SU(N) CGC pipeline, routed exclusively
//! through `tenferro-linalg`'s public traced surface (no hand-rolled numeric
//! kernels).
//!
//! Three operations are needed, each mapping the reference's dense linear
//! algebra to a public tenferro API:
//!
//! - **nullspace** (`clebschgordan.jl:_nullspace!`, `svd!(A; full=true)`):
//!   full-matrices SVD via [`tenferro_linalg::TracedTensorLinalgExt::svd_full`];
//!   the trailing `n - rank` rows of `Vh` span the right nullspace.
//! - **positive-diagonal QR** (`clebschgordan.jl:qrpos!`): QR with
//!   [`tenferro_linalg::QrGauge::PositiveDiagonal`], returning the sign-fixed
//!   `Q`.
//! - **least squares** (`clebschgordan.jl:lower_weight_CGC!`,
//!   `ldiv!(qr!(eqs), rhs)`): [`tenferro_linalg::TracedTensorLinalgExt::lstsq`].
//!
//! Matrices are passed as plain column-major `f64` buffers so the rest of the
//! crate never sees a tenferro type. The active provider is the CPU faer
//! backend (the only one implementing full-matrices SVD in this tenferro
//! slice), pinned by the `cgc-gen` feature.

use tenferro_cpu::CpuBackend;
use tenferro_linalg::{QrGauge, QrOptions, TracedTensorLinalgExt};
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

use super::SunError;

fn linalg_err(context: &str, e: impl std::fmt::Display) -> SunError {
    SunError::Linalg(format!("{context}: {e}"))
}

fn traced(rows: usize, cols: usize, data: Vec<f64>) -> Result<TracedTensor, SunError> {
    TracedTensor::from_vec_col_major(vec![rows, cols], data)
        .map_err(|e| linalg_err("build input", e))
}

/// Compile and execute the traced outputs on a fresh CPU faer backend, in the
/// order given. Each returned [`Tensor`] is a concrete host tensor.
fn run(outputs: &[&TracedTensor]) -> Result<Vec<Tensor>, SunError> {
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

fn f64_out(t: &Tensor) -> Result<Vec<f64>, SunError> {
    Ok(t.as_slice::<f64>()
        .map_err(|e| linalg_err("read", e))?
        .to_vec())
}

/// A column-major dense matrix of shape `rows x cols`.
pub(crate) struct Mat {
    pub rows: usize,
    pub cols: usize,
    /// Column-major: element `(i, j)` at `data[i + j * rows]`.
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.data[i + j * self.rows] += v;
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.rows]
    }

    fn traced(&self) -> Result<TracedTensor, SunError> {
        traced(self.rows, self.cols, self.data.clone())
    }
}

/// Right nullspace basis of `a` at absolute singular-value tolerance `atol`,
/// returned as the columns of an `a.cols x k` matrix (`k = cols - rank`).
///
/// Ports `_nullspace!(A; atol)`: `rank = #{ s_i > atol }`, and the nullspace is
/// the trailing `Vh` rows (right singular vectors of zero singular value). With
/// `atol > 0` the reference's relative term vanishes (`rtol = … * iszero(atol)`),
/// so the cut is purely `s_i > atol`.
pub(crate) fn nullspace(a: &Mat, atol: f64) -> Result<Mat, SunError> {
    let n = a.cols;
    // Reference `_nullspace!`: an empty system (no rows or no columns) has the
    // whole space as its nullspace -> the n x n identity. Also avoids handing a
    // 0-row matrix to svd_full.
    if a.rows == 0 || n == 0 {
        let mut id = Mat::zeros(n, n);
        for i in 0..n {
            id.data[i + i * n] = 1.0;
        }
        return Ok(id);
    }
    let ta = a.traced()?;
    let (_u, s, vh) = ta.svd_full().map_err(|e| linalg_err("svd_full", e))?;
    let out = run(&[&s, &vh])?;
    let sv = f64_out(&out[0])?;
    let vhd = f64_out(&out[1])?; // column-major n x n
    let rank = sv.iter().filter(|&&x| x > atol).count();
    let k = n - rank;
    // Nullspace vector alpha = row (rank + alpha) of Vh: v[j] = Vh[rank+alpha, j].
    // Column-major (n x n): Vh[t, j] at index t + j * n.
    let mut ns = Mat::zeros(n, k);
    for alpha in 0..k {
        let t = rank + alpha;
        for j in 0..n {
            ns.data[j + alpha * n] = vhd[t + j * n];
        }
    }
    Ok(ns)
}

/// Positive-diagonal `Q` of the thin QR of `a` (`qrpos!` returning `first`).
///
/// Ports `qrpos!(C)`: QR with the R-diagonal signs folded into `Q` so each
/// `R[i,i] >= 0`; the gauge-fixing step keeps only `Q` (`gaugefix! = first ∘
/// qrpos! ∘ cref!`). `Q` has shape `a.rows x min(a.rows, a.cols)`.
pub(crate) fn qr_positive_q(a: &Mat) -> Result<Mat, SunError> {
    let ta = a.traced()?;
    let (q, _r) = ta
        .qr_with_options(QrOptions::default().gauge(QrGauge::PositiveDiagonal))
        .map_err(|e| linalg_err("qr", e))?;
    let out = run(&[&q])?;
    let qd = f64_out(&out[0])?;
    let cols = a.rows.min(a.cols);
    Ok(Mat {
        rows: a.rows,
        cols,
        data: qd,
    })
}

/// Least-squares solution `X` of `A X = B` for tall/square full-column-rank
/// `A` (`ldiv!(qr!(A), B)`). `X` has shape `A.cols x B.cols`.
pub(crate) fn lstsq(a: &Mat, b: &Mat) -> Result<Mat, SunError> {
    let ta = a.traced()?;
    let tb = b.traced()?;
    let x = ta.lstsq(&tb).map_err(|e| linalg_err("lstsq", e))?;
    let out = run(&[&x])?;
    let xd = f64_out(&out[0])?;
    Ok(Mat {
        rows: a.cols,
        cols: b.cols,
        data: xd,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullspace_of_1x2_recovers_kernel() {
        // A = [1, 1] (1x2). Kernel spanned by (1, -1)/sqrt2. Thin SVD would drop
        // this; full SVD via svd_full must recover it -- the minimal SU(2)
        // singlet system.
        let a = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0, 1.0],
        };
        let ns = nullspace(&a, 1e-13).unwrap();
        assert_eq!((ns.rows, ns.cols), (2, 1));
        let (v0, v1) = (ns.at(0, 0), ns.at(1, 0));
        assert!((v0 + v1).abs() < 1e-10, "A v = {}", v0 + v1);
        assert!((v0 * v0 + v1 * v1 - 1.0).abs() < 1e-10, "not unit");
    }

    #[test]
    fn qr_positive_q_is_orthonormal_with_positive_r_diagonal() {
        // Single column (2x1): Q must be that column normalized, oriented so the
        // (only) R diagonal is >= 0, i.e. Q points the same way as the input.
        let a = Mat {
            rows: 2,
            cols: 1,
            data: vec![-3.0, 4.0],
        };
        let q = qr_positive_q(&a).unwrap();
        assert_eq!((q.rows, q.cols), (2, 1));
        // R[0,0] = <q, a> must be >= 0.
        let r00 = q.at(0, 0) * a.at(0, 0) + q.at(1, 0) * a.at(1, 0);
        assert!(r00 >= 0.0, "R diagonal not positive: {r00}");
        assert!((q.at(0, 0).powi(2) + q.at(1, 0).powi(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lstsq_recovers_consistent_tall_system() {
        // A (3x2) full column rank, b = A x_true -> lstsq recovers x_true.
        let a = Mat {
            rows: 3,
            cols: 2,
            data: vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], // col-major
        };
        let x_true = [2.0, -1.0];
        let mut bdata = vec![0.0; 3];
        for (i, bi) in bdata.iter_mut().enumerate() {
            for (j, xj) in x_true.iter().enumerate() {
                *bi += a.at(i, j) * xj;
            }
        }
        let b = Mat {
            rows: 3,
            cols: 1,
            data: bdata,
        };
        let x = lstsq(&a, &b).unwrap();
        assert_eq!((x.rows, x.cols), (2, 1));
        assert!((x.at(0, 0) - 2.0).abs() < 1e-9);
        assert!((x.at(1, 0) + 1.0).abs() < 1e-9);
    }
}
