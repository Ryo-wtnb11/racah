//! SU(N) F- and R-symbols derived from Layer 2 Clebsch-Gordan coefficients
//! (Layer 3 of the `cgc-gen` track).
//!
//! Ported from SUNRepresentations.jl v0.4.0 `src/sector.jl`:
//! - [`f_symbol`] ports `_Fsymbol` (`sector.jl:58-89`): the F-symbol as the
//!   contraction of four CGC over all magnetic indices, leaving the four outer
//!   multiplicity indices `[μ, ν, κ, λ]`.
//! - [`r_symbol`] ports `_Rsymbol` (`sector.jl:91-110`): the braiding matrix.
//!
//! The four-CGC contraction, the `[μ, ν, κ, λ]` axis order (TensorKitSectors
//! `sectors.jl:Fsymbol_from_fusiontensor`, `:406-418`), and the pentagon/hexagon/
//! F-unitarity gates all live in the **family-generic** `frcore` core
//! (issue #27); this module is the SU(N) binding of that core — the
//! [`SunFamily`] provider plus the per-family public API. The SU(N) behavior is
//! unchanged by the genericization: [`SunFamily`] returns exactly the same
//! multiplicities and CGC entries the old inline contraction consumed.
//!
//! # Conjugation
//!
//! `_Fsymbol`/`_Rsymbol` conjugate two of the CGC. SUNRepresentations' SU(N) CGC
//! are real `Float64` in the standard gauge (`sectorscalartype = Float64`), so
//! conjugation is the identity and is elided; the port is value-identical (see
//! the `frcore` core).

use std::sync::Arc;

use super::{cgc, directproduct, Irrep, SunError};
use crate::frcore::{
    self, f_block_raw, f_unitarity_residual, hexagon_residual, pentagon_residual, r_block_raw,
    Family, MEntry,
};

pub use crate::frcore::{FBlock, RBlock};

/// The SU(N) binding of the generic F/R core: a stateless zero-sized provider.
///
/// Its `&mut self` methods (required by [`Family`]) delegate to the free
/// functions [`cgc`] / [`directproduct`], which are backed by the process-global
/// CGC cache. The `&mut` is vacuous here — no interior mutability, no lock — so
/// the shared core's `&mut`-provider seam (needed by the `&mut CanonicalCatalog`
/// B/C/D provider) costs SU(N) nothing.
struct SunFamily;

impl Family for SunFamily {
    type Irrep = Irrep;
    type Error = SunError;

    fn mult(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, SunError> {
        mult(a, b, c)
    }

    fn cgc_entries(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<Vec<MEntry>, SunError> {
        Ok(cgc(a, b, c)?
            .entries()
            .iter()
            .map(|e| MEntry {
                m1: e.m1,
                m2: e.m2,
                m3: e.m3,
                mu: e.mu,
                value: e.value,
            })
            .collect())
    }

    fn products(&mut self, a: &Irrep, b: &Irrep) -> Result<Vec<Irrep>, SunError> {
        Ok(directproduct(a, b)?.into_keys().collect())
    }
}

// ---------------------------------------------------------------------------
// Multiplicity / rank helpers (the reference's compile-time `SUNIrrep{N}` and
// `Nsymbol` guards, re-erected as runtime typed errors -- issue #15).
// ---------------------------------------------------------------------------

/// `N^c_{ab}`, the fusion multiplicity of `a ⊗ b → c`. Errors
/// [`SunError::RankMismatch`] if `a`, `b`, `c` are not all SU(N) for one `N`
/// (the reference relies on the `SUNIrrep{N}` type parameter for this).
fn mult(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, SunError> {
    if c.rank() != a.rank() {
        return Err(SunError::RankMismatch {
            a: a.rank(),
            b: c.rank(),
        });
    }
    Ok(directproduct(a, b)?.get(c).copied().unwrap_or(0) as usize)
}

/// All labels of an F/R request share one rank, or [`SunError::RankMismatch`].
fn require_same_rank(labels: &[&Irrep]) -> Result<(), SunError> {
    let n = labels[0].rank();
    for s in &labels[1..] {
        if s.rank() != n {
            return Err(SunError::RankMismatch { a: n, b: s.rank() });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// F-symbol.
// ---------------------------------------------------------------------------

/// The F-symbol $F^{abc}_d[e, f]$ as a dense $[\mu, \nu, \kappa, \lambda]$ block.
///
/// Ports `sector.jl:_Fsymbol` (the contraction lives in the `frcore` core). The
/// four vertices are $a\otimes b\to e$ ($\mu$), $e\otimes c\to d$ ($\nu$),
/// $b\otimes c\to f$ ($\kappa$), $a\otimes f\to d$ ($\lambda$).
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if the six labels are not all SU(N) for one `N`.
/// - [`SunError::ZeroFusionChannel`] if any of the four vertices is empty. (The
///   reference returns an all-zero block here; this query API returns a typed
///   error — issue #15.)
/// - [`SunError::NullspaceDimMismatch`] / [`SunError::NotOrthonormal`] /
///   [`SunError::LadderInconsistent`] / [`SunError::Linalg`] surfaced from an
///   underlying CGC generation.
pub fn f_symbol(
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
    e: &Irrep,
    f: &Irrep,
) -> Result<FBlock, SunError> {
    require_same_rank(&[a, b, c, d, e, f])?;
    // Guard: every vertex must be non-empty (the reference's
    // `Nsymbol(...) == 0 && return zeros` short-circuit becomes a typed error).
    let vertices = [
        (a, b, e), // N1 = μ
        (e, c, d), // N2 = ν
        (b, c, f), // N3 = κ
        (a, f, d), // N4 = λ
    ];
    for (x, y, z) in vertices {
        if mult(x, y, z)? == 0 {
            return Err(SunError::ZeroFusionChannel {
                a: x.dynkin(),
                b: y.dynkin(),
                c: z.dynkin(),
            });
        }
    }

    // Derived-f64 cache: key = the plain ordered six-label tuple.
    //
    // Why no Regge-style canonicalization: the 6j symmetry group that lets the
    // SU(2) F cache key on a canonical class (`su2.rs:FKey`) has no analogue for
    // GT-basis SU(N) F blocks — there is no implemented tetrahedral/Regge
    // canonical form for a rank-4 multiplicity block in this gauge, and the
    // block is not invariant under the naive label permutations anyway. So the
    // key is the six labels as given; distinct requests never collide, and no
    // canonicalization can silently merge two genuinely different blocks.
    let cache = crate::cache::cache_sun_f();
    let key = (
        a.clone(),
        b.clone(),
        c.clone(),
        d.clone(),
        e.clone(),
        f.clone(),
    );
    if let Some(hit) = cache.get(&key) {
        return Ok((*hit).clone());
    }
    let block = f_block_raw(&mut SunFamily, a, b, c, d, e, f)?;
    let stored = cache.insert(key, Arc::new(block));
    Ok((*stored).clone())
}

// ---------------------------------------------------------------------------
// R-symbol.
// ---------------------------------------------------------------------------

/// The R-symbol $R^{ab}_c$ as a dense $N^c_{ab} \times N^c_{ba}$ matrix.
///
/// Ports `sector.jl:_Rsymbol` (the contraction lives in the `frcore` core).
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if `a`, `b`, `c` are not all SU(N) for one `N`.
/// - [`SunError::ZeroFusionChannel`] if $a \otimes b \to c$ is empty.
/// - CGC generation errors are surfaced.
pub fn r_symbol(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<RBlock, SunError> {
    require_same_rank(&[a, b, c])?;
    if mult(a, b, c)? == 0 {
        return Err(SunError::ZeroFusionChannel {
            a: a.dynkin(),
            b: b.dynkin(),
            c: c.dynkin(),
        });
    }
    r_block_raw(&mut SunFamily, a, b, c)
}

// ---------------------------------------------------------------------------
// Gates (shipped as public API: generation gates and oracle harnesses).
// ---------------------------------------------------------------------------

/// Verify that the F-move for fixed outer labels `(a, b, c, d)` is unitary
/// (`M Mᵀ = I` over the two orthonormal associativity bases). See
/// the `frcore` unitarity gate for the matrix layout.
///
/// # Errors
///
/// [`SunError::FNotUnitary`] with the worst `|(M Mᵀ - I)_{ij}|` if the gate
/// fails; [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_f_unitarity(a: &Irrep, b: &Irrep, c: &Irrep, d: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c, d])?;
    let worst = f_unitarity_residual(&mut SunFamily, a, b, c, d)?;
    if worst > frcore::TOL_F_UNITARY {
        return Err(SunError::FNotUnitary { residual: worst });
    }
    Ok(())
}

/// Verify the pentagon identity for the quadruple `(a, b, c, d)` (see
/// the `frcore` pentagon gate).
///
/// # Errors
///
/// [`SunError::PentagonViolation`] (worst residual) on failure;
/// [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_pentagon(a: &Irrep, b: &Irrep, c: &Irrep, d: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c, d])?;
    let worst = pentagon_residual(&mut SunFamily, a, b, c, d)?;
    if worst > frcore::TOL_PENTAGON {
        return Err(SunError::PentagonViolation { residual: worst });
    }
    Ok(())
}

/// Verify both hexagon identities for the triple `(a, b, c)` (see
/// the `frcore` hexagon gate).
///
/// # Errors
///
/// [`SunError::HexagonViolation`] (worst residual) on failure;
/// [`SunError::RankMismatch`] on mixed ranks; CGC errors surfaced.
pub fn check_hexagon(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<(), SunError> {
    require_same_rank(&[a, b, c])?;
    let worst = hexagon_residual(&mut SunFamily, a, b, c)?;
    if worst > frcore::TOL_HEXAGON {
        return Err(SunError::HexagonViolation { residual: worst });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(d: &[i64]) -> Irrep {
        Irrep::from_dynkin(d).unwrap()
    }

    // ---- guard inventory: red-first ill-posed inputs ----

    #[test]
    fn f_symbol_zero_vertex_is_typed_error() {
        // SU(3): a⊗b→e with e ∉ a⊗b. 3⊗3 = 6 ⊕ 3̄, so e = 8 is empty.
        let three = irr(&[1, 0]);
        let eight = irr(&[1, 1]);
        let err = f_symbol(&three, &three, &three, &three, &eight, &three).unwrap_err();
        assert!(matches!(err, SunError::ZeroFusionChannel { .. }));
    }

    #[test]
    fn f_symbol_rank_mismatch_is_typed_error() {
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        let err = f_symbol(&su3, &su3, &su3, &su3, &su4, &su3).unwrap_err();
        assert!(matches!(err, SunError::RankMismatch { .. }));
    }

    #[test]
    fn r_symbol_zero_vertex_is_typed_error() {
        // 3 ⊗ 3 → 8 is empty (3⊗3 = 6 ⊕ 3̄).
        let three = irr(&[1, 0]);
        let eight = irr(&[1, 1]);
        let err = r_symbol(&three, &three, &eight).unwrap_err();
        assert!(matches!(err, SunError::ZeroFusionChannel { .. }));
    }

    #[test]
    fn r_symbol_rank_mismatch_is_typed_error() {
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        let err = r_symbol(&su3, &su4, &su3).unwrap_err();
        assert!(matches!(err, SunError::RankMismatch { .. }));
    }

    // ---- shapes ----

    #[test]
    fn su3_trivial_f_is_scalar_one() {
        // F with a = trivial: F^{1,b,c}_d[e=b, f=c] should be the identity
        // scalar (1×1×1×1 block, value 1) for admissible b,c.
        // a = 1: e = b = 3 forced, f = d ∈ 3⊗3 (take 6), so all vertices hold.
        let triv = Irrep::trivial(3).unwrap();
        let three = irr(&[1, 0]);
        let six = irr(&[2, 0]);
        let block = f_symbol(&triv, &three, &three, &six, &three, &six).unwrap();
        assert_eq!(block.dims(), [1, 1, 1, 1]);
        assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn su3_octet_cubed_f_block_is_2x2x2x2() {
        // 8⊗8 → 8 has multiplicity 2, so F(8,8,8,8,8,8) is a 2×2×2×2 block.
        let eight = irr(&[1, 1]);
        let block = f_symbol(&eight, &eight, &eight, &eight, &eight, &eight).unwrap();
        assert_eq!(block.dims(), [2, 2, 2, 2]);
    }

    #[test]
    fn su3_octet_r_block_is_2x2() {
        // 8⊗8 → 8 : R is a 2×2 braiding matrix.
        let eight = irr(&[1, 1]);
        let block = r_symbol(&eight, &eight, &eight).unwrap();
        assert_eq!(block.dim(), 2);
    }

    // ---- gates on small SU(3) families (self-consistency) ----

    #[test]
    fn su3_f_unitarity_multiplicity_free() {
        // 3⊗3⊗3 → various d: multiplicity-free F-move must be orthogonal.
        let three = irr(&[1, 0]);
        check_f_unitarity(&three, &three, &three, &irr(&[1, 0])).unwrap();
    }

    #[test]
    fn su3_f_unitarity_with_multiplicity() {
        // 8⊗8⊗8 → 8: the F-move mixes multiplicity indices; still orthogonal.
        let eight = irr(&[1, 1]);
        check_f_unitarity(&eight, &eight, &eight, &eight).unwrap();
    }

    #[test]
    fn su3_pentagon_multiplicity_free() {
        let three = irr(&[1, 0]);
        check_pentagon(&three, &three, &three, &three).unwrap();
    }

    #[test]
    fn su3_hexagon_multiplicity_free() {
        let three = irr(&[1, 0]);
        check_hexagon(&three, &three, &three).unwrap();
    }
}
