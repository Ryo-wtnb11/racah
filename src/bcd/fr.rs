//! B/C/D (SO(2r+1)/Sp(2r)/SO(2r)) F- and R-symbols from catalog-driven CGC
//! (Stage 3 S3.4; design authority: issue #18; spec: issue #27).
//!
//! The four-CGC contraction and the pentagon/hexagon/F-unitarity gates are the
//! **family-generic** [`crate::frcore`] core, shared with `crate::sun::fr`; this
//! module is the B/C/D binding of that core. The provider is [`BcdFamily`],
//! wrapping a `&mut CanonicalCatalog`: it materializes canonical generator bases
//! on demand and adapts each dense [`CatalogCgc`] isometry into the sparse
//! `(m1, m2, m3, mu, value)` entries the core consumes.
//!
//! # Magnetic-index decomposition (the Kronecker gauge)
//!
//! A [`CatalogCgc`] copy is a dense column-major `d1·d2 × d3` isometry. Its row
//! index is the product-basis index `m1 + d1·m2` — the **first factor fast** —
//! fixed by `Generators::product` (QSpace `wbsparray::setRec_kron`, and pinned in
//! `docs/gauge_soN.md`); the column is the coupled index `m3`. So a row `row`
//! decomposes as `m1 = row % d1`, `m2 = row / d1`, with `d1 = dim(s1)`. Because
//! the catalog stores exactly one canonical generator set per irrep, a factor's
//! magnetic index means the same basis state across every CGC it appears in — the
//! shared-index joins in the core (`ma` in `CGC(a,b,e)` vs `CGC(a,f,d)`, etc.) are
//! therefore consistent.
//!
//! # Real-valued CGC
//!
//! The sweep produces real orthogonal isometries, so the reference conjugations
//! are the identity and elided (as for SU(N)); see [`crate::frcore`].

use std::sync::Arc;

use super::catalog::CatalogCgc;
use super::{directproduct, CanonicalCatalog, CatalogError, Irrep};
use crate::frcore::{
    self, f_block_raw, f_unitarity_residual, hexagon_residual, pentagon_residual, r_block_raw,
    Family, MEntry,
};

pub use crate::frcore::{FBlock, RBlock};

/// Failure of a B/C/D F/R request or verification gate.
///
/// Wraps the catalog's [`CatalogError`] (materialization, budget, sweep-gate,
/// wrong-group, and the red-first zero-fusion-channel guard) and adds the three
/// self-consistency gate violations. Kept separate from [`CatalogError`] so the
/// catalog's contract stays "a catalog request outcome"; the gate residuals are a
/// property of the F/R algebra, not of a catalog lookup.
///
/// Not `Eq`: several variants carry an `f64` residual.
#[derive(Clone, Debug, PartialEq)]
pub enum FrError {
    /// An underlying catalog request failed (includes
    /// [`CatalogError::ZeroFusionChannel`] for an ill-posed vertex and
    /// [`CatalogError::WrongGroup`] for a foreign label — the guard-inventory
    /// typed errors, issue #15).
    Catalog(CatalogError),
    /// The F-move matrix (rows `(e, μ, ν)`, cols `(f, κ, λ)` for fixed outer
    /// labels `a, b, c, d`) failed the unitarity gate. Worst `|(M Mᵀ - I)_{ij}|`.
    FNotUnitary {
        /// Worst unitarity residual.
        residual: f64,
    },
    /// The pentagon identity spot check exceeded tolerance. Worst residual.
    PentagonViolation {
        /// Worst pentagon residual.
        residual: f64,
    },
    /// A hexagon identity spot check exceeded tolerance. Worst residual.
    HexagonViolation {
        /// Worst hexagon residual.
        residual: f64,
    },
}

impl std::fmt::Display for FrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrError::Catalog(e) => write!(f, "{e}"),
            FrError::FNotUnitary { residual } => {
                write!(f, "B/C/D F-move matrix not unitary (residual {residual:e})")
            }
            FrError::PentagonViolation { residual } => {
                write!(
                    f,
                    "B/C/D pentagon identity violated (residual {residual:e})"
                )
            }
            FrError::HexagonViolation { residual } => {
                write!(f, "B/C/D hexagon identity violated (residual {residual:e})")
            }
        }
    }
}

impl std::error::Error for FrError {}

impl From<CatalogError> for FrError {
    fn from(e: CatalogError) -> Self {
        FrError::Catalog(e)
    }
}

/// Count of product-decomposition sweeps actually run (tier misses). A
/// performance-contract counter: the F/R gates route every CGC request through
/// the value tier, so this rises once per distinct `(s1, s2)` product, not once
/// per `cgc_entries` call. Exercised by the `warm_state_does_not_resweep` test.
pub(crate) static CGC_SWEEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The B/C/D binding of the generic F/R core: a `&mut CanonicalCatalog` provider.
///
/// The `&mut` is real here (materializing a canonical-parent chain mutates the
/// append-only catalog), which is why the [`Family`] seam takes `&mut self`.
struct BcdFamily<'a> {
    cat: &'a mut CanonicalCatalog,
}

impl Family for BcdFamily<'_> {
    type Irrep = Irrep;
    type Error = CatalogError;

    fn mult(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, CatalogError> {
        bcd_mult(a, b, c)
    }

    /// Sparse CGC entries for `a ⊗ b → c`, served from the process-global B/C/D
    /// CGC value tier ([`crate::cache::cache_bcd_cgc`]).
    ///
    /// On a miss the whole `a ⊗ b` product is decomposed **once**
    /// ([`CanonicalCatalog::cgc_product`]) and every coupled channel is cached, so
    /// the gates' many `(a, b, ·)` requests share one sweep instead of
    /// re-sweeping per coupled irrep (issue #27 P1 review).
    fn cgc_entries(
        &mut self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
    ) -> Result<Vec<MEntry>, CatalogError> {
        let tier = crate::cache::cache_bcd_cgc();
        if let Some(hit) = tier.get(&(a.clone(), b.clone(), c.clone())) {
            return Ok(sparse_entries(a, &hit));
        }
        // Miss: one sweep for the whole product; cache every channel.
        CGC_SWEEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let channels = self.cat.cgc_product(a, b)?;
        let mut wanted: Option<Arc<CatalogCgc>> = None;
        for ch in channels {
            let key = (a.clone(), b.clone(), ch.s3().clone());
            let stored = tier.insert(key, Arc::new(ch));
            if stored.s3() == c {
                wanted = Some(stored);
            }
        }
        let cgc = wanted.expect("c is a coupled channel of a⊗b (mult>0 checked by caller)");
        Ok(sparse_entries(a, &cgc))
    }

    fn products(&mut self, a: &Irrep, b: &Irrep) -> Result<Vec<Irrep>, CatalogError> {
        Ok(directproduct(a, b)?.into_keys().collect())
    }
}

/// `N^c_{ab}` from the exact S3.0 decomposition (no float work). A group/rank
/// mismatch surfaces as [`CatalogError::Label`] via [`super::BcdError`].
fn bcd_mult(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, CatalogError> {
    Ok(directproduct(a, b)?.get(c).copied().unwrap_or(0) as usize)
}

/// Adapt a dense [`CatalogCgc`] into sparse [`MEntry`] entries, decomposing each
/// column-major row index into `(m1, m2)` per the Kronecker gauge (module docs).
fn sparse_entries(s1: &Irrep, cgc: &CatalogCgc) -> Vec<MEntry> {
    let (rows, d3) = cgc.copy_shape();
    // d1 = dim(s1); the row index is m1 + d1·m2 (first factor fast).
    let d1 = usize::try_from(s1.dim()).expect("irrep dim fits usize for tractable ranks");
    let mult = cgc.multiplicity();
    let mut out = Vec::new();
    for mu in 0..mult {
        let copy = cgc.copy(mu);
        for col in 0..d3 {
            for row in 0..rows {
                // ponytail: keep every exact-nonzero coefficient; no purge
                // threshold, so no arbitrary cutoff can drop a small-but-real
                // entry. The dense buffer is small for the ranks in scope.
                let v = copy[col * rows + row];
                if v != 0.0 {
                    out.push(MEntry {
                        m1: (row % d1) as u32,
                        m2: (row / d1) as u32,
                        m3: col as u32,
                        mu: mu as u32,
                        value: v,
                    });
                }
            }
        }
    }
    out
}

/// Verify every label belongs to the catalog's family (series and rank),
/// red-first, before any materialization — the guard-inventory series/rank
/// mismatch check (issue #15).
fn require_catalog_family(cat: &CanonicalCatalog, labels: &[&Irrep]) -> Result<(), CatalogError> {
    for s in labels {
        if s.series() != cat.series() || s.rank() != cat.rank() {
            return Err(CatalogError::WrongGroup {
                catalog: (cat.series(), cat.rank()),
                got: (s.series(), s.rank()),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// F-symbol.
// ---------------------------------------------------------------------------

/// The B/C/D F-symbol $F^{abc}_d[e, f]$ as a dense $[\mu, \nu, \kappa, \lambda]$
/// block, over the catalog's canonical CGC gauge.
///
/// The four vertices are $a\otimes b\to e$ ($\mu$), $e\otimes c\to d$ ($\nu$),
/// $b\otimes c\to f$ ($\kappa$), $a\otimes f\to d$ ($\lambda$). Cached in the
/// derived-f64 B/C/D F tier
/// (the derived-f64 B/C/D F cache (`cache::cache_bcd_f`)) on the plain six-label key.
///
/// # Errors
///
/// - [`FrError::Catalog`] wrapping [`CatalogError::WrongGroup`] if any label is
///   not of the catalog's family.
/// - [`FrError::Catalog`] wrapping [`CatalogError::ZeroFusionChannel`] if any of
///   the four vertices is empty (the reference returns an all-zero block; this
///   query API returns a typed error — issue #15).
/// - [`FrError::Catalog`] wrapping [`CatalogError::BudgetExceeded`] /
///   [`CatalogError::Sweep`] from an underlying materialization.
#[allow(clippy::too_many_arguments)]
pub fn f_symbol(
    cat: &mut CanonicalCatalog,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
    e: &Irrep,
    f: &Irrep,
) -> Result<FBlock, FrError> {
    require_catalog_family(cat, &[a, b, c, d, e, f])?;
    // Guard: every vertex non-empty, decided by the exact S3.0 decomposition
    // before any float work (PR #14 lesson; issue #15). Mirrors sun::f_symbol.
    let vertices = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)];
    for (x, y, z) in vertices {
        if bcd_mult(x, y, z)? == 0 {
            return Err(FrError::Catalog(CatalogError::ZeroFusionChannel {
                a: x.dynkin(),
                b: y.dynkin(),
                c: z.dynkin(),
            }));
        }
    }

    let cache = crate::cache::cache_bcd_f();
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
    let block = {
        let mut fam = BcdFamily { cat };
        f_block_raw(&mut fam, a, b, c, d, e, f)?
    };
    let stored = cache.insert(key, Arc::new(block));
    Ok((*stored).clone())
}

// ---------------------------------------------------------------------------
// R-symbol (uncached — a single sparse join of two CGC).
// ---------------------------------------------------------------------------

/// The B/C/D R-symbol $R^{ab}_c$ as a dense $N^c_{ab} \times N^c_{ba}$ matrix.
///
/// # Errors
///
/// - [`FrError::Catalog`] wrapping [`CatalogError::WrongGroup`] if a label is
///   foreign, or [`CatalogError::ZeroFusionChannel`] if $a \otimes b \to c$ is empty.
/// - Materialization errors surfaced through [`FrError::Catalog`].
pub fn r_symbol(
    cat: &mut CanonicalCatalog,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
) -> Result<RBlock, FrError> {
    require_catalog_family(cat, &[a, b, c])?;
    if bcd_mult(a, b, c)? == 0 {
        return Err(FrError::Catalog(CatalogError::ZeroFusionChannel {
            a: a.dynkin(),
            b: b.dynkin(),
            c: c.dynkin(),
        }));
    }
    let mut fam = BcdFamily { cat };
    Ok(r_block_raw(&mut fam, a, b, c)?)
}

// ---------------------------------------------------------------------------
// Gates (self-consistency oracles for the B/C/D surface).
// ---------------------------------------------------------------------------

/// Verify that the F-move for fixed outer labels `(a, b, c, d)` is unitary.
///
/// # Errors
///
/// [`FrError::FNotUnitary`] with the worst residual on failure;
/// [`FrError::Catalog`] on a foreign label or a materialization failure.
pub fn check_f_unitarity(
    cat: &mut CanonicalCatalog,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
) -> Result<(), FrError> {
    require_catalog_family(cat, &[a, b, c, d])?;
    let mut fam = BcdFamily { cat };
    let worst = f_unitarity_residual(&mut fam, a, b, c, d)?;
    if worst > frcore::TOL_F_UNITARY {
        return Err(FrError::FNotUnitary { residual: worst });
    }
    Ok(())
}

/// Verify the pentagon identity for the quadruple `(a, b, c, d)`.
///
/// # Errors
///
/// [`FrError::PentagonViolation`] (worst residual) on failure; [`FrError::Catalog`]
/// on a foreign label or a materialization failure.
pub fn check_pentagon(
    cat: &mut CanonicalCatalog,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
) -> Result<(), FrError> {
    require_catalog_family(cat, &[a, b, c, d])?;
    let mut fam = BcdFamily { cat };
    let worst = pentagon_residual(&mut fam, a, b, c, d)?;
    if worst > frcore::TOL_PENTAGON {
        return Err(FrError::PentagonViolation { residual: worst });
    }
    Ok(())
}

/// Verify both hexagon identities for the triple `(a, b, c)`.
///
/// # Errors
///
/// [`FrError::HexagonViolation`] (worst residual) on failure; [`FrError::Catalog`]
/// on a foreign label or a materialization failure.
pub fn check_hexagon(
    cat: &mut CanonicalCatalog,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
) -> Result<(), FrError> {
    require_catalog_family(cat, &[a, b, c])?;
    let mut fam = BcdFamily { cat };
    let worst = hexagon_residual(&mut fam, a, b, c)?;
    if worst > frcore::TOL_HEXAGON {
        return Err(FrError::HexagonViolation { residual: worst });
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[doc(hidden)]
pub fn cgc_sweeps() -> u64 {
    CGC_SWEEPS.load(std::sync::atomic::Ordering::Relaxed)
}
