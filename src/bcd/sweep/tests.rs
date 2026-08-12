//! Tests for the S3.2 decomposition sweep.
//!
//! Oracles are independent of the code under test: the discovered labels and
//! multiplicities are gated against S3.0's exact [`crate::bcd::directproduct`]
//! (a separate implementation, itself cross-checked against Sage/OSCAR
//! fixtures); the per-block weight structure is checked against the irrep's
//! exact weight system (Freudenthal + Weyl orbits, S3.0); orthonormality and
//! determinism are self-consistency invariants.

use std::collections::HashSet;

use super::*;
use crate::bcd::{defining_seed, directproduct};

fn defining_irrep(series: Series, r: usize) -> Irrep {
    let mut dynkin = vec![0i64; r];
    dynkin[0] = 1;
    Irrep::from_dynkin(series, &dynkin).unwrap()
}

/// Number of distinct weights of an irrep and the total (= dim), computed
/// independently from S3.0's dominant-weight multiplicities expanded over Weyl
/// orbits (signed permutations for B/C, even-signed for D). Basis-independent,
/// so it can be compared to the sweep block's Cartan-eigenvalue rows.
fn weight_counts(a: &Irrep) -> (usize, usize) {
    let series = a.series();
    let mut distinct: HashSet<Vec<i64>> = HashSet::new();
    let mut total = 0usize;
    for (mu, &m) in &a.weight_multiplicities() {
        let orbit = weyl_orbit(series, mu);
        for w in &orbit {
            distinct.insert(w.clone());
        }
        total += orbit.len() * m as usize;
    }
    (distinct.len(), total)
}

fn weyl_orbit(series: Series, mu: &[i64]) -> Vec<Vec<i64>> {
    let r = mu.len();
    let mut set: HashSet<Vec<i64>> = HashSet::new();
    for signs in 0u32..(1u32 << r) {
        if series == Series::D && signs.count_ones() % 2 != 0 {
            continue;
        }
        let signed: Vec<i64> = (0..r)
            .map(|i| if signs & (1 << i) != 0 { -mu[i] } else { mu[i] })
            .collect();
        let mut idx: Vec<usize> = (0..r).collect();
        permute(&signed, &mut idx, 0, &mut set);
    }
    set.into_iter().collect()
}

fn permute(v: &[i64], idx: &mut Vec<usize>, k: usize, set: &mut HashSet<Vec<i64>>) {
    let n = idx.len();
    if k == n {
        set.insert(idx.iter().map(|&i| v[i]).collect());
        return;
    }
    for i in k..n {
        idx.swap(k, i);
        permute(v, idx, k + 1, set);
        idx.swap(k, i);
    }
}

// ---- 1. defining ⊗ defining across all series/ranks -----------------------

#[test]
fn defining_squared_matches_exact_decomposition() {
    // B2/B3/C2/C3/D3 plus one rank-4 per family (the spec's required set).
    for (series, r) in [
        (Series::B, 2),
        (Series::B, 3),
        (Series::B, 4),
        (Series::C, 2),
        (Series::C, 3),
        (Series::C, 4),
        (Series::D, 3),
        (Series::D, 4),
    ] {
        let seed = defining_seed(series, r).unwrap();
        let d = seed.dim();
        let a = defining_irrep(series, r);
        let expected = directproduct(&a, &a).unwrap();
        let decomp = decompose_defining_product(&seed, &seed).unwrap();

        // The sweep multiplicities equal the exact decomposition (this is the
        // Ruling 1 gate, but assert it explicitly as a test too).
        assert_eq!(
            decomp.multiplicities(),
            expected,
            "{series:?}{r} discovered multiplicities != exact"
        );

        // Dimension bookkeeping: Σ block dims == d1·d2.
        let total: usize = decomp.blocks().iter().map(|b| b.dim()).sum();
        assert_eq!(total, d * d, "{series:?}{r} tiling");

        for b in decomp.blocks() {
            // CGC isometry: columns orthonormal (VᵀV == I within tier).
            assert_isometry(b);
            // Block Cartan-eigenvalue rows reproduce the irrep's weight system:
            // same number of distinct weights and same total (= dim).
            let (distinct, tot) = weight_counts(b.irrep());
            assert_eq!(tot, b.dim(), "{series:?}{r} block total weights");
            assert_eq!(
                distinct_weight_rows(b),
                distinct,
                "{series:?}{r} block {:?} distinct-weight count",
                b.irrep().dynkin()
            );
        }
    }
}

fn assert_isometry(b: &Block) {
    let (rows, cols) = b.cgc_shape();
    let data = b.cgc();
    for i in 0..cols {
        for j in 0..cols {
            let dot: f64 = (0..rows)
                .map(|k| data[k + i * rows] * data[k + j * rows])
                .sum();
            let target = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - target).abs() < 1e-9,
                "CGC not isometric at ({i},{j}): {dot}"
            );
        }
    }
}

/// Number of distinct Cartan-eigenvalue rows in a block (its distinct weights).
fn distinct_weight_rows(b: &Block) -> usize {
    let mut set: HashSet<Vec<i64>> = HashSet::new();
    let d3 = b.dim();
    let nz = b.irrep().rank();
    for s in 0..d3 {
        let row: Vec<i64> = (0..nz).map(|j| b.weight(s, j).round() as i64).collect();
        set.insert(row);
    }
    set.len()
}

// ---- 2. determinism -------------------------------------------------------

#[test]
fn determinism_bitwise_identical() {
    for (series, r) in [(Series::B, 2), (Series::C, 3), (Series::D, 3)] {
        let seed = defining_seed(series, r).unwrap();
        let run1 = decompose_defining_product(&seed, &seed).unwrap();
        let run2 = decompose_defining_product(&seed, &seed).unwrap();
        assert_eq!(run1.blocks().len(), run2.blocks().len());
        for (b1, b2) in run1.blocks().iter().zip(run2.blocks()) {
            assert_eq!(b1.irrep(), b2.irrep());
            assert_eq!(
                b1.cgc().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b2.cgc().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{series:?}{r} CGC not bitwise identical across runs"
            );
        }
    }
}

// ---- 3. outer multiplicity ≥ 2 --------------------------------------------

#[test]
fn outer_multiplicity_two_d3_adjoint_squared() {
    // D3 (0,1,1) is the adjoint, obtainable from the vector ⊗ vector block.
    // (0,1,1) ⊗ (0,1,1) has N^{(0,1,1)} = 2 (exact layer predicts OM ≥ 2).
    let adj = Irrep::from_dynkin(Series::D, &[0, 1, 1]).unwrap();
    let expected = directproduct(&adj, &adj).unwrap();
    assert_eq!(
        expected.get(&adj).copied(),
        Some(2),
        "exact layer must predict OM=2 for (0,1,1)²"
    );

    let seed = defining_seed(Series::D, 3).unwrap();
    let vv = decompose_defining_product(&seed, &seed).unwrap();
    let adj_gen = vv
        .blocks()
        .iter()
        .find(|b| b.irrep() == &adj)
        .unwrap()
        .generators()
        .clone();

    let prod = Generators::product(&adj_gen, &adj_gen).unwrap();
    let decomp = decompose(&prod, &expected).unwrap();

    // Two (0,1,1) blocks, carrying OM indices (0,2) and (1,2).
    let om_blocks: Vec<(usize, usize)> = decomp
        .blocks()
        .iter()
        .filter(|b| b.irrep() == &adj)
        .map(|b| b.outer_multiplicity())
        .collect();
    assert_eq!(om_blocks.len(), 2);
    assert!(om_blocks.contains(&(0, 2)) && om_blocks.contains(&(1, 2)));

    for b in decomp.blocks() {
        if b.irrep() != &adj {
            assert_eq!(b.outer_multiplicity().1, 1);
        }
    }
}

// ---- 4. sign convention ---------------------------------------------------

#[test]
fn cgc_first_significant_entry_positive() {
    // rangeSignConvention: each block's first significant CGC entry is positive.
    let seed = defining_seed(Series::B, 3).unwrap();
    let decomp = decompose_defining_product(&seed, &seed).unwrap();
    for b in decomp.blocks() {
        if let Some(v) = b.cgc().iter().copied().find(|x| x.abs() > 1e-10) {
            assert!(v > 0.0, "first significant CGC entry not positive: {v}");
        }
    }
}

// ---- 5. ill-posed input guards (typed errors) -----------------------------

#[test]
fn product_of_mismatched_generators_is_typed_error() {
    let b2 = Generators::from_seed(&defining_seed(Series::B, 2).unwrap());
    let c2 = Generators::from_seed(&defining_seed(Series::C, 2).unwrap());
    assert_eq!(
        Generators::product(&b2, &c2),
        Err(SweepError::GeneratorMismatch)
    );
    let b3 = Generators::from_seed(&defining_seed(Series::B, 3).unwrap());
    assert_eq!(
        Generators::product(&b2, &b3),
        Err(SweepError::GeneratorMismatch)
    );
}

#[test]
fn so3_exclusion_inherited() {
    // B_1 = SO(3), C_1 = Sp(2), D_2 = SO(4) are excluded low-rank isomorphisms;
    // the seed layer rejects them, so the sweep never receives such generators.
    assert!(defining_seed(Series::B, 1).is_err());
    assert!(defining_seed(Series::C, 1).is_err());
    assert!(defining_seed(Series::D, 2).is_err());
}

#[test]
fn spurious_multiplicity_expectation_is_gated() {
    // A wrong `expected` map (a spurious extra irrep) must fail the Ruling 1
    // gate — proving the gate is live on the production path.
    let seed = defining_seed(Series::C, 2).unwrap();
    let ga = Generators::from_seed(&seed);
    let prod = Generators::product(&ga, &ga).unwrap();
    let a = defining_irrep(Series::C, 2);
    let mut expected = directproduct(&a, &a).unwrap();
    let bogus = Irrep::from_dynkin(Series::C, &[2, 2]).unwrap();
    expected.insert(bogus.clone(), 1);
    match decompose(&prod, &expected) {
        Err(SweepError::MultiplicityMismatch { dynkin, .. }) => {
            assert_eq!(dynkin, bogus.dynkin());
        }
        other => panic!("expected MultiplicityMismatch, got {other:?}"),
    }
}

#[test]
fn missing_multiplicity_expectation_is_gated() {
    // Dropping a real block from `expected` must also fail (both directions).
    let seed = defining_seed(Series::B, 2).unwrap();
    let ga = Generators::from_seed(&seed);
    let prod = Generators::product(&ga, &ga).unwrap();
    let a = defining_irrep(Series::B, 2);
    let mut expected = directproduct(&a, &a).unwrap();
    let trivial = Irrep::trivial(Series::B, 2).unwrap();
    expected.remove(&trivial);
    assert!(matches!(
        decompose(&prod, &expected),
        Err(SweepError::MultiplicityMismatch { .. })
    ));
}

// ---- 7. weight-sort tie-break (site test) ---------------------------------

#[test]
fn descending_weight_sort_tie_break_is_ascending_index() {
    // Two states share weight row [0,0] (indices 1 and 2); one MW row [1,0]
    // (index 0). Descending sort must put the MW first, then break the tie by
    // ascending original index: [0, 1, 2] (not [0, 2, 1]). This pins the gauge
    // tie-break at its site — no label/multiplicity oracle can catch a change to
    // it, since it only reorders equal-weight states.
    let mut z = super::super::linalg::Dense::zeros(3, 2);
    z.set(0, 0, 1.0); // row 0 = [1, 0] (highest)
    z.set(2, 0, -1.0); // row 2 = [-1, 0]
                       // row 1 = [0, 0]
    let perm = super::descending_weight_perm(&z);
    assert_eq!(perm, vec![0, 1, 2]);

    // A degenerate tie at the top: rows 0 and 1 identical -> ascending index
    // keeps 0 before 1.
    let mut z2 = super::super::linalg::Dense::zeros(2, 1);
    z2.set(0, 0, 5.0);
    z2.set(1, 0, 5.0);
    assert_eq!(super::descending_weight_perm(&z2), vec![0, 1]);
}

// ---- 8. Kronecker composition convention ----------------------------------

#[test]
fn kronecker_composition_dimension_and_group() {
    let a = Generators::from_seed(&defining_seed(Series::B, 2).unwrap());
    let prod = Generators::product(&a, &a).unwrap();
    assert_eq!(prod.dim(), a.dim() * a.dim());
    assert_eq!(prod.rank(), a.rank());
    assert_eq!(prod.series(), Series::B);
}

#[test]
fn kronecker_first_factor_is_fast_index() {
    // Pin the composite-index convention (gauge): composite(m_a, m_b) =
    // m_a + d_a·m_b, so the product Cartan diagonal at that index equals
    // Sz_a[m_a] + Sz_b[m_b]. The convention is only observable for *distinct*
    // factors (a ⊗ a is swap-symmetric), so use B2 vector (d=5) ⊗ adjoint
    // (d=10); were the second factor the fast index, the diagonal would be
    // transposed and this exact reconstruction would fail on the asymmetric
    // cells.
    let seed = defining_seed(Series::B, 2).unwrap();
    let a = Generators::from_seed(&seed);
    let vv = decompose_defining_product(&seed, &seed).unwrap();
    let b = vv
        .blocks()
        .iter()
        .find(|blk| blk.irrep().dynkin() == vec![0, 2])
        .unwrap()
        .generators()
        .clone();
    let (da, db) = (a.dim(), b.dim());
    assert_ne!(da, db, "factors must be distinct to observe the convention");
    let prod = Generators::product(&a, &b).unwrap();
    for mb in 0..db {
        for ma in 0..da {
            let composite = ma + da * mb; // first factor fast
            assert_eq!(
                prod.cartan_diag(0)[composite],
                a.cartan_diag(0)[ma] + b.cartan_diag(0)[mb],
                "Kronecker convention broken at (m_a={ma}, m_b={mb})"
            );
        }
    }
}

// ---- 9. golden CGC value snapshot (gauge regression guard) ----------------

/// A **regression snapshot** (not an independent oracle) pinning specific CGC
/// entries of `B2 vector²`. It guards the CGC *values* against
/// value-affecting gauge mutations that no label oracle can catch (the
/// decomposition — labels/dims/multiplicities/isometry — is gauge-invariant); an
/// external QSpace CGC fixture (S3.5) is the independent oracle.
///
/// Not every gauge choice is catchable here, and the distinction is by
/// *magnitude*:
/// - **O(1)-catchable** (this snapshot catches them): the sign convention, the
///   seed order, the weight-sort tie-break, and the QR gauge (PositiveDiagonal)
///   — a mutation flips a sign or permutes/re-mixes states, shifting a pinned
///   entry by O(1), far above the 1e-9 tolerance. Verified: disabling the sign
///   convention fails this test.
/// - **round-off-neutral** (this snapshot cannot catch them, and correctly so):
///   the Gram–Schmidt *order* within pass 1 and the *second* orthonormalization.
///   `U`, `V`, and the current level are mutually orthogonal, so projecting them
///   out commutes and the second pass is a no-op once the first converges; a
///   mutation to either shifts values only at the `~1e-13` round-off floor (see
///   the round-off-neutrality note in `docs/gauge_soN.md` §4). These are
///   genuinely value-neutral, like `docs/gauge.md`'s value-neutral `cref`
///   tie-break, and are pinned by the doc, not by a value test.
///
/// Columns 1/4/9 of the adjoint block are *descended* states (they would move
/// under a sign/tie-break/QR-gauge change to the descent).
#[test]
fn cgc_value_snapshot_b2_regression() {
    let seed = defining_seed(Series::B, 2).unwrap();
    let d = decompose_defining_product(&seed, &seed).unwrap();
    let block = |dynkin: Vec<i64>| {
        d.blocks()
            .iter()
            .find(|b| b.irrep().dynkin() == dynkin)
            .unwrap()
    };
    let s2 = 0.5_f64.sqrt(); // 1/√2

    // Highest-weight (pre-descent) columns.
    assert_close(block(vec![2, 0]).cgc(), 14, 0, 12, 1.0);
    assert_close(block(vec![0, 2]).cgc(), 10, 0, 2, s2);
    assert_close(block(vec![0, 2]).cgc(), 10, 0, 10, -s2);

    // Descended columns of the adjoint block (exercise the descent gauge).
    assert_close(block(vec![0, 2]).cgc(), 10, 1, 14, -s2);
    assert_close(block(vec![0, 2]).cgc(), 10, 1, 22, s2);
    assert_close(block(vec![0, 2]).cgc(), 10, 4, 1, -0.5);
    assert_close(block(vec![0, 2]).cgc(), 10, 4, 5, 0.5);
    assert_close(block(vec![0, 2]).cgc(), 10, 4, 13, -0.5);
    assert_close(block(vec![0, 2]).cgc(), 10, 4, 17, 0.5);
    assert_close(block(vec![0, 2]).cgc(), 10, 9, 8, -s2);
    assert_close(block(vec![0, 2]).cgc(), 10, 9, 16, s2);
}

/// Assert `cgc[row, col] ≈ v` for a `25 × cols` column-major block (rows = 25).
fn assert_close(cgc: &[f64], _cols: usize, col: usize, row: usize, v: f64) {
    let got = cgc[row + col * 25];
    assert!(
        (got - v).abs() < 1e-9,
        "CGC[{row},{col}] = {got}, expected {v}"
    );
}

// ---- coherence guard: deterministic synthetic rotation (issue #15 instance 5) ----

/// Pins `Generators::coherence_residual` (the restored QSpace `normDiff`
/// cross-copy measure) on EVERY platform, independent of which real ill-
/// conditioned irrep happens to rotate on a given target. Two embeddings that
/// differ by a known orthogonal rotation `W` inside a degenerate weight space —
/// exactly the incoherence class — must register an O(1) residual, well above
/// `TOL_BASIS_COHERENT` (1e-10). The `findabsmax` precedent: test the decision
/// function with constructed input, not an accident of a numeric sweep.
#[test]
fn coherence_residual_detects_degenerate_rotation() {
    use super::super::linalg::Dense;
    // A 2-dim carrier at a single (degenerate) weight 0, with one raising op.
    let raising = |m: [[f64; 2]; 2]| {
        let mut d = Dense::zeros(2, 2);
        for (r, row) in m.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                d.set(r, c, v);
            }
        }
        d
    };
    let canonical = Generators {
        series: Series::D,
        rank: 1,
        dim: 2,
        sp: vec![raising([[0.0, 1.0], [0.0, 0.0]])],
        sz: vec![vec![0.0, 0.0]], // degenerate weight: rotation is a real gauge freedom
    };
    // Rotate the degenerate 2-space by 45°: Sp' = W Sp Wᵀ, weights unchanged.
    let c = std::f64::consts::FRAC_1_SQRT_2;
    let rotated = Generators {
        series: Series::D,
        rank: 1,
        dim: 2,
        sp: vec![raising([[-c * c, c * c], [-c * c, c * c]])],
        sz: vec![vec![0.0, 0.0]],
    };
    let residual = canonical.coherence_residual(&rotated);
    assert!(
        residual > 1e-3,
        "a degenerate-space rotation must register O(1), got {residual}"
    );
    // Sanity: a set is coherent with itself.
    assert_eq!(canonical.coherence_residual(&canonical), 0.0);
}

// ---- intertwiner alignment (issue #29) -------------------------------------

/// A real coupled block with a degenerate weight space: the SO(5) adjoint (0,2),
/// dim 10, from vector⊗vector. Its zero-weight space has multiplicity `rank = 2`,
/// so a within-space rotation is a genuine gauge freedom — the incoherence class.
fn so5_adjoint_block() -> Block {
    let seed = defining_seed(Series::B, 2).unwrap();
    let g = Generators::from_seed(&seed);
    let prod = Generators::product(&g, &g).unwrap();
    let v = defining_irrep(Series::B, 2);
    let expected = directproduct(&v, &v).unwrap();
    let decomp = decompose(&prod, &expected).unwrap();
    let adj = Irrep::from_dynkin(Series::B, &[0, 2]).unwrap();
    decomp
        .blocks()
        .iter()
        .find(|b| b.irrep() == &adj)
        .expect("vector⊗vector must contain the adjoint")
        .clone()
}

/// The `d × d` identity except a 45° rotation on the two `zero`-weight indices.
fn rotation_on(d: usize, zero: [usize; 2]) -> Dense {
    let mut w = Dense::zeros(d, d);
    for i in 0..d {
        w.set(i, i, 1.0);
    }
    let c = std::f64::consts::FRAC_1_SQRT_2;
    // [[c,-c],[c,c]] on the 2×2 sub-block.
    w.set(zero[0], zero[0], c);
    w.set(zero[0], zero[1], -c);
    w.set(zero[1], zero[0], c);
    w.set(zero[1], zero[1], c);
    w
}

/// The two zero-weight state indices of a block (all Cartan diagonals zero).
fn zero_weight_indices(g: &Generators) -> [usize; 2] {
    let zeros: Vec<usize> = (0..g.dim())
        .filter(|&s| (0..g.rank()).all(|j| g.cartan_diag(j)[s].abs() < 1e-9))
        .collect();
    assert_eq!(
        zeros.len(),
        2,
        "SO(5) adjoint has a doubly-degenerate zero weight"
    );
    [zeros[0], zeros[1]]
}

/// PR #28 template — rotate, align, require exact recovery of the canonical
/// values. A block put into an O(1)-rotated frame inside its degenerate weight
/// space (the platform-fragile incoherence class) aligns back to the canonical
/// frame: the aligned generators match the canonical set below the guard
/// tolerance and the aligned CGC recovers the canonical CGC — independent of the
/// rotation, so every platform's rotated embedding maps onto one answer.
#[test]
fn alignment_recovers_canonical_frame_after_rotation() {
    let b0 = so5_adjoint_block();
    let g0 = b0.generators().clone();
    let zero = zero_weight_indices(&g0);
    let w0 = rotation_on(g0.dim(), zero);

    // Rotated frame: G1 = W0ᵀ·G0·W0, V1 = V0·W0 (V1ᵀ Sp V1 = W0ᵀ (V0ᵀ Sp V0) W0).
    let g1 = conjugate_generators(&g0, &w0.transpose()).unwrap();
    let v1 = matmul(&b0.cgc, &w0).unwrap();
    let b1 = Block {
        irrep: b0.irrep().clone(),
        cgc: v1,
        gens: g1.clone(),
        z: b0.z.clone(),
        om: b0.outer_multiplicity(),
    };

    // The rotated frame WOULD brick the guard (O(1) residual).
    assert!(
        g1.coherence_residual(&g0) > 1e-3,
        "the synthetic rotation must be a real incoherence"
    );

    let (aligned_cgc, residual) = align_block(&b1, &g0).unwrap();
    assert!(
        residual < 1e-10,
        "alignment must drive the generator residual under the guard tol, got {residual:e}"
    );
    // Aligned CGC recovers the canonical CGC element-wise.
    let mut worst = 0.0f64;
    for (a, b) in aligned_cgc.data.iter().zip(b0.cgc.data.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-9,
        "aligned CGC must recover canonical, worst {worst:e}"
    );
}

/// A well-conditioned (already-coherent) block aligns to `W = identity` up to
/// sign: the aligned CGC equals the input CGC, and the residual is at the sweep's
/// round-off floor. (This is why the catalog fast-path — skip alignment when the
/// raw residual already passes — is bit-exact.)
#[test]
fn alignment_of_coherent_block_is_identity() {
    let b0 = so5_adjoint_block();
    let g0 = b0.generators().clone();
    let w = intertwiner(&g0, &g0).unwrap();
    // W ≈ I.
    for i in 0..g0.dim() {
        for j in 0..g0.dim() {
            let target = if i == j { 1.0 } else { 0.0 };
            assert!(
                (w.at(i, j) - target).abs() < 1e-9,
                "intertwiner(G,G) must be I"
            );
        }
    }
    let (aligned_cgc, residual) = align_block(&b0, &g0).unwrap();
    assert!(residual < 1e-9, "self-alignment residual {residual:e}");
    let mut worst = 0.0f64;
    for (a, b) in aligned_cgc.data.iter().zip(b0.cgc.data.iter()) {
        worst = worst.max((a - b).abs());
    }
    assert!(
        worst < 1e-9,
        "self-alignment must not move the CGC, worst {worst:e}"
    );
}

/// Determinism: aligning the SAME rotated block twice yields bit-identical CGC
/// (the SVD-based Procrustes is a deterministic function of its inputs).
#[test]
fn alignment_is_deterministic() {
    let b0 = so5_adjoint_block();
    let g0 = b0.generators().clone();
    let zero = zero_weight_indices(&g0);
    let w0 = rotation_on(g0.dim(), zero);
    let g1 = conjugate_generators(&g0, &w0.transpose()).unwrap();
    let v1 = matmul(&b0.cgc, &w0).unwrap();
    let b1 = Block {
        irrep: b0.irrep().clone(),
        cgc: v1,
        gens: g1,
        z: b0.z.clone(),
        om: b0.outer_multiplicity(),
    };
    let (c1, _) = align_block(&b1, &g0).unwrap();
    let (c2, _) = align_block(&b1, &g0).unwrap();
    assert_eq!(c1.data, c2.data, "alignment must be bitwise deterministic");
}
