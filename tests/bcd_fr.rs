//! B/C/D F/R self-consistency oracles (Stage 3 S3.4, issue #27).
//!
//! Two tiers, mirroring the SU(N) F/R suites:
//! - CI-fast (`#[test]`): F-move unitarity on seeded small families, and the
//!   Sp(4) ≅ SO(5) F-magnitude spot check.
//! - Heavy (`#[ignore]`, run with `cargo test --release --features cgc-gen --
//!   --ignored`): the pentagon and hexagon spot checks, including the OM ≥ 2
//!   D3-adjoint family. These materialize CGC (SVD sweeps) for many intermediate
//!   irreps and cost seconds-to-minutes each — the same treatment as the heavy
//!   SU(3) F/R table oracle.
//!
//! All values are racah-generated on both sides; this is an intra-stage
//! self-consistency check. The SU(N)-anchored isomorphism battery (SO(6) ≅ SU(4)
//! etc., QSpace fixtures) belongs to S3.5 and is not duplicated here.

#![cfg(feature = "cgc-gen")]

use racah::bcd::{
    check_f_unitarity, check_hexagon, check_pentagon, directproduct, f_symbol, CanonicalCatalog,
    CatalogError, FrError, Irrep, Series,
};

fn irr(s: Series, d: &[i64]) -> Irrep {
    Irrep::from_dynkin(s, d).unwrap()
}

// ---------------------------------------------------------------------------
// The crate contract, asserted platform-independently.
//
// An F/R gate on a family EITHER closes (Ok — the F-move is unitary / the
// pentagon-hexagon identity holds within tolerance) OR fail-loud with a typed
// `BasisIncoherent` — it must NEVER return a silently-wrong value (a non-unitary
// F-move, a violated identity). WHICH of the two happens can be platform-
// dependent: an ill-conditioned coupled channel's frame is a deterministic
// resolution of a NEAR-TIE, so a given channel may embed coherently on one
// target (CI Linux x86) and in a rotated frame on another (dev macOS ARM). So we
// assert the DISJUNCTION, never a specific channel's incoherence. (The guard's
// firing itself is pinned deterministically by the synthetic rotation test in
// `bcd::sweep`.) See the P1b review round on PR #28.
// ---------------------------------------------------------------------------

/// The contract: `Ok` (gate closed) or `BasisIncoherent` (fail-loud) — anything
/// else (a violation slipping past the coherence guard) is a real defect.
#[track_caller]
fn assert_closes_or_bricks(r: Result<(), FrError>) {
    match r {
        Ok(()) => {}
        Err(FrError::Catalog(CatalogError::BasisIncoherent { .. })) => {}
        other => panic!(
            "contract violated: a gate must close (Ok) or fail-loud with \
             BasisIncoherent, never return a silent wrong value — got {other:?}"
        ),
    }
}

#[test]
fn c2_vector_f_move_closes_or_bricks() {
    // C2 = Sp(4), vector (0,1) dim 5. (Coherent, closes, on the dev macOS ARM box.)
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    let adj = irr(Series::C, &[2, 0]);
    assert_closes_or_bricks(check_f_unitarity(&mut cat, &v, &v, &v, &adj));
}

#[test]
fn b2_vector_f_move_closes_or_bricks() {
    // B2 = SO(5), vector (1,0). Its F-move pulls the (1,2) dim-35 channel, whose
    // frame is ill-conditioned: it bricks with BasisIncoherent on dev macOS ARM
    // (residual √6 ≈ 2.449) but embeds coherently and closes on CI Linux x86 —
    // hence the disjunction (a channel-specific assertion here is platform-fragile).
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let v = irr(Series::B, &[1, 0]);
    let adj = irr(Series::B, &[0, 2]);
    assert_closes_or_bricks(check_f_unitarity(&mut cat, &v, &v, &v, &adj));
}

#[test]
fn d3_vector_f_move_closes_or_bricks() {
    // D3 = SO(6), vector (1,0,0) dim 6; vector^2 -> 1 + adjoint + (2,0,0).
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let v = irr(Series::D, &[1, 0, 0]);
    let d = irr(Series::D, &[2, 0, 0]);
    assert_closes_or_bricks(check_f_unitarity(&mut cat, &v, &v, &v, &d));
}

// ---------------------------------------------------------------------------
// Sp(4) ≅ SO(5) F-level spot check (CI-fast), sign-sensitive.
//
// Label dictionary (derived from the S3.0 Dynkin↔partition maps and confirmed by
// matching Weyl dims across the whole (p, q even) grid): the SO(5)/Sp(4) diagram
// isomorphism swaps the two nodes, so a B2 label (a1, a2) corresponds to the C2
// label (a2, a1). Under it, B2 vector (1,0) ↔ C2 vector (0,1), B2 adjoint (0,2) ↔
// C2 adjoint (2,0), etc.
//
// The two catalogs generate the same fusion category from independent defining
// seeds, so their CGC gauges differ by a per-vertex sign (a coboundary): for a
// multiplicity-free sextet, F_C = ε(a,b,e)·ε(e,c,d)·ε(b,c,f)·ε(a,f,d) · F_B, with
// one sign ε ∈ {±1} per fusion vertex. Comparing |F| (or unitarity) is blind to
// those signs. Instead we FIT the coboundary: one GF(2) unknown per vertex, one
// equation per sextet (XOR of the 4 vertex bits = the observed relative sign),
// solved by Gaussian elimination. A consistent fit that leaves no residual proves
// the two F tensors are gauge-EQUIVALENT INCLUDING sign; a contradiction (0 = 1)
// means a genuine sign mismatch and fails. We also assert the magnitudes match.
// (Full sign/basis alignment for OM ≥ 2 needs the S3.5 fitted-unitary harness.)
// ---------------------------------------------------------------------------

/// A GF(2) linear system in row-echelon form: each row is `(variable set, rhs)`
/// keyed by its minimal pivot variable.
type Gf2Rows = Vec<(std::collections::BTreeSet<usize>, bool)>;

/// Add one GF(2) equation (`vars` XOR-sum = `rhs`) to `rows`. Returns `false`
/// only on a `0 = 1` contradiction (an inconsistent coboundary).
fn gf2_add(rows: &mut Gf2Rows, mut vars: std::collections::BTreeSet<usize>, mut rhs: bool) -> bool {
    while let Some(&pivot) = vars.iter().next() {
        if let Some((rv, rr)) = rows.iter().find(|(rv, _)| rv.iter().next() == Some(&pivot)) {
            for v in rv {
                if !vars.insert(*v) {
                    vars.remove(v); // symmetric difference over GF(2)
                }
            }
            rhs ^= *rr;
        } else {
            rows.push((vars, rhs));
            return true;
        }
    }
    !rhs // no variables left: consistent iff rhs is 0
}

#[test]
fn sp4_so5_vector_f_coboundary_fit_signed() {
    let mut cb = CanonicalCatalog::new(Series::B, 2).unwrap();
    let mut cc = CanonicalCatalog::new(Series::C, 2).unwrap();
    let vb = irr(Series::B, &[1, 0]);
    let vc = irr(Series::C, &[0, 1]);
    // B2 label (a1,a2) -> C2 label (a2,a1).
    let swap = |x: &Irrep| {
        irr(Series::C, &{
            let d = x.dynkin();
            vec![d[1], d[0]]
        })
    };

    // Intern each fusion vertex (keyed by its C2/swapped triple) to a GF(2) var.
    type VertexKey = (Vec<i64>, Vec<i64>, Vec<i64>);
    let mut vertex_id: std::collections::HashMap<VertexKey, usize> =
        std::collections::HashMap::new();
    let mut var = |x: &Irrep, y: &Irrep, z: &Irrep| -> usize {
        let key = (swap(x).dynkin(), swap(y).dynkin(), swap(z).dynkin());
        let n = vertex_id.len();
        *vertex_id.entry(key).or_insert(n)
    };

    let mut rows: Gf2Rows = Vec::new();
    let vv = directproduct(&vb, &vb).unwrap(); // {1, (0,2), (2,0)}
    let mut compared = 0usize;
    let mut worst_mag = 0.0f64;
    for e in vv.keys() {
        for f in vv.keys() {
            let ev = directproduct(e, &vb).unwrap();
            let vf = directproduct(&vb, f).unwrap();
            for d in ev.keys() {
                if !vf.contains_key(d) {
                    continue; // d must lie in both e⊗v and v⊗f
                }
                // Skip sextets that pull an incoherent channel (the guard's
                // typed error) — the signed comparison is over coherence-verified
                // sextets only. Any other error is a real failure.
                let fb = match f_symbol(&mut cb, &vb, &vb, &vb, d, e, f) {
                    Ok(x) => x,
                    Err(FrError::Catalog(CatalogError::BasisIncoherent { .. })) => continue,
                    Err(e) => panic!("unexpected B2 error: {e:?}"),
                };
                if fb.dims() != [1, 1, 1, 1] {
                    continue; // coboundary sign is per-vertex only for mult-free
                }
                let fc = match f_symbol(&mut cc, &vc, &vc, &vc, &swap(d), &swap(e), &swap(f)) {
                    Ok(x) => x,
                    Err(FrError::Catalog(CatalogError::BasisIncoherent { .. })) => continue,
                    Err(e) => panic!("unexpected C2 error: {e:?}"),
                };
                let (b, c) = (fb.at(0, 0, 0, 0), fc.at(0, 0, 0, 0));
                worst_mag = worst_mag.max((b.abs() - c.abs()).abs());
                if b.abs() < 1e-9 {
                    continue; // both ~0: the sign equation would be vacuous
                }
                // Equation: ε(a,b,e)⊕ε(e,c,d)⊕ε(b,c,f)⊕ε(a,f,d) = [signs differ].
                let vars: std::collections::BTreeSet<usize> = [
                    var(&vb, &vb, e),
                    var(e, &vb, d),
                    var(&vb, &vb, f),
                    var(&vb, f, d),
                ]
                .into_iter()
                .collect();
                let rhs = b.signum() != c.signum();
                assert!(
                    gf2_add(&mut rows, vars, rhs),
                    "Sp(4)/SO(5) sign mismatch at sextet e={:?} f={:?} d={:?}: \
                     F_B={b:+.6} F_C={c:+.6} is not a vertex-sign coboundary",
                    e.dynkin(),
                    f.dynkin(),
                    d.dynkin()
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared >= 3,
        "expected several coherent matched sextets, got {compared}"
    );
    assert!(
        worst_mag < 1e-9,
        "Sp(4)/SO(5) |F| mismatch: worst ||F_B|-|F_C|| = {worst_mag:e}"
    );
}

// ---------------------------------------------------------------------------
// Pentagon / hexagon, the same closes-or-bricks contract (heavy — run with
// --release --ignored).
//
// A gate CLOSES only when every channel its enumeration pulls is well-
// conditioned; a gate that pulls any ill-conditioned channel fail-LOUD with a
// typed BasisIncoherent (never a silently-wrong value). WHICH happens is
// platform-dependent for near-tie channels, so these assert the disjunction, not
// a specific channel. Dev-macOS-ARM measurements (not assertions): C2 vector
// hexagon closes; C2 vector pentagon bricks at the (2,2)-type channel; B2 vector
// hexagon bricks at (1,2) (√6); the D3 adjoint OM≥2 (0,1,1)² battery bricks at
// the 84 = (0,2,2) channel (residual 3.65). Closing the ill-conditioned families
// awaits the intertwiner-alignment leaf.
//
// (Earlier "~5 s green" figures in this file's history were a measurement bug —
// the timing ran through a pipe whose exit code was grep's, not the test's, so
// the gates had never actually run.)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn c2_vector_hexagon_closes_or_bricks() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    assert_closes_or_bricks(check_hexagon(&mut cat, &v, &v, &v));
}

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn c2_vector_pentagon_closes_or_bricks() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let v = irr(Series::C, &[0, 1]);
    assert_closes_or_bricks(check_pentagon(&mut cat, &v, &v, &v, &v));
}

#[test]
#[ignore = "heavy: materializes many CGC; run with --release --ignored"]
fn b2_vector_hexagon_closes_or_bricks() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let v = irr(Series::B, &[1, 0]);
    assert_closes_or_bricks(check_hexagon(&mut cat, &v, &v, &v));
}

/// OM ≥ 2 on the D3 adjoint g = (0,1,1): `g⊗g → g` has multiplicity 2 (exact
/// S3.0). On dev macOS ARM the g⊗g decomposition's 84 = (0,2,2) channel is
/// near-rank-deficient in QR (PR #24) and embeds in an O(1)-rotated frame, so the
/// braiding battery bricks with `BasisIncoherent` there (residual 3.65) — the
/// restored QSpace coherence guard (issue #15 instance 5). On another platform
/// the near-tie may resolve coherently and the battery close; the assertion is
/// the disjunction. A guaranteed-closing OM ≥ 2 battery is the intertwiner-
/// alignment leaf's acceptance gate, not this PR's.
///
/// `#[ignore]` (release-only): materializes the adjoint CGC chain; seconds in
/// release, minutes under the unoptimized debug SVD.
#[test]
#[ignore = "heavy OM>=2 family (minutes in debug SVD): run with --release --ignored"]
fn d3_adjoint_om2_closes_or_bricks() {
    let g = irr(Series::D, &[0, 1, 1]);
    assert_eq!(
        directproduct(&g, &g).unwrap().get(&g).copied().unwrap(),
        2,
        "exact layer must predict OM=2 for the D3 adjoint square"
    );
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    assert_closes_or_bricks(check_hexagon(&mut cat, &g, &g, &g));
}
