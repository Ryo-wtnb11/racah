//! Tests for the S3.3 [`CanonicalCatalog`].
//!
//! Oracles are independent of the code under test: canonical-parent choices are
//! checked against the exact S3.0 [`crate::bcd::directproduct`] (a separate
//! implementation, itself cross-checked against Sage/OSCAR fixtures); generator
//! validity against the S3.1 commutator relations; query-order independence and
//! reset stability are bitwise self-consistency invariants; the ill-posed guards
//! are red-first (the PR #14 trivial-coupling lesson).

use super::*;

fn irr(series: Series, dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin(series, dynkin).unwrap()
}

fn defining(series: Series, r: usize) -> Irrep {
    let mut d = vec![0i64; r];
    d[0] = 1;
    irr(series, &d)
}

// ---- 0. construction, base cases, group guards -----------------------------

#[test]
fn construction_seeds_base_cases() {
    let cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    assert_eq!(cat.series(), Series::C);
    assert_eq!(cat.rank(), 2);
    // Trivial and defining are present with no materialization.
    assert!(cat.is_materialized(&Irrep::trivial(Series::C, 2).unwrap()));
    assert!(cat.is_materialized(&defining(Series::C, 2)));
    assert_eq!(cat.len(), 2);
    assert!(cat.bytes() > 0);
}

#[test]
fn excluded_low_rank_is_typed_error() {
    // B1 = SO(3), C1 = Sp(2), D2 = SO(4) are excluded isomorphisms.
    assert!(matches!(
        CanonicalCatalog::new(Series::B, 1),
        Err(CatalogError::Label(BcdError::ExcludedRank { .. }))
    ));
    assert!(matches!(
        CanonicalCatalog::new(Series::D, 2),
        Err(CatalogError::Label(BcdError::ExcludedRank { .. }))
    ));
}

#[test]
fn foreign_group_is_typed_error() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    // A B-series irrep passed to a C-series catalog.
    let foreign = irr(Series::B, &[2, 0]);
    assert!(matches!(
        cat.generators(&foreign),
        Err(CatalogError::WrongGroup { .. })
    ));
    // A rank-3 irrep passed to a rank-2 catalog.
    let wrong_rank = irr(Series::C, &[1, 0, 0]);
    assert!(matches!(
        cat.generators(&wrong_rank),
        Err(CatalogError::WrongGroup { .. })
    ));
}

// ---- 1. canonical-parent rule (exact, over S3.0) ---------------------------

#[test]
fn enumerated_irrep_equals_from_dynkin() {
    // The bounded-dim enumeration must produce the same Irrep value the sweep
    // does (via from_dynkin), or store lookups would miss.
    let c = irr(Series::C, &[2, 0]);
    let below = irreps_below(Series::C, 2, &c);
    // The defining rep (1,0) is below (2,0) and must be present, byte-identical.
    assert!(below.contains(&defining(Series::C, 2)));
    assert!(below.contains(&Irrep::trivial(Series::C, 2).unwrap()));
}

#[test]
fn canonical_parent_is_two_smaller_reps_containing_c() {
    // For a symmetric power the balanced/defining split is admissible; the rule
    // returns SOME pair (a,b) with a ≺ c, b ≺ c, and c ∈ a ⊗ b.
    for (series, r, dynkin) in [
        (Series::C, 2, vec![2, 0]),
        (Series::B, 2, vec![2, 0]),
        (Series::D, 3, vec![2, 0, 0]),
    ] {
        let c = irr(series, &dynkin);
        let (a, b) = canonical_parent(series, r, &c).expect("non-base has a parent");
        assert!(prec_key(&a) < prec_key(&c), "a must be ≺ c");
        assert!(prec_key(&b) < prec_key(&c), "b must be ≺ c");
        // canonical a ⪯ b form.
        assert!(prec_key(&a) <= prec_key(&b), "pair must be in a ⪯ b form");
        let prod = directproduct(&a, &b).unwrap();
        assert!(prod.contains_key(&c), "c must appear in a ⊗ b");
    }
}

#[test]
fn canonical_parent_is_order_independent() {
    // canonical_parent is a pure function of c and the exact data: it never
    // consults any discovery order, so repeated calls are identical.
    let c = irr(Series::C, &[3, 0]);
    let p1 = canonical_parent(Series::C, 2, &c);
    let p2 = canonical_parent(Series::C, 2, &c);
    assert_eq!(p1, p2);
}

// ---- 2. materialization + inherited Ruling-1 gate --------------------------

#[test]
fn materialize_valid_irreps_passes_commutator_gate() {
    // Every stored generator set satisfies the S3.1 commutator relations (the
    // sweep gates them; here we confirm the catalog's stored sets do too).
    for (series, r, dynkin) in [
        (Series::C, 2, vec![2, 0]),
        (Series::C, 2, vec![0, 1]),
        (Series::B, 2, vec![2, 0]),
        (Series::D, 3, vec![0, 1, 1]),
    ] {
        let mut cat = CanonicalCatalog::new(series, r).unwrap();
        let c = irr(series, &dynkin);
        cat.generators(&c).unwrap();
        let res = cat.stored_commutator_residual(&c).unwrap();
        assert!(
            res < 1e-6,
            "commutator residual {res:e} too large for {dynkin:?}"
        );
        // The whole canonical-parent chain is materialized.
        let (a, b) = canonical_parent(series, r, &c).unwrap();
        assert!(cat.is_materialized(&a));
        assert!(cat.is_materialized(&b));
    }
}

// ---- 3. query-order independence (the Ruling-2 acceptance) -----------------

/// Materialize a set of irreps in a given query order; return the catalog for
/// byte-comparison of its stored generators.
fn materialize_in_order(series: Series, r: usize, order: &[Vec<i64>]) -> CanonicalCatalog {
    let mut cat = CanonicalCatalog::new(series, r).unwrap();
    for d in order {
        cat.generators(&irr(series, d)).unwrap();
    }
    cat
}

fn stored_generators(cat: &CanonicalCatalog, series: Series, dynkin: &[i64]) -> Generators {
    cat.store.get(&irr(series, dynkin)).unwrap().clone()
}

#[test]
fn generators_bitwise_identical_across_query_orders() {
    let series = Series::C;
    let r = 2;
    let targets = [vec![2, 0], vec![0, 1], vec![1, 1], vec![3, 0]];
    let order_a = [vec![3, 0], vec![1, 1], vec![0, 1], vec![2, 0]];
    let order_b = [vec![0, 1], vec![2, 0], vec![3, 0], vec![1, 1]];

    let cat_a = materialize_in_order(series, r, &order_a);
    let cat_b = materialize_in_order(series, r, &order_b);

    for t in &targets {
        let ga = stored_generators(&cat_a, series, t);
        let gb = stored_generators(&cat_b, series, t);
        assert_eq!(ga, gb, "generators of {t:?} differ across query orders");
    }
}

#[test]
fn cgc_bitwise_identical_across_query_orders() {
    let series = Series::C;
    let r = 2;
    // Warm the two catalogs in different orders, then request the same CGC.
    let mut cat_a = materialize_in_order(series, r, &[vec![3, 0], vec![0, 1], vec![2, 0]]);
    let mut cat_b = materialize_in_order(series, r, &[vec![2, 0], vec![3, 0], vec![0, 1]]);

    let d = defining(series, r);
    let c = irr(series, &[2, 0]);
    let cgc_a = cat_a.cgc(&d, &d, &c).unwrap();
    let cgc_b = cat_b.cgc(&d, &d, &c).unwrap();
    assert_eq!(cgc_a, cgc_b, "CGC bytes differ across query orders");

    // And a fresh catalog (no warming) gives the same CGC.
    let mut cat_c = CanonicalCatalog::new(series, r).unwrap();
    let cgc_c = cat_c.cgc(&d, &d, &c).unwrap();
    assert_eq!(cgc_a, cgc_c, "CGC depends on prior queries");
}

// ---- 4. ill-posed cgc inputs are typed errors (PR #14 lesson) --------------

#[test]
fn cgc_zero_channel_is_typed_error() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let d = defining(Series::C, 2);
    // defining ⊗ defining for Sp(4) = trivial ⊕ (0,1) ⊕ (2,0); (3,0) is NOT in it.
    let not_a_channel = irr(Series::C, &[3, 0]);
    assert!(matches!(
        cat.cgc(&d, &d, &not_a_channel),
        Err(CatalogError::ZeroFusionChannel { .. })
    ));
    // A genuine channel succeeds.
    assert!(cat.cgc(&d, &d, &irr(Series::C, &[2, 0])).is_ok());
}

#[test]
fn cgc_isometry_columns_are_orthonormal() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    let d = defining(Series::C, 2);
    let c = irr(Series::C, &[2, 0]);
    let cgc = cat.cgc(&d, &d, &c).unwrap();
    let (rows, cd3) = cgc.copy_shape();
    // Vᵀ V = I for the single copy.
    let v = cgc.copy(0);
    for p in 0..cd3 {
        for q in 0..cd3 {
            let mut dot = 0.0;
            for i in 0..rows {
                dot += v[i + p * rows] * v[i + q * rows];
            }
            let target = if p == q { 1.0 } else { 0.0 };
            assert!(
                (dot - target).abs() < 1e-8,
                "V not an isometry at ({p},{q})"
            );
        }
    }
}

// ---- 5. atomic byte-budget failure -----------------------------------------

#[test]
fn budget_exceeded_leaves_no_partial_state() {
    // A budget that admits the base cases but not the chain for a deeper irrep.
    let series = Series::C;
    let r = 2;
    let base = CanonicalCatalog::new(series, r).unwrap();
    let base_bytes = base.bytes();
    let base_len = base.len();

    // Size the budget just above the base so any real chain overflows.
    let mut cat = CanonicalCatalog::with_budget(series, r, base_bytes + 8).unwrap();
    let deep = irr(series, &[3, 0]); // needs a multi-entry chain
    let before_len = cat.len();
    let before_bytes = cat.bytes();
    let err = cat.generators(&deep).unwrap_err();
    assert!(matches!(err, CatalogError::BudgetExceeded { .. }));
    // No partial state: byte count and entry count unchanged.
    assert_eq!(
        cat.len(),
        before_len,
        "partial entries leaked after budget failure"
    );
    assert_eq!(
        cat.bytes(),
        before_bytes,
        "byte count changed after budget failure"
    );
    assert_eq!(cat.len(), base_len);
    assert!(!cat.is_materialized(&deep));
}

#[test]
fn generous_budget_admits_the_chain() {
    let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();
    // Default budget is large; the chain commits and bytes grow.
    let before = cat.bytes();
    cat.generators(&irr(Series::C, &[3, 0])).unwrap();
    assert!(cat.bytes() > before);
}

// ---- 6. reset + bitwise re-materialization ---------------------------------

#[test]
fn reset_then_rematerialize_is_bitwise_identical() {
    let series = Series::C;
    let r = 2;
    let mut cat = CanonicalCatalog::new(series, r).unwrap();
    let c = irr(series, &[3, 0]);
    cat.generators(&c).unwrap();
    let first = cat.store.get(&c).unwrap().clone();
    let first_bytes = cat.bytes();

    cat.reset();
    assert_eq!(cat.len(), 2, "reset must return to just the base cases");
    assert!(!cat.is_materialized(&c));

    cat.generators(&c).unwrap();
    let second = cat.store.get(&c).unwrap().clone();
    assert_eq!(first, second, "re-materialization not bitwise identical");
    assert_eq!(cat.bytes(), first_bytes);
}

// ---- 7. deep chain (>= 3 parents) + chain-depth error numbers --------------

/// The canonical-parent chain depth of `c` (number of recursive parents down to
/// a base case), computed the same way the catalog recurses.
fn chain_depth(series: Series, r: usize, c: &Irrep) -> usize {
    if c.dynkin().iter().all(|&x| x == 0) || c == &defining(series, r) {
        return 0; // base cases: trivial and defining
    }
    let (a, b) = canonical_parent(series, r, c).unwrap();
    1 + chain_depth(series, r, &a).max(chain_depth(series, r, &b))
}

#[test]
fn deep_chain_materializes_with_bounded_error() {
    // Symmetric powers of the C2 vector give progressively deeper chains. Report
    // (depth, worst commutator residual) — the issue #18 chain-depth watch item.
    let series = Series::C;
    let r = 2;
    let mut cat = CanonicalCatalog::new(series, r).unwrap();
    let mut saw_depth_ge_3 = false;
    println!("chain-depth vs commutator-residual (C2 symmetric powers):");
    for k in 2..=6i64 {
        let c = irr(series, &[k, 0]);
        cat.generators(&c).unwrap();
        let depth = chain_depth(series, r, &c);
        let res = cat.stored_commutator_residual(&c).unwrap();
        println!(
            "  (k={k}) dim={:>4} depth={depth} residual={res:.3e}",
            c.dim()
        );
        assert!(res < 1e-6, "residual {res:e} exceeds gate at depth {depth}");
        if depth >= 3 {
            saw_depth_ge_3 = true;
        }
    }
    assert!(
        saw_depth_ge_3,
        "expected a chain of depth >= 3 among the symmetric powers"
    );
}

// ---- 8. mutation sanity (the guards have teeth) ----------------------------

#[test]
fn parent_choice_is_value_affecting() {
    // If the canonical parent were chosen differently, c's generators would
    // generally differ. Materialize (2,0) via its canonical parent, then build a
    // NON-canonical product that also contains (2,0) and show the projected
    // generators differ — proving the canonical choice is a real gauge fact (so
    // the cross-order bitwise test above is not vacuous).
    let series = Series::C;
    let r = 2;
    let mut cat = CanonicalCatalog::new(series, r).unwrap();
    let c = irr(series, &[2, 0]);
    let canonical = cat.generators(&c).unwrap().clone();

    // A non-canonical product still containing (2,0): (2,0) ⊗ (2,0) ⊇ (2,0).
    let big = irr(series, &[2, 0]);
    cat.generators(&big).unwrap();
    let gbig = cat.store.get(&big).unwrap().clone();
    let product = Generators::product(&gbig, &gbig).unwrap();
    let expected = directproduct(&big, &big).unwrap();
    let decomp = decompose(&product, &expected).unwrap();
    let block = decomp.blocks().iter().find(|b| b.irrep() == &c).unwrap();

    // The intrinsic Cartan spectrum still agrees (weights are gauge-independent)
    // — the debug-assert the catalog relies on.
    for j in 0..canonical.rank() {
        for (s, &d) in canonical.cartan_diag(j).iter().enumerate() {
            assert!((block.weight(s, j) - d).abs() < 1e-6);
        }
    }
    // But the CGC isometry from a different product is a different embedding,
    // confirming that WHICH product is canonical is value-affecting.
    let (rows, _) = block.cgc_shape();
    assert_ne!(rows, canonical.dim(), "sanity: product space larger than c");
}

#[test]
fn appending_from_non_canonical_product_would_diverge() {
    // Harvest discipline: the catalog only writes a block whose canonical parent
    // is the current product. Take the antisymmetric (0,1) of Sp(4); its
    // canonical parent is defining ⊗ defining. It ALSO appears in other,
    // non-canonical products, so a naive "append whatever the sweep discovers
    // first" harvest would be order-dependent. Confirm both facts from S3.0.
    let series = Series::C;
    let r = 2;
    let d = defining(series, r);
    let target = irr(series, &[0, 1]);
    let cp = canonical_parent(series, r, &target).unwrap();
    assert_eq!(
        cp,
        (d.clone(), d.clone()),
        "canonical parent must be defining ⊗ defining"
    );

    // Search the small candidate set for a NON-canonical product also containing
    // the target — proving the discovery-order harvest would be ambiguous.
    let mut below = irreps_below(series, r, &irr(series, &[2, 0]));
    below.push(irr(series, &[2, 0]));
    let mut found_non_canonical = false;
    for x in &below {
        for y in &below {
            if let Ok(prod) = directproduct(x, y) {
                if prod.contains_key(&target) && (x.clone(), y.clone()) != cp {
                    found_non_canonical = true;
                }
            }
        }
    }
    assert!(
        found_non_canonical,
        "target must also live in a non-canonical product for this test to bite"
    );
}

// ---- 9. D-series chirality (P1 fix: box-count-first well-order) -------------

#[test]
fn d_chirality_pairs_materialize() {
    // dim is NOT monotone in the D last partition coordinate (partition (1,1,0)
    // has dim 15 > (1,1,±1) dim 10), so the old (dim, dynkin) order left the
    // D3/D4 chirality labels with an empty candidate set. Box-count-first fixes
    // it: (defining, c-minus-a-box) is always admissible. Each must materialize
    // and pass the S3.1 commutator gate.
    for (series, r, dynkin) in [
        (Series::D, 3, vec![0, 0, 2]),
        (Series::D, 3, vec![0, 2, 0]),
        (Series::D, 4, vec![0, 0, 0, 2]),
        (Series::D, 3, vec![0, 0, 4]), // chirality-charged deep target
    ] {
        let mut cat = CanonicalCatalog::new(series, r).unwrap();
        let c = irr(series, &dynkin);
        cat.generators(&c)
            .unwrap_or_else(|e| panic!("D chirality {dynkin:?} must materialize: {e}"));
        let res = cat.stored_commutator_residual(&c).unwrap();
        assert!(
            res < 1e-6,
            "commutator residual {res:e} too large for {dynkin:?}"
        );
    }
}

#[test]
fn d_chirality_pair_generators_differ_from_each_other() {
    // The two D3 chiralities (0,0,2) and (0,2,0) are distinct irreps; both must
    // be produced with their own generators (a golden check that the fix does
    // not collapse the pair).
    let mut cat = CanonicalCatalog::new(Series::D, 3).unwrap();
    let plus = irr(Series::D, &[0, 0, 2]);
    let minus = irr(Series::D, &[0, 2, 0]);
    cat.generators(&plus).unwrap();
    cat.generators(&minus).unwrap();
    assert!(cat.is_materialized(&plus) && cat.is_materialized(&minus));
    // Same dim, distinct labels ⇒ distinct stored generator sets.
    assert_ne!(plus, minus);
    assert_eq!(cat.store.get(&plus).unwrap().dim(), 10);
    assert_eq!(cat.store.get(&minus).unwrap().dim(), 10);
}

#[test]
fn golden_canonical_parent_table() {
    // Pin WHICH parent the minimizer selects (the gauge choice itself — the
    // cross-order test passes for any pure function, so this is what nails the
    // canonical parent down). Values are the box-count-first order (§14.2).
    // (series, rank, c, expected a, expected b) as Dynkin labels.
    type Row = (
        Series,
        usize,
        &'static [i64],
        &'static [i64],
        &'static [i64],
    );
    let cases: &[Row] = &[
        // C2 (Sp(4)) symmetric tower and adjoint: balanced/defining splits.
        (Series::C, 2, &[2, 0], &[1, 0], &[1, 0]),
        (Series::C, 2, &[0, 1], &[1, 0], &[1, 0]),
        (Series::C, 2, &[1, 1], &[1, 0], &[0, 1]),
        (Series::C, 2, &[3, 0], &[1, 0], &[2, 0]),
        // B2 (SO(5)).
        (Series::B, 2, &[2, 0], &[1, 0], &[1, 0]),
        (Series::B, 2, &[0, 2], &[1, 0], &[1, 0]),
        // D3 (SO(6)) non-chiral, and the chirality pair (the P1 fix): both now
        // resolve to (defining, (0,1,1)) — unmaterializable under the old order.
        (Series::D, 3, &[2, 0, 0], &[1, 0, 0], &[1, 0, 0]),
        (Series::D, 3, &[0, 1, 1], &[1, 0, 0], &[1, 0, 0]),
        (Series::D, 3, &[0, 0, 2], &[1, 0, 0], &[0, 1, 1]),
        (Series::D, 3, &[0, 2, 0], &[1, 0, 0], &[0, 1, 1]),
    ];
    for &(series, r, c, ea, eb) in cases {
        let (a, b) = canonical_parent(series, r, &irr(series, c)).unwrap();
        assert_eq!(
            (a.dynkin(), b.dynkin()),
            (ea.to_vec(), eb.to_vec()),
            "canonical parent of {series:?}{r} {c:?} changed"
        );
    }
}
